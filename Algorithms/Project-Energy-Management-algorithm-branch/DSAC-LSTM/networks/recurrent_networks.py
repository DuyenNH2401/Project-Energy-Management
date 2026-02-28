"""Recurrent Neural Networks for DSAC-LSTM.

This module contains LSTM-based Actor and Critic networks designed for
Partially Observable Markov Decision Processes (POMDPs).

Key Design Decisions:
1. Input Enrichment: LSTM receives [obs, prev_action, prev_reward] to encode history.
2. Causal Q-Network: Current action is NOT passed to LSTM, only concatenated after.
3. Stochastic Q-Values: Outputs (mean, std) for distributional RL.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions import Normal
import math


def init_weights(module: nn.Module, gain: float = 1.0):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)


class RecurrentFeatureExtractor(nn.Module):
    """LSTM-based feature extractor for encoding observation history.
    
    This module processes sequences of enriched inputs [obs, prev_action, prev_reward]
    and outputs a sequence of hidden features representing the encoded history.
    
    Args:
        obs_dim: Dimension of observations.
        action_dim: Dimension of actions.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of LSTM layers.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input: [obs, prev_action, prev_reward]
        input_dim = obs_dim + action_dim + 1  # +1 for reward (scalar)
        
        # Optional input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Output dimension
        self.output_dim = hidden_dim
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=1.0))
    
    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the feature extractor.
        
        Args:
            obs: Observation tensor of shape (batch, seq_len, obs_dim).
            prev_action: Previous action tensor of shape (batch, seq_len, action_dim).
            prev_reward: Previous reward tensor of shape (batch, seq_len, 1).
            hidden: Optional initial hidden state (h_0, c_0).
            
        Returns:
            features: Encoded features of shape (batch, seq_len, hidden_dim).
            hidden: Updated hidden state (h_n, c_n).
        """
        batch_size, seq_len, _ = obs.shape
        
        # Concatenate input components: [obs, prev_action, prev_reward]
        lstm_input = torch.cat([obs, prev_action, prev_reward], dim=-1)
        
        # Project input
        lstm_input = self.input_projection(lstm_input)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)
        
        # LSTM forward
        features, new_hidden = self.lstm(lstm_input, hidden)
        
        return features, new_hidden
    
    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state to zeros.
        
        Args:
            batch_size: Batch size.
            device: Device to create tensors on.
            
        Returns:
            Tuple of (h_0, c_0) tensors.
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


class RecurrentPolicy(nn.Module):
    """Recurrent stochastic policy network (Actor).
    
    Outputs a Gaussian distribution over continuous actions.
    Uses tanh squashing for bounded action spaces.
    
    Args:
        obs_dim: Dimension of observations.
        action_dim: Dimension of actions.
        hidden_dim: Hidden layer dimension.
        lstm_hidden_dim: LSTM hidden state dimension.
        lstm_layers: Number of LSTM layers.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
    """
    
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extractor (LSTM backbone)
        self.feature_extractor = RecurrentFeatureExtractor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
        )
        
        # MLP head for policy
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layers with small weights
        init_weights(self.mean_layer, gain=0.01)
        init_weights(self.log_std_layer, gain=0.01)
    
    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass to get action distribution parameters.
        
        Args:
            obs: Observations (batch, seq_len, obs_dim).
            prev_action: Previous actions (batch, seq_len, action_dim).
            prev_reward: Previous rewards (batch, seq_len, 1).
            hidden: Optional LSTM hidden state.
            
        Returns:
            mean: Action mean (batch, seq_len, action_dim).
            log_std: Action log std (batch, seq_len, action_dim).
            hidden: Updated hidden state.
        """
        # Extract features from LSTM
        features, new_hidden = self.feature_extractor(
            obs, prev_action, prev_reward, hidden
        )
        
        # MLP forward
        x = self.mlp(features)
        
        # Get mean and log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, new_hidden
    
    def sample(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample actions from the policy.
        
        Args:
            obs: Observations.
            prev_action: Previous actions.
            prev_reward: Previous rewards.
            hidden: LSTM hidden state.
            deterministic: If True, return mean action.
            
        Returns:
            action: Sampled action (tanh squashed).
            log_prob: Log probability of the action.
            hidden: Updated hidden state.
        """
        mean, log_std, new_hidden = self.forward(obs, prev_action, prev_reward, hidden)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            # Log prob for deterministic action is not well-defined, return 0
            log_prob = torch.zeros_like(action[..., 0:1])
        else:
            # Reparameterization trick
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Sample with reparameterization
            action = torch.tanh(x_t)
            
            # Compute log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            # Enforcing action bounds: correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, new_hidden


class RecurrentQNetwork(nn.Module):
    """Recurrent Q-Network (Critic) for distributional RL.
    
    Outputs (mean, std) of Q-value distribution.
    Current action is concatenated AFTER LSTM encoding to preserve causality.
    
    Args:
        obs_dim: Dimension of observations.
        action_dim: Dimension of actions.
        hidden_dim: MLP hidden dimension.
        lstm_hidden_dim: LSTM hidden state dimension.
        lstm_layers: Number of LSTM layers.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # Feature extractor (LSTM backbone)
        self.feature_extractor = RecurrentFeatureExtractor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
        )
        
        # MLP head: takes [LSTM_features, current_action]
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output layer: (mean, std) of Q-distribution
        self.output_layer = nn.Linear(hidden_dim, 2)  # [mean, std]
        
        # Initialize
        self.apply(lambda m: init_weights(m, gain=1.0))
        init_weights(self.output_layer, gain=0.01)
    
    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        current_action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass to compute Q-value distribution.
        
        Args:
            obs: Observations (batch, seq_len, obs_dim).
            prev_action: Previous actions (batch, seq_len, action_dim).
            prev_reward: Previous rewards (batch, seq_len, 1).
            current_action: Current actions (batch, seq_len, action_dim).
            hidden: Optional LSTM hidden state.
            
        Returns:
            q_output: Q-value distribution params (batch, seq_len, 2) -> [mean, std].
            hidden: Updated hidden state.
        """
        # Extract features from LSTM (history encoding)
        features, new_hidden = self.feature_extractor(
            obs, prev_action, prev_reward, hidden
        )
        
        # Concatenate LSTM features with CURRENT action
        # This preserves causality: features encode past, action is present
        critic_input = torch.cat([features, current_action], dim=-1)
        
        # MLP forward
        x = self.mlp(critic_input)
        
        # Output: [mean, std]
        output = self.output_layer(x)
        mean = output[..., 0:1]
        std = torch.nn.functional.softplus(output[..., 1:2]) + 1e-4  # Ensure positive
        
        return torch.cat([mean, std], dim=-1), new_hidden
    
    def get_q_value(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        current_action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Get mean, std, and sampled Q-value.
        
        Returns:
            mean: Q-value mean.
            std: Q-value std.
            q_sample: Sampled Q-value using reparameterization.
            hidden: Updated hidden state.
        """
        output, new_hidden = self.forward(
            obs, prev_action, prev_reward, current_action, hidden
        )
        mean = output[..., 0:1]
        std = output[..., 1:2]
        
        # Sample Q-value using reparameterization
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)  # Clip for stability
        q_sample = mean + z * std
        
        return mean, std, q_sample, new_hidden
