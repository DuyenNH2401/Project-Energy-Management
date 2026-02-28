"""Hybrid Neural Networks for PASACLag.

This module contains Actor and Critic networks for hybrid action spaces
(discrete + continuous) with safety constraints.

Key Features:
1. HybridActor: Outputs discrete logits + continuous (mean, log_std)
2. HybridCritic: Double Q-learning for reward estimation
3. CostCritic: Estimates expected cumulative cost for safety constraints
"""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedOneHotCategorical


def init_weights(module: nn.Module, gain: float = 1.0):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class HybridActor(nn.Module):
    """Hybrid Actor Network for mixed discrete-continuous action spaces.
    
    Outputs a flattened composite vector containing:
    - Logits for discrete actions (processed via Gumbel-Softmax for differentiability)
    - Mean and log_std for continuous actions (Reparameterization trick)
    
    Args:
        state_dim: Dimension of state space.
        continuous_dim: Dimension of continuous action space.
        discrete_dims: List of sizes for each discrete action (e.g., [4, 2] for gear and clutch).
        hidden_dim: Hidden layer dimension.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
    """
    
    def __init__(
        self,
        state_dim: int,
        continuous_dim: int,
        discrete_dims: List[int],
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Total discrete logits
        self.total_discrete_dim = sum(discrete_dims)
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Discrete action head (logits for each discrete action)
        self.discrete_head = nn.Linear(hidden_dim, self.total_discrete_dim)
        
        # Continuous action head (mean and log_std)
        self.continuous_mean = nn.Linear(hidden_dim, continuous_dim)
        self.continuous_log_std = nn.Linear(hidden_dim, continuous_dim)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=1.0))
        init_weights(self.continuous_mean, gain=0.01)
        init_weights(self.continuous_log_std, gain=0.01)
    
    def forward(
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass to get action distribution parameters.
        
        Args:
            state: State tensor (batch, state_dim).
            temperature: Gumbel-Softmax temperature for discrete actions.
            
        Returns:
            discrete_logits: Logits for discrete actions (batch, total_discrete_dim).
            continuous_mean: Mean for continuous actions (batch, continuous_dim).
            continuous_log_std: Log std for continuous actions (batch, continuous_dim).
        """
        features = self.feature_net(state)
        
        # Discrete logits
        discrete_logits = self.discrete_head(features)
        
        # Continuous parameters
        continuous_mean = self.continuous_mean(features)
        continuous_log_std = self.continuous_log_std(features)
        continuous_log_std = torch.clamp(
            continuous_log_std, self.log_std_min, self.log_std_max
        )
        
        return discrete_logits, continuous_mean, continuous_log_std
    
    def sample(
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hybrid actions from the policy.
        
        Args:
            state: State tensor (batch, state_dim).
            temperature: Gumbel-Softmax temperature.
            deterministic: If True, use argmax for discrete and mean for continuous.
            
        Returns:
            discrete_actions: One-hot encoded discrete actions (batch, total_discrete_dim).
            continuous_actions: Continuous actions after tanh squashing (batch, continuous_dim).
            log_prob: Combined log probability (batch, 1).
            hybrid_action: Concatenated action for critic input (batch, total_dim).
        """
        discrete_logits, cont_mean, cont_log_std = self.forward(state, temperature)
        
        # Sample discrete actions using Gumbel-Softmax
        discrete_actions_list = []
        discrete_log_probs = []
        
        idx = 0
        for dim in self.discrete_dims:
            logits = discrete_logits[:, idx:idx + dim]
            
            if deterministic:
                # Hard argmax
                discrete_action = F.one_hot(
                    logits.argmax(dim=-1), num_classes=dim
                ).float()
                log_prob = F.log_softmax(logits, dim=-1)
                discrete_log_probs.append(
                    (discrete_action * log_prob).sum(dim=-1, keepdim=True)
                )
            else:
                # Gumbel-Softmax (differentiable)
                dist = RelaxedOneHotCategorical(
                    temperature=torch.tensor(temperature, device=state.device),
                    logits=logits
                )
                discrete_action = dist.rsample()
                
                # Straight-through estimator: use hard one-hot in forward, soft in backward
                hard_action = F.one_hot(
                    discrete_action.argmax(dim=-1), num_classes=dim
                ).float()
                discrete_action = hard_action - discrete_action.detach() + discrete_action
                
                # Log probability using categorical distribution
                log_prob = F.log_softmax(logits, dim=-1)
                discrete_log_probs.append(
                    (discrete_action * log_prob).sum(dim=-1, keepdim=True)
                )
            
            discrete_actions_list.append(discrete_action)
            idx += dim
        
        discrete_actions = torch.cat(discrete_actions_list, dim=-1)
        discrete_log_prob = torch.cat(discrete_log_probs, dim=-1).sum(dim=-1, keepdim=True)
        
        # Sample continuous actions using Reparameterization trick
        cont_std = cont_log_std.exp()
        
        if deterministic:
            continuous_actions = torch.tanh(cont_mean)
            cont_log_prob = torch.zeros_like(discrete_log_prob)
        else:
            dist = Normal(cont_mean, cont_std)
            x_t = dist.rsample()  # Reparameterization trick
            continuous_actions = torch.tanh(x_t)
            
            # Log probability with tanh squashing correction
            cont_log_prob = dist.log_prob(x_t)
            cont_log_prob -= torch.log(1 - continuous_actions.pow(2) + 1e-6)
            cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)
        
        # Combined log probability
        log_prob = discrete_log_prob + cont_log_prob
        
        # Concatenate for critic input: [discrete_one_hot, continuous]
        hybrid_action = torch.cat([discrete_actions, continuous_actions], dim=-1)
        
        return discrete_actions, continuous_actions, log_prob, hybrid_action
    
    def get_action_dim(self) -> int:
        """Get total action dimension for critic input."""
        return self.total_discrete_dim + self.continuous_dim


class HybridCritic(nn.Module):
    """Double Q-Network for hybrid action spaces.
    
    Takes state and hybrid action (one-hot discrete + continuous) as input.
    Outputs scalar Q-value.
    
    Args:
        state_dim: Dimension of state space.
        action_dim: Total action dimension (discrete one-hot + continuous).
        hidden_dim: Hidden layer dimension.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.apply(lambda m: init_weights(m, gain=1.0))
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute Q-values.
        
        Args:
            state: State tensor (batch, state_dim).
            action: Hybrid action tensor (batch, action_dim).
            
        Returns:
            q1: Q-value from network 1 (batch, 1).
            q2: Q-value from network 2 (batch, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through Q1 only (for policy update)."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class CostCritic(nn.Module):
    """Cost Q-Network for safety constraint estimation.
    
    Estimates expected cumulative cost Q_c(s, a) = E[sum of future costs].
    Used with Lagrangian method to enforce safety constraints.
    
    Args:
        state_dim: Dimension of state space.
        action_dim: Total action dimension (discrete one-hot + continuous).
        hidden_dim: Hidden layer dimension.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.apply(lambda m: init_weights(m, gain=1.0))
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to compute cost Q-value.
        
        Args:
            state: State tensor (batch, state_dim).
            action: Hybrid action tensor (batch, action_dim).
            
        Returns:
            q_cost: Cost Q-value (batch, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
