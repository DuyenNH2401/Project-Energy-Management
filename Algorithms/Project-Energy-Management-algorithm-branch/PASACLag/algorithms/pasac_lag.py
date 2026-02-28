"""PASACLag: Parameterized Soft Actor-Critic with Lagrangian Safety Constraints.

This module implements SAC for hybrid action spaces (discrete + continuous)
with safety constraints enforced via the Lagrangian method.

Key Features:
1. Hybrid action handling with Gumbel-Softmax and Reparameterization
2. Double Q-learning for reward estimation
3. Cost critic for constraint violation estimation
4. Lagrangian multiplier for safe RL
5. Rescaled objective (1/(1+λ)) for stable gradient updates

Reference:
- SAC: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy RL"
- Lagrangian: Stooke et al., "Responsive Safety in RL via PID Lagrangian"
"""
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ..networks.hybrid_networks import HybridActor, HybridCritic, CostCritic
from ..buffers.replay_buffer import ReplayBuffer


class PASACLagAgent:
    """Lagrangian-based Parameterized Soft Actor-Critic Agent.
    
    Implements SAC for hybrid action spaces with safety constraints using
    the Lagrangian method and rescaled objective for stable training.
    
    Args:
        state_dim: Dimension of state space.
        continuous_dim: Dimension of continuous action space.
        discrete_dims: List of sizes for each discrete action.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        alpha: Entropy temperature (or initial value if auto_alpha=True).
        auto_alpha: Whether to automatically tune entropy temperature.
        cost_limit: Safety constraint limit (epsilon in E[sum c_t] <= epsilon).
        lambda_lr: Learning rate for Lagrangian multiplier.
        actor_lr: Learning rate for actor network.
        critic_lr: Learning rate for critic networks.
        cost_critic_lr: Learning rate for cost critic.
        alpha_lr: Learning rate for entropy temperature.
        hidden_dim: Hidden layer dimension for networks.
        gumbel_temperature: Temperature for Gumbel-Softmax.
        device: Device for tensor operations.
    """
    
    def __init__(
        self,
        state_dim: int,
        continuous_dim: int,
        discrete_dims: List[int],
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        cost_limit: float = 25.0,
        lambda_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        cost_critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        hidden_dim: int = 256,
        gumbel_temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.gamma = gamma
        self.tau = tau
        self.cost_limit = cost_limit
        self.gumbel_temperature = gumbel_temperature
        self.device = device
        self.auto_alpha = auto_alpha
        
        # Calculate action dimension
        self.total_discrete_dim = sum(discrete_dims)
        self.action_dim = self.total_discrete_dim + continuous_dim
        
        # Initialize Actor
        self.actor = HybridActor(
            state_dim, continuous_dim, discrete_dims, hidden_dim
        ).to(device)
        
        # Initialize Critics (Double Q-learning)
        self.critic = HybridCritic(state_dim, self.action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Initialize Cost Critic
        self.cost_critic = CostCritic(state_dim, self.action_dim, hidden_dim).to(device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        
        # Freeze target network parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False
        for param in self.cost_critic_target.parameters():
            param.requires_grad = False
        
        # Initialize Lagrangian multiplier (learnable, clipped to >= 0)
        self.log_lambda = torch.zeros(1, requires_grad=True, device=device)
        
        # Initialize entropy temperature
        if auto_alpha:
            # Target entropy: -dim(A) for continuous
            self.target_entropy = -continuous_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), device=device)
        
        # Initialize optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optimizer = Adam(self.cost_critic.parameters(), lr=cost_critic_lr)
        self.lambda_optimizer = Adam([self.log_lambda], lr=lambda_lr)
        
        # Training step counter
        self.total_steps = 0
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get current entropy temperature."""
        return self.log_alpha.exp()
    
    @property
    def lagrangian(self) -> torch.Tensor:
        """Get current Lagrangian multiplier (clipped to >= 0)."""
        return F.softplus(self.log_lambda)
    
    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> Tuple[List[int], np.ndarray]:
        """Select action for environment interaction.
        
        Args:
            state: Current state (state_dim,).
            evaluate: If True, use deterministic action.
            
        Returns:
            discrete_indices: List of discrete action indices.
            continuous_action: Continuous action array.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Sample from actor
            discrete_actions, continuous_actions, _, _ = self.actor.sample(
                state_tensor,
                temperature=self.gumbel_temperature,
                deterministic=evaluate,
            )
            
            # Decode discrete actions to indices
            discrete_indices = []
            idx = 0
            discrete_np = discrete_actions.cpu().numpy()[0]
            for dim in self.discrete_dims:
                one_hot = discrete_np[idx:idx + dim]
                discrete_indices.append(int(np.argmax(one_hot)))
                idx += dim
            
            continuous_action = continuous_actions.cpu().numpy()[0]
        
        return discrete_indices, continuous_action
    
    def select_action_with_hybrid(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> np.ndarray:
        """Select action and return as hybrid action vector.
        
        Args:
            state: Current state (state_dim,).
            evaluate: If True, use deterministic action.
            
        Returns:
            hybrid_action: Full action vector for replay buffer.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            _, _, _, hybrid_action = self.actor.sample(
                state_tensor,
                temperature=self.gumbel_temperature,
                deterministic=evaluate,
            )
            
        return hybrid_action.cpu().numpy()[0]
    
    def update_parameters(
        self,
        memory: ReplayBuffer,
        batch_size: int,
    ) -> Dict[str, float]:
        """Perform one update step for all networks.
        
        Implements the exact update rules from the specification:
        1. Critic loss (MSE with target networks)
        2. Cost critic loss (Bellman update for cost)
        3. Lagrangian multiplier update
        4. Actor loss (rescaled objective with 1/(1+λ))
        5. Temperature update (if auto_alpha)
        
        Args:
            memory: Replay buffer to sample from.
            batch_size: Number of transitions to sample.
            
        Returns:
            Dictionary of loss values for logging.
        """
        self.total_steps += 1
        
        # Sample batch
        batch = memory.sample(batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        # ===== Critic Loss (Double Q-learning) =====
        with torch.no_grad():
            # Sample next actions from current policy
            _, _, next_log_prob, next_hybrid_action = self.actor.sample(
                next_states, temperature=self.gumbel_temperature
            )
            
            # Compute target Q-values
            next_q1, next_q2 = self.critic_target(next_states, next_hybrid_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            
            # Target: y = r + γ * (min Q - α * log π)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # MSE loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Cost Critic Loss =====
        with torch.no_grad():
            # Sample next actions
            _, _, _, next_hybrid_action = self.actor.sample(
                next_states, temperature=self.gumbel_temperature
            )
            
            # Target cost Q-value: y_c = c + γ * Q_c(s', a')
            next_cost_q = self.cost_critic_target(next_states, next_hybrid_action)
            target_cost_q = costs + (1 - dones) * self.gamma * next_cost_q
        
        current_cost_q = self.cost_critic(states, actions)
        cost_critic_loss = F.mse_loss(current_cost_q, target_cost_q)
        
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()
        
        # ===== Lagrangian Multiplier Update =====
        # λ ← λ - η * (ε - Q_c(s, a))
        # Using gradient descent: minimize λ * (Q_c - ε)
        with torch.no_grad():
            mean_cost_q = current_cost_q.mean()
        
        lambda_loss = -self.lagrangian * (self.cost_limit - mean_cost_q)
        
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        # ===== Actor Loss (Rescaled Objective) =====
        # Sample new actions from current policy
        _, _, log_prob, new_hybrid_action = self.actor.sample(
            states, temperature=self.gumbel_temperature
        )
        
        # Q-values for new actions
        q1, q2 = self.critic(states, new_hybrid_action)
        min_q = torch.min(q1, q2)
        
        # Cost Q-value for new actions
        cost_q = self.cost_critic(states, new_hybrid_action)
        
        # Actor objective:
        # J = E[min Q - α * log π - λ * (Q_c - ε)]
        # We maximize this, so minimize the negative
        actor_loss = (
            self.alpha.detach() * log_prob
            - min_q
            + self.lagrangian.detach() * (cost_q - self.cost_limit)
        ).mean()
        
        # Apply rescaled gradient: 1 / (1 + λ)
        # This is done by scaling the loss
        rescale_factor = 1.0 / (1.0 + self.lagrangian.detach())
        actor_loss = rescale_factor * actor_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== Temperature (Alpha) Update =====
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            # Target: log_α * (H - H_target)
            alpha_loss = (
                -self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ===== Soft Update Target Networks =====
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.cost_critic, self.cost_critic_target)
        
        return {
            "critic_loss": critic_loss.item(),
            "cost_critic_loss": cost_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha": self.alpha.item(),
            "lambda": self.lagrangian.item(),
            "mean_q": min_q.mean().item(),
            "mean_cost_q": cost_q.mean().item(),
        }
    
    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module,
    ):
        """Soft update target network parameters."""
        for source_param, target_param in zip(
            source.parameters(), target.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "cost_critic": self.cost_critic.state_dict(),
                "cost_critic_target": self.cost_critic_target.state_dict(),
                "log_lambda": self.log_lambda,
                "log_alpha": self.log_alpha,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "cost_critic_optimizer": self.cost_critic_optimizer.state_dict(),
                "lambda_optimizer": self.lambda_optimizer.state_dict(),
                "total_steps": self.total_steps,
                "config": {
                    "state_dim": self.state_dim,
                    "continuous_dim": self.continuous_dim,
                    "discrete_dims": self.discrete_dims,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "cost_limit": self.cost_limit,
                },
            },
            path,
        )
    
    def load(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.cost_critic.load_state_dict(checkpoint["cost_critic"])
        self.cost_critic_target.load_state_dict(checkpoint["cost_critic_target"])
        
        self.log_lambda = checkpoint["log_lambda"]
        self.log_alpha = checkpoint["log_alpha"]
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.cost_critic_optimizer.load_state_dict(checkpoint["cost_critic_optimizer"])
        self.lambda_optimizer.load_state_dict(checkpoint["lambda_optimizer"])
        
        self.total_steps = checkpoint["total_steps"]
