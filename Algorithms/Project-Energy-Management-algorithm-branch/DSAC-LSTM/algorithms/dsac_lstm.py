"""Distributional Soft Actor-Critic with LSTM (DSAC-LSTM).

This module implements DSAC combined with LSTM networks for learning in
Partially Observable environments (POMDPs).

Key Features:
1. Recurrent networks for history encoding.
2. Distributional critic with (mean, std) output.
3. Burn-in strategy for stable LSTM training.
4. Automatic temperature (alpha) adjustment.

Based on DSAC-v2 paper: https://arxiv.org/abs/2310.05858
"""
from copy import deepcopy
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import huber_loss
from torch.optim import Adam

from my_code.networks.recurrent_networks import RecurrentPolicy, RecurrentQNetwork


class ApproxContainer(nn.Module):
    """Container for all approximation networks in DSAC-LSTM.
    
    Contains recurrent policy and two Q-networks with their targets.
    
    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: MLP hidden dimension.
        lstm_hidden_dim: LSTM hidden dimension.
        lstm_layers: Number of LSTM layers.
        policy_lr: Policy learning rate.
        value_lr: Value learning rate.
        alpha_lr: Alpha (temperature) learning rate.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        policy_lr: float = 3e-4,
        value_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
    ):
        super().__init__()
        
        # Create Q-networks (Critic)
        self.q1 = RecurrentQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
        )
        self.q2 = RecurrentQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
        )
        
        # Target Q-networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        
        # Create Policy (Actor)
        self.policy = RecurrentPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
        )
        
        # Target Policy
        self.policy_target = deepcopy(self.policy)
        
        # Freeze target network gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False
        
        # Entropy coefficient (log for numerical stability)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # Optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=value_lr)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=value_lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)


class DSAC_LSTM:
    """Distributional Soft Actor-Critic with LSTM.
    
    Implements recurrent SAC with distributional critic and burn-in strategy.
    
    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        auto_alpha: Whether to automatically adjust temperature.
        alpha: Initial/fixed temperature value.
        target_entropy: Target entropy for auto alpha (default: -action_dim).
        delay_update: Policy update delay (updates per critic update).
        burn_in_steps: Number of steps to skip at sequence start for loss.
        hidden_dim: MLP hidden dimension.
        lstm_hidden_dim: LSTM hidden dimension.
        lstm_layers: Number of LSTM layers.
        policy_lr: Policy learning rate.
        value_lr: Value learning rate.
        alpha_lr: Alpha learning rate.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        delay_update: int = 2,
        burn_in_steps: int = 20,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        policy_lr: float = 3e-4,
        value_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        
        # Create networks
        self.networks = ApproxContainer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            policy_lr=policy_lr,
            value_lr=value_lr,
            alpha_lr=alpha_lr,
        ).to(self.device)
        
        # Algorithm parameters
        self.gamma = gamma
        self.tau = tau
        self.tau_b = tau  # For moving average of std
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.delay_update = delay_update
        self.burn_in_steps = burn_in_steps
        self.action_dim = action_dim
        
        # Moving averages for distributional loss
        self.mean_std1 = -1.0
        self.mean_std2 = -1.0
    
    def get_alpha(self, requires_grad: bool = False) -> float:
        """Get current temperature value."""
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            return alpha if requires_grad else alpha.item()
        return self.alpha
    
    def select_action(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Select action from policy.
        
        Args:
            obs: Observation tensor (1, 1, obs_dim) for single step.
            prev_action: Previous action (1, 1, action_dim).
            prev_reward: Previous reward (1, 1, 1).
            hidden: Policy LSTM hidden state.
            deterministic: Whether to use deterministic action.
            
        Returns:
            action: Selected action.
            log_prob: Log probability of action.
            hidden: Updated hidden state.
        """
        with torch.no_grad():
            action, log_prob, new_hidden = self.networks.policy.sample(
                obs, prev_action, prev_reward, hidden, deterministic
            )
        return action, log_prob, new_hidden
    
    def update(self, data: Dict[str, torch.Tensor], iteration: int) -> Dict[str, float]:
        """Perform one update step.
        
        Args:
            data: Batch of sequence data from replay buffer.
            iteration: Current iteration number.
            
        Returns:
            Dictionary of logging metrics.
        """
        start_time = time.time()
        
        # Unpack data
        obs = data["obs"]  # (batch, seq_len, obs_dim)
        actions = data["actions"]  # (batch, seq_len, action_dim)
        rewards = data["rewards"]  # (batch, seq_len, 1)
        next_obs = data["next_obs"]
        dones = data["dones"]
        prev_actions = data["prev_actions"]
        prev_rewards = data["prev_rewards"]
        masks = data["masks"]  # (batch, seq_len, 1)
        
        batch_size, seq_len, _ = obs.shape
        burn_in = self.burn_in_steps
        valid_len = seq_len - burn_in
        
        # === Compute New Actions for Current Observations ===
        new_actions, new_log_probs, _ = self.networks.policy.sample(
            obs, prev_actions, prev_rewards, hidden=None
        )
        
        # === Critic Loss ===
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        
        critic_loss, q1_mean, q2_mean, std1, std2 = self._compute_critic_loss(
            obs, next_obs, actions, rewards, dones,
            prev_actions, prev_rewards, new_actions, new_log_probs,
            masks, burn_in
        )
        
        critic_loss.backward()
        
        # Freeze critics for policy update
        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        
        # === Policy Loss ===
        self.networks.policy_optimizer.zero_grad()
        policy_loss, entropy = self._compute_policy_loss(
            obs, prev_actions, prev_rewards, masks, burn_in
        )
        policy_loss.backward()
        
        # Unfreeze critics
        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        
        # === Alpha Loss ===
        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            alpha_loss = self._compute_alpha_loss(
                obs, prev_actions, prev_rewards, masks, burn_in
            )
            alpha_loss.backward()
        
        # === Apply Updates ===
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()
        
        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()
            
            if self.auto_alpha:
                self.networks.alpha_optimizer.step()
            
            # Soft update target networks
            self._soft_update_targets()
        
        # Logging
        tb_info = {
            "loss/critic": critic_loss.item(),
            "loss/policy": policy_loss.item(),
            "q/q1_mean": q1_mean,
            "q/q2_mean": q2_mean,
            "q/std1": std1,
            "q/std2": std2,
            "policy/entropy": entropy.item(),
            "policy/alpha": self.get_alpha(),
            "time/update_ms": (time.time() - start_time) * 1000,
        }
        
        return tb_info
    
    def _compute_critic_loss(
        self,
        obs, next_obs, actions, rewards, dones,
        prev_actions, prev_rewards, new_actions, new_log_probs,
        masks, burn_in
    ) -> Tuple[torch.Tensor, float, float, float, float]:
        """Compute distributional critic loss with burn-in."""
        batch_size, seq_len, _ = obs.shape
        
        # Get Q-values for current state-action pairs
        q1_out, _ = self.networks.q1(obs, prev_actions, prev_rewards, actions)
        q2_out, _ = self.networks.q2(obs, prev_actions, prev_rewards, actions)
        
        q1_mean_all = q1_out[..., 0:1]
        q1_std_all = q1_out[..., 1:2]
        q2_mean_all = q2_out[..., 0:1]
        q2_std_all = q2_out[..., 1:2]
        
        # === Compute Target Q-values ===
        with torch.no_grad():
            # Get next actions from target policy
            next_prev_actions = actions  # Previous action for next step is current action
            next_prev_rewards = rewards
            
            next_actions, next_log_probs, _ = self.networks.policy_target.sample(
                next_obs, next_prev_actions, next_prev_rewards, hidden=None
            )
            
            # Get target Q-values
            q1_next_out, _ = self.networks.q1_target(
                next_obs, next_prev_actions, next_prev_rewards, next_actions
            )
            q2_next_out, _ = self.networks.q2_target(
                next_obs, next_prev_actions, next_prev_rewards, next_actions
            )
            
            q1_next_mean = q1_next_out[..., 0:1]
            q2_next_mean = q2_next_out[..., 0:1]
            q1_next_std = q1_next_out[..., 1:2]
            q2_next_std = q2_next_out[..., 1:2]
            
            # Sample from distributional Q
            normal = Normal(torch.zeros_like(q1_next_mean), torch.ones_like(q1_next_std))
            z = torch.clamp(normal.sample(), -3, 3)
            q1_next_sample = q1_next_mean + z * q1_next_std
            q2_next_sample = q2_next_mean + z * q2_next_std
            
            # Min Q target
            q_next = torch.min(q1_next_mean, q2_next_mean)
            q_next_sample = torch.where(
                q1_next_mean < q2_next_mean, q1_next_sample, q2_next_sample
            )
            
            # Compute target
            alpha = self.get_alpha()
            target_q = rewards + (1 - dones) * self.gamma * (q_next - alpha * next_log_probs)
            target_q_sample = rewards + (1 - dones) * self.gamma * (q_next_sample - alpha * next_log_probs)
        
        # Update moving average of std
        if self.mean_std1 < 0:
            self.mean_std1 = q1_std_all[:, burn_in:].detach().mean()
            self.mean_std2 = q2_std_all[:, burn_in:].detach().mean()
        else:
            self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * q1_std_all[:, burn_in:].detach().mean()
            self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * q2_std_all[:, burn_in:].detach().mean()
        
        # Compute loss for each Q-network
        q1_std_detach = torch.clamp(q1_std_all, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std_all, min=0.).detach()
        bias = 0.1
        
        ratio1 = (self.mean_std1.pow(2) / (q1_std_detach.pow(2) + bias)).clamp(0.1, 10)
        ratio2 = (self.mean_std2.pow(2) / (q2_std_detach.pow(2) + bias)).clamp(0.1, 10)
        
        # Bounded target for variance
        td_bound1 = 3 * q1_std_detach
        td_bound2 = 3 * q2_std_detach
        diff1 = torch.clamp(target_q_sample - q1_mean_all.detach(), -td_bound1, td_bound1)
        diff2 = torch.clamp(target_q_sample - q2_mean_all.detach(), -td_bound2, td_bound2)
        target_q_bound1 = q1_mean_all.detach() + diff1
        target_q_bound2 = q2_mean_all.detach() + diff2
        
        # Huber loss for mean + variance term
        q1_loss = ratio1 * (
            huber_loss(q1_mean_all, target_q, delta=50, reduction='none') +
            q1_std_all * (q1_std_detach.pow(2) - huber_loss(q1_mean_all.detach(), target_q_bound1, delta=50, reduction='none')) / (q1_std_detach + bias)
        )
        q2_loss = ratio2 * (
            huber_loss(q2_mean_all, target_q, delta=50, reduction='none') +
            q2_std_all * (q2_std_detach.pow(2) - huber_loss(q2_mean_all.detach(), target_q_bound2, delta=50, reduction='none')) / (q2_std_detach + bias)
        )
        
        # Apply burn-in: only compute loss after burn_in steps
        valid_mask = masks[:, burn_in:]
        q1_loss_valid = q1_loss[:, burn_in:] * valid_mask
        q2_loss_valid = q2_loss[:, burn_in:] * valid_mask
        
        total_loss = (q1_loss_valid.sum() + q2_loss_valid.sum()) / valid_mask.sum()
        
        # Metrics
        q1_mean_metric = q1_mean_all[:, burn_in:].detach().mean().item()
        q2_mean_metric = q2_mean_all[:, burn_in:].detach().mean().item()
        std1_metric = q1_std_all[:, burn_in:].detach().mean().item()
        std2_metric = q2_std_all[:, burn_in:].detach().mean().item()
        
        return total_loss, q1_mean_metric, q2_mean_metric, std1_metric, std2_metric
    
    def _compute_policy_loss(
        self, obs, prev_actions, prev_rewards, masks, burn_in
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy loss with entropy regularization."""
        # Sample new actions
        new_actions, new_log_probs, _ = self.networks.policy.sample(
            obs, prev_actions, prev_rewards, hidden=None
        )
        
        # Get Q-values for new actions
        q1_out, _ = self.networks.q1(obs, prev_actions, prev_rewards, new_actions)
        q2_out, _ = self.networks.q2(obs, prev_actions, prev_rewards, new_actions)
        
        q1_mean = q1_out[..., 0:1]
        q2_mean = q2_out[..., 0:1]
        q_min = torch.min(q1_mean, q2_mean)
        
        # Policy loss: maximize Q - alpha * log_prob
        alpha = self.get_alpha(requires_grad=False)
        policy_loss = alpha * new_log_probs - q_min
        
        # Apply burn-in
        valid_mask = masks[:, burn_in:]
        policy_loss_valid = policy_loss[:, burn_in:] * valid_mask
        
        total_loss = policy_loss_valid.sum() / valid_mask.sum()
        entropy = -new_log_probs[:, burn_in:].detach().mean()
        
        return total_loss, entropy
    
    def _compute_alpha_loss(
        self, obs, prev_actions, prev_rewards, masks, burn_in
    ) -> torch.Tensor:
        """Compute temperature (alpha) loss."""
        with torch.no_grad():
            _, new_log_probs, _ = self.networks.policy.sample(
                obs, prev_actions, prev_rewards, hidden=None
            )
        
        # Alpha loss
        valid_mask = masks[:, burn_in:]
        log_probs_valid = new_log_probs[:, burn_in:]
        
        alpha_loss = -(
            self.networks.log_alpha * 
            (log_probs_valid + self.target_entropy) * valid_mask
        ).sum() / valid_mask.sum()
        
        return alpha_loss
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        polyak = 1 - self.tau
        
        with torch.no_grad():
            for p, p_targ in zip(self.networks.q1.parameters(), self.networks.q1_target.parameters()):
                p_targ.data.mul_(polyak).add_((1 - polyak) * p.data)
            
            for p, p_targ in zip(self.networks.q2.parameters(), self.networks.q2_target.parameters()):
                p_targ.data.mul_(polyak).add_((1 - polyak) * p.data)
            
            for p, p_targ in zip(self.networks.policy.parameters(), self.networks.policy_target.parameters()):
                p_targ.data.mul_(polyak).add_((1 - polyak) * p.data)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "networks": self.networks.state_dict(),
            "mean_std1": self.mean_std1,
            "mean_std2": self.mean_std2,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.networks.load_state_dict(checkpoint["networks"])
        self.mean_std1 = checkpoint.get("mean_std1", -1.0)
        self.mean_std2 = checkpoint.get("mean_std2", -1.0)
