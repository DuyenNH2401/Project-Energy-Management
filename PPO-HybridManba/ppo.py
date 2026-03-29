import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import HybridActorCritic


class RolloutBuffer:
    """
    Stores rollout data for PPO updates.
    Handles sequence assembly for the Hybrid Mamba model.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_seqs = []  # stores the full state sequence context for each step
    
    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.state_seqs.clear()


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 d_model=64, seq_len=16, d_state=16, d_conv=4, expand=2,
                 n_mamba_layers=3, n_heads=4, n_kv_heads=2, device='cpu'):
        """
        PPO agent with Hybrid Mamba Actor-Critic.
        
        Args:
            state_dim: Observation space dimension
            action_dim: Action space dimension (discrete)
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            K_epochs: Number of PPO update epochs
            eps_clip: PPO clipping parameter
            d_model: Model hidden dimension
            seq_len: Sequence length (context window) for Mamba
            d_state: SSM state dimension
            d_conv: Convolution kernel size for Mamba
            expand: Mamba expansion factor
            n_mamba_layers: Number of Mamba blocks
            n_heads: Number of GQA query heads
            n_kv_heads: Number of GQA key/value heads
            device: Compute device
        """
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        self.buffer = RolloutBuffer()

        # Sequence buffer for maintaining context during action selection
        self._episode_states = []

        # Initialize Hybrid Actor-Critic
        self.policy = HybridActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_mamba_layers=n_mamba_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        ).to(device)

        self.optimizer = optim.Adam([
            {'params': self.policy.encoder.parameters(), 'lr': lr_actor},
            {'params': self.policy.mamba.parameters(), 'lr': lr_actor},
            {'params': self.policy.attn.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])

        # Old policy for clipping objective
        self.policy_old = HybridActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_mamba_layers=n_mamba_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def _build_state_seq(self, state):
        """
        Build a state sequence from the episode history for temporal context.
        Pads with zeros if the episode is shorter than seq_len.
        
        Args:
            state: Current state (numpy array or tensor)
        Returns:
            state_seq: (1, seq_len, state_dim) tensor
        """
        if isinstance(state, np.ndarray):
            state_t = torch.FloatTensor(state).to(self.device)
        else:
            state_t = state.to(self.device)

        self._episode_states.append(state_t)

        # Get the last seq_len states
        recent = self._episode_states[-self.seq_len:]

        # Pad if needed
        if len(recent) < self.seq_len:
            pad_len = self.seq_len - len(recent)
            padding = [torch.zeros_like(recent[0]) for _ in range(pad_len)]
            recent = padding + recent

        state_seq = torch.stack(recent, dim=0).unsqueeze(0)  # (1, seq_len, state_dim)
        return state_seq

    def select_action(self, state):
        """
        Select action using old policy with sequence context.
        """
        with torch.no_grad():
            state_seq = self._build_state_seq(state)
            action, action_logprob = self.policy_old.act(state_seq)
        
        self.buffer.states.append(torch.FloatTensor(state).to(self.device))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        # Store a clone of the full sequence context
        self.buffer.state_seqs.append(state_seq.squeeze(0).clone())  # (seq_len, state_dim)

        return action.item()

    def reset_episode(self):
        """Reset the episode state buffer (call on episode boundaries)."""
        self._episode_states.clear()

    def update(self):
        """PPO update with sequence-based evaluation."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_state_seqs = torch.stack(self.buffer.state_seqs, dim=0).detach().to(self.device)  # (N, seq_len, state_dim)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values using state sequences
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_state_seqs, old_actions)

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()
