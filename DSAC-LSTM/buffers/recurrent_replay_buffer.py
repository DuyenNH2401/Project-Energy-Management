"""Recurrent Replay Buffer for DSAC-LSTM.

This module implements an episode-based replay buffer that stores complete
episodes and samples fixed-length sequences for training recurrent networks.

Key Features:
1. Episode-based storage: Stores complete episodes for proper sequence handling.
2. Sequence sampling: Samples contiguous sequences of fixed length.
3. Input enrichment: Automatically computes prev_action and prev_reward.
4. Padding and masking: Handles episodes shorter than sequence length.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class Episode:
    """Container for a single episode's data."""
    
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_observations: List[np.ndarray] = []
        self.dones: List[bool] = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a transition to the episode."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert episode data to numpy arrays."""
        return {
            "obs": np.array(self.observations, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "next_obs": np.array(self.next_observations, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }


class RecurrentReplayBuffer:
    """Episode-based replay buffer for recurrent policy learning.
    
    Stores complete episodes and samples fixed-length sequences.
    Automatically computes prev_action and prev_reward for input enrichment.
    
    Args:
        max_episodes: Maximum number of episodes to store.
        obs_dim: Dimension of observations.
        action_dim: Dimension of actions.
        sequence_length: Length of sequences to sample.
        device: Device to move tensors to.
    """
    
    def __init__(
        self,
        max_episodes: int = 1000,
        obs_dim: int = 1,
        action_dim: int = 1,
        sequence_length: int = 80,
        device: str = "cpu",
    ):
        self.max_episodes = max_episodes
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        
        # Storage
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.current_episode: Optional[Episode] = None
        
        # Track total transitions for statistics
        self.total_transitions = 0
    
    def start_episode(self):
        """Start a new episode."""
        self.current_episode = Episode()
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a transition to the current episode.
        
        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.
        """
        if self.current_episode is None:
            self.start_episode()
        
        self.current_episode.add(obs, action, reward, next_obs, done)
        self.total_transitions += 1
        
        # If episode is done, finalize it
        if done:
            self.end_episode()
    
    def end_episode(self):
        """Finalize the current episode and add to storage."""
        if self.current_episode is None or len(self.current_episode) == 0:
            return
        
        # Convert to numpy and store
        episode_data = self.current_episode.to_numpy()
        self.episodes.append(episode_data)
        
        # Remove oldest episode if buffer is full
        if len(self.episodes) > self.max_episodes:
            removed = self.episodes.pop(0)
            self.total_transitions -= len(removed["obs"])
        
        self.current_episode = None
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of sequences.
        
        Args:
            batch_size: Number of sequences to sample.
            
        Returns:
            Dictionary containing batched tensors:
                - obs: (batch, seq_len, obs_dim)
                - actions: (batch, seq_len, action_dim)
                - rewards: (batch, seq_len, 1)
                - next_obs: (batch, seq_len, obs_dim)
                - dones: (batch, seq_len, 1)
                - prev_actions: (batch, seq_len, action_dim)
                - prev_rewards: (batch, seq_len, 1)
                - masks: (batch, seq_len, 1) validity mask
        """
        # Filter episodes long enough for sampling
        valid_episodes = [
            ep for ep in self.episodes if len(ep["obs"]) >= self.sequence_length
        ]
        
        if len(valid_episodes) < batch_size:
            # Allow sampling with replacement if not enough valid episodes
            valid_episodes = self.episodes
        
        batch_data = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "prev_actions": [],
            "prev_rewards": [],
            "masks": [],
        }
        
        for _ in range(batch_size):
            # Randomly select an episode
            ep_idx = np.random.randint(len(valid_episodes))
            episode = valid_episodes[ep_idx]
            ep_len = len(episode["obs"])
            
            # Sample a starting index for the sequence
            if ep_len >= self.sequence_length:
                start_idx = np.random.randint(0, ep_len - self.sequence_length + 1)
                seq_len = self.sequence_length
                pad_len = 0
            else:
                # Episode shorter than sequence_length: use entire episode + padding
                start_idx = 0
                seq_len = ep_len
                pad_len = self.sequence_length - ep_len
            
            # Extract sequence
            obs = episode["obs"][start_idx : start_idx + seq_len]
            actions = episode["actions"][start_idx : start_idx + seq_len]
            rewards = episode["rewards"][start_idx : start_idx + seq_len]
            next_obs = episode["next_obs"][start_idx : start_idx + seq_len]
            dones = episode["dones"][start_idx : start_idx + seq_len]
            
            # Compute prev_actions and prev_rewards
            # At t=0 within sequence: use start_idx-1 from episode or zeros
            if start_idx > 0:
                first_prev_action = episode["actions"][start_idx - 1]
                first_prev_reward = episode["rewards"][start_idx - 1]
            else:
                first_prev_action = np.zeros(self.action_dim, dtype=np.float32)
                first_prev_reward = 0.0
            
            # Prev actions: [a_{t-1} for t in sequence]
            prev_actions = np.concatenate(
                [[first_prev_action], actions[:-1]], axis=0
            )
            prev_rewards = np.concatenate(
                [[first_prev_reward], rewards[:-1]], axis=0
            )
            
            # Create validity mask (1 for valid, 0 for padding)
            masks = np.ones((seq_len, 1), dtype=np.float32)
            
            # Pad if necessary
            if pad_len > 0:
                obs = np.pad(obs, ((0, pad_len), (0, 0)), mode="constant")
                actions = np.pad(actions, ((0, pad_len), (0, 0)), mode="constant")
                rewards = np.pad(rewards, (0, pad_len), mode="constant")
                next_obs = np.pad(next_obs, ((0, pad_len), (0, 0)), mode="constant")
                dones = np.pad(dones, (0, pad_len), mode="constant", constant_values=1)
                prev_actions = np.pad(prev_actions, ((0, pad_len), (0, 0)), mode="constant")
                prev_rewards = np.pad(prev_rewards, (0, pad_len), mode="constant")
                masks = np.pad(masks, ((0, pad_len), (0, 0)), mode="constant")
            
            # Reshape for correct dimensions
            rewards = rewards.reshape(-1, 1)
            prev_rewards = prev_rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)
            
            batch_data["obs"].append(obs)
            batch_data["actions"].append(actions)
            batch_data["rewards"].append(rewards)
            batch_data["next_obs"].append(next_obs)
            batch_data["dones"].append(dones)
            batch_data["prev_actions"].append(prev_actions)
            batch_data["prev_rewards"].append(prev_rewards)
            batch_data["masks"].append(masks)
        
        # Stack and convert to tensors
        result = {}
        for key, value in batch_data.items():
            tensor = torch.tensor(np.stack(value), dtype=torch.float32)
            result[key] = tensor.to(self.device)
        
        return result
    
    def __len__(self) -> int:
        """Return number of stored episodes."""
        return len(self.episodes)
    
    def num_transitions(self) -> int:
        """Return total number of transitions stored."""
        return self.total_transitions
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough data to sample."""
        return len(self.episodes) >= batch_size
