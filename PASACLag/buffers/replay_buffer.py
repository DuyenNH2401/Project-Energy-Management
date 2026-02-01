"""Replay Buffer for PASACLag.

Stores transitions with hybrid actions (discrete + continuous) and cost signals
for safe reinforcement learning with Lagrangian constraints.
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch


class ReplayBuffer:
    """Experience replay buffer for hybrid action spaces with cost signals.
    
    Stores transitions: (state, action, reward, cost, next_state, done)
    where action is a hybrid vector containing one-hot discrete + continuous.
    
    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Dimension of state space.
        continuous_dim: Dimension of continuous action space.
        discrete_dims: List of sizes for each discrete action.
        device: Device for tensor operations.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        continuous_dim: int,
        discrete_dims: List[int],
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.device = device
        
        # Calculate total action dimension
        self.total_discrete_dim = sum(discrete_dims)
        self.action_dim = self.total_discrete_dim + continuous_dim
        
        # Initialize buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state (state_dim,).
            action: Hybrid action (action_dim,) - [discrete_one_hot, continuous].
            reward: Reward received.
            cost: Cost signal for safety constraint.
            next_state: Next state (state_dim,).
            done: Whether episode ended.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_with_separate_actions(
        self,
        state: np.ndarray,
        discrete_actions: List[int],
        continuous_action: np.ndarray,
        reward: float,
        cost: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition with separate discrete and continuous actions.
        
        Converts discrete actions to one-hot encoding internally.
        
        Args:
            state: Current state (state_dim,).
            discrete_actions: List of discrete action indices.
            continuous_action: Continuous action (continuous_dim,).
            reward: Reward received.
            cost: Cost signal for safety constraint.
            next_state: Next state (state_dim,).
            done: Whether episode ended.
        """
        # Convert discrete actions to one-hot
        one_hot_list = []
        for i, (action_idx, dim) in enumerate(zip(discrete_actions, self.discrete_dims)):
            one_hot = np.zeros(dim, dtype=np.float32)
            one_hot[action_idx] = 1.0
            one_hot_list.append(one_hot)
        
        discrete_one_hot = np.concatenate(one_hot_list)
        hybrid_action = np.concatenate([discrete_one_hot, continuous_action])
        
        self.add(state, hybrid_action, reward, cost, next_state, done)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Dictionary containing:
                - states: (batch, state_dim)
                - actions: (batch, action_dim)
                - rewards: (batch, 1)
                - costs: (batch, 1)
                - next_states: (batch, state_dim)
                - dones: (batch, 1)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "states": torch.from_numpy(self.states[indices]).to(self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "costs": torch.from_numpy(self.costs[indices]).to(self.device),
            "next_states": torch.from_numpy(self.next_states[indices]).to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).to(self.device),
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size
    
    def get_action_dim(self) -> int:
        """Get total action dimension."""
        return self.action_dim
    
    def decode_action(
        self, hybrid_action: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """Decode hybrid action to separate discrete indices and continuous.
        
        Args:
            hybrid_action: Hybrid action vector (action_dim,).
            
        Returns:
            discrete_indices: List of discrete action indices.
            continuous_action: Continuous action array.
        """
        discrete_indices = []
        idx = 0
        
        for dim in self.discrete_dims:
            one_hot = hybrid_action[idx:idx + dim]
            discrete_indices.append(int(np.argmax(one_hot)))
            idx += dim
        
        continuous_action = hybrid_action[idx:]
        
        return discrete_indices, continuous_action
