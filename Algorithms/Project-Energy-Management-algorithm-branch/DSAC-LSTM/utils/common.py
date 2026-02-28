"""Common utilities for DSAC-LSTM training."""
import random
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_id: str, seed: int = 0) -> gym.Env:
    """Create a Gymnasium environment.
    
    Args:
        env_id: Environment ID (e.g., 'Pendulum-v1').
        seed: Random seed.
        
    Returns:
        Gymnasium environment instance.
    """
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


class RunningMeanStd:
    """Running mean and standard deviation for normalization.
    
    Used for reward normalization during training.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    @property
    def std(self):
        return np.sqrt(self.var)
