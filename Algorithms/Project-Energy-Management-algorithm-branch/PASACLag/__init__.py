"""PASACLag: Parameterized Soft Actor-Critic with Lagrangian Safety Constraints.

A reinforcement learning algorithm for hybrid action spaces (discrete + continuous)
with safety constraints for Hybrid Electric Vehicle energy management.
"""
from .algorithms.pasac_lag import PASACLagAgent
from .buffers.replay_buffer import ReplayBuffer
from .networks.hybrid_networks import HybridActor, HybridCritic, CostCritic

__all__ = [
    "PASACLagAgent",
    "ReplayBuffer",
    "HybridActor",
    "HybridCritic",
    "CostCritic",
]

__version__ = "1.0.0"
