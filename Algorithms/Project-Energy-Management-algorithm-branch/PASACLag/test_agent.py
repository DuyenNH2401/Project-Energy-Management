"""Quick test script for PASACLag algorithm."""
import numpy as np
from .algorithms.pasac_lag import PASACLagAgent
from .buffers.replay_buffer import ReplayBuffer

# Create agent
agent = PASACLagAgent(
    state_dim=10,
    continuous_dim=1,
    discrete_dims=[4, 2],
    cost_limit=25.0,
)
print("Agent created successfully")

# Create buffer
buffer = ReplayBuffer(
    capacity=1000,
    state_dim=10,
    continuous_dim=1,
    discrete_dims=[4, 2],
)

# Test action selection
state = np.random.randn(10)
disc, cont = agent.select_action(state)
print(f"Discrete actions: {disc}")
print(f"Continuous action: {cont}")

# Fill buffer with random data
for i in range(256):
    buffer.add_with_separate_actions(
        np.random.randn(10),
        [np.random.randint(0, 4), np.random.randint(0, 2)],
        np.random.randn(1).astype(np.float32),
        np.random.random(),
        np.random.random(),
        np.random.randn(10),
        False,
    )

# Test update
metrics = agent.update_parameters(buffer, 64)
print(f"Update metrics:")
print(f"  Critic loss: {metrics['critic_loss']:.4f}")
print(f"  Cost critic loss: {metrics['cost_critic_loss']:.4f}")
print(f"  Actor loss: {metrics['actor_loss']:.4f}")
print(f"  Lambda: {metrics['lambda']:.4f}")
print(f"  Alpha: {metrics['alpha']:.4f}")

print("\nAll tests PASSED!")
