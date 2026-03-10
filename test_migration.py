"""Quick test: Run 3 episodes to verify libtraci migration works correctly."""
import time
import numpy as np
from sumo_env import SumoEnv, _SUMO_BACKEND

print(f"\n{'='*60}")
print(f"  MIGRATION TEST: Backend = {_SUMO_BACKEND}")
print(f"{'='*60}\n")

env = SumoEnv(render=False)

for ep in range(1, 4):
    obs, info = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    start = time.time()
    
    done = False
    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_steps += 1
        done = terminated or truncated
        
        # Safety: Max 50 steps per test episode
        if ep_steps >= 50:
            break
    
    elapsed = time.time() - start
    steps_per_sec = ep_steps / max(elapsed, 0.001)
    reason = info.get("success_reason", "timeout")
    
    print(
        f"  EP{ep}: steps={ep_steps:>3} | "
        f"reason={reason:<12} | "
        f"speed={steps_per_sec:.1f} steps/s | "
        f"obs_shape={obs.shape}"
    )

env.close()
print(f"\n[OK] Migration test PASSED with backend: {_SUMO_BACKEND}")
