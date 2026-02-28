import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import your env
from simulation.continuous_sumo_env import SumoEnv

# --- CONFIGURATION ---
MAP_CONFIG = ["maps/CTU_map/osm.sumocfg", "maps/NinhKieuBridge/osm.sumocfg", "maps/NJ_TurnPike/osm.sumocfg"]

class VisualLogCallback(BaseCallback):
    """
    A simple hook to print data and slow down the loop 
    so you can see what is happening in the GUI.
    """
    def __init__(self, verbose=0):
        super(VisualLogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 1. Get the latest observation (State)
        # SB3 wraps envs in a Vector, so we use [0]
        obs = self.locals['new_obs'][0]
        
        # Extract specific data based on your 20-dim array:
        # Index 0: Speed (Normalized)
        # Index 1: Acceleration
        # Index 2: Energy Cost (Normalized)
        speed = obs[0]
        accel = obs[1]
        energy = obs[2]
        
        # 2. Get the latest reward
        reward = self.locals['rewards'][0]
        
        # 3. Print to Console
        print(f"Step {self.n_calls:04d} | "
              f"Speed: {speed:.2f} | "
              f"Accel: {accel:.2f} | "
              f"Energy Cost: {energy:.2f} | "
              f"Reward: {reward:.3f}")
        
        # 4. Slow down Python slightly so the GUI animation looks smooth
        # (Otherwise it runs too fast to appreciate)
        time.sleep(0.05) 
        
        return True

def main():
    print("--- STARTING VISUAL PPO TEST ---")

    # 1. Initialize Env with Render=True (Opens GUI)
    env = SumoEnv(
        render=True, 
        map_config=MAP_CONFIG,
        TRAFFIC_SCALE=7.0  # Lower scale slightly to see clearly
    )

    # 2. Initialize PPO
    # We use MlpPolicy because inputs are just numbers (Vector)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    print("\nStarting Interaction Loop...")
    print("Watch the SUMO Window and this Console.")
    print("Press Ctrl+C in terminal to stop.\n")

    try:
        # 3. Run!
        # This will drive the car, collect data, and print your logs
        model.learn(total_timesteps=5000, callback=VisualLogCallback())
        
        print("Finished 5000 steps.")
        
    except KeyboardInterrupt:
        print("\nStopped by User.")
    finally:
        env.close()
        print("Environment Closed.")

if __name__ == "__main__":
    main()