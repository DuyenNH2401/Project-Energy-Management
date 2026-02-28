import os
import time
import glob
import numpy as np
from stable_baselines3 import PPO
from simulation.continuous_sumo_env import SumoEnv

# --- CONFIGURATION ---
MAP_CONFIG = ["maps/CTU_map/osm.sumocfg", "maps/NinhKieuBridge/osm.sumocfg", "maps/NJ_TurnPike/osm.sumocfg"]
MODEL_ROOT_DIR = "./models/sa_model"

def get_model_path():
    """
    Scans the models directory recursively for .zip files 
    and lets the user choose one via command line.
    """
    # 1. Find all .zip files
    # recursive=True allows finding files in subfolders like 'checkpoints' or 'highest_reward'
    files = glob.glob(os.path.join(MODEL_ROOT_DIR, "**/*.zip"), recursive=True)
    print([entry.path for entry in os.scandir(MODEL_ROOT_DIR) if entry.is_dir()]
)
    if not files:
        print(f"No models found in {MODEL_ROOT_DIR}!")
        return None

    print("\n--- AVAILABLE MODELS ---")
    for i, f in enumerate(files):
        # clean up path for display
        display_name = os.path.relpath(f, MODEL_ROOT_DIR)
        print(f"[{i}] {display_name}")
    print("------------------------")

    # 2. Ask User
    while True:
        try:
            selection = input("Enter the number of the model to load: ")
            idx = int(selection)
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # 1. SELECT MODEL
    model_path = get_model_path()
    if not model_path:
        return

    print(f"\nLoading model from: {model_path}")
    print("Initializing GUI Environment...")

    # 2. SETUP ENVIRONMENT (GUI ON)
    env = SumoEnv(
        render=True, 
        map_config=MAP_CONFIG,
        TRAFFIC_SCALE=5.0,
        delay=100
    )

    # 3. LOAD MODEL
    # We don't need to pass the env here for prediction, but it's good practice
    model = PPO.load(model_path)

    # 4. RUNNING LOOP
    episodes = 5
    
    try:
        for ep in range(episodes):
            print(f"\n=== STARTING EPISODE {ep + 1}/{episodes} ===")
            obs, info = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done:
                # Get Action from AI
                # deterministic=True means "Pick the absolute best action", no randomness.
                # Use False if you want to see exploration behavior.
                action, _states = model.predict(obs, deterministic=True)
                
                # Execute Step
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step_counter += 1
                
                # Extract Data for Reporting
                speed = obs[0]
                energy = obs[2]
                
                # REPORTING (Matches Train Format)
                # We print every step in testing so you can sync with GUI
                print(f"Step {step_counter:04d} | Act: {action} | Rew: {reward:6.2f} | "
                      f"Spd: {speed:.2f} | Egy: {energy:.2f}")

                # Slow down slightly for visual clarity (0.05s = 20 FPS)
                time.sleep(0.05)

                if terminated or truncated:
                    print("-" * 60)
                    reason = "CRASH/ARRIVED" if terminated else "TIMEOUT"
                    print(f">>> EPISODE FINISHED ({reason}) <<< | Total Reward: {total_reward:.2f} | Steps: {step_counter}")
                    print("-" * 60)
                    done = True
                    
                    # Small pause before next episode
                    time.sleep(2)

    except KeyboardInterrupt:
        print("\nTest Stopped by User.")
    finally:
        env.close()
        print("Environment Closed.")

if __name__ == "__main__":
    main()