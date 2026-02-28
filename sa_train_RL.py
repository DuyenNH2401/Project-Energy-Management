# single agent train file
import os
import time
import gymnasium as gym
import numpy as np
import csv
from datetime import datetime 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 

# Import your custom environment
from simulation.continuous_sumo_env import SumoEnv

# --- CONFIGURATION ---
MAP_CONFIG = ["maps/CTU_map/osm.sumocfg", "maps/NinhKieuBridge/osm.sumocfg", "maps/NJ_TurnPike/osm.sumocfg"]
LOG_DIR = "./reports/sa_model"
MODEL_DIR = "./models/sa_model"
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "highest_reward")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
DATA_LOG_DIR = os.path.join(LOG_DIR, "data")

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
TOTAL_TIMESTEPS = 2_000_000  # Increased for overnight run (approx 5-10 hours depending on speed)
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64

TIME_NOW = datetime.now().strftime("%Y-%m-%d-%Hh%Mm-%Ss")

class ResearchLogCallback(BaseCallback):
    """
    Logs data to a CSV file for research papers.
    Tracks: Episode, Reward, Length, Avg Speed, Total Energy, Success
    """
    def __init__(self, log_dir, verbose=0):
        super(ResearchLogCallback, self).__init__(verbose)
        self.file_name = f"training_data_{TIME_NOW}.csv"
        self.log_path = os.path.join(log_dir, self.file_name)
        
        self.ep_speed_sum = 0.0
        self.ep_energy_sum = 0.0
        self.ep_steps = 0
        self.episode_count = 0

        # Create the CSV file and write headers
        with open(self.log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Steps", "Total_Reward", "Avg_Speed_mps", "Total_Energy_Wh", "Success"])

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0] 
        
        # Accumulate data
        if "real_speed" in infos:
            self.ep_speed_sum += infos['real_speed']
        if "real_energy" in infos:
            self.ep_energy_sum += infos['real_energy']
        
        self.ep_steps += 1

        if self.locals['dones'][0]:
            ep_reward = infos.get('episode', {}).get('r', 0)
            avg_speed = self.ep_speed_sum / max(1, self.ep_steps)
            total_energy = self.ep_energy_sum 
            is_success = infos.get("is_success", 0)

            with open(self.log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.episode_count,
                    self.ep_steps,
                    f"{ep_reward:.2f}",
                    f"{avg_speed:.2f}",
                    f"{total_energy:.2f}",
                    is_success
                ])
                file.flush() 

            # Reset Accumulators
            self.ep_speed_sum = 0.0
            self.ep_energy_sum = 0.0
            self.ep_steps = 0
            self.episode_count += 1

        return True

class DetailedLogCallback(BaseCallback):
    """
    Custom callback strictly for PRINTING detailed stats to console.
    CSV logging is handled by ResearchLogCallback.
    """
    def __init__(self, verbose=0):
        super(DetailedLogCallback, self).__init__(verbose)
        self.current_ep_reward = 0

    def _on_step(self) -> bool:
        dones = self.locals['dones'][0]
        reward = self.locals['rewards'][0]
        action = self.locals['actions'][0]
        obs = self.locals['new_obs'][0]
        
        self.current_ep_reward += reward

        # Print episode finish
        if dones:
            infos = self.locals['infos'][0]
            if 'episode' in infos:
                ep_r = infos['episode']['r']
                ep_l = infos['episode']['l']
                print(f"\n>>> EPISODE FINISHED <<< | Total Reward: {ep_r:.2f} | Steps: {ep_l}")
                print("-" * 60)
            self.current_ep_reward = 0

        # Print detailed stats every 5 steps
        if self.n_calls % 5 == 0:
            speed = obs[0]
            energy = obs[2] # Assuming index 2 is energy
            print(f"Step {self.n_calls:06d} | Act: {action} | Rew: {reward:6.2f} | "
                  f"Spd: {speed:.2f} | Egy: {energy:.2f}")

        return True


def make_env(rank: int, seed: int = 0):
    def _init():
        env = SumoEnv(
            render=False, 
            map_config=MAP_CONFIG,
            TRAFFIC_SCALE=5.0,
            delay=0 
        )
        env = Monitor(env, filename=os.path.join(LOG_DIR, str(rank)))
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("--- STARTING OVERNIGHT TRAINING ---")
    
    # 1. Environment
    env = DummyVecEnv([make_env(rank=0)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    eval_env = DummyVecEnv([make_env(rank=1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 2. Callbacks
    
    # Save the BEST model (highest reward)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000, # Check every ~5-10 mins
        deterministic=True,
        render=False
    )

    # Save a CHECKPOINT every 50,000 steps (safety save every ~30-60 mins)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_sumo_ckpt"
    )

    csv_callback = ResearchLogCallback(log_dir=DATA_LOG_DIR)
    log_callback = DetailedLogCallback()
    
    # Combine ALL callbacks
    callback_list = CallbackList([eval_callback, checkpoint_callback, csv_callback, log_callback])

    # 3. Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        device='cpu' 
    )

    print(f"Training for {TOTAL_TIMESTEPS} steps...")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
        model.save(os.path.join(MODEL_DIR, "ppo_sumo_final"))
        print("Training Finished Successfully.")

    except KeyboardInterrupt:
        print("\n\n!!! INTERRUPTED BY USER !!!")
        model.save(os.path.join(CHECKPOINT_DIR, "ppo_sumo_interrupted"))
        env.close()
        
    except Exception as e:
        print(f"\n\n!!! CRITICAL ERROR: {e} !!!")
        # Try to save emergency backup
        model.save(os.path.join(CHECKPOINT_DIR, "ppo_sumo_CRASH_BACKUP"))
        env.close()
        raise e  # Re-raise to see the traceback

if __name__ == "__main__":
    main()