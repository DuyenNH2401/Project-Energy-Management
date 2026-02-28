# Dual agent train file
# da_train_RL.py

import os
import time
import pathlib
import gymnasium as gym
import numpy as np
import csv
import traci
from datetime import datetime 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 

# Import the custom environment
from simulation.continuous_sumo_env import SumoEnv
from da_env_wrapper import SteeringOnlyWrapper, ThrottleOnlyWrapper

# --- CONFIGURATION ---
MAP_CONFIG = ["maps/CTU_map/osm.sumocfg", "maps/NinhKieuBridge/osm.sumocfg", "maps/NJ_TurnPike/osm.sumocfg"]

STEER_LOG_DIR = "./reports/da_model/steer"
THROTTLE_LOG_DIR = "./reports/da_model/throttle"


STEER_MODEL_DIR = "./models/da_model/steer"
STEER_BEST_MODEL_DIR = os.path.join(STEER_MODEL_DIR, "highest_reward")
STEER_CHECKPOINT_DIR = os.path.join(STEER_MODEL_DIR, "checkpoints")\

THROTTLE_MODEL_DIR = "./models/da_model/throttle"
THROTTLE_BEST_MODEL_DIR = os.path.join(THROTTLE_MODEL_DIR, "highest_reward")
THROTTLE_CHECKPOINT_DIR = os.path.join(THROTTLE_MODEL_DIR, "checkpoints")

STEER_DATA_LOG_DIR = os.path.join(STEER_LOG_DIR, "data")
THROTTLE_DATA_LOG_DIR = os.path.join(THROTTLE_LOG_DIR, "data")


# Create directories
os.makedirs(STEER_LOG_DIR, exist_ok=True)
os.makedirs(STEER_DATA_LOG_DIR, exist_ok=True)

os.makedirs(THROTTLE_LOG_DIR, exist_ok=True)
os.makedirs(THROTTLE_DATA_LOG_DIR, exist_ok=True)

os.makedirs(STEER_BEST_MODEL_DIR, exist_ok=True)
os.makedirs(STEER_CHECKPOINT_DIR, exist_ok=True)

os.makedirs(THROTTLE_BEST_MODEL_DIR, exist_ok=True)
os.makedirs(THROTTLE_CHECKPOINT_DIR, exist_ok=True)


# Hyperparameters
TOTAL_TIMESTEPS = 200_000  
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64

TIME_NOW = datetime.now().strftime("%Y-%m-%d-%Hh%Mm-%Ss")

class ResearchLogCallback(BaseCallback):
    """
    Logs data to a CSV file for research papers.
    Tracks: Episode, Reward, Length, Avg Speed, Total Energy, Success
    """
    def __init__(self, log_dir, log_name = "training_data", verbose=0):
        super(ResearchLogCallback, self).__init__(verbose)
        self.file_name = f"{log_name}_{datetime.now().strftime('%Y-%m-%d-%Hh%Mm-%Ss')}.csv"
        self.log_path = os.path.join(log_dir, self.file_name)
        
        self.ep_speed_sum = 0.0
        self.ep_energy_sum = 0.0
        self.ep_steps = 0
        self.episode_count = 0
        self.current_ep_reward = 0


        # Create the CSV file and write headers
        with open(self.log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Steps", "Total_Reward", "Avg_Speed_mps", "Total_Energy_Wh", "Success"])

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0] 
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
                print(f"\n>>> EPISODE {self.episode_count} FINISHED <<< | Total Reward: {ep_r:.2f} | Steps: {ep_l}")
                print("-" * 60)
            self.current_ep_reward = 0

        # Print detailed stats every 5 steps
        if self.n_calls % 5 == 0:
            speed = obs[0]
            energy = obs[2] # Assuming index 2 is energy
            print(f"Step {self.n_calls:06d} | Act: {action} | Rew: {reward:6.2f} | "
                  f"Spd: {speed:.2f} | Egy: {energy:.2f}")
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
    

def make_env_steering(rank: int, seed: int = 0):
    def _init():
        env = SumoEnv(
            render=False,
            map_config=MAP_CONFIG,
            TRAFFIC_SCALE=5.0,
            delay=0)
        env = SteeringOnlyWrapper(env=env)
        env = Monitor(env, filename=os.path.join(STEER_LOG_DIR, str(rank)))
        env.reset(seed=seed + rank)
        return env
    return _init

def make_env_throttle(rank: int, seed: int = 0, steering_model = None):
    def _init():
        env = SumoEnv(
            render=False,
            map_config=MAP_CONFIG,
            TRAFFIC_SCALE=5.0,
            delay=0)
        env = ThrottleOnlyWrapper(env=env, steering_model=steering_model)
        env = Monitor(env, filename=os.path.join(THROTTLE_LOG_DIR, str(rank)))
        env.reset(seed=seed + rank)
        return env
    return _init

def main(step_per_agent=TOTAL_TIMESTEPS//2, skip_steer_train = False):

    steering_model = None 
    steering_model_path = os.path.join(STEER_MODEL_DIR, "steering_ppo_sumo_final")
    interrupted_path = os.path.join(STEER_CHECKPOINT_DIR, "steering_ppo_sumo_interrupted")

    if not skip_steer_train and not os.path.exists(steering_model_path + ".zip"):

        print("---Steering Agent---")
        print("-"*60)
        
        steer_envs = DummyVecEnv([make_env_steering(rank=0)])
        # no VecNormalize as we did it in the env
        #steer_envs = VecNormalize(steer_envs, norm_obs=True, norm_reward=True, clip_obs=10.)

        steer_eval_env = DummyVecEnv([make_env_steering(rank=1)])
        #steer_eval_env = VecNormalize(steer_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

        # Save the BEST model (highest reward)
        eval_callback = EvalCallback(
            steer_eval_env,
            best_model_save_path=STEER_BEST_MODEL_DIR,
            log_path=STEER_LOG_DIR,
            eval_freq=5000, # Check every ~5-10 mins
            deterministic=True,
            render=False
        )

        # Save a CHECKPOINT every 50,000 steps (safety save every ~30-60 mins)
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=STEER_CHECKPOINT_DIR,
            name_prefix="ppo_sumo_steering_ckpt"
        )


        callbacks = ResearchLogCallback(log_dir=STEER_LOG_DIR, log_name="steering_training_data")
        # Combine ALL callbacks
        callback_list = CallbackList([eval_callback, checkpoint_callback, callbacks])

        steering_model = PPO(
            "MlpPolicy",
            steer_envs,
            verbose=0,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            device='cpu', 
        )

        try:
            steering_model.learn(total_timesteps=step_per_agent, callback=callback_list)
            steering_model.save(os.path.join(STEER_MODEL_DIR, "steering_ppo_sumo_final"))
            print("Training Finished Successfully.")

        except KeyboardInterrupt:
            print("\n\n!!! STEERING TRAINING INTERRUPTED BY USER !!!")
            steering_model.save(os.path.join(STEER_CHECKPOINT_DIR, "steering_ppo_sumo_interrupted"))
            steer_envs.close()
            
        except Exception as e:
            print(f"\n\n!!! CRITICAL ERROR: {e} !!!")
            # Try to save emergency backup
            steering_model.save(os.path.join(STEER_CHECKPOINT_DIR, "steering_ppo_sumo_CRASH_BACKUP"))
            steer_envs.close()
            raise e  # Re-raise to see the traceback
    

    print("Cleaning up SUMO connection before Phase 2...")
    try:
        if traci.isLoaded():
            traci.close()
    except Exception:
        pass
    time.sleep(2.0) 

    print("---Throttle Agent---")
    print("-"*60)
    # Define the path variable so we don't make typos
    # MUST MATCH PHASE 1 SAVE NAME
    steer_final_path = os.path.join(STEER_MODEL_DIR, "steering_ppo_sumo_final.zip") 
    steer_interrupted_path = os.path.join(STEER_CHECKPOINT_DIR, "steering_ppo_sumo_interrupted.zip")

    if os.path.exists(steer_final_path):
        print("Loading Steering Final Model...")
        expert_steer = PPO.load(steer_final_path)   
    elif os.path.exists(steer_interrupted_path):
        print("Loading Steering Interrupted Model...")
        expert_steer = PPO.load(steer_interrupted_path)
    else:
        # If running sequentially in one go, use the memory object
        if 'steering_model' in locals():
             print("Using Steering Model from memory...")
             expert_steer = steering_model
        else:
             raise FileNotFoundError(f"No Steering Model found at {steer_final_path}! Cannot train Throttle Agent.")


    throttle_envs = DummyVecEnv([make_env_throttle(rank=0, steering_model=expert_steer)])
    #throttle_envs = VecNormalize(throttle_envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    throttle_eval_env = DummyVecEnv([make_env_throttle(rank=1, steering_model=expert_steer)])
    #throttle_eval_env = VecNormalize(throttle_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Save the BEST model (highest reward)
    eval_callback = EvalCallback(
        throttle_eval_env,
        best_model_save_path=THROTTLE_BEST_MODEL_DIR,
        log_path=THROTTLE_LOG_DIR,
        eval_freq=5000, # Check every ~5-10 mins
        deterministic=True,
        render=False
    )

        # Save a CHECKPOINT every 50,000 steps (safety save every ~30-60 mins)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=THROTTLE_CHECKPOINT_DIR,
        name_prefix="ppo_sumo_throttle_ckpt"
    )


    callbacks = ResearchLogCallback(log_dir=THROTTLE_LOG_DIR, log_name="throttle_training_data")
    # Combine ALL callbacks
    callback_list = CallbackList([eval_callback, checkpoint_callback, callbacks])

    throttle_model = PPO(
        "MlpPolicy",
        throttle_envs,
        verbose=0,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        device='cpu', 
    )

    try:
        throttle_model.learn(total_timesteps=step_per_agent, callback=callback_list)
        throttle_model.save(os.path.join(THROTTLE_MODEL_DIR, "throttle_ppo_sumo_final"))
        print("Training Finished Successfully.")

    except KeyboardInterrupt:
        print("\n\n!!! THROTTLE TRAINING INTERRUPTED BY USER !!!")
        throttle_model.save(os.path.join(THROTTLE_CHECKPOINT_DIR, "throttle_ppo_sumo_interrupted"))
        throttle_envs.close()
        
    except Exception as e:
        print(f"\n\n!!! CRITICAL ERROR: {e} !!!")
        # Try to save emergency backup
        throttle_model.save(os.path.join(THROTTLE_CHECKPOINT_DIR, "throttle_ppo_sumo_CRASH_BACKUP"))
        throttle_envs.close()
        raise e  # Re-raise to see the traceback
    

if __name__ == "__main__":
    main()