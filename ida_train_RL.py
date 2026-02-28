import os
import csv
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from simulation.continuous_sumo_env import SumoEnv
from ida_env_wrapper import SwitchableWrapper

# --- CONFIG ---
ROUNDS = 100_000         
BUFFER_SIZE = 2048        
TARGET_STEPS_PER_ROUND = 2048  # Minimum steps before looking for an exit
MAP_CONFIG = ["maps/CTU_map/osm.sumocfg", "maps/NinhKieuBridge/osm.sumocfg"]

BASE_DIR = "./reports/ida_model"
MODEL_DIR = "./models/ida_model"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class SwitchLogicCallback(BaseCallback):
	def __init__(self, log_path, target_steps, model_type="?", verbose=0):
		super().__init__(verbose)
		self.log_path = log_path
		self.target_steps = target_steps
		self.model_type = model_type
		
		# Counters
		self.round_steps = 0 # Steps in THIS specific round
		self.total_episodes = 0
		self.ep_len = 0
		self.ep_reward = 0
		
		# CSV Setup
		mode = 'a' if os.path.exists(log_path) else 'w'
		with open(self.log_path, mode=mode, newline='') as f:
			writer = csv.writer(f)
			if mode == 'w': writer.writerow(["Episode", "Steps", "Reward", "AvgSpeed", "Energy", "Success"])

	def _on_step(self) -> bool:
		self.round_steps += 1
		self.ep_len += 1
		
		# Get Info
		current_reward = self.locals['rewards'][0]
		current_action = self.locals['actions'][0] # This is [action]
		infos = self.locals['infos'][0]
		
		# Extract metrics
		real_speed = infos.get('real_speed', 0.0)
		real_energy = infos.get('real_energy', 0.0)
		
		self.ep_reward += current_reward
		
		# --- 1. LIVE STATUS UPDATE ---
		# Extract scalar value from action array for clean printing
		act_val = current_action[0] if isinstance(current_action, (list, np.ndarray)) else current_action
		
		print(f"[{self.model_type}] Ep: {self.total_episodes} | "
			  f"Step: {self.round_steps}/{self.target_steps} | "
			  f"Act: [{act_val:.2f}] | Rew: {current_reward:.2f} | "
			  f"Spd: {real_speed:.2f} | Egy: {real_energy:.2f}")

		# --- 2. EPISODE FINISH & STOPPING LOGIC ---
		if self.locals['dones'][0]:
			self.total_episodes += 1
			
			# Print Summary Line (So it doesn't get overwritten by the \r)
			print(f"\n   >>> [Ep {self.total_episodes} FINISHED] "
				  f"Total Reward: {self.ep_reward:.2f} | Total Steps: {self.ep_len} | "
				  f"Success: {infos.get('is_success', False)}")

			# Log to CSV
			is_success = infos.get('is_success', 0)
			with open(self.log_path, mode='a', newline='') as f:
				writer = csv.writer(f)
				writer.writerow([self.total_episodes, self.ep_len, f"{self.ep_reward:.2f}", 
								 f"{real_speed:.2f}", f"{infos.get('real_energy', 0):.2f}", is_success])
			
			# Reset Episode trackers
			self.ep_len = 0
			self.ep_reward = 0

			# Check if we should Switch Agents
			if self.round_steps >= self.target_steps:
				print(f"[{self.model_type}] Target reached ({self.round_steps} steps). Episode Done. Switching...")
				return False # <--- This stops .learn() cleanly
		
		return True # Continue training

def load_brain_into_fast_shell(path, env, buffer_size):
	# Standard PPO setup
	new_model = PPO("MlpPolicy", env, verbose=0, n_steps=buffer_size, batch_size=64, learning_rate=0.0003)
	
	if os.path.exists(f"{path}.zip"):
		print(f"   > Found existing save: {path}.zip")
		try:
			old_model = PPO.load(f"{path}.zip", device="auto")
			new_model.set_parameters(old_model.get_parameters())
			del old_model 
		except Exception as e:
			print(f"   ! Warning: Could not load weights ({e}). Starting fresh.")
	else:
		print(f"   > No save found. Created Fresh Model.")
	return new_model

def main():
	print("Launching SUMO (Soft-Stop Mode)...")
	
	# Setup Env
	raw_env = SumoEnv(map_config=MAP_CONFIG, render=True)
	switch_env = SwitchableWrapper(raw_env)
	vec_env = DummyVecEnv([lambda: switch_env])

	now = datetime.now().strftime("%Y_%m_%d__%Hh%Mm%Ss")
	new_steer_save_path = os.path.join(MODEL_DIR, "steer_model_" + now)
	new_throttle_save_path = os.path.join(MODEL_DIR, "throttle_model_" + now)

	# 2. Define where we want to LOAD from (Fixed names or specific previous run)
	# CHANGE THIS to the actual file you want to resume from!
	# If you want to start fresh, leave these paths pointing to non-existent files.
	load_steer_path = os.path.join(MODEL_DIR, "steer_interleaved_latest") 
	load_throttle_path = os.path.join(MODEL_DIR, "throttle_interleaved_latest")

	# Setup Models (Try to load from fixed path)
	print("\n--- Setup Steering ---")
	model_steer = load_brain_into_fast_shell(load_steer_path, vec_env, BUFFER_SIZE)
	print("\n--- Setup Throttle ---")
	model_throttle = load_brain_into_fast_shell(load_throttle_path, vec_env, BUFFER_SIZE)

	# Sync obs
	obs = vec_env.reset()
	model_steer._last_obs = obs
	model_throttle._last_obs = obs

	now = datetime.now().strftime("%Y_%m_%d__%Hh%Mm%Ss")
	steer_file_name = "steer_log_" + now + ".csv"
	throttle_file_name = "throttle_log_" + now + ".csv"

	# Initialize Callbacks ONCE
	cb_steer = SwitchLogicCallback(os.path.join(BASE_DIR, steer_file_name), TARGET_STEPS_PER_ROUND, "STR")
	cb_throttle = SwitchLogicCallback(os.path.join(BASE_DIR, throttle_file_name), TARGET_STEPS_PER_ROUND, "THR")

	try:
		for i in range(ROUNDS):
			
			# === PART A: THROTTLE ===
			print(f"\n>>> ROUND {i} | PART A: THROTTLE TURN")
			cb_throttle.round_steps = 0 # Reset round counter
			
			helper = model_steer if i > 0 else None
			vec_env.envs[0].set_mode('THROTTLE', helper_model=helper)
			if i > 0: model_throttle._last_obs = model_steer._last_obs 
			
			# TRICK: Set total_timesteps huge, let the Callback decide when to stop
			model_throttle.learn(total_timesteps=1_000_000, callback=cb_throttle, reset_num_timesteps=False)

			# === PART B: STEERING ===
			print(f"\n>>> ROUND {i} | PART B: STEERING TURN")
			cb_steer.round_steps = 0 

			vec_env.envs[0].set_mode('STEER', helper_model=model_throttle)
			model_steer._last_obs = model_throttle._last_obs 
			
			model_steer.learn(total_timesteps=1_000_000, callback=cb_steer, reset_num_timesteps=False)

			# Save periodically
			if True: # Save every round
				# 1. Save with Timestamp (History)
				model_steer.save(new_steer_save_path)
				model_throttle.save(new_throttle_save_path)
				
				# 2. Save as "Latest" (For easy resuming next time)
				model_steer.save(os.path.join(MODEL_DIR, "steer_interleaved_latest"))
				model_throttle.save(os.path.join(MODEL_DIR, "throttle_interleaved_latest"))

	except KeyboardInterrupt:
		print("\nInterrupted! Saving models...")
		model_steer.save(new_steer_save_path)
		model_throttle.save(new_throttle_save_path)
		vec_env.close()

	vec_env.close()

if __name__ == "__main__":
	main()