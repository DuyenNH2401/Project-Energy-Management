# daft_train_RL.py
# Dual-agent sequential and hierarchical training and joint fine-tuning.
# training stages:
# 1. Steering model training with fixxed throttle
# 2. Throttle model training with frozen steering model
# 3. Fine tuning Steering and Throttle model together (joint fine-tuning)

# Needed libraries
import os
import time
import pathlib
import gymnasium as gym
import numpy as np
import csv
import traci
from datetime import datetime
# golden standard of DRL while SAC might be better advanced
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Custom env
from simulation.continuous_sumo_env import SumoEnv
# pretty useful here
from da_env_wrapper import SteeringOnlyWrapper, ThrottleOnlyWrapper
from ida_env_wrapper import SwitchableWrapper


STEER_STEP = 200_000
THROTTLE_STEP = 200_000
JOINT_STEP = 100_000
MAP_CONFIG = [
    "maps/map1/run.sumocfg"
]

LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64

BASE_DIR =  "./reports/daft_model/"
MODEL_DIR = "./models/daft_model/"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
NOW_TIME = datetime.now().strftime("%d%m%Y_%H%M%S")

class DAFT_Callback(BaseCallback):
	def __init__(
		self,
		log_path,
		target_timesteps,
		model_type: str,
		verbose: int = 0,
	):
		super().__init__(verbose)

		now = datetime.now().strftime("%d%m%Y_%H%M%S")
		self.target_timesteps = target_timesteps
		self.model_type = model_type
		self.file_name = f"{log_path}/{model_type}_{now}.csv"

		# --- Soft stop ---
		self.soft_stop_armed = False

		# --- Episode stats ---
		self.ep_len = 0
		self.ep_reward = 0.0
		self.total_episodes = 0
		self.print_every = 1
		self.last_print = 0
		self.sum_speed = 0.0
		self.sum_energy = 0.0
		self.sum_wiggle = 0.0
		self.sum_safety = 0.0

		# Open file once
		self.file = open(self.file_name, "w")
		self.file.write(
			"episode,steps,ep_reward,avg_speed,total_energy,wiggle,safety,success,reason\n"
		)

	def _on_step(self) -> bool:
		self.ep_len += 1

		infos = self.locals["infos"][0]
		rewards = float(self.locals["rewards"][0])
		done = self.locals["dones"][0]
		action = self.locals["actions"][0]

		self.sum_speed += infos.get("real_speed", 0.036)
		self.sum_energy += infos.get("real_energy", 0.0)
		self.sum_wiggle += infos.get("wiggle", 0.0)
		self.sum_safety += infos.get("safety", 0.0)

		self.ep_reward += rewards

		# ---- STEP HEARTBEAT PRINT ----
		if self.verbose and (self.num_timesteps % self.print_every == 0):
			action = self.locals.get("actions")
			if action is not None:
				action = action[0]  # unwrap VecEnv

			print(
				f"[DAFT][STEP] t={self.num_timesteps} | "
				f"ep_len={self.ep_len} | "
				f"reward={rewards:.3f} | "
				f"action={action}"
			)

			self.last_print = self.num_timesteps

		# ---- ARM SOFT STOP ----
		if self.num_timesteps >= self.target_timesteps:
			self.soft_stop_armed = True

		# ---- EPISODE END ----
		if done:
			self.total_episodes += 1

			avg_speed = self.sum_speed / max(1, self.ep_len)
			energy = self.sum_energy
			wiggle = self.sum_wiggle / max(1, self.ep_len)
			safety = self.sum_safety / max(1, self.ep_len)
			success = infos.get("is_success", 0)
			success_reason = infos.get("success_reason", "not_found")

			self.file.write(
				f"{self.total_episodes},"
				f"{self.ep_len},"
				f"{self.ep_reward:.3f},"
				f"{avg_speed:.3f},"
				f"{energy:.3f},"
				f"{wiggle:.3f},"
				f"{safety:.3f},"
				f"{success},"
				f"{success_reason}\n"
			)
			self.file.flush()

			self.sum_speed = 0.0
			self.sum_energy = 0.0
			self.sum_wiggle = 0.0
			self.sum_safety = 0.0

			if self.verbose:
				print(
					f"[DAFT:{self.model_type}] "
					f"EP {self.total_episodes} DONE | "
					f"steps={self.ep_len} | "
					f"ep_reward={self.ep_reward:+.2f} | "
					f"success={success}"
				)

			# Reset episode stats
			self.ep_len = 0
			self.ep_reward = 0.0

			# ---- SOFT STOP EXIT ----
			if self.soft_stop_armed:
				print(
					f"[DAFT:{self.model_type}] "
					f"SOFT STOP @ t={self.num_timesteps:,} — waiting episode end "
				)
				return False

		return True

	def _on_training_end(self):
		self.file.close()

def make_env(phase="STEER", steering_model=None):
	if phase == "STEER":
		env = SumoEnv(render=False, map_config=MAP_CONFIG, TRAFFIC_SCALE=7.0)
		env = SteeringOnlyWrapper(env)
	elif phase == "THROTTLE":
		env = SumoEnv(render=False, map_config=MAP_CONFIG, TRAFFIC_SCALE=7.0)
		env = ThrottleOnlyWrapper(env, steering_model=steering_model)
	return env

def safe_save(model, path, tag):
	now = datetime.now().strftime("%d%m%Y_%H%M%S")
	save_path = os.path.join(path, f"{now}_{tag}")
	model.save(save_path)
	print(f"Model saved: {save_path}")


def cleanup_sumo():
	print("Cleaning up SUMO connection...")
	try:
		if traci.isLoaded():
			traci.close()
	except Exception:
		pass
	time.sleep(2.0)

def find_best_model(model_dir, keyword):
	"""
	Priority order:
	1. BestModel
	2. CRASH_BACKUP
	3. interrupted
	4. checkpoint
	"""
	if not os.path.exists(model_dir):
		raise FileNotFoundError(f"{model_dir} does not exist")

	files = os.listdir(model_dir)

	priority_patterns = [
		"BestModel",
		"CRASH_BACKUP",
		"interrupted",
		"checkpoint",
	]

	for pattern in priority_patterns:
		candidates = [
			os.path.join(model_dir, f)
			for f in files
			if keyword in f and pattern in f and f.endswith(".zip")
		]
		if candidates:
			return max(candidates, key=os.path.getmtime)

	raise FileNotFoundError(
		f"No suitable model found for keyword='{keyword}' in {model_dir}"
	)


def phase1_steering():
	print("Launching SUMO (Soft-Stop Mode)...")
	print("Phase 1 - Steering Only!!!")

	steer_only_env = make_env(phase="STEER")
	steer_eval_env = make_env(phase="STEER")

	steering_models_path = os.path.join(MODEL_DIR, "steer_only/")
	steering_checkpoint_path = os.path.join(MODEL_DIR, "steer_only/")
	steering_log_path = os.path.join(BASE_DIR, "steer_only/")
	final_path = os.path.join(steering_models_path, "steering_ppo_sumo_final")

	os.makedirs(steering_models_path, exist_ok=True)
	os.makedirs(steering_log_path, exist_ok=True)

	steer_eval_callback = EvalCallback(
		steer_eval_env,
		best_model_save_path=os.path.join(
			steering_models_path, f"{NOW_TIME}_BestModel"
		),
		log_path=steering_models_path,
		eval_freq=5000,
		deterministic=True,
		render=False
	)

	steer_checkpoint_callback = CheckpointCallback(
		save_freq=5000,
		save_path=os.path.join(
			steering_models_path, f"{NOW_TIME}_checkpoint"
		),
		name_prefix="ppo_sumo_steering_ckpt"
	)

	steer_daft_callback = DAFT_Callback(
		log_path=steering_log_path,
		target_timesteps=STEER_STEP,
		model_type="STEER",
		verbose=1,
	)

	callback_list = CallbackList([
		steer_eval_callback,
		steer_checkpoint_callback,
		steer_daft_callback
	])

	steering_model = PPO(
		"MlpPolicy",
		steer_only_env,
		verbose=0,
		learning_rate=LEARNING_RATE,
		n_steps=N_STEPS,
		batch_size=BATCH_SIZE,
		device="cpu"
	)

	try:
		steering_model.learn(
			total_timesteps=STEER_STEP,
			callback=callback_list
		)
		steering_model.save(final_path)
		print("Training Finished Successfully.")

	except KeyboardInterrupt:
		print("\n\n!!! STEERING TRAINING INTERRUPTED BY USER !!!")
		checkpoint_time = datetime.now().strftime("%d%m%Y_%H%M%S")
		steering_model.save(
			os.path.join(
				steering_checkpoint_path,
				f"{checkpoint_time}_steering_ppo_sumo_interrupted"
			)
		)

	except Exception as e:
		print(f"\n\n!!! CRITICAL ERROR: {e} !!!")
		checkpoint_time = datetime.now().strftime("%d%m%Y_%H%M%S")
		steering_model.save(
			os.path.join(
				steering_checkpoint_path,
				f"{checkpoint_time}_steering_ppo_sumo_CRASH_BACKUP"
			)
		)
		raise e

	finally:
		steer_only_env.close()
		steer_eval_env.close()
		cleanup_sumo()

	return steering_model

def phase2_throttle(steering_model):
	print("Phase 2 - Throttle Only!!!")

	steering_model.policy.set_training_mode(False)

	throttle_only_env = make_env(phase="THROTTLE", steering_model=steering_model)

	throttle_eval_env = ThrottleOnlyWrapper(
		SumoEnv(render=False, map_config=MAP_CONFIG, TRAFFIC_SCALE=7.0),
		steering_model
	)

	throttle_models_path = os.path.join(MODEL_DIR, "throttle_only/")
	throttle_checkpoint_path = throttle_models_path
	throttle_log_path = os.path.join(BASE_DIR, "throttle_only/")
	final_path = os.path.join(throttle_models_path, "throttle_ppo_sumo_final")

	os.makedirs(throttle_models_path, exist_ok=True)
	os.makedirs(throttle_log_path, exist_ok=True)

	throttle_eval_callback = EvalCallback(
		throttle_eval_env,
		best_model_save_path=os.path.join(
			throttle_models_path, f"{NOW_TIME}_BestModel"
		),
		log_path=throttle_models_path,
		eval_freq=5000,
		deterministic=True,
		render=False
	)

	throttle_checkpoint_callback = CheckpointCallback(
		save_freq=5000,
		save_path=os.path.join(
			throttle_models_path, f"{NOW_TIME}_checkpoint"
		),
		name_prefix="ppo_sumo_throttle_ckpt"
	)

	throttle_daft_callback = DAFT_Callback(
		log_path=throttle_log_path,
		target_timesteps=THROTTLE_STEP,
		model_type="THROTTLE",
		verbose=1
	)

	callback_list = CallbackList([
		throttle_eval_callback,
		throttle_checkpoint_callback,
		throttle_daft_callback
	])

	throttle_model = PPO(
		"MlpPolicy",
		throttle_only_env,
		verbose=0,
		learning_rate=LEARNING_RATE,
		n_steps=N_STEPS,
		batch_size=BATCH_SIZE,
		device="cpu"
	)

	try:
		throttle_model.learn(
			total_timesteps=THROTTLE_STEP,
			callback=callback_list
		)
		throttle_model.save(final_path)
		print("Training Finished Successfully.")

	except KeyboardInterrupt:
		print("\n\n!!! THROTTLE TRAINING INTERRUPTED BY USER !!!")
		checkpoint_time = datetime.now().strftime("%d%m%Y_%H%M%S")
		throttle_model.save(
			os.path.join(
				throttle_checkpoint_path,
				f"{checkpoint_time}_throttle_ppo_sumo_interrupted"
			)
		)

	except Exception as e:
		print(f"\n\n!!! CRITICAL ERROR: {e} !!!")
		checkpoint_time = datetime.now().strftime("%d%m%Y_%H%M%S")
		throttle_model.save(
			os.path.join(
				throttle_checkpoint_path,
				f"{checkpoint_time}_throttle_ppo_sumo_CRASH_BACKUP"
			)
		)
		raise e

	finally:
		throttle_only_env.close()
		throttle_eval_env.close()
		cleanup_sumo()

	return throttle_model


def phase3_joint():
	print("Phase 3 - Alternating Joint Fine-Tuning!!!")

	joint_models_path = os.path.join(MODEL_DIR, "joint/")
	joint_log_path = os.path.join(BASE_DIR, "joint/")
	os.makedirs(joint_models_path, exist_ok=True)
	os.makedirs(joint_log_path, exist_ok=True)

	# --- Load best available models ---
	steer_model_path = find_best_model(
		os.path.join(MODEL_DIR, "steer_only"), "steering"
	)
	throttle_model_path = find_best_model(
		os.path.join(MODEL_DIR, "throttle_only"), "throttle"
	)

	steer_model = PPO.load(steer_model_path, device="cpu")
	throttle_model = PPO.load(throttle_model_path, device="cpu")

	base_env = SumoEnv(
		render=False,
		map_config=MAP_CONFIG,
		TRAFFIC_SCALE=7.0
	)

	switch_env = SwitchableWrapper(base_env)

	CYCLE_STEPS = 10_000
	NUM_CYCLES = JOINT_STEP // (2 * CYCLE_STEPS)

	try:
		for cycle in range(NUM_CYCLES):
			print(f"\n=== Joint Cycle {cycle+1}/{NUM_CYCLES} ===")

			# ---------- STEER ----------
			print("[JOINT] Steering learns, throttle helps")
			switch_env.set_mode("STEER")
			switch_env.load_helper(throttle_model_path)

			steer_cb = DAFT_Callback(
				log_path=joint_log_path,
				target_timesteps=CYCLE_STEPS,
				model_type=f"JOINT_STEER_C{cycle}",
				verbose=1
			)

			steer_model.set_env(switch_env)
			steer_model.learn(CYCLE_STEPS, callback=steer_cb)

			steer_model_path = os.path.join(
				joint_models_path, f"steer_joint_cycle_{cycle}.zip"
			)
			steer_model.save(steer_model_path)

			# ---------- THROTTLE ----------
			print("[JOINT] Throttle learns, steering helps")
			switch_env.set_mode("THROTTLE")
			switch_env.load_helper(steer_model_path)

			throttle_cb = DAFT_Callback(
				log_path=joint_log_path,
				target_timesteps=CYCLE_STEPS,
				model_type=f"JOINT_THROTTLE_C{cycle}",
				verbose=1
			)

			throttle_model.set_env(switch_env)
			throttle_model.learn(CYCLE_STEPS, callback=throttle_cb)

			throttle_model_path = os.path.join(
				joint_models_path, f"throttle_joint_cycle_{cycle}.zip"
			)
			throttle_model.save(throttle_model_path)

	except KeyboardInterrupt:
		print("\n\n!!! JOINT TRAINING INTERRUPTED BY USER !!!")
		ts = datetime.now().strftime("%d%m%Y_%H%M%S")

		steer_model.save(
			os.path.join(joint_models_path, f"{ts}_steer_CRASH_BACKUP.zip")
		)
		throttle_model.save(
			os.path.join(joint_models_path, f"{ts}_throttle_CRASH_BACKUP.zip")
		)

	except Exception as e:
		print(f"\n\n!!! JOINT TRAINING CRASHED: {e} !!!")
		ts = datetime.now().strftime("%d%m%Y_%H%M%S")

		steer_model.save(
			os.path.join(joint_models_path, f"{ts}_steer_CRASH_BACKUP.zip")
		)
		throttle_model.save(
			os.path.join(joint_models_path, f"{ts}_throttle_CRASH_BACKUP.zip")
		)
		raise e

	finally:
		switch_env.close()
		cleanup_sumo()



if __name__ == "__main__":
	steering_model = phase1_steering()
	throttle_model = phase2_throttle(steering_model)
	phase3_joint()