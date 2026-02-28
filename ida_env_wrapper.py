# ida_env_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import traci
from simulation.continuous_sumo_env import SumoEnv

class SwitchableWrapper(gym.Wrapper):
	def __init__(self, env: SumoEnv):
		super().__init__(env)
		self.mode = 'STEER' 
		self.helper_model = None
		self.current_obs = None
		
		# Keep your original Action Space
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
		
	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.current_obs = obs
		return obs, info

	def step(self, action):
		student_val = action[0]
		
		# Get Teacher prediction
		if self.helper_model is not None:
			if self.current_obs is None:
				teacher_val = 0.0
			else:
				teacher_val, _ = self.helper_model.predict(self.current_obs)
			if isinstance(teacher_val, np.ndarray): teacher_val = teacher_val.item()
		else:
			teacher_val = 0.3 if self.mode == 'STEER' else 0 # Default forward speed

		# --- ORIGINAL LOGIC (RESTORED) ---
		if self.mode == 'STEER':
			steer = student_val
			# Use teacher's throttle directly (no forced mapping)
			throttle = float(teacher_val) 
			
		else: # THROTTLE
			# Use student's throttle directly
			throttle = student_val
			steer = float(teacher_val)

		# DEBUG: Print what is actually being sent to SUMO
		# (This helps us see if the agent is just hitting the brakes)
		#if np.random.rand() < 0.05: # Print only 5% of the time to avoid spam
		   #print(f"[{self.mode}] Student Output: {student_val:.3f} -> SUMO Throttle: {throttle:.3f}")

		full_action = np.array([steer, throttle], dtype=np.float32)
		next_obs, reward, terminated, truncated, info = self.env.step(full_action)
		
		self.current_obs = next_obs
		return next_obs, reward, terminated, truncated, info
	
	def load_helper(self, model_path, algo_class=PPO):
		"""
		Loads a helper model from a file path inside the subprocess.
		"""
		if model_path is None:
			self.helper_model = None
			return
			
		try:
			# We load the model onto the CPU to avoid GPU conflicts in subprocesses
			self.helper_model = algo_class.load(model_path, device='cpu')
		except Exception as e:
			print(f"Error loading helper in subprocess: {e}")
			self.helper_model = None

	def set_mode(self, mode, helper_model=None):
		"""
		Updated set_mode. 
		Note: We ignore helper_model arg here if we are using the 
		parallel 'load_helper' method, but keep it for backward compatibility.
		"""
		self.mode = mode
		if helper_model is not None:
			self.helper_model = helper_model
		# If helper_model is None, we assume it was already loaded via load_helper()


class RewardWrapper(gym.Wrapper):
	def __init__(self, env: SumoEnv | SwitchableWrapper, phase = "STEER"):
		super().__init__(env)
		self.phase = phase
		self.prev_action = None
		self.stuck_time = 0
		self.weights = {
					"STEER": {
						"speed":   0.0,
						"energy": -0.2,
						"wiggle": -1.0,
						"safety": -1.0,
					},

					"THROTTLE": {
						"speed":   1.0,
						"energy": -1.0,
						"wiggle": -0.1,
						"safety": -0.8,
					},

					"JOINT": {
						"speed":   0.5,
						"energy": -0.6,
						"wiggle": -0.6,
						"safety": -0.8,
					}
				}

	def set_phase(self, phase):
		assert phase in self.weights.keys(), f"Unknown phase: {phase}"
		self.phase = phase

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.prev_action = None
		return obs, info
	
	def set_mode(self, mode):
		return self.env.set_mode(mode)

	def load_helper(self, path):
		return self.env.load_helper(path)

	def _veh_exists(self, veh_id):
		try:
			return veh_id in traci.vehicle.getIDList()
		except Exception:
			return False
		
	def _get_terminal_obs(self):
		return np.zeros(self.observation_space.shape, dtype=np.float32)
	
	def safe_traci(self, fn, default=0.0):
		if not self._veh_exists():
			return default
		try:
			return fn()
		except Exception:
			return default
	
	def step(self, action):
		obs, _, terminated, truncated, info = self.env.step(action)

		# ------------------------------
		# 1. BASE reward from env
		# ------------------------------
		reward = 0.0

		# ------------------------------
		# 2. Phase-weighted shaping
		# ------------------------------
		env = self.env
		while hasattr(env, "env"):
			if hasattr(env, "VEH_ID"):
				break
			env = env.env

		veh_id = env.VEH_ID
		if self._veh_exists(veh_id):

			speed = traci.vehicle.getSpeed(veh_id)
			speed_r = np.clip(speed / env.MAX_SPEED, 0.0, 1.0)

			elec = traci.vehicle.getElectricityConsumption(veh_id) or 0.0
			elec_p = elec / env.MAX_ELEC

			if self.prev_action is None:
				wiggle_p = 0.0
			else:
				wiggle_p = np.mean(np.abs(action - self.prev_action))
			self.prev_action = action

			leader = traci.vehicle.getLeader(veh_id, env.MAX_DIST)
			if leader is not None:
				leader_dist = leader[1]
				if leader_dist < env.TARGET_DIST:
					safety_p = np.clip(
						1 - leader_dist / env.TARGET_DIST, 0.0, 1.0
					)
				else:
					safety_p = np.clip(
						0.1 * (leader_dist - env.TARGET_DIST) / env.TARGET_DIST,
						0.0, 1.0
					)
			else:
				safety_p = 0.0
		
			speed_r = np.nan_to_num(speed_r, nan=0.0, posinf=0.0, neginf=0.0) 
			elec_p = np.nan_to_num(elec_p, nan=0.0, posinf=0.0, neginf=0.0) 
			wiggle_p = np.nan_to_num(wiggle_p, nan=0.0, posinf=0.0, neginf=0.0) 
			safety_p = np.nan_to_num(safety_p, nan=0.0, posinf=0.0, neginf=0.0)

			W = self.weights[self.phase]
			reward += (
				W["speed"]  * speed_r +
				W["energy"] * elec_p +
				W["wiggle"] * wiggle_p +
				W["safety"] * safety_p
			)

		# ------------------------------
		# 3. Terminal bonus / penalty
		# ------------------------------
		avg_speed = info.get("real_speed", 0.0)
		if avg_speed < 0.3:
			self.stuck_time += 1
		else:
			self.stuck_time = 0 
		
		if self.stuck_time > 50:
			terminated = True

		if terminated:
			success = bool(info.get("is_success", 0))
			
			if success:
				reward += 90.0
			else:
				reward -= 90.0

		return obs, reward, terminated, truncated, info