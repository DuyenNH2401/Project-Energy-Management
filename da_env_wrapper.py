# Env wrapper for dual agent training
# da_env_wrapper.py
from simulation.continuous_sumo_env import SumoEnv
import gymnasium as gym
import numpy as np


class SteeringOnlyWrapper(gym.ActionWrapper):
	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.env = env
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(1,),
			dtype=np.float32,
		)

	def action(self, action):

		steering = action[0]
		throttle = 0.3

		# steer_cmd = action[0], accel_cmd = action[1]
		return np.array([steering, throttle], dtype=np.float32)
	

class ThrottleOnlyWrapper(gym.Wrapper):
	def __init__(self, env: gym.Env, steering_model):
		super().__init__(env)
		self.env = env
		self.steering_model = steering_model
		self.action_space = gym.spaces.Box(
			low=-1.0, high=1.0, shape=(1,), dtype=np.float32
		)
		self.last_obs = None

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.last_obs = obs
		return obs, info
	
	def step(self, action):
		throttle = action[0]

		steering, _ = self.steering_model.predict(self.last_obs, deterministic=True)
		if isinstance(steering, np.ndarray):
			steering = steering.item()
		steering = float(steering)

		action_all = np.array([steering, throttle], dtype=np.float32)
		
		if isinstance(steering, np.ndarray):
			steering = steering[0]

		obs, reward, terminated, truncated, info = self.env.step(action=action_all)
		self.last_obs = obs

		return obs, reward, terminated, truncated, info