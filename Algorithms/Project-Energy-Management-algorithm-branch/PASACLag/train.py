"""Training script for PASACLag algorithm.

This script demonstrates how to train the PASACLag agent on a hybrid
electric vehicle energy management environment with safety constraints.

Usage:
    python train.py
"""
import os
import time
import yaml
import numpy as np
import torch

from .algorithms.pasac_lag import PASACLagAgent
from .buffers.replay_buffer import ReplayBuffer


class DummyHEVEnv:
    """Dummy HEV Environment for testing.
    
    Replace this with your actual environment implementation.
    
    State space: [velocity, acceleration, SOC, engine_speed, ...]
    Action space: 
        - Discrete: gear shift (4 options), clutch (2 options)
        - Continuous: engine torque [-1, 1]
    """
    
    def __init__(
        self,
        state_dim: int = 10,
        continuous_dim: int = 1,
        discrete_dims: list = [4, 2],
    ):
        self.state_dim = state_dim
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.max_steps = 500
        self.step_count = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.step_count = 0
        self.soc = 0.8  # Initial State of Charge
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def step(
        self,
        discrete_actions: list,
        continuous_action: np.ndarray,
    ) -> tuple:
        """Take action and return (next_state, reward, cost, done, info).
        
        Args:
            discrete_actions: List of discrete action indices.
            continuous_action: Continuous action array.
            
        Returns:
            next_state: Next observation.
            reward: Reward signal (e.g., fuel efficiency).
            cost: Cost signal for safety (e.g., SOC violation).
            done: Whether episode ended.
            info: Additional information.
        """
        self.step_count += 1
        
        # Simulate dynamics (replace with actual HEV model)
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        
        # Reward: fuel efficiency (dummy)
        reward = -0.1 * np.abs(continuous_action[0]) + 0.5
        
        # Cost: SOC constraint violation
        # Cost = 1 if SOC drops below threshold, else 0
        self.soc -= 0.001 * np.abs(continuous_action[0])
        cost = 1.0 if self.soc < 0.2 else 0.0
        
        done = self.step_count >= self.max_steps or self.soc < 0.1
        info = {"soc": self.soc}
        
        return next_state, reward, cost, done, info


def train(config_path: str = "config.yaml"):
    """Main training loop.
    
    Args:
        config_path: Path to configuration file.
    """
    # Load configuration
    with open(os.path.join(os.path.dirname(__file__), config_path), 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get environment configuration
    env_config = config["environment"]
    state_dim = env_config["state_dim"]
    continuous_dim = env_config["continuous_dim"]
    discrete_dims = env_config["discrete_dims"]
    
    # Initialize environment
    env = DummyHEVEnv(state_dim, continuous_dim, discrete_dims)
    
    # Initialize agent
    agent_config = config["agent"]
    agent = PASACLagAgent(
        state_dim=state_dim,
        continuous_dim=continuous_dim,
        discrete_dims=discrete_dims,
        gamma=agent_config["gamma"],
        tau=agent_config["tau"],
        alpha=agent_config["alpha"],
        auto_alpha=agent_config["auto_alpha"],
        cost_limit=agent_config["cost_limit"],
        lambda_lr=agent_config["lambda_lr"],
        actor_lr=agent_config["actor_lr"],
        critic_lr=agent_config["critic_lr"],
        cost_critic_lr=agent_config["cost_critic_lr"],
        alpha_lr=agent_config["alpha_lr"],
        hidden_dim=agent_config["hidden_dim"],
        gumbel_temperature=agent_config["gumbel_temperature"],
        device=device,
    )
    
    # Initialize replay buffer
    training_config = config["training"]
    buffer = ReplayBuffer(
        capacity=training_config["buffer_size"],
        state_dim=state_dim,
        continuous_dim=continuous_dim,
        discrete_dims=discrete_dims,
        device=device,
    )
    
    # Create directories
    paths = config["paths"]
    os.makedirs(paths["log_dir"], exist_ok=True)
    os.makedirs(paths["model_dir"], exist_ok=True)
    
    # Training loop
    total_steps = 0
    episode_rewards = []
    episode_costs = []
    
    print("=" * 60)
    print("PASACLag Training Started")
    print("=" * 60)
    
    for episode in range(training_config["max_episodes"]):
        state = env.reset()
        episode_reward = 0
        episode_cost = 0
        start_time = time.time()
        
        for step in range(training_config["max_steps"]):
            total_steps += 1
            
            # Select action
            if total_steps < training_config["warmup_steps"]:
                # Random exploration during warmup
                discrete_actions = [
                    np.random.randint(0, dim) for dim in discrete_dims
                ]
                continuous_action = np.random.uniform(-1, 1, continuous_dim).astype(np.float32)
            else:
                # Use policy
                discrete_actions, continuous_action = agent.select_action(state)
            
            # Step environment
            next_state, reward, cost, done, info = env.step(
                discrete_actions, continuous_action
            )
            
            # Store transition
            buffer.add_with_separate_actions(
                state, discrete_actions, continuous_action,
                reward, cost, next_state, done
            )
            
            # Update agent
            if (
                total_steps >= training_config["warmup_steps"]
                and total_steps % training_config["update_frequency"] == 0
                and buffer.can_sample(training_config["batch_size"])
            ):
                metrics = agent.update_parameters(
                    buffer, training_config["batch_size"]
                )
            
            episode_reward += reward
            episode_cost += cost
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        elapsed = time.time() - start_time
        
        # Logging
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-10:])
            mean_cost = np.mean(episode_costs[-10:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {mean_reward:7.2f} | "
                f"Cost: {mean_cost:6.2f} | "
                f"Lambda: {agent.lagrangian.item():6.3f} | "
                f"Alpha: {agent.alpha.item():6.3f} | "
                f"Time: {elapsed:.2f}s"
            )
        
        # Evaluation
        if (episode + 1) % training_config["eval_frequency"] == 0:
            eval_reward, eval_cost = evaluate(env, agent, num_episodes=5)
            print(
                f"  [EVAL] Reward: {eval_reward:.2f} | Cost: {eval_cost:.2f}"
            )
        
        # Save checkpoint
        if (episode + 1) % training_config["save_frequency"] == 0:
            save_path = os.path.join(
                paths["model_dir"], f"pasac_lag_ep{episode + 1}.pt"
            )
            agent.save(save_path)
            print(f"  [SAVED] {save_path}")
    
    # Final save
    final_path = os.path.join(paths["model_dir"], "pasac_lag_final.pt")
    agent.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


def evaluate(
    env,
    agent: PASACLagAgent,
    num_episodes: int = 5,
) -> tuple:
    """Evaluate agent performance.
    
    Args:
        env: Environment instance.
        agent: Trained agent.
        num_episodes: Number of evaluation episodes.
        
    Returns:
        mean_reward: Average episode reward.
        mean_cost: Average episode cost.
    """
    rewards = []
    costs = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_cost = 0
        
        for _ in range(env.max_steps):
            discrete_actions, continuous_action = agent.select_action(
                state, evaluate=True
            )
            next_state, reward, cost, done, _ = env.step(
                discrete_actions, continuous_action
            )
            
            episode_reward += reward
            episode_cost += cost
            state = next_state
            
            if done:
                break
        
        rewards.append(episode_reward)
        costs.append(episode_cost)
    
    return np.mean(rewards), np.mean(costs)


if __name__ == "__main__":
    train()
