"""Training script for DSAC-LSTM.

Example usage:
    python train.py

This script demonstrates training DSAC-LSTM on continuous control tasks.
It handles episode-based data collection, sequence sampling, and training.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from algorithms.dsac_lstm import DSAC_LSTM
from buffers.recurrent_replay_buffer import RecurrentReplayBuffer
from utils.common import set_seed, make_env



# Environment
ENV_ID = "Pendulum-v1"
SEED = 42

# Training
TOTAL_STEPS = 100000
BATCH_SIZE = 32
START_STEPS = 5000          # Random exploration steps before training
UPDATE_AFTER = 1000         # Start training after this many steps
UPDATE_EVERY = 50           # Update model every N steps
GRADIENT_STEPS = 50         # Gradient steps per update

# Algorithm
GAMMA = 0.99                # Discount factor
TAU = 0.005                 # Soft update coefficient
LEARNING_RATE = 3e-4        # Learning rate for all networks
HIDDEN_DIM = 256            # MLP hidden dimension
LSTM_HIDDEN_DIM = 128       # LSTM hidden dimension
LSTM_LAYERS = 1             # Number of LSTM layers
SEQUENCE_LENGTH = 64        # Sequence length for training
BURN_IN_STEPS = 16          # Burn-in steps for LSTM hidden state

# Replay Buffer
BUFFER_SIZE = 1000          # Maximum number of episodes in buffer

# Logging
LOG_INTERVAL = 1000         # Log metrics every N steps
SAVE_INTERVAL = 10000       # Save model every N steps
SAVE_DIR = "checkpoints"    # Directory to save models

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    """Main training loop."""
    # Setup
    set_seed(SEED)
    
    # Create environment
    env = make_env(ENV_ID, SEED)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    print(f"Environment: {ENV_ID}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    
    # Create agent
    agent = DSAC_LSTM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=GAMMA,
        tau=TAU,
        auto_alpha=True,
        delay_update=2,
        burn_in_steps=BURN_IN_STEPS,
        hidden_dim=HIDDEN_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        policy_lr=LEARNING_RATE,
        value_lr=LEARNING_RATE,
        alpha_lr=LEARNING_RATE,
        device=DEVICE,
    )
    
    # Create replay buffer
    buffer = RecurrentReplayBuffer(
        max_episodes=BUFFER_SIZE,
        obs_dim=obs_dim,
        action_dim=action_dim,
        sequence_length=SEQUENCE_LENGTH,
        device=DEVICE,
    )
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Training state
    total_steps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_length = 0
    
    # Hidden state management for inference
    policy_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    prev_action = np.zeros(action_dim, dtype=np.float32)
    prev_reward = 0.0
    
    # Reset environment
    obs, _ = env.reset(seed=SEED)
    buffer.start_episode()
    
    # Training loop with progress bar
    pbar = tqdm(total=TOTAL_STEPS, desc="Training")
    
    while total_steps < TOTAL_STEPS:
        # Select action
        if total_steps < START_STEPS:
            # Random exploration
            action = env.action_space.sample()
        else:
            # Policy action
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
            
            prev_action_tensor = torch.tensor(prev_action, dtype=torch.float32, device=DEVICE)
            prev_action_tensor = prev_action_tensor.unsqueeze(0).unsqueeze(0)
            
            prev_reward_tensor = torch.tensor([[prev_reward]], dtype=torch.float32, device=DEVICE)
            prev_reward_tensor = prev_reward_tensor.unsqueeze(-1)  # (1, 1, 1)
            
            action_tensor, _, policy_hidden = agent.select_action(
                obs_tensor, prev_action_tensor, prev_reward_tensor,
                hidden=policy_hidden, deterministic=False
            )
            action = action_tensor.squeeze().cpu().numpy()
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action * action_high)
        done = terminated or truncated
        
        # Store transition
        buffer.add(obs, action, reward, next_obs, done)
        
        # Update counters
        episode_reward += reward
        episode_length += 1
        total_steps += 1
        pbar.update(1)
        
        # Handle episode end
        if done:
            # Log episode stats
            pbar.set_postfix({
                "ep": episode_num,
                "reward": f"{episode_reward:.2f}",
                "len": episode_length,
            })
            
            # Reset for new episode
            obs, _ = env.reset()
            buffer.start_episode()
            policy_hidden = None
            prev_action = np.zeros(action_dim, dtype=np.float32)
            prev_reward = 0.0
            
            episode_num += 1
            episode_reward = 0.0
            episode_length = 0
        else:
            obs = next_obs
            prev_action = action.copy()
            prev_reward = reward
        
        # Training updates
        if total_steps >= UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
            if buffer.can_sample(BATCH_SIZE):
                for _ in range(GRADIENT_STEPS):
                    batch = buffer.sample(BATCH_SIZE)
                    metrics = agent.update(batch, total_steps)
                
                # Log training metrics
                if total_steps % LOG_INTERVAL == 0:
                    print(f"\nStep {total_steps}:")
                    print(f"  Critic Loss: {metrics['loss/critic']:.4f}")
                    print(f"  Policy Loss: {metrics['loss/policy']:.4f}")
                    print(f"  Alpha: {metrics['policy/alpha']:.4f}")
                    print(f"  Entropy: {metrics['policy/entropy']:.4f}")
        
        # Save checkpoint
        if total_steps % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dsac_lstm_{total_steps}.pt")
            agent.save(save_path)
            print(f"\nSaved checkpoint: {save_path}")
    
    pbar.close()
    
    # Final save
    final_path = os.path.join(SAVE_DIR, "dsac_lstm_final.pt")
    agent.save(final_path)
    print(f"Training complete! Final model saved: {final_path}")
    
    env.close()


if __name__ == "__main__":
    train()
