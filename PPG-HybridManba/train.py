import gymnasium as gym
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppo import PPG
from sumo_discrete_env import SumoDiscreteEnv
from tqdm import tqdm
from collections import deque

def train():
    env = SumoDiscreteEnv(use_gui=False)
    
    # Environment dimension parameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Log and update frequencies
    max_ep_len = 500              # Max timesteps in one episode
    max_training_timesteps = int(3e5)   # Break training loop if timesteps > max_training_timesteps
    print_freq = max_ep_len * 2   # Print avg reward in the interval (in num timesteps)
    update_timestep = max_ep_len * 4    # Update policy every n timesteps
    
    # PPO hyperparameters
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 40
    eps_clip = 0.2

    # Hybrid Mamba architecture hyperparameters
    d_model = 64          # Hidden dimension
    seq_len = 16          # Context window length (how many past states to use)
    d_state = 16          # SSM state dimension
    d_conv = 4            # Convolution kernel size
    expand = 2            # Mamba expansion factor
    n_mamba_layers = 3    # Number of Mamba blocks
    n_heads = 4           # GQA query heads
    n_kv_heads = 2        # GQA key/value heads

    # PPG-specific hyperparameters
    gae_lambda = 0.95     # GAE lambda for advantage estimation
    mini_batch_size = 128 # Mini-batch size for gradient updates
    n_pi = 4              # Number of policy phases per auxiliary phase
    aux_epochs = 6        # Training epochs for auxiliary phase
    beta_clone = 1.0      # KL divergence penalty weight in auxiliary phase
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")
    print(f"Architecture: Encoder -> Mamba x{n_mamba_layers} -> GQA(heads={n_heads}, kv={n_kv_heads}) -> Actor/Critic")
    print(f"Sequence length: {seq_len}, d_model: {d_model}")
    print(f"PPG Config: N_pi={n_pi}, aux_epochs={aux_epochs}, beta_clone={beta_clone}")
    print(f"GAE lambda: {gae_lambda}, mini_batch_size: {mini_batch_size}")
        
    # Initialize PPG agent with Hybrid Mamba
    ppg_agent = PPG(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        d_model=d_model,
        seq_len=seq_len,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_mamba_layers=n_mamba_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        # PPG-specific
        gae_lambda=gae_lambda,
        mini_batch_size=mini_batch_size,
        n_pi=n_pi,
        aux_epochs=aux_epochs,
        beta_clone=beta_clone,
        buffer_size=update_timestep,
        device=device,
    )

    # Count parameters
    total_params = sum(p.numel() for p in ppg_agent.policy.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    time_step = 0
    i_episode = 0
    
    # Logging variables
    print_running_reward = 0
    print_running_episodes = 0
    
    # For visualization
    history_episodes = []
    history_rewards = []
    current_ep_reward = 0
    recent_rewards = deque(maxlen=100)
    
    state, info = env.reset()
    ppg_agent.reset_episode()  # Reset rolling sequence buffer
    
    # Training loop
    pbar = tqdm(total=max_training_timesteps, desc="Training", unit="step")
    
    while time_step <= max_training_timesteps:
        # Select action with sequence context
        action = ppg_agent.select_action(state)
        
        # Take step in environment
        state, reward, done, truncated, info = env.step(action)
        
        # Store reward and terminal flag
        ppg_agent.buffer.store_reward_terminal(reward, done or truncated)
        
        time_step += 1
        pbar.update(1)
        
        print_running_reward += reward
        current_ep_reward += float(reward) # Ensure it's a python float
        
        # Update progress bar postfix with current reward
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        pbar.set_postfix({"EP": i_episode, "Reward": f"{current_ep_reward:.1f}", "Avg100": f"{avg_reward:.1f}"})
        
        # PPG Update (policy phase + potentially auxiliary phase)
        if time_step % update_timestep == 0:
            metrics = ppg_agent.update()
            
            # Log update info
            log_msg = f"  [Update] policy_loss={metrics.get('policy_loss', 0):.4f}"
            log_msg += f"  value_loss={metrics.get('value_loss', 0):.4f}"
            log_msg += f"  entropy={metrics.get('entropy', 0):.4f}"
            if metrics.get('aux_phase_triggered', False):
                log_msg += f"  | AUX_PHASE: aux_loss={metrics.get('aux_value_loss', 0):.4f}"
                log_msg += f"  main_v_loss={metrics.get('main_value_loss_aux', 0):.4f}"
                log_msg += f"  kl_div={metrics.get('kl_divergence', 0):.6f}"
            pbar.write(log_msg)
            
        # Logging
        if time_step % print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
            pbar.write(f"Episode: {i_episode} \t Timestep: {time_step} \t Average Reward: {print_avg_reward:.2f}")
            print_running_reward = 0
            print_running_episodes = 0
            
        # Break episode on completion
        if done or truncated:
            history_episodes.append(i_episode)
            history_rewards.append(current_ep_reward)
            recent_rewards.append(current_ep_reward)
            current_ep_reward = 0
            
            state, info = env.reset()
            ppg_agent.reset_episode()  # Reset rolling sequence buffer for new episode
            print_running_episodes += 1
            i_episode += 1
            
    pbar.close()
    env.close()
    print("Training finished.")
    
    # Save the model
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'ppg_hybridmamba_final.pth')
    
    torch.save(ppg_agent.policy.state_dict(), model_path)
    print(f"Model successfully saved to: {model_path}")
    
    # Visualize results
    visualize_results(history_episodes, history_rewards)

def visualize_results(episodes, rewards):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label='Episode Reward', alpha=0.4, color='royalblue')
        
        # Calculate moving average
        if len(rewards) >= 2:
            window = max(2, min(100, len(rewards) // 10))
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(episodes, moving_avg, label=f'Moving Average (window={window})', color='darkorange', linewidth=2)
            
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards over Episodes')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        save_path = os.path.join(os.path.dirname(__file__), 'training_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results plot saved to {save_path}")
        
    except ImportError:
        print("matplotlib or pandas is not installed. Skipping visualization.")

if __name__ == '__main__':
    train()