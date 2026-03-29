import gymnasium as gym
import torch
import numpy as np
from ppo import PPO

def train():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
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
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")
    print(f"Architecture: Encoder -> Mamba x{n_mamba_layers} -> GQA(heads={n_heads}, kv={n_kv_heads}) -> Actor/Critic")
    print(f"Sequence length: {seq_len}, d_model: {d_model}")
        
    # Initialize PPO agent with Hybrid Mamba
    ppo_agent = PPO(
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
        device=device,
    )

    # Count parameters
    total_params = sum(p.numel() for p in ppo_agent.policy.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    time_step = 0
    i_episode = 0
    
    # Logging variables
    print_running_reward = 0
    print_running_episodes = 0
    
    state, info = env.reset()
    ppo_agent.reset_episode()  # Reset episode state buffer
    
    # Training loop
    while time_step <= max_training_timesteps:
        # Select action with sequence context
        action = ppo_agent.select_action(state)
        
        # Take step in environment
        state, reward, done, truncated, info = env.step(action)
        
        # Save reward and is_terminal
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done or truncated)
        
        time_step += 1
        print_running_reward += reward
        
        # Update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()
            
        # Logging
        if time_step % print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
            print(f"Episode: {i_episode} \t Timestep: {time_step} \t Average Reward: {print_avg_reward:.2f}")
            print_running_reward = 0
            print_running_episodes = 0
            
        # Break episode on completion
        if done or truncated:
            state, info = env.reset()
            ppo_agent.reset_episode()  # Reset episode state buffer for new episode
            print_running_episodes += 1
            i_episode += 1
            
    env.close()
    print("Training finished.")

if __name__ == '__main__':
    train()
