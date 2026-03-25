# =============================================================================
# Train script: PPO + xLSTM trên CartPole-v1
# =============================================================================
# Script mẫu để chạy thử nghiệm thuật toán PPO với mạng Actor-Critic xLSTM.
# Sử dụng CartPole-v1 (discrete action) làm môi trường kiểm thử.
# =============================================================================

import sys
import os
import time
import numpy as np
import config

# --- Path setup ---
_PPO_DIR = os.path.dirname(__file__)
if _PPO_DIR not in sys.path:
    sys.path.insert(0, _PPO_DIR)

import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from actor_critic import ActorCriticxLSTMConfig
from ppo_agent import PPOAgent




def make_env(env_id: str, seed: int):
    """Tạo Gym environment."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def get_obs_action_dims(env):
    """Lấy kích thước observation và action từ environment."""
    obs_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, "n"):
        # Discrete
        action_dim = env.action_space.n
        action_type = "discrete"
    else:
        # Continuous
        action_dim = env.action_space.shape[0]
        action_type = "continuous"

    return obs_dim, action_dim, action_type


def build_slstm_at(block_type: str, num_blocks: int):
    """Xác định vị trí đặt sLSTM blocks."""
    if block_type == "slstm":
        return "all"
    elif block_type == "mlstm":
        return []
    elif block_type == "mixed":
        # Xen kẽ: sLSTM ở các vị trí chẵn
        return list(range(0, num_blocks, 2))
    return []


def train():
    """Vòng lặp training chính."""
    # ── Setup ──
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    env = make_env(config.ENV_ID, config.SEED)
    obs_dim, action_dim, action_type = get_obs_action_dims(env)

    print("=" * 60)
    print(f"  PPO + xLSTM Training")
    print("=" * 60)
    print(f"  Environment:    {config.ENV_ID}")
    print(f"  Obs dim:        {obs_dim}")
    print(f"  Action dim:     {action_dim} ({action_type})")
    print(f"  Block type:     {config.BLOCK_TYPE}")
    print(f"  Num blocks:     {config.NUM_BLOCKS}")
    print(f"  Embedding dim:  {config.EMBEDDING_DIM}")
    print(f"  Num heads:      {config.NUM_HEADS}")
    print(f"  Total steps:    {config.TOTAL_TIMESTEPS:,}")
    print(f"  Rollout steps:  {config.ROLLOUT_STEPS}")
    print("=" * 60)

    # ── ActorCritic config ──
    slstm_at = build_slstm_at(config.BLOCK_TYPE, config.NUM_BLOCKS)

    ac_config = ActorCriticxLSTMConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        embedding_dim=config.EMBEDDING_DIM,
        num_blocks=config.NUM_BLOCKS,
        num_heads=config.NUM_HEADS,
        context_length=config.ROLLOUT_STEPS,
        slstm_at=slstm_at,
        use_feedforward=(config.BLOCK_TYPE != "mlstm"),
    )

    # ── Agent ──
    agent = PPOAgent(
        config=ac_config,
        lr=config.LR,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_eps=config.CLIP_EPS,
        value_loss_coef=config.VALUE_LOSS_COEF,
        entropy_coef=config.ENTROPY_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        ppo_epochs=config.PPO_EPOCHS,
        device=config.DEVICE,
    )

    total_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"  Total params:   {total_params:,}")
    print(f"  Device:         {agent.device}")
    print("=" * 60 + "\n")

    # ── Training Loop ──
    obs, _ = env.reset()
    agent.reset_hidden_state()

    global_step = 0
    num_updates = 0
    episode_rewards = []
    current_ep_reward = 0.0
    start_time = time.time()
    recent_rewards = []

    while global_step < config.TOTAL_TIMESTEPS:
        # ── Rollout phase ──
        for step in range(config.ROLLOUT_STEPS):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_ep_reward += reward

            agent.store_transition(obs, action, log_prob, reward, value, done)

            obs = next_obs
            global_step += 1

            if done:
                episode_rewards.append(current_ep_reward)
                recent_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()
                agent.reset_hidden_state()

            if global_step >= config.TOTAL_TIMESTEPS:
                break

        # ── Compute last value cho GAE ──
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            _, _, last_value, _ = agent.policy.step(last_obs_t, state=agent.hidden_state)
            last_value = last_value.cpu().numpy().item()

        # ── PPO Update ──
        losses = agent.update(last_value=last_value, last_done=done)
        num_updates += 1

        # ── Logging ──
        if num_updates % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed if elapsed > 0 else 0

            if recent_rewards:
                mean_reward = np.mean(recent_rewards[-100:])
                max_reward = np.max(recent_rewards[-100:])
            else:
                mean_reward = 0.0
                max_reward = 0.0

            print(
                f"Update {num_updates:4d} | "
                f"Steps: {global_step:7,} | "
                f"Episodes: {len(episode_rewards):4d} | "
                f"Mean Reward: {mean_reward:8.2f} | "
                f"Max Reward: {max_reward:8.2f} | "
                f"Policy Loss: {losses['policy_loss']:8.4f} | "
                f"Value Loss: {losses['value_loss']:8.4f} | "
                f"Entropy: {losses['entropy']:6.4f} | "
                f"FPS: {fps:5.0f}"
            )

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Training Complete!")
    print(f"  Total time:     {elapsed:.1f}s")
    print(f"  Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(f"  Mean reward (last 100):  {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  Max reward:              {np.max(episode_rewards):.2f}")
    print("=" * 60)

    # ── Save model ──
    if config.SAVE_PATH:
        os.makedirs(os.path.dirname(config.SAVE_PATH) or ".", exist_ok=True)
        torch.save({
            "policy_state_dict": agent.policy.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "config": ac_config,
            "episode_rewards": episode_rewards,
        }, config.SAVE_PATH)
        print(f"  Model saved to: {config.SAVE_PATH}")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    train()
