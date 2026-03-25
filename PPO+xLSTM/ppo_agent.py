# =============================================================================
# PPOAgent: PPO algorithm with xLSTM Actor-Critic
# =============================================================================
# Triển khai thuật toán Proximal Policy Optimization (PPO) với:
#   - RolloutBuffer quản lý trajectory + hidden states
#   - Clipped Surrogate Objective
#   - Value Function Loss (clipped hoặc MSE)
#   - Entropy Bonus
#   - Generalized Advantage Estimation (GAE)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn

from actor_critic import ActorCriticxLSTM, ActorCriticxLSTMConfig


# ──────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Lưu trữ trajectory thu thập được trong quá trình rollout.
    Quản lý cả hidden states của xLSTM để đảm bảo tính liên tục của bộ nhớ.
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.hidden_states = []

    def store(self, obs, action, log_prob, reward, value, done, hidden_state=None):
        """Lưu 1 transition vào buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        if hidden_state is not None:
            self.hidden_states.append(self._detach_state(hidden_state))

    def clear(self):
        """Xóa toàn bộ buffer sau khi update xong."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.hidden_states.clear()

    @staticmethod
    def _detach_state(state):
        """Detach hidden state khỏi computation graph để tiết kiệm bộ nhớ."""
        if state is None:
            return None
        detached = {}
        for key, val in state.items():
            if isinstance(val, dict):
                detached[key] = RolloutBuffer._detach_state(val)
            elif isinstance(val, torch.Tensor):
                detached[key] = val.detach().clone()
            elif isinstance(val, tuple):
                detached[key] = tuple(
                    t.detach().clone() if isinstance(t, torch.Tensor) else t
                    for t in val
                )
            else:
                detached[key] = val
        return detached

    def __len__(self):
        return len(self.observations)


# ──────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO Agent sử dụng mạng ActorCriticxLSTM.

    Bao gồm:
        - Thu thập trajectory (rollout) với xLSTM step-by-step
        - Tính Generalized Advantage Estimation (GAE)
        - PPO Clipped update với nhiều epoch
    """

    def __init__(
        self,
        config: ActorCriticxLSTMConfig,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        num_mini_batches: int = 4,
        target_kl: float = None,
        device: str = "auto",
    ):
        """
        Args:
            config:           Cấu hình mạng ActorCriticxLSTM.
            lr:               Learning rate.
            gamma:            Hệ số chiết khấu.
            gae_lambda:       Lambda cho GAE.
            clip_eps:         Epsilon cho PPO clipping.
            value_loss_coef:  Hệ số value loss.
            entropy_coef:     Hệ số entropy bonus.
            max_grad_norm:    Gradient clipping norm.
            ppo_epochs:       Số lần lặp PPO update trên cùng batch dữ liệu.
            num_mini_batches: Số mini-batch chia từ buffer.
            target_kl:        KL divergence tối đa cho early stopping (None = tắt).
            device:           "cpu", "cuda", hoặc "auto".
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.target_kl = target_kl

        # ── Networks ──
        self.policy = ActorCriticxLSTM(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # ── Rollout ──
        self.buffer = RolloutBuffer()
        self.hidden_state = None

    # ------------------------------------------------------------------
    # Chọn hành động (dùng trong rollout)
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray):
        """
        Chọn hành động dựa trên observation hiện tại (step-by-step).

        Args:
            obs: observation (obs_dim,)

        Returns:
            action, log_prob, value (tất cả dạng numpy scalar)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, obs_dim)

        action, log_prob, value, self.hidden_state = self.policy.step(
            obs_tensor, state=self.hidden_state
        )

        return (
            action.cpu().numpy().item(),
            log_prob.cpu().numpy().item(),
            value.cpu().numpy().item(),
        )

    # ------------------------------------------------------------------
    # Lưu transition
    # ------------------------------------------------------------------
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """Lưu 1 transition vào buffer."""
        self.buffer.store(obs, action, log_prob, reward, value, done, self.hidden_state)

    # ------------------------------------------------------------------
    # Reset hidden state (khi episode mới)
    # ------------------------------------------------------------------
    def reset_hidden_state(self):
        """Reset hidden state của xLSTM về None (đầu episode mới)."""
        self.hidden_state = None

    # ------------------------------------------------------------------
    # Tính GAE
    # ------------------------------------------------------------------
    def compute_gae(self, last_value: float, last_done: bool):
        """
        Tính Generalized Advantage Estimation.

        Args:
            last_value: V(s_{T+1}) - giá trị cuối cùng sau rollout.
            last_done:  episode đã kết thúc hay không.

        Returns:
            advantages: np.array (T,)
            returns:    np.array (T,)
        """
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def update(self, last_value: float, last_done: bool):
        """
        Thực hiện PPO update trên dữ liệu trong buffer.

        Args:
            last_value: V(s_{T+1}).
            last_done:  episode kết thúc hay chưa.

        Returns:
            dict chứa các loss trung bình.
        """
        advantages, returns = self.compute_gae(last_value, last_done)

        # Chuẩn bị tensors
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)       # (T, obs_dim)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)          # (T,)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device) # (T,)
        advantages_t = torch.FloatTensor(advantages).to(self.device)                       # (T,)
        returns_t = torch.FloatTensor(returns).to(self.device)                             # (T,)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Reshape cho xLSTM: (1, T, obs_dim) - xử lý toàn bộ chuỗi như 1 batch
        obs_seq = obs.unsqueeze(0)         # (1, T, obs_dim)
        actions_seq = actions.unsqueeze(0)  # (1, T)

        # ── PPO epochs ──
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            # Forward qua toàn bộ chuỗi
            log_probs, values, entropy = self.policy.evaluate_actions(obs_seq, actions_seq)

            log_probs = log_probs.squeeze(0)  # (T,)
            values = values.squeeze(0)         # (T,)
            entropy = entropy.squeeze(0)       # (T,)

            # ── Policy Loss (Clipped Surrogate) ──
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Value Loss ──
            value_loss = nn.functional.mse_loss(values, returns_t)

            # ── Entropy Bonus ──
            entropy_loss = -entropy.mean()

            # ── Total Loss ──
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            # ── Backpropagation ──
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            total_loss_val += loss.item()
            num_updates += 1

            # ── Early stopping nếu KL quá lớn ──
            if self.target_kl is not None:
                approx_kl = (old_log_probs - log_probs).mean().item()
                if approx_kl > 1.5 * self.target_kl:
                    break

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "total_loss": total_loss_val / num_updates,
        }
