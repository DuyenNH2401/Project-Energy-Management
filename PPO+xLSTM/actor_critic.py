# =============================================================================
# ActorCriticxLSTM: PPO Actor-Critic network powered by xLSTM
# =============================================================================
# Thay thế LSTM truyền thống trong kiến trúc Actor-Critic bằng xLSTMBlockStack,
# tận dụng sLSTM (exponential gating) và mLSTM (matrix memory) để cải thiện
# khả năng ghi nhớ dài hạn cho Reinforcement Learning.
# =============================================================================

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# --- Path setup to import xlstm from current directory ---
_XLSTM_ROOT = os.path.dirname(__file__)
if _XLSTM_ROOT not in sys.path:
    sys.path.insert(0, _XLSTM_ROOT)

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from xlstm.components.feedforward import FeedForwardConfig


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActorCriticxLSTMConfig:
    """
    Cấu hình cho mạng Actor-Critic sử dụng xLSTM.

    Attributes:
        obs_dim:          Kích thước observation của môi trường.
        action_dim:       Số lượng action (discrete) hoặc kích thước action vector (continuous).
        action_type:      'discrete' hoặc 'continuous'.
        embedding_dim:    Kích thước embedding bên trong xLSTM (nên là bội số của num_heads).
        num_blocks:       Số lượng xLSTM block xếp chồng.
        num_heads:        Số lượng head cho sLSTM/mLSTM.
        context_length:   Độ dài context cho mLSTM (không cần thiết cho sLSTM).
        slstm_at:         Danh sách vị trí đặt sLSTM block; các vị trí còn lại dùng mLSTM.
                          Dùng "all" nếu muốn toàn bộ là sLSTM.
        use_feedforward:  Có dùng FeedForward block trong sLSTM hay không.
        dropout:          Tỷ lệ dropout.
        conv1d_kernel_size: Kích thước kernel cho CausalConv1d bên trong sLSTM.
    """
    obs_dim: int = 4
    action_dim: int = 2
    action_type: Literal["discrete", "continuous"] = "discrete"

    # xLSTM hyperparameters
    embedding_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 4
    context_length: int = 256
    slstm_at: list = field(default_factory=lambda: [])  # empty = all mLSTM; "all" = all sLSTM
    use_feedforward: bool = True
    dropout: float = 0.0
    conv1d_kernel_size: int = 4


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────

class ActorCriticxLSTM(nn.Module):
    """
    Mạng Actor-Critic sử dụng xLSTMBlockStack làm backbone.

    Data flow:
        obs -> FeatureExtractor (Linear) -> xLSTMBlockStack -> Actor Head (policy)
                                                             -> Critic Head (value)
    """

    def __init__(self, config: ActorCriticxLSTMConfig):
        super().__init__()
        self.config = config

        # ── Feature Extractor: map obs_dim → embedding_dim ──
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.embedding_dim),
            nn.Tanh(),
        )

        # ── Build xLSTMBlockStackConfig ──
        xlstm_cfg = self._build_xlstm_config(config)

        # ── xLSTM core ──
        self.xlstm = xLSTMBlockStack(xlstm_cfg)

        # ── Actor Head ──
        if config.action_type == "discrete":
            self.actor_head = nn.Linear(config.embedding_dim, config.action_dim)
        else:
            self.actor_mean = nn.Linear(config.embedding_dim, config.action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(config.action_dim))

        # ── Critic Head ──
        self.critic_head = nn.Linear(config.embedding_dim, 1)

        # ── Khởi tạo trọng số ──
        self._init_weights()

    # ------------------------------------------------------------------
    # Xây dựng xLSTM Config
    # ------------------------------------------------------------------
    @staticmethod
    def _build_xlstm_config(cfg: ActorCriticxLSTMConfig) -> xLSTMBlockStackConfig:
        """
        Tạo xLSTMBlockStackConfig từ ActorCriticxLSTMConfig.
        Hỗ trợ cả sLSTM-only, mLSTM-only, và pha trộn.
        """
        slstm_at = cfg.slstm_at if cfg.slstm_at else []

        # --- Determine which block types we need ---
        all_slstm = (slstm_at == "all") or (
            isinstance(slstm_at, list) and sorted(slstm_at) == list(range(cfg.num_blocks))
        )
        all_mlstm = isinstance(slstm_at, list) and len(slstm_at) == 0

        # sLSTM block config
        slstm_block_cfg = None
        if not all_mlstm:
            ff_cfg = FeedForwardConfig(
                proj_factor=1.3,
                act_fn="gelu",
                embedding_dim=cfg.embedding_dim,
                dropout=cfg.dropout,
            ) if cfg.use_feedforward else None

            slstm_block_cfg = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    embedding_dim=cfg.embedding_dim,
                    num_heads=cfg.num_heads,
                    conv1d_kernel_size=cfg.conv1d_kernel_size,
                    dropout=cfg.dropout,
                ),
                feedforward=ff_cfg,
            )

        # mLSTM block config
        mlstm_block_cfg = None
        if not all_slstm:
            mlstm_block_cfg = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    embedding_dim=cfg.embedding_dim,
                    num_heads=cfg.num_heads,
                    context_length=cfg.context_length,
                    dropout=cfg.dropout,
                ),
            )

        return xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_cfg,
            slstm_block=slstm_block_cfg,
            context_length=cfg.context_length,
            num_blocks=cfg.num_blocks,
            embedding_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
            slstm_at=slstm_at if slstm_at else [],
            add_post_blocks_norm=True,
        )

    # ------------------------------------------------------------------
    # Khởi tạo trọng số
    # ------------------------------------------------------------------
    def _init_weights(self):
        """Khởi tạo trọng số cho Feature Extractor, Actor Head, Critic Head."""
        for module in [self.feature_extractor, self.critic_head]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Sequential):
                for sub in module:
                    if isinstance(sub, nn.Linear):
                        nn.init.orthogonal_(sub.weight, gain=1.0)
                        if sub.bias is not None:
                            nn.init.zeros_(sub.bias)

        if self.config.action_type == "discrete":
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)
        else:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)

    # ------------------------------------------------------------------
    # Forward: xử lý batch chuỗi (cho PPO update)
    # ------------------------------------------------------------------
    def forward(
        self,
        obs_seq: torch.Tensor,
        state: dict = None,
    ):
        """
        Forward qua toàn bộ chuỗi observation (dùng cho PPO update).

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            state:   Hidden state dict (có thể None).

        Returns:
            action_dist: phân phối hành động
            value:       (batch, seq_len, 1)
            state:       hidden state mới (nếu cần)
        """
        # Feature extraction
        x = self.feature_extractor(obs_seq)  # (B, S, embedding_dim)

        # xLSTM forward (full sequence)
        x = self.xlstm(x)  # (B, S, embedding_dim)

        # Heads
        value = self.critic_head(x)  # (B, S, 1)
        action_dist = self._get_action_dist(x)

        return action_dist, value.squeeze(-1)

    # ------------------------------------------------------------------
    # Step: xử lý từng bước (cho rollout thu thập dữ liệu)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        obs: torch.Tensor,
        state: dict = None,
    ):
        """
        Step qua 1 observation duy nhất (dùng cho rollout).

        Args:
            obs:   (batch, obs_dim) hoặc (batch, 1, obs_dim)
            state: Hidden state dict từ step trước.

        Returns:
            action:    (batch,)
            log_prob:  (batch,)
            value:     (batch,)
            state:     Hidden state mới
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)

        x = self.feature_extractor(obs)  # (B, 1, embedding_dim)

        # xLSTM step (single timestep, recurrent)
        x, state = self.xlstm.step(x, state=state)  # (B, 1, embedding_dim)

        value = self.critic_head(x).squeeze(-1).squeeze(-1)  # (B,)
        action_dist = self._get_action_dist(x)

        action = action_dist.sample()
        if self.config.action_type == "discrete":
            action = action.squeeze(-1)  # (B,)
            log_prob = action_dist.log_prob(action)
        else:
            log_prob = action_dist.log_prob(action).sum(dim=-1)

        return action, log_prob.squeeze(-1), value, state

    # ------------------------------------------------------------------
    # Evaluate actions (cho PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Tính log_prob, value và entropy cho một batch chuỗi quan sát và hành động.
        Dùng trong PPO update.

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            actions: (batch, seq_len) cho discrete, (batch, seq_len, action_dim) cho continuous

        Returns:
            log_probs: (batch, seq_len)
            values:    (batch, seq_len)
            entropy:   (batch, seq_len)
        """
        action_dist, values = self.forward(obs_seq)

        if self.config.action_type == "discrete":
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
        else:
            log_probs = action_dist.log_prob(actions).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Helper: tạo phân phối hành động
    # ------------------------------------------------------------------
    def _get_action_dist(self, features: torch.Tensor):
        """Tạo phân phối hành động từ features sau xLSTM."""
        if self.config.action_type == "discrete":
            logits = self.actor_head(features)  # (B, S, action_dim)
            return Categorical(logits=logits)
        else:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            return Normal(mean, std)
