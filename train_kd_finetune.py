# train_kd_finetune.py
# Phase C — Core Research Contribution: Multi-Teacher Knowledge Distillation
#
# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH CONTRIBUTION (Novel element for the paper)
# ═══════════════════════════════════════════════════════════════════════════
# Standard "Best Teacher Transfer" (train_finetune.py) simply copies weights
# and fine-tunes. This is essentially just warm-start — not a research novelty.
#
# This script implements PROPER Multi-Teacher Knowledge Distillation (KD):
#
#   Student loss = L_SAC_standard
#                + α(t) · KL[ π_student(·|s)[steer]   ‖ π_teacher1(·|s)[steer] ]
#                + β(t) · KL[ π_student(·|s)[throttle] ‖ π_teacher2(·|s)[throttle] ]
#
# Where:
#   π_student  = Full 2D Gaussian policy (student SAC, action=[steer, throttle])
#   π_teacher1 = Frozen Stage-1 lateral model (1D Gaussian over steer only)
#   π_teacher2 = Frozen Stage-2 longitudinal model (1D Gaussian over throttle only)
#   α(t), β(t) = Linearly annealed from KD_ALPHA_START → 0 over training
#
# ─────────────────────────────────────────────────────────────────────────
# KL DIVERGENCE (Gaussian, analytical form)
# ─────────────────────────────────────────────────────────────────────────
#   KL( N(μ₁,σ₁) ‖ N(μ₂,σ₂) )
#     = log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²) / (2σ₂²) − ½
#
# Because both student and teachers use Gaussian policies (SAC default),
# we can compute KL exactly without sampling — numerically stable and efficient.
#
# ─────────────────────────────────────────────────────────────────────────
# ANNEALING SCHEDULE (Key to KD success)
# ─────────────────────────────────────────────────────────────────────────
# At t=0:   α=β=1.0  →  Student heavily guided by teachers
# At t=T/2: α=β=0.5  →  Student balances teacher guidance + own policy
# At t=T:   α=β=0.0  →  Student fully autonomous (pure SAC)
#
# This allows the student to:
#   1. Start training grounded in expert knowledge (no catastrophic exploration)
#   2. Gradually become independent as it gains experience
#   3. Eventually surpass teachers by combining both skill sets
#
# ─────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────
#   python train_kd_finetune.py
#
# OUTPUT:
#   models/kd_finetune/  — model checkpoints
#   reports/kd_finetune/ — CSV with standard metrics + KL divergence curves

import os
import csv
import random
from datetime import datetime
from collections import OrderedDict, deque

import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import GymEnv

import config as cfg
from sumo_env import SumoEnv
from wrappers import FullControlWrapper
from callbacks import CurriculumLogCallback

# ─────────────────────────────────────────────────────────────────────────
# KD CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────
KD_ALPHA_START = cfg.KD_ALPHA_START
KD_BETA_START  = cfg.KD_BETA_START
KD_ANNEAL_FRAC = cfg.KD_ANNEAL_FRAC

KD_TIMESTEPS   = cfg.KD_TIMESTEPS

# Directories
KD_MODEL_DIR = cfg.KD_MODEL_DIR
KD_LOG_DIR   = cfg.KD_LOG_DIR

for d in [
    KD_MODEL_DIR, KD_LOG_DIR,
    os.path.join(KD_MODEL_DIR, "best/"),
    os.path.join(KD_MODEL_DIR, "checkpoints/"),
]:
    os.makedirs(d, exist_ok=True)

# Teacher action dimension indices
STEER_DIM    = 0  # Teacher1 (lateral) → student dimension 0
THROTTLE_DIM = 1  # Teacher2 (longitudinal) → student dimension 1


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICAL KL DIVERGENCE (Gaussian)
# ═══════════════════════════════════════════════════════════════════════════
def gaussian_kl_div(
    mu_p: torch.Tensor, log_std_p: torch.Tensor,
    mu_q: torch.Tensor, log_std_q: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(μ_p, σ_p) ‖ N(μ_q, σ_q) ) — analytically computed.

    All inputs: shape [batch, 1]
    Returns:    shape [batch], mean KL per sample.

    Formula:
        KL = log(σ_q/σ_p) + (σ_p² + (μ_p−μ_q)²) / (2σ_q²) − 0.5

    Works in log-space to avoid numerical issues:
        log(σ_q/σ_p) = log_std_q − log_std_p
    """
    # Clamp log_stds to avoid extreme values
    log_std_p = torch.clamp(log_std_p, -10, 2)
    log_std_q = torch.clamp(log_std_q, -10, 2)

    var_p = torch.exp(2 * log_std_p)  # σ_p²
    var_q = torch.exp(2 * log_std_q)  # σ_q²

    kl = (log_std_q - log_std_p) + (var_p + (mu_p - mu_q).pow(2)) / (2 * var_q) - 0.5
    return kl.squeeze(-1)  # [batch]


# ═══════════════════════════════════════════════════════════════════════════
# KD-AWARE SAC (Subclasses stable-baselines3 SAC)
# ═══════════════════════════════════════════════════════════════════════════
class KDSAC(SAC):
    """
    Knowledge Distillation SAC.

    Extends SB3's SAC by injecting KL divergence terms into the actor loss.
    The two frozen teacher models guide the student's policy:
      - teacher_lateral    → guides steering (action dim 0)
      - teacher_longitudinal → guides throttle (action dim 1)

    α and β are annealed from their start values to 0 over training.
    KL metrics are stored in `self.kd_metrics` for callback logging.
    """

    def __init__(
        self,
        policy,
        env: GymEnv,
        teacher_lateral: SAC,
        teacher_longitudinal: SAC,
        kd_alpha: float = KD_ALPHA_START,
        kd_beta: float  = KD_BETA_START,
        kd_anneal_steps: int = KD_TIMESTEPS,
        **kwargs,
    ):
        super().__init__(policy, env, **kwargs)

        self.teacher_lateral = teacher_lateral
        self.teacher_longitudinal = teacher_longitudinal

        self.kd_alpha_start = kd_alpha
        self.kd_beta_start  = kd_beta
        self.kd_anneal_steps = max(kd_anneal_steps, 1)

        # Current values (will be updated during training)
        self.kd_alpha = kd_alpha
        self.kd_beta  = kd_beta

        # Metrics exposed to callbacks
        self.kd_metrics = {
            "alpha": kd_alpha,
            "beta": kd_beta,
            "kl_steer": 0.0,
            "kl_throttle": 0.0,
        }

        # Freeze teachers permanently
        self._freeze_teacher(self.teacher_lateral)
        self._freeze_teacher(self.teacher_longitudinal)

    @staticmethod
    def _freeze_teacher(model: SAC):
        """Freeze all parameters of a teacher model."""
        model.policy.set_training_mode(False)
        for param in model.policy.parameters():
            param.requires_grad_(False)

    def _update_kd_weights(self):
        """Linear annealing: α,β decrease from start → 0 over kd_anneal_steps."""
        frac = min(self.num_timesteps / self.kd_anneal_steps, 1.0)
        self.kd_alpha = self.kd_alpha_start * (1.0 - frac)
        self.kd_beta  = self.kd_beta_start  * (1.0 - frac)

    @torch.no_grad()
    def _get_teacher_distribution(
        self, actor, obs_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Query a teacher actor's Gaussian distribution parameters.

        Uses actor.get_action_dist_params() — the official SB3 v2.x API
        which handles feature extraction, mu/log_std computation, and
        log_std clamping internally.

        Returns:
            mu:       [batch, action_dim] — mean action (pre-squash)
            log_std:  [batch, action_dim] — log standard deviation (clamped)
        """
        mu, log_std, _ = actor.get_action_dist_params(obs_tensor)
        return mu, log_std

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Override SB3 v2.7.1's SAC.train() to inject KD loss into the actor update.

        This method mirrors the upstream SAC.train() EXACTLY, then adds KL
        divergence terms to the actor loss. All other logic (critic, entropy
        coefficient, target network updates) is unchanged from upstream.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update KD annealing schedule
        self._update_kd_weights()

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        kl_steers, kl_throttles = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # For n-step replay, discount factor is gamma**n_steps
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # ─── ENTROPY COEFFICIENT ──────────────────────────────────────
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ─── CRITIC UPDATE ────────────────────────────────────────────
            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ), dim=1
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ─── ACTOR UPDATE (with KD loss injection) ───────────────────
            # Compute actor loss
            # Min over all critic networks
            q_values_pi = torch.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss_sac = (ent_coef * log_prob - min_qf_pi).mean()

            # ─── KL Divergence terms (KD contribution) ────────────────────
            kd_loss = torch.tensor(0.0, device=self.device)
            kl_steer_val = 0.0
            kl_throttle_val = 0.0

            obs_tensor = replay_data.observations.to(self.device)

            # Student distribution (pre-squash Gaussian parameters)
            student_mu, student_log_std = self._get_teacher_distribution(
                self.actor, obs_tensor
            )
            # Shape: [batch, 2] for full 2D student

            # ── Term 1: KL(student_steer ‖ teacher_lateral_steer) ─────────
            if self.kd_alpha > 1e-6:
                t1_mu, t1_log_std = self._get_teacher_distribution(
                    self.teacher_lateral.actor, obs_tensor
                )
                # t1_mu: [batch, 1]  student_mu[:, 0:1]: [batch, 1]
                kl_steer = gaussian_kl_div(
                    student_mu[:, STEER_DIM:STEER_DIM+1],
                    student_log_std[:, STEER_DIM:STEER_DIM+1],
                    t1_mu,
                    t1_log_std,
                )
                kl_steer_mean = kl_steer.mean()
                kd_loss = kd_loss + self.kd_alpha * kl_steer_mean
                kl_steer_val = kl_steer_mean.item()

            # ── Term 2: KL(student_throttle ‖ teacher_longitudinal_throttle) ─
            if self.kd_beta > 1e-6:
                t2_mu, t2_log_std = self._get_teacher_distribution(
                    self.teacher_longitudinal.actor, obs_tensor
                )
                # t2_mu: [batch, 1]  student_mu[:, 1:2]: [batch, 1]
                kl_throttle = gaussian_kl_div(
                    student_mu[:, THROTTLE_DIM:THROTTLE_DIM+1],
                    student_log_std[:, THROTTLE_DIM:THROTTLE_DIM+1],
                    t2_mu,
                    t2_log_std,
                )
                kl_throttle_mean = kl_throttle.mean()
                kd_loss = kd_loss + self.kd_beta * kl_throttle_mean
                kl_throttle_val = kl_throttle_mean.item()

            # ── Total actor loss ──────────────────────────────────────────
            total_actor_loss = actor_loss_sac + kd_loss
            actor_losses.append(total_actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor.optimizer.step()

            # ── Soft update target networks ───────────────────────────────
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0
                )

            kl_steers.append(kl_steer_val)
            kl_throttles.append(kl_throttle_val)

        # SB3 increments _n_updates once after the loop, not per step
        self._n_updates += gradient_steps

        # Update exposed KD metrics
        self.kd_metrics.update({
            "alpha":        self.kd_alpha,
            "beta":         self.kd_beta,
            "kl_steer":     float(np.mean(kl_steers)) if kl_steers else 0.0,
            "kl_throttle":  float(np.mean(kl_throttles)) if kl_throttles else 0.0,
        })

        # SB3 internal logger (matching upstream format)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("kd/alpha",       self.kd_alpha)
        self.logger.record("kd/beta",        self.kd_beta)
        self.logger.record("kd/kl_steer",    self.kd_metrics["kl_steer"])
        self.logger.record("kd/kl_throttle", self.kd_metrics["kl_throttle"])

        self.policy.set_training_mode(False)


# ═══════════════════════════════════════════════════════════════════════════
# KD LOGGING CALLBACK
# ═══════════════════════════════════════════════════════════════════════════
class KDLogCallback(BaseCallback):
    """
    Logs KD-specific metrics to a separate CSV alongside standard training stats.

    Columns: timestep, alpha, beta, kl_steer, kl_throttle, rolling_avg_kl_total
    This CSV is the key evidence for the paper showing the student "graduating"
    from teacher guidance (KL values decreasing over time by design).
    """

    def __init__(self, log_dir: str, model: "KDSAC", rolling_window: int = 50):
        super().__init__(verbose=0)
        self._kd_model = model
        self.rolling_window = rolling_window

        self._kl_history = deque(maxlen=rolling_window)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"kd_metrics_{now}.csv")
        os.makedirs(log_dir, exist_ok=True)

        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "timestep", "alpha", "beta",
            "kl_steer", "kl_throttle", "kl_total",
            "rolling_avg_kl_total",
        ])
        print(f"  [KD] Logging KL metrics → {self.csv_path}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        """Log after every rollout collection."""
        m = self._kd_model.kd_metrics
        kl_total = m["kl_steer"] + m["kl_throttle"]
        self._kl_history.append(kl_total)
        rolling = float(np.mean(self._kl_history))

        self._w.writerow([
            self.model.num_timesteps,
            f"{m['alpha']:.5f}",
            f"{m['beta']:.5f}",
            f"{m['kl_steer']:.5f}",
            f"{m['kl_throttle']:.5f}",
            f"{kl_total:.5f}",
            f"{rolling:.5f}",
        ])
        self._f.flush()

    def _on_training_end(self):
        self._f.close()
        print(f"\n  [KD] Final α={self._kd_model.kd_alpha:.4f}, β={self._kd_model.kd_beta:.4f}")
        print(f"  [KD] Metrics CSV: {self.csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def find_model(model_dir: str, keyword: str) -> str:
    """Locate the best available model by priority: final > best > latest."""
    candidates = [
        os.path.join(model_dir, f"sac_{keyword}_final.zip"),
        os.path.join(model_dir, f"sac_{keyword}_final"),
        os.path.join(model_dir, "best", "best_model.zip"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    if os.path.isdir(model_dir):
        zips = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir) if f.endswith(".zip")
        ]
        if zips:
            return max(zips, key=os.path.getmtime)
    raise FileNotFoundError(f"No model found for '{keyword}' in {model_dir}")


def transfer_weights(student: KDSAC, teacher: SAC, teacher_name: str) -> int:
    """
    Initialize student weights from teacher where shapes match.
    Same logic as train_finetune.py — provides good starting point before KD.
    """
    new_params  = student.policy.state_dict()
    tchr_params = teacher.policy.state_dict()
    transferred = OrderedDict()
    count = 0

    for key in new_params:
        if key in tchr_params and tchr_params[key].shape == new_params[key].shape:
            transferred[key] = tchr_params[key].clone()
            count += 1
        else:
            transferred[key] = new_params[key]

    student.policy.load_state_dict(transferred)
    print(f"  [{teacher_name}] Transferred {count}/{len(new_params)} param tensors")
    return count


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  PHASE C: MULTI-TEACHER KNOWLEDGE DISTILLATION")
    print(f"  Algorithm: KD-SAC | Timesteps: {KD_TIMESTEPS:,}")
    print(f"  α_start={KD_ALPHA_START}  β_start={KD_BETA_START}  anneal=linear→0")
    print("=" * 70)

    # --- Reproducibility ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    # --- Locate teacher models ---
    lateral_path      = find_model(cfg.STAGE1_MODEL_DIR, "lateral")
    longitudinal_path = find_model(cfg.STAGE2_MODEL_DIR, "longitudinal")
    print(f"\n  Teacher 1 (Lateral):      {lateral_path}")
    print(f"  Teacher 2 (Longitudinal): {longitudinal_path}")

    # --- Load frozen teachers ---
    print("\n  Loading teachers... ", end="", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_lateral      = SAC.load(lateral_path, device=device)
    teacher_longitudinal = SAC.load(longitudinal_path, device=device)
    print("✓")

    from stable_baselines3.common.monitor import Monitor
    # --- Environments ---
    train_env = FullControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    )
    eval_env = Monitor(FullControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    ))

    # --- KD-SAC hyperparameters ---
    # Slightly lower LR for stable fine-tuning; more learning_starts for buffer warm-up
    kd_sac_params = cfg.SAC_PARAMS.copy()
    kd_sac_params["learning_rate"]   = 1e-4   # conservative for fine-tuning
    kd_sac_params["learning_starts"] = 2_000  # wait for buffer before first update
    kd_sac_params["batch_size"]      = 256

    # --- Build KD-SAC student ---
    print("\n  Building KD-SAC student model...")
    model = KDSAC(
        policy="MlpPolicy",
        env=train_env,
        teacher_lateral=teacher_lateral,
        teacher_longitudinal=teacher_longitudinal,
        kd_alpha=KD_ALPHA_START,
        kd_beta=KD_BETA_START,
        kd_anneal_steps=KD_TIMESTEPS,
        verbose=0,
        device="auto",
        **kd_sac_params,
    )

    # --- Weight initialization from best teacher (Stage 2) ---
    print("\n  Weight transfer from Stage 2 (Longitudinal) → Student...")
    n_from_lon = transfer_weights(model, teacher_longitudinal, "Stage2")

    # Also fill from Stage 1 where Stage 2 couldn't (mismatched action heads)
    print("  Weight transfer from Stage 1 (Lateral) → gaps...")
    lon_sd  = teacher_longitudinal.policy.state_dict()
    lat_sd  = teacher_lateral.policy.state_dict()
    cur_sd  = model.policy.state_dict()
    extra = 0
    for key in cur_sd:
        lon_ok = key in lon_sd and lon_sd[key].shape == cur_sd[key].shape
        lat_ok = key in lat_sd and lat_sd[key].shape == cur_sd[key].shape
        if not lon_ok and lat_ok:
            cur_sd[key] = lat_sd[key].clone()
            extra += 1
    if extra > 0:
        model.policy.load_state_dict(cur_sd)
        print(f"  [Stage1] Additional {extra} tensors from lateral teacher")

    total_transferred = n_from_lon + extra
    total_params = len(model.policy.state_dict())
    print(f"\n  Weight Transfer Summary: {total_transferred}/{total_params} tensors initialized")
    print(f"  Action heads (randomly initialized): {total_params - total_transferred}/{total_params}")

    # --- Callbacks ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(KD_MODEL_DIR, "best/"),
        log_path=KD_LOG_DIR,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(KD_MODEL_DIR, "checkpoints/"),
        name_prefix="kd_sac",
    )

    # Standard episode log (inherits rich terminal output + CSV)
    episode_log_cb = CurriculumLogCallback(
        log_dir=KD_LOG_DIR,
        stage_name="KD_FINETUNE",
        target_timesteps=KD_TIMESTEPS,
        verbose=1,
    )

    # KD-specific metrics log (α, β, KL curves)
    kd_log_cb = KDLogCallback(
        log_dir=KD_LOG_DIR,
        model=model,
        rolling_window=50,
    )

    all_callbacks = CallbackList([eval_cb, ckpt_cb, episode_log_cb, kd_log_cb])

    # --- Train ---
    final_path = os.path.join(KD_MODEL_DIR, "kd_sac_final")

    print(f"\n{'─' * 70}")
    print(f"  Starting KD training:")
    print(f"    α: {KD_ALPHA_START} → 0  (linear over {KD_TIMESTEPS:,} steps)")
    print(f"    β: {KD_BETA_START} → 0  (linear over {KD_TIMESTEPS:,} steps)")
    print(f"    Budget: {KD_TIMESTEPS:,} timesteps")
    print(f"    Output: {KD_MODEL_DIR}")
    print(f"{'─' * 70}\n")

    try:
        model.learn(
            total_timesteps=KD_TIMESTEPS,
            callback=all_callbacks,
        )
        model.save(final_path)
        print(f"\n[KD] Training complete! Model saved: {final_path}")
        print(f"  KD metrics CSV:  {KD_LOG_DIR}")

    except KeyboardInterrupt:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(KD_MODEL_DIR, f"kd_sac_interrupted_{ts}")
        model.save(path)
        print(f"\n[KD] Interrupted! Saved: {path}")

    except Exception as e:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(KD_MODEL_DIR, f"kd_sac_CRASH_{ts}")
        model.save(path)
        print(f"\n[KD] CRASH! Emergency save: {path}")
        raise e

    finally:
        train_env.close()
        eval_env.close()
        del teacher_lateral, teacher_longitudinal


if __name__ == "__main__":
    main()
