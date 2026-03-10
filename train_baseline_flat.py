# train_baseline_flat.py
# Phase B — Baseline #1: Flat SAC (No Curriculum Learning)
#
# PURPOSE: Scientific BASELINE for comparison with the Curriculum approach.
# This script trains a single PPO agent DIRECTLY on the full 2D action space
# and Stage-3 reward, WITHOUT any curriculum warm-up or transfer learning.
#
# RESEARCH QUESTION: Does multi-stage curriculum learning provide a measurable
# advantage over end-to-end flat training?
#
# EXPERIMENTAL SETUP:
#   - PPO hyperparameters optimized for fast convergence
#   - Training budget: 200,000 timesteps
#   - Same reward function: FullControlWrapper (Stage-3 weights)
#   - Same seed for reproducibility
#
# USAGE:
#   python train_baseline_flat.py
#
# OUTPUT:
#   models/baselines/flat_ppo/  — model checkpoints + best model
#   reports/baselines/flat_ppo/ — CSV logs for analysis and plotting

import os
import random
from datetime import datetime

import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

import config as cfg
from sumo_env import SumoEnv
from wrappers import FullControlWrapper
from callbacks import CurriculumLogCallback


# =============================================================================
# FLAT BASELINE CONFIGURATION
# =============================================================================
# Total budget = 500,000 timesteps for sufficient exploration
FLAT_TOTAL_TIMESTEPS = 100_000

# Directories — sourced from config.py for single source of truth
FLAT_MODEL_DIR  = cfg.FLAT_MODEL_DIR
FLAT_LOG_DIR    = cfg.FLAT_LOG_DIR

# Extra sub-dirs (created by config, but ensure sub-folders exist)
os.makedirs(os.path.join(FLAT_MODEL_DIR, "best/"), exist_ok=True)
os.makedirs(os.path.join(FLAT_MODEL_DIR, "checkpoints/"), exist_ok=True)


def main():
    print("=" * 60)
    print("  BASELINE: FLAT SAC (No Curriculum Learning)")
    print(f"  Total Timesteps: {FLAT_TOTAL_TIMESTEPS:,}")
    print("  Wrapper: FullControlWrapper (Stage-3 reward)")
    print("=" * 60)

    # --- Reproducibility (same seed as curriculum experiments) ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    from stable_baselines3.common.monitor import Monitor
    # --- Environment ---
    # IMPORTANT: Uses FullControlWrapper (Stage-3 reward + full 2D action space)
    # This is intentionally the HARDEST starting point — no curriculum warm-up.
    train_env = FullControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    )
    eval_env = Monitor(FullControlWrapper(
        SumoEnv(render=False, traffic_scale=cfg.TRAFFIC_SCALE)
    ))

    # --- Model: SAC from scratch optimized for sample efficiency ---
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=0,
        device="auto",
        **cfg.SAC_PARAMS
    )

    # --- Callbacks ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(FLAT_MODEL_DIR, "best/"),
        log_path=FLAT_LOG_DIR,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(FLAT_MODEL_DIR, "checkpoints/"),
        name_prefix="sac_flat_baseline",
    )

    log_cb = CurriculumLogCallback(
        log_dir=FLAT_LOG_DIR,
        stage_name="BASELINE_FLAT_SAC",
        target_timesteps=FLAT_TOTAL_TIMESTEPS,
        verbose=1,
    )

    all_callbacks = CallbackList([eval_cb, ckpt_cb, log_cb])

    # --- Train ---
    final_path = os.path.join(FLAT_MODEL_DIR, "sac_flat_baseline_final")

    try:
        model.learn(
            total_timesteps=FLAT_TOTAL_TIMESTEPS,
            callback=all_callbacks,
        )
        model.save(final_path)
        print(f"\n[BASELINE] Flat SAC training complete! Model saved: {final_path}")
        print(f"  CSV logs: {FLAT_LOG_DIR}")

    except KeyboardInterrupt:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_path = os.path.join(FLAT_MODEL_DIR, f"sac_flat_interrupted_{ts}")
        model.save(interrupted_path)
        print(f"\n[BASELINE] Interrupted! Saved: {interrupted_path}")

    except Exception as e:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_path = os.path.join(FLAT_MODEL_DIR, f"sac_flat_CRASH_{ts}")
        model.save(crash_path)
        print(f"\n[BASELINE] CRASH! Emergency save: {crash_path}")
        raise e

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
