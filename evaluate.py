# evaluate.py
# Phase B — Unified Evaluation Script
#
# PURPOSE: Standard evaluation harness for comparing ALL methods under
# identical conditions. Generates side-by-side metrics suitable for
# inclusion in a research paper Table.
#
# SUPPORTS:
#   1. Curriculum SAC (Stage 3 final model)     [--mode curriculum]
#   2. KD-SAC (Phase C final method)             [--mode kd]
#   3. Flat PPO Baseline                         [--mode flat]
#   4. Rule-Based Baseline (IDM + Pure Pursuit)  [--mode rulebased]
#   5. All four at once                          [--mode all]
#
# METRICS REPORTED (standard RL/AD evaluation):
#   - Success Rate (%)           — reaches destination without collision
#   - Collision Rate (%)         — fraction of episodes with collision
#   - Avg Episode Reward         — total reward per episode
#   - Avg Speed (m/s)            — mean driving speed
#   - Energy Efficiency (Wh/m)   — energy per meter (lower is better)
#   - Avg Episode Length (steps) — how long before termination
#
# USAGE:
#   python evaluate.py --mode all --episodes 30
#   python evaluate.py --mode curriculum --model_path ./models/curriculum/stage3_finetune/sac_finetune_final.zip
#   python evaluate.py --mode flat --episodes 20
#   python evaluate.py --mode rulebased --episodes 20 --render
#
# OUTPUT:
#   reports/evaluation/  — per-mode CSVs + summary comparison table

import os
import sys
import csv
import argparse
import random
from datetime import datetime

import numpy as np
import torch

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set 'SUMO_HOME' environment variable.")

# Use same SUMO backend as sumo_env.py (evaluate.py doesn't call traci directly,
# but some imports may expect it in the namespace)
try:
    import libsumo as traci
except (ImportError, OSError, Exception):
    try:
        import libtraci as traci
    except (ImportError, OSError, Exception):
        import traci
from stable_baselines3 import SAC, PPO

import config as cfg
from sumo_env import SumoEnv
from wrappers import FullControlWrapper
from evaluate_rulebased import IDMController, PurePursuitController, run_episode as rulebased_episode

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------
EVAL_OUTPUT_DIR = cfg.EVAL_OUTPUT_DIR
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# =============================================================================
# RL EVALUATOR — shared by curriculum SAC, KD SAC and flat PPO
# =============================================================================
# Fixed map for evaluation — always use map1
EVAL_MAP = os.path.join(cfg.BASE_DIR, "maps/map2/run.sumocfg")


def evaluate_rl_model(
    model_path: str,
    mode_name: str,
    algo_class=SAC,
    n_episodes: int = 20,
    render: bool = False,
    seed: int = cfg.SEED,
) -> list[dict]:
    """
    Run `n_episodes` of deterministic RL evaluation.

    Returns list of episode result dicts.
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n  Loading model: {model_path}")
    if not os.path.exists(model_path):
        # Try appending .zip
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    # Build a throw-away env purely for loading the model
    dummy_env = FullControlWrapper(SumoEnv(render=False, map_config=EVAL_MAP))
    model = algo_class.load(model_path, env=dummy_env, device="cpu")
    model.policy.set_training_mode(False)
    dummy_env.close()

    # --- Fixed route discovery ---
    # Run one episode to discover a route, then reuse it for all episodes.
    fixed_route = None

    results = []

    for ep in range(1, n_episodes + 1):
        # Use fixed map (map1) and fixed route for all episodes
        env = FullControlWrapper(SumoEnv(
            render=render,
            map_config=EVAL_MAP,
            traffic_scale=cfg.TRAFFIC_SCALE,
        ))

        # If we already discovered a route, inject it before reset
        if fixed_route is not None:
            env.unwrapped.fixed_route_edges = list(fixed_route)

        obs, _ = env.reset()

        # Save the route from the first episode for reuse
        if fixed_route is None and hasattr(env.unwrapped, 'fixed_route_edges'):
            fixed_route = list(env.unwrapped.fixed_route_edges)
            print(f"  🛣️  Fixed route discovered: {len(fixed_route)} edges")

        ep_reward = 0.0
        ep_steps = 0
        ep_speed = 0.0
        ep_energy = 0.0
        success = 0
        reason = "timeout"

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_steps += 1
            ep_speed += info.get("real_speed", 0.0)
            ep_energy += info.get("real_energy", 0.0)

            done = terminated or truncated

            if done:
                success = info.get("is_success", 0)
                reason = info.get("success_reason", "timeout")

        env.close()

        avg_speed = ep_speed / max(1, ep_steps)
        # Efficiency: Wh/m
        dist_approx = avg_speed * ep_steps * cfg.SIM_STEPS_PER_ACTION * 0.1  # steps * delta_t(0.1s)
        eff = ep_energy / max(dist_approx, 1.0)

        result = {
            "episode": ep,
            "mode": mode_name,
            "success": success,
            "reason": reason,
            "steps": ep_steps,
            "reward": ep_reward,
            "avg_speed": avg_speed,
            "total_energy": ep_energy,
            "energy_efficiency_wh_per_m": eff,
            "collisions": 1 if reason == "collision" else 0,
        }
        results.append(result)

        icon = "✅" if success else "❌"
        print(
            f"    EP{ep:>3}/{n_episodes} {icon} "
            f"reason={reason:<12} | "
            f"reward={ep_reward:>+8.1f} | "
            f"speed={avg_speed:.2f}m/s | "
            f"E={ep_energy:.0f}Wh"
        )

    return results


# =============================================================================
# RULE-BASED EVALUATOR
# =============================================================================
def evaluate_rulebased(
    n_episodes: int = 20,
    render: bool = False,
    seed: int = cfg.SEED,
) -> list[dict]:
    """Wrap evaluate_rulebased.run_episode for unified output format."""
    random.seed(seed)
    np.random.seed(seed)

    idm = IDMController()
    pp  = PurePursuitController()
    results = []

    for ep in range(1, n_episodes + 1):
        active_map = EVAL_MAP  # Fixed map for consistent evaluation
        result = rulebased_episode(active_map, idm, pp, render=render)

        result["episode"] = ep
        result["mode"]    = "rule_based"
        result["reward"]  = 0.0  # no reward signal for rule-based
        results.append(result)

        icon = "✅" if result["success"] else "❌"
        print(
            f"    EP{ep:>3}/{n_episodes} {icon} "
            f"reason={result['reason']:<12} | "
            f"speed={result['avg_speed']:.2f}m/s | "
            f"E={result['total_energy']:.0f}Wh"
        )

    return results


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
def compute_summary(results: list[dict], mode_name: str) -> dict:
    """Compute mean ± std for all key metrics."""
    n = len(results)
    if n == 0:
        return {}

    success_rate  = np.mean([r["success"] for r in results]) * 100
    collision_rate = np.mean([r["collisions"] for r in results]) * 100
    avg_reward    = np.mean([r["reward"] for r in results])
    std_reward    = np.std([r["reward"] for r in results])
    avg_speed     = np.mean([r["avg_speed"] for r in results])
    std_speed     = np.std([r["avg_speed"] for r in results])
    avg_energy    = np.mean([r["total_energy"] for r in results])
    std_energy    = np.std([r["total_energy"] for r in results])
    avg_eff       = np.mean([r["energy_efficiency_wh_per_m"] for r in results])
    std_eff       = np.std([r["energy_efficiency_wh_per_m"] for r in results])
    avg_steps     = np.mean([r["steps"] for r in results])

    return {
        "mode": mode_name,
        "n_episodes": n,
        "success_rate_%": f"{success_rate:.1f}",
        "collision_rate_%": f"{collision_rate:.1f}",
        "avg_reward": f"{avg_reward:.2f} ± {std_reward:.2f}",
        "avg_speed_ms": f"{avg_speed:.3f} ± {std_speed:.3f}",
        "avg_energy_wh": f"{avg_energy:.1f} ± {std_energy:.1f}",
        "avg_efficiency_wh_per_m": f"{avg_eff:.4f} ± {std_eff:.4f}",
        "avg_ep_length": f"{avg_steps:.1f}",
    }


# =============================================================================
# SAVE CSV
# =============================================================================
def save_episode_csv(results: list[dict], mode_name: str, out_dir: str) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"eval_{mode_name}_{now}.csv")

    fields = ["episode", "mode", "success", "reason", "steps",
              "reward", "avg_speed", "total_energy", "energy_efficiency_wh_per_m", "collisions"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    return path


def save_summary_csv(summaries: list[dict], out_dir: str) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"eval_comparison_{now}.csv")

    if not summaries:
        return path

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    return path


# =============================================================================
# PRINT COMPARISON TABLE
# =============================================================================
def print_comparison_table(summaries: list[dict]):
    if not summaries:
        return

    print("\n" + "=" * 90)
    print("  📊 EVALUATION COMPARISON TABLE")
    print("=" * 90)

    # Header
    col_w = 22
    metrics = [
        ("Success Rate (%)",           "success_rate_%"),
        ("Collision Rate (%)",          "collision_rate_%"),
        ("Avg Reward",                  "avg_reward"),
        ("Avg Speed (m/s)",             "avg_speed_ms"),
        ("Avg Energy (Wh)",             "avg_energy_wh"),
        ("Efficiency (Wh/m) ↓",        "avg_efficiency_wh_per_m"),
        ("Avg Ep Length (steps)",       "avg_ep_length"),
    ]

    # Method names
    methods = [s["mode"] for s in summaries]
    print(f"  {'Metric':<30}" + "".join(f"  {m:<{col_w}}" for m in methods))
    print(f"  {'─' * 30}" + ("  " + "─" * col_w) * len(methods))

    for label, key in metrics:
        row = f"  {label:<30}"
        for s in summaries:
            val = s.get(key, "N/A")
            row += f"  {val:<{col_w}}"
        print(row)

    print("=" * 90)


# =============================================================================
# FIND MODEL PATH HELPER
# =============================================================================
def find_model(model_dir: str, keyword: str) -> str:
    candidates = [
        os.path.join(model_dir, f"sac_{keyword}_final.zip"),
        os.path.join(model_dir, f"sac_{keyword}_final"),
        os.path.join(model_dir, f"ppo_{keyword}_final.zip"),
        os.path.join(model_dir, f"ppo_{keyword}_final"),
        os.path.join(model_dir, "best", "best_model.zip"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback: any .zip
    if os.path.isdir(model_dir):
        zips = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir) if f.endswith(".zip")
        ]
        if zips:
            return max(zips, key=os.path.getmtime)
    raise FileNotFoundError(f"No model found for '{keyword}' in: {model_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Evaluation — Curriculum SAC vs Flat SAC vs Rule-Based"
    )
    parser.add_argument(
        "--mode", choices=["curriculum", "kd", "flat", "rulebased", "all"],
        default="all", help="Which method(s) to evaluate"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per method")
    parser.add_argument("--render",   action="store_true", help="Show SUMO-GUI")
    parser.add_argument("--seed",     type=int, default=cfg.SEED, help="Random seed")
    parser.add_argument(
        "--curriculum_model",
        default=None,
        help="Path to curriculum SAC model (auto-detected if omitted)"
    )
    parser.add_argument(
        "--kd_model",
        default=None,
        help="Path to KD SAC model (auto-detected if omitted)"
    )
    parser.add_argument(
        "--flat_model",
        default=None,
        help="Path to flat SAC model (auto-detected if omitted)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  UNIFIED EVALUATION SCRIPT")
    print(f"  Mode: {args.mode}  |  Episodes/method: {args.episodes}")
    print(f"  Seed: {args.seed}  |  Render: {args.render}")
    print("=" * 60)

    all_summaries = []
    run_modes = (
        ["curriculum", "kd", "flat", "rulebased"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in run_modes:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {mode.upper()}")
        print(f"{'─' * 60}")

        if mode in ("curriculum", "flat", "kd"):
            # Locate model
            if mode == "curriculum":
                model_path = (
                    args.curriculum_model
                    or find_model(cfg.STAGE3_MODEL_DIR, "finetune")
                )
                mode_name = "curriculum_sac"
            elif mode == "kd":
                model_path = (
                    args.kd_model
                    or find_model(cfg.KD_MODEL_DIR, "kd_sac")
                )
                mode_name = "kd_sac"
            else:
                model_path = (
                    args.flat_model
                    or find_model(cfg.FLAT_MODEL_DIR, "flat_baseline")
                )
                mode_name = "flat_ppo"

            algo = PPO if mode_name == "flat_ppo" else SAC

            results = evaluate_rl_model(
                model_path=model_path,
                mode_name=mode_name,
                algo_class=algo,
                n_episodes=args.episodes,
                render=args.render,
                seed=args.seed,
            )

        else:  # rulebased
            mode_name = "rule_based"
            results = evaluate_rulebased(
                n_episodes=args.episodes,
                render=args.render,
                seed=args.seed,
            )

        # Save episode-level CSV
        csv_path = save_episode_csv(results, mode_name, EVAL_OUTPUT_DIR)
        print(f"\n  Episode CSV saved: {csv_path}")

        # Compute summary
        summary = compute_summary(results, mode_name)
        all_summaries.append(summary)

    # Save comparison table CSV
    summary_csv = save_summary_csv(all_summaries, EVAL_OUTPUT_DIR)

    # Print comparison table
    print_comparison_table(all_summaries)

    print(f"\n  Comparison CSV: {summary_csv}")
    print(f"  All outputs in: {EVAL_OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
