"""
test_agent.py
=============
Evaluate the trained PPO agent on the E1 → E13 route in map1.
This is the exact same route used during training (test_mode=False).

Usage
-----
# Auto-find latest best checkpoint, GUI on, 20 episodes:
    python test_agent.py

# Headless, 50 episodes:
    python test_agent.py --no-render --episodes 50

# Specify a checkpoint manually:
    python test_agent.py --model models/tianshou_ppo/best_policy_XYZ.pth

# Also run random-action baseline for comparison:
    python test_agent.py --no-render --baseline

# Print every step (debug single episode):
    python test_agent.py --episodes 1 --verbose
"""

import os
import sys
import csv
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# ── Tianshou — exact same setup as ppo.py ────────────────────────────────────
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

# ── SUMO Environment — same import chain as ppo.py ───────────────────────────
try:
    from simulation.continuous_sumo_env_tum_lum import SumoEnv
except ImportError:
    from continuous_sumo_env_tum_lum import SumoEnv


# =============================================================================
# CONFIG — must stay in sync with ppo.py
# =============================================================================

MAP_CONFIGS = ["maps/map1/run.sumocfg"]
MODEL_DIR   = "models/tianshou_ppo/"
REPORT_DIR  = "reports/tianshou_ppo/eval/"

# Network architecture — MUST match training exactly
HIDDEN_SIZES = [256, 256]
OBS_SHAPE    = (22,)        # SumoEnv observation_space.shape
ACTION_SHAPE = (2,)         # [steering, throttle]

# PPO hyperparams (only needed to reconstruct the policy object)
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
MAX_GRAD_NORM = 0.5
VF_COEF       = 0.25
ENT_COEF      = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_DT = 0.1  # seconds per SUMO simulation step


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def find_latest_checkpoint(model_dir: str) -> str | None:
    """Return the newest best_policy_*.pth, or any .pth if none found."""
    d = Path(model_dir)
    if not d.exists():
        return None
    all_ckpts  = sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    best_ckpts = [p for p in all_ckpts if "best_policy" in p.name]
    chosen     = (best_ckpts or all_ckpts)
    return str(chosen[0]) if chosen else None


def load_policy(checkpoint_path: str) -> PPOPolicy:
    """Rebuild the PPO policy and load saved weights."""
    actor_net  = Net(OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    critic_net = Net(OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=DEVICE)

    actor  = ActorProb(actor_net,  ACTION_SHAPE, device=DEVICE, unbounded=True).to(DEVICE)
    critic = Critic(critic_net, device=DEVICE).to(DEVICE)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )

    def dist_fn(mu, sigma):
        return torch.distributions.Independent(
            torch.distributions.Normal(mu, sigma), 1
        )

    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist_fn,
        action_space=None,
        discount_factor=GAMMA, gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM, vf_coef=VF_COEF, ent_coef=ENT_COEF,
        action_scaling=True, action_bound_method="clip",
    )

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"[Test] Loaded : {checkpoint_path}")
    print(f"[Test] Device : {DEVICE}")
    return policy


# =============================================================================
# ACTION SELECTION
# =============================================================================

def get_deterministic_action(policy: PPOPolicy, obs: np.ndarray) -> np.ndarray:
    """
    Deterministic eval action: use actor mean (mu), no exploration noise.
    This is the correct way to evaluate a trained PPO agent.
    """
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        (mu, _sigma), _ = policy.actor(obs_t, state=None)
    return np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)


# =============================================================================
# SINGLE EPISODE
# =============================================================================

def run_episode(
    env: SumoEnv,
    policy: PPOPolicy | None,  # None = random baseline
    episode_idx: int,
    verbose: bool,
) -> dict:
    obs, _ = env.reset()

    # ── Accumulators ──────────────────────────────────────────────────────────
    total_reward   = 0.0
    total_energy   = 0.0
    speed_sum      = 0.0
    wiggle_sum     = 0.0
    safety_sum     = 0.0
    n_steps        = 0
    speeds         = []
    steers         = []
    throttles      = []
    terminated     = truncated = False
    reason         = "running"
    success        = 0
    t_start        = time.perf_counter()

    while not (terminated or truncated):
        if policy is not None:
            action = get_deterministic_action(policy, obs)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        spd = info.get("real_speed",  0.0)
        total_reward += reward
        total_energy += info.get("real_energy", 0.0)
        speed_sum    += spd
        wiggle_sum   += abs(info.get("wiggle",  0.0))
        safety_sum   += info.get("safety",      0.0)
        n_steps      += 1
        speeds.append(spd)
        steers.append(float(action[0]))
        throttles.append(float(action[1]))

        if verbose:
            print(
                f"    step {n_steps:>4d} | "
                f"rew={reward:+.3f} | "
                f"spd={spd:.1f} m/s ({spd*3.6:.0f} km/h) | "
                f"energy={info.get('real_energy',0):.2f} Wh | "
                f"safety={info.get('safety',0):.3f} | "
                f"wiggle={info.get('wiggle',0):.4f} | "
                f"steer={action[0]:+.2f} throttle={action[1]:+.2f}"
            )

        reason  = info.get("reason",     "running")
        success = info.get("is_success", 0)

    # ── Compute derived metrics ───────────────────────────────────────────────
    wall_s      = time.perf_counter() - t_start
    n           = max(1, n_steps)
    avg_spd_ms  = speed_sum / n
    sim_s       = n_steps * SIM_DT
    dist_m      = avg_spd_ms * sim_s
    dist_km     = max(dist_m / 1000.0, 1e-9)

    speeds_arr  = np.array(speeds, dtype=np.float32)
    peak_spd_ms = float(speeds_arr.max()) if len(speeds_arr) else 0.0
    pct_fast    = float(np.mean(speeds_arr > 13.9)) * 100  # % steps over 50 km/h

    return {
        "episode":          episode_idx,
        "agent":            "PPO" if policy is not None else "Random",
        "success":          success,
        "reason":           reason,
        "steps":            n_steps,
        "sim_time_s":       round(sim_s,       1),
        "wall_time_s":      round(wall_s,       2),
        "total_reward":     round(total_reward, 2),
        "avg_speed_ms":     round(avg_spd_ms,   2),
        "avg_speed_kmh":    round(avg_spd_ms * 3.6, 2),
        "peak_speed_ms":    round(peak_spd_ms,  2),
        "peak_speed_kmh":   round(peak_spd_ms * 3.6, 2),
        "pct_above_50kmh":  round(pct_fast,     1),
        "total_energy_wh":  round(total_energy, 1),
        "est_dist_m":       round(dist_m,        1),
        "energy_per_km_wh": round(total_energy / dist_km, 1),
        "avg_wiggle":       round(wiggle_sum / n, 4),
        "avg_safety":       round(safety_sum / n, 4),
        "steer_std":        round(float(np.std(steers)),    4),
        "throttle_std":     round(float(np.std(throttles)), 4),
    }


# =============================================================================
# AGGREGATE STATISTICS
# =============================================================================

def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}

    def _stat(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return {
            "mean": round(float(np.mean(vals)), 3),
            "std":  round(float(np.std(vals)),  3),
            "min":  round(float(np.min(vals)),  3),
            "max":  round(float(np.max(vals)),  3),
        } if vals else {}

    reasons = {}
    for r in results:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    return {
        "n":              len(results),
        "success_rate":   round(np.mean([r["success"] for r in results]) * 100, 1),
        "reasons":        reasons,
        "reward":         _stat("total_reward"),
        "steps":          _stat("steps"),
        "sim_time_s":     _stat("sim_time_s"),
        "avg_speed_kmh":  _stat("avg_speed_kmh"),
        "peak_speed_kmh": _stat("peak_speed_kmh"),
        "pct_above_50":   _stat("pct_above_50kmh"),
        "energy_wh":      _stat("total_energy_wh"),
        "energy_per_km":  _stat("energy_per_km_wh"),
        "wiggle":         _stat("avg_wiggle"),
        "safety":         _stat("avg_safety"),
        "steer_std":      _stat("steer_std"),
        "throttle_std":   _stat("throttle_std"),
    }


# =============================================================================
# CONSOLE DISPLAY
# =============================================================================

W = 62

def print_episode(r: dict, idx: int, total: int) -> None:
    icon = "✓" if r["success"] else "✗"
    print(
        f"  Ep {idx:>3}/{total} {icon} | "
        f"rew={r['total_reward']:>+8.1f} | "
        f"steps={r['steps']:>4} | "
        f"{r['avg_speed_kmh']:>5.1f} km/h | "
        f"{r['total_energy_wh']:>6.0f} Wh | "
        f"safety={r['avg_safety']:.3f} | "
        f"{r['reason']}"
    )


def print_summary(agg: dict, label: str) -> None:
    def _m(key):  return (agg.get(key) or {}).get("mean", 0)
    def _s(key):  return (agg.get(key) or {}).get("std",  0)
    def _mn(key): return (agg.get(key) or {}).get("min",  0)
    def _mx(key): return (agg.get(key) or {}).get("max",  0)

    def bar(val, lo, hi, w=18):
        pct = max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9)))
        f   = int(pct * w)
        return "█" * f + "░" * (w - f)

    def row(name, key, unit="", fmt=".2f"):
        m = _m(key); s = _s(key); mn = _mn(key); mx = _mx(key)
        b = bar(m, mn, mx)
        print(f"  {name:<24} {m:{fmt}} ± {s:{fmt}} {unit:<6}  {b}")

    print()
    print("═" * W)
    print(f"  {label}")
    print(f"  Route: E1 → E13  |  Map: map1")
    print("═" * W)
    print(f"  {'Episodes':<24} {agg['n']}")
    print(f"  {'Success rate':<24} {agg['success_rate']} %")
    print(f"  {'Termination breakdown':<24}", end="")
    for rsn, cnt in agg["reasons"].items():
        pct = cnt / agg["n"] * 100
        print(f"  {rsn}: {cnt} ({pct:.0f}%)", end="")
    print()
    print("─" * W)
    row("Total reward",       "reward",         "",       ".1f")
    row("Steps",              "steps",          "steps",  ".0f")
    row("Sim time",           "sim_time_s",     "s",      ".1f")
    print("─" * W)
    row("Avg speed",          "avg_speed_kmh",  "km/h",   ".1f")
    row("Peak speed",         "peak_speed_kmh", "km/h",   ".1f")
    row("Time > 50 km/h",     "pct_above_50",   "%",      ".1f")
    print("─" * W)
    row("Total energy",       "energy_wh",      "Wh",     ".0f")
    row("Energy / km",        "energy_per_km",  "Wh/km",  ".0f")
    print("─" * W)
    row("Avg wiggle",         "wiggle",         "",       ".4f")
    row("Avg safety score",   "safety",         "",       ".4f")
    row("Steering std-dev",   "steer_std",      "",       ".4f")
    row("Throttle std-dev",   "throttle_std",   "",       ".4f")
    print("═" * W)


def print_comparison(a: dict, b: dict) -> None:
    rows = [
        # (key,              label,               higher_better, fmt)
        ("success_rate",     "Success rate",       True,  ".1f"),
        ("reward",           "Total reward",       True,  ".1f"),
        ("avg_speed_kmh",    "Avg speed (km/h)",   True,  ".1f"),
        ("peak_speed_kmh",   "Peak speed (km/h)",  True,  ".1f"),
        ("energy_wh",        "Total energy (Wh)",  False, ".0f"),
        ("energy_per_km",    "Energy/km (Wh/km)",  False, ".0f"),
        ("wiggle",           "Avg wiggle",         False, ".4f"),
        ("safety",           "Avg safety score",   True,  ".4f"),
        ("steps",            "Steps to finish",    False, ".0f"),
    ]

    print()
    print("═" * W)
    print("  PPO Agent  vs  Random Baseline  —  E1→E13 map1")
    print("═" * W)
    print(f"  {'Metric':<24} {'Agent':>9} {'Random':>9} {'Δ':>9}  ")
    print("─" * W)

    for key, label, higher_better, fmt in rows:
        if key == "success_rate":
            av = a.get(key, 0); bv = b.get(key, 0)
        else:
            av = (a.get(key) or {}).get("mean", 0)
            bv = (b.get(key) or {}).get("mean", 0)
        delta  = av - bv
        better = (delta >= 0) if higher_better else (delta <= 0)
        sign   = "+" if delta >= 0 else ""
        tag    = "✓" if better else "✗"
        print(f"  {label:<24} {av:>9{fmt}} {bv:>9{fmt}}  {sign}{delta:>{fmt}}  {tag}")

    print("═" * W)
    print()


# =============================================================================
# CSV WRITERS
# =============================================================================

EPISODE_FIELDS = [
    "episode", "agent", "success", "reason", "steps",
    "sim_time_s", "wall_time_s", "total_reward",
    "avg_speed_ms", "avg_speed_kmh", "peak_speed_ms", "peak_speed_kmh",
    "pct_above_50kmh", "total_energy_wh", "est_dist_m", "energy_per_km_wh",
    "avg_wiggle", "avg_safety", "steer_std", "throttle_std",
]

def save_episodes_csv(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[Report] Episodes → {path}")


def save_summary_csv(agg: dict, label: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stat_keys = [
        ("reward",         ""),
        ("steps",          "steps"),
        ("sim_time_s",     "s"),
        ("avg_speed_kmh",  "km/h"),
        ("peak_speed_kmh", "km/h"),
        ("pct_above_50",   "%"),
        ("energy_wh",      "Wh"),
        ("energy_per_km",  "Wh/km"),
        ("wiggle",         ""),
        ("safety",         ""),
        ("steer_std",      ""),
        ("throttle_std",   ""),
    ]
    rows = [["metric", "mean", "std", "min", "max", "unit", "agent"]]
    rows.append(["n_episodes",    agg["n"],            "", "", "", "episodes", label])
    rows.append(["success_rate",  agg["success_rate"], "", "", "", "%",        label])
    for rkey, unit in stat_keys:
        s = agg.get(rkey, {})
        rows.append([rkey, s.get("mean",""), s.get("std",""),
                     s.get("min",""), s.get("max",""), unit, label])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[Report] Summary  → {path}")


# =============================================================================
# MAIN
# =============================================================================

def evaluate(args) -> None:
    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt = args.model or find_latest_checkpoint(MODEL_DIR)
    if ckpt is None:
        sys.exit(
            f"[Error] No .pth checkpoint found in '{MODEL_DIR}'.\n"
            f"        Train the agent first with ppo.py, or pass --model <path>."
        )

    print()
    print("═" * W)
    print("  SUMO PPO EVALUATION — E1 → E13  (map1)")
    print("═" * W)
    print(f"  Checkpoint : {ckpt}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Render     : {args.render}")
    print(f"  Baseline   : {args.baseline}")
    print(f"  Device     : {DEVICE}")
    print("═" * W)

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = load_policy(ckpt)

    # ── Shared env factory ────────────────────────────────────────────────────
    # test_mode=False is intentional: this is EXACTLY the training route (E1→E13)
    # and the training environment setup. No XML route file needed.
    def make_env(render: bool) -> SumoEnv:
        return SumoEnv(
            render      = render,
            map_config  = MAP_CONFIGS,
            test_mode   = False,    # uses same E1→E13 fixed route as training
            delay       = 50 if render else 0,
        )

    # ── PPO agent episodes ────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  Running {args.episodes} episodes with trained PPO agent…")
    print(f"{'─'*W}")

    agent_env     = make_env(args.render)
    agent_results = []

    for ep in range(1, args.episodes + 1):
        if args.verbose:
            print(f"\n  ── Episode {ep}/{args.episodes} ──────────────────────")
        r = run_episode(agent_env, policy, ep, args.verbose)
        agent_results.append(r)
        print_episode(r, ep, args.episodes)

    agent_env.close()
    agent_agg = aggregate(agent_results)
    print_summary(agent_agg, "PPO AGENT SUMMARY")

    # ── Random baseline (optional) ────────────────────────────────────────────
    rand_results = []
    rand_agg     = {}

    if args.baseline:
        print(f"\n{'─'*W}")
        print(f"  Running {args.episodes} episodes with random baseline…")
        print(f"{'─'*W}")

        rand_env = make_env(render=False)
        for ep in range(1, args.episodes + 1):
            r = run_episode(rand_env, None, ep, verbose=False)
            rand_results.append(r)
            print_episode(r, ep, args.episodes)
        rand_env.close()

        rand_agg = aggregate(rand_results)
        print_summary(rand_agg, "RANDOM BASELINE SUMMARY")
        print_comparison(agent_agg, rand_agg)

    # ── Save reports ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    os.makedirs(REPORT_DIR, exist_ok=True)

    all_results = agent_results + rand_results
    save_episodes_csv(all_results,
                      os.path.join(REPORT_DIR, f"episodes_{ts}.csv"))
    save_summary_csv(agent_agg,  "PPO",
                     os.path.join(REPORT_DIR, f"summary_agent_{ts}.csv"))
    if rand_agg:
        save_summary_csv(rand_agg, "Random",
                         os.path.join(REPORT_DIR, f"summary_baseline_{ts}.csv"))

    print(f"\n[Test] Reports saved to: {REPORT_DIR}")
    print("[Test] Done.")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PPO agent on the E1→E13 training route (map1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",     type=str,  default=None,
                   help="Path to .pth checkpoint. Auto-selects newest if omitted.")
    p.add_argument("--episodes",  type=int,  default=20,
                   help="Number of evaluation episodes (default: 20).")
    p.add_argument("--render",    dest="render", action="store_true",  default=True,
                   help="Show SUMO GUI (default: on).")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Run headless — much faster.")
    p.add_argument("--verbose",   action="store_true", default=False,
                   help="Print every step within each episode.")
    p.add_argument("--baseline",  action="store_true", default=False,
                   help="Also run random-action baseline and show comparison table.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())