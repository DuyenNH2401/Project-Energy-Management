# visualize.py
# Reward & Telemetry Visualization for 3-Stage Curriculum Learning.
#
# Usage:
#   python visualize.py                     # Auto-detect latest CSV logs
#   python visualize.py --stage 1           # Plot only Stage 1
#   python visualize.py --stage 2           # Plot only Stage 2
#   python visualize.py --stage 3           # Plot only Stage 3
#   python visualize.py --all               # Plot all stages combined
#   python visualize.py --csv path/to.csv   # Plot a specific CSV file
#   python visualize.py --live              # Live mode: auto-refresh every 30s

import os
import sys
import glob
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import config as cfg

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================
COLORS = {
    "stage1": "#4FC3F7",   # Light Blue
    "stage2": "#81C784",   # Green
    "stage3": "#FFB74D",   # Orange
    "reward": "#E91E63",   # Pink
    "speed":  "#2196F3",   # Blue
    "energy": "#FF5722",   # Deep Orange
    "success": "#4CAF50",  # Green
    "wiggle": "#9C27B0",   # Purple
    "safety": "#F44336",   # Red
    "rolling": "#FFFFFF",  # White (for rolling avg line)
    "raw":    "#FFFFFF22", # Transparent white (for raw data)
}

STAGE_NAMES = {
    0: "Baseline: Flat PPO",
    1: "Stage 1: Lateral Control",
    2: "Stage 2: Longitudinal (Energy)",
    3: "Stage 3: Fine-Tuning (Fusion)",
}

STAGE_COLORS = {
    0: "#E0E0E0",
    1: COLORS["stage1"],
    2: COLORS["stage2"],
    3: COLORS["stage3"],
}

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e94560",
    "axes.labelcolor":  "#eee",
    "axes.grid":        True,
    "grid.color":       "#ffffff15",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#eee",
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#e94560",
    "font.family":      "sans-serif",
    "font.size":        10,
})


# =============================================================================
# CSV DISCOVERY
# =============================================================================
def find_latest_csv(log_dir: str) -> str | None:
    """Find the most recently modified CSV file in a directory."""
    pattern = os.path.join(log_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_latest_csv_multi(log_dirs: list[str]) -> str | None:
    """Find the most recently modified CSV across multiple directories.
    Excludes kd_metrics_* files which contain KL divergence data, not episode logs."""
    all_files = []
    for d in log_dirs:
        pattern = os.path.join(d, "*.csv")
        for f in glob.glob(pattern):
            # Skip KD metrics CSVs — they don't have episode/reward columns
            if os.path.basename(f).startswith("kd_metrics"):
                continue
            all_files.append(f)
    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and clean a training CSV log."""
    df = pd.read_csv(csv_path)

    # Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Convert numeric columns
    numeric_cols = ["episode", "timestep", "steps", "reward",
                    "avg_speed", "total_energy", "avg_wiggle", "avg_safety",
                    "success", "rolling_avg_reward", "rolling_avg_speed",
                    "rolling_avg_energy", "rolling_success_rate"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# ROLLING AVERAGE HELPER
# =============================================================================
def smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute a rolling average with edge padding."""
    if len(data) < window:
        window = max(1, len(data) // 2)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


# =============================================================================
# SINGLE STAGE PLOT — 6-panel detailed view
# =============================================================================
def plot_single_stage(df: pd.DataFrame, stage_num: int, save_path: str | None = None):
    """Create a detailed 6-panel plot for one training stage."""

    color = STAGE_COLORS.get(stage_num, COLORS["stage1"])
    title = STAGE_NAMES.get(stage_num, f"Stage {stage_num}")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"📊  {title}  —  Training Report",
        fontsize=18, fontweight="bold", color=color, y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25,
                           left=0.07, right=0.95, top=0.92, bottom=0.06)

    episodes = df["episode"].values
    x_col = "episode"
    x = df[x_col].values

    # ---- 1. REWARD ----
    ax1 = fig.add_subplot(gs[0, 0])
    reward = df["reward"].values
    ax1.fill_between(x, reward, alpha=0.15, color=COLORS["reward"])
    ax1.plot(x, reward, alpha=0.3, linewidth=0.5, color=COLORS["reward"], label="Raw")
    ax1.plot(x, smooth(reward), linewidth=2, color=COLORS["reward"], label="Rolling Avg")
    ax1.set_title("Episode Reward", fontweight="bold")
    ax1.set_xlabel(x_col.capitalize())
    ax1.set_ylabel("Reward")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.axhline(y=0, color="#ffffff30", linestyle="-", linewidth=0.5)

    # ---- 2. AVERAGE SPEED ----
    ax2 = fig.add_subplot(gs[0, 1])
    speed = df["avg_speed"].values
    ax2.fill_between(x, speed, alpha=0.15, color=COLORS["speed"])
    ax2.plot(x, speed, alpha=0.3, linewidth=0.5, color=COLORS["speed"], label="Raw")
    ax2.plot(x, smooth(speed), linewidth=2, color=COLORS["speed"], label="Rolling Avg")
    ax2.set_title("Average Speed (m/s)", fontweight="bold")
    ax2.set_xlabel(x_col.capitalize())
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend(loc="lower right", fontsize=8)

    # ---- 3. ENERGY CONSUMPTION ----
    ax3 = fig.add_subplot(gs[1, 0])
    energy = df["total_energy"].values
    ax3.fill_between(x, energy, alpha=0.15, color=COLORS["energy"])
    ax3.plot(x, energy, alpha=0.3, linewidth=0.5, color=COLORS["energy"], label="Raw")
    ax3.plot(x, smooth(energy), linewidth=2, color=COLORS["energy"], label="Rolling Avg")
    ax3.set_title("Energy Consumption per Episode (Wh)", fontweight="bold")
    ax3.set_xlabel(x_col.capitalize())
    ax3.set_ylabel("Energy (Wh)")
    ax3.legend(loc="upper right", fontsize=8)

    # ---- 4. SUCCESS RATE ----
    ax4 = fig.add_subplot(gs[1, 1])
    if "rolling_success_rate" in df.columns:
        sr = df["rolling_success_rate"].values
        ax4.plot(x, sr, linewidth=2, color=COLORS["success"])
        ax4.fill_between(x, sr, alpha=0.15, color=COLORS["success"])
    else:
        success = df["success"].values.astype(float)
        sr = smooth(success, window=30) * 100
        ax4.plot(x, sr, linewidth=2, color=COLORS["success"])
        ax4.fill_between(x, sr, alpha=0.15, color=COLORS["success"])
    ax4.set_title("Success Rate (%)", fontweight="bold")
    ax4.set_xlabel(x_col.capitalize())
    ax4.set_ylabel("Success %")
    ax4.set_ylim(-5, 105)
    ax4.axhline(y=50, color="#ffffff20", linestyle="--", linewidth=0.5)

    # ---- 5. WIGGLE (Comfort) ----
    ax5 = fig.add_subplot(gs[2, 0])
    if "avg_wiggle" in df.columns:
        wiggle = df["avg_wiggle"].values
        ax5.fill_between(x, wiggle, alpha=0.15, color=COLORS["wiggle"])
        ax5.plot(x, wiggle, alpha=0.3, linewidth=0.5, color=COLORS["wiggle"], label="Raw")
        ax5.plot(x, smooth(wiggle), linewidth=2, color=COLORS["wiggle"], label="Rolling Avg")
        ax5.set_title("Wiggle / Action Smoothness", fontweight="bold")
        ax5.set_xlabel(x_col.capitalize())
        ax5.set_ylabel("Wiggle (lower = smoother)")
        ax5.legend(loc="upper right", fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No wiggle data", transform=ax5.transAxes,
                 ha="center", va="center", fontsize=14, color="#666")

    # ---- 6. EPISODE LENGTH ----
    ax6 = fig.add_subplot(gs[2, 1])
    steps = df["steps"].values
    ax6.fill_between(x, steps, alpha=0.15, color=color)
    ax6.plot(x, steps, alpha=0.3, linewidth=0.5, color=color, label="Raw")
    ax6.plot(x, smooth(steps), linewidth=2, color=color, label="Rolling Avg")
    ax6.set_title("Episode Length (steps)", fontweight="bold")
    ax6.set_xlabel(x_col.capitalize())
    ax6.set_ylabel("Steps")
    ax6.legend(loc="lower right", fontsize=8)

    # Add stats text box
    total_eps = len(df)
    total_success = df["success"].sum() if "success" in df.columns else 0
    stats_text = (
        f"Episodes: {total_eps} | "
        f"Success: {total_success}/{total_eps} ({total_success/max(1,total_eps)*100:.1f}%) | "
        f"Best Reward: {reward.max():+.1f} | "
        f"Final Avg Reward: {reward[-min(20,len(reward)):].mean():+.1f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, color="#aaa",
             style="italic")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 Saved: {save_path}")

    return fig


# =============================================================================
# CROSS-STAGE COMPARISON — 4-panel overview
# =============================================================================
def plot_comparison(dfs: dict[int, pd.DataFrame], save_path: str | None = None):
    """Compare training metrics across stages in a single figure."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "🔬  3-Stage Curriculum Learning  —  Comparison Dashboard",
        fontsize=18, fontweight="bold", color="#FFD700", y=0.98,
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25,
                        left=0.07, right=0.95, top=0.90, bottom=0.08)

    metrics = [
        ("reward",       "Episode Reward",           axes[0, 0]),
        ("avg_speed",    "Average Speed (m/s)",       axes[0, 1]),
        ("total_energy", "Energy Consumption (Wh)",   axes[1, 0]),
    ]

    for col, title, ax in metrics:
        for stage, df in sorted(dfs.items()):
            x = df["episode"].values
            y = df[col].values
            color = STAGE_COLORS[stage]
            label = STAGE_NAMES[stage]

            ax.plot(x, y, alpha=0.15, linewidth=0.5, color=color)
            ax.plot(x, smooth(y), linewidth=2.5, color=color, label=label)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.legend(fontsize=8, loc="best")
        ax.axhline(y=0, color="#ffffff20", linestyle="-", linewidth=0.5)

    # Success rate comparison
    ax_sr = axes[1, 1]
    for stage, df in sorted(dfs.items()):
        color = STAGE_COLORS[stage]
        label = STAGE_NAMES[stage]

        if "rolling_success_rate" in df.columns:
            sr = df["rolling_success_rate"].values
        else:
            success = df["success"].values.astype(float)
            sr = smooth(success, window=30) * 100

        x = df["episode"].values
        ax_sr.plot(x, sr, linewidth=2.5, color=color, label=label)

    ax_sr.set_title("Success Rate (%)", fontweight="bold")
    ax_sr.set_xlabel("Episode")
    ax_sr.set_ylabel("Success %")
    ax_sr.set_ylim(-5, 105)
    ax_sr.legend(fontsize=8, loc="best")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 Saved: {save_path}")

    return fig


# =============================================================================
# REWARD BREAKDOWN — Stacked area chart concept
# =============================================================================
def plot_reward_analysis(df: pd.DataFrame, stage_num: int, save_path: str | None = None):
    """Detailed reward analysis: distribution, histogram, and trend."""

    color = STAGE_COLORS.get(stage_num, COLORS["stage1"])
    title = STAGE_NAMES.get(stage_num, f"Stage {stage_num}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"🎯  {title}  —  Reward Analysis",
        fontsize=16, fontweight="bold", color=color, y=1.02,
    )
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.97, top=0.88, bottom=0.12)

    reward = df["reward"].values
    episodes = df["episode"].values

    # 1. Reward distribution (Histogram)
    ax1 = axes[0]
    ax1.hist(reward, bins=50, color=color, alpha=0.7, edgecolor="#ffffff30")
    ax1.axvline(x=np.mean(reward), color="#FFD700", linestyle="--",
                linewidth=2, label=f"Mean: {np.mean(reward):+.1f}")
    ax1.axvline(x=np.median(reward), color="#E91E63", linestyle="--",
                linewidth=2, label=f"Median: {np.median(reward):+.1f}")
    ax1.set_title("Reward Distribution", fontweight="bold")
    ax1.set_xlabel("Reward")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)

    # 2. Reward over time with confidence band
    ax2 = axes[1]
    window = 20
    roll_mean = pd.Series(reward).rolling(window, min_periods=1).mean().values
    roll_std = pd.Series(reward).rolling(window, min_periods=1).std().fillna(0).values

    ax2.fill_between(episodes, roll_mean - roll_std, roll_mean + roll_std,
                     alpha=0.2, color=color, label="±1 Std Dev")
    ax2.plot(episodes, roll_mean, linewidth=2, color=color, label="Rolling Mean")
    ax2.plot(episodes, reward, alpha=0.1, linewidth=0.5, color=color)
    ax2.set_title("Reward Trend ± Std", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.legend(fontsize=8)

    # 3. Cumulative reward
    ax3 = axes[2]
    cumulative = np.cumsum(reward)
    ax3.plot(episodes, cumulative, linewidth=2, color=color)
    ax3.fill_between(episodes, cumulative, alpha=0.15, color=color)
    ax3.set_title("Cumulative Reward", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cumulative Reward")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 Saved: {save_path}")

    return fig


# =============================================================================
# LIVE MODE — Auto-refresh
# =============================================================================
def live_plot(csv_path: str, stage_num: int, refresh_sec: int = 30):
    """Continuously refresh the plot as training progresses."""
    print(f"\n🔴 LIVE MODE — Refreshing every {refresh_sec}s")
    print(f"   Watching: {csv_path}")
    print(f"   Press Ctrl+C to stop.\n")

    plt.ion()
    fig = None

    try:
        while True:
            if os.path.exists(csv_path):
                df = load_csv(csv_path)
                if len(df) > 0:
                    if fig is not None:
                        plt.close(fig)
                    fig = plot_single_stage(df, stage_num)
                    plt.draw()
                    plt.pause(0.1)
                    print(f"  🔄 Refreshed at {datetime.now().strftime('%H:%M:%S')} "
                          f"| Episodes: {len(df)}")
            else:
                print(f"  ⏳ Waiting for CSV to appear...")

            time.sleep(refresh_sec)

    except KeyboardInterrupt:
        print("\n  🛑 Live mode stopped.")
        plt.ioff()


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3-Stage Curriculum Learning and Baseline training logs"
    )
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3],
                        help="Plot a specific stage (0=Baseline, 1, 2, or 3)")
    parser.add_argument("--baseline", action="store_true",
                        help="Plot the baseline (Flat PPO) training log")
    parser.add_argument("--all", action="store_true",
                        help="Plot all stages comparison")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a specific CSV file")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: auto-refresh every 30s")
    parser.add_argument("--refresh", type=int, default=30,
                        help="Refresh interval for live mode (seconds)")
    parser.add_argument("--save", action="store_true",
                        help="Save plots as PNG files")

    args = parser.parse_args()

    save_dir = os.path.join(cfg.BASE_REPORT_DIR, "plots/")
    os.makedirs(save_dir, exist_ok=True)

    stage_dirs = {
        0: cfg.FLAT_LOG_DIR,
        1: cfg.STAGE1_LOG_DIR,
        2: cfg.STAGE2_LOG_DIR,
        3: cfg.STAGE3_LOG_DIR,
    }

    # Stage 3 fallback: also check KD log directory
    # train_kd_finetune.py saves CSV to cfg.KD_LOG_DIR, not STAGE3_LOG_DIR
    stage3_fallback_dirs = [cfg.STAGE3_LOG_DIR, cfg.KD_LOG_DIR]

    if args.baseline:
        args.stage = 0

    # --- Specific CSV file ---
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"❌ File not found: {args.csv}")
            sys.exit(1)
        df = load_csv(args.csv)
        stage_num = args.stage if args.stage is not None else 1

        if args.live:
            live_plot(args.csv, stage_num, args.refresh)
        else:
            save_path = os.path.join(save_dir, f"stage{stage_num}_detail.png") if args.save else None
            plot_single_stage(df, stage_num, save_path)
            save_path2 = os.path.join(save_dir, f"stage{stage_num}_reward_analysis.png") if args.save else None
            plot_reward_analysis(df, stage_num, save_path2)
            plt.show()
        return

    # --- Single stage ---
    if args.stage is not None:
        if args.stage == 3:
            csv_path = find_latest_csv_multi(stage3_fallback_dirs)
        else:
            csv_path = find_latest_csv(stage_dirs[args.stage])

        if csv_path is None:
            search_dirs = stage3_fallback_dirs if args.stage == 3 else [stage_dirs[args.stage]]
            print(f"❌ No CSV found in {search_dirs}")
            print(f"   Run train_{'baseline_flat' if args.stage == 0 else 'lateral' if args.stage == 1 else 'longitudinal' if args.stage == 2 else 'kd_finetune'}.py first.")
            sys.exit(1)

        print(f"📂 Loading: {csv_path}")

        if args.live:
            live_plot(csv_path, args.stage, args.refresh)
        else:
            df = load_csv(csv_path)
            save_p1 = os.path.join(save_dir, f"stage{args.stage}_detail.png") if args.save else None
            save_p2 = os.path.join(save_dir, f"stage{args.stage}_reward_analysis.png") if args.save else None
            plot_single_stage(df, args.stage, save_p1)
            plot_reward_analysis(df, args.stage, save_p2)
            plt.show()
        return

    # --- All stages comparison (default if no args) ---
    dfs = {}
    for stage, log_dir in stage_dirs.items():
        if stage == 3:
            csv_path = find_latest_csv_multi(stage3_fallback_dirs)
        else:
            csv_path = find_latest_csv(log_dir)
        if csv_path:
            print(f"📂 Stage {stage}: {csv_path}")
            dfs[stage] = load_csv(csv_path)
        else:
            search_info = f"{stage3_fallback_dirs}" if stage == 3 else f"{log_dir}"
            print(f"⚠️  Stage {stage}: No CSV found in {search_info}")

    if not dfs:
        print("\n❌ No training logs found. Run a training script first!")
        print("   python train_lateral.py")
        sys.exit(1)

    # Plot each available stage
    for stage, df in dfs.items():
        save_p = os.path.join(save_dir, f"stage{stage}_detail.png") if args.save else None
        plot_single_stage(df, stage, save_p)

    # Comparison (only if 2+ stages)
    if len(dfs) >= 2 or args.all:
        save_comp = os.path.join(save_dir, "comparison_dashboard.png") if args.save else None
        plot_comparison(dfs, save_comp)

    plt.show()


if __name__ == "__main__":
    main()
