# visualize_paper.py
# Phase E — Paper-Quality Visualizations (IEEE Style)
#
# PURPOSE:
# Generates high-quality, publication-ready figures for the research paper.
# It enforces IEEE standards (Times New Roman, clear contrast, high DPI).
#
# PLOTS GENERATED:
#   1. KL Divergence & Annealing (KD-SAC specific)
#   2. Performance Radar Chart (Comparing all methods)
#   3. Action Distribution (Violin plot: Steer/Throttle smoothness)
#   4. Generalization Bar Chart (Zero-shot performance)
#
# USAGE:
#   python visualize_paper.py

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Try to use SUMO to run a quick episode for action collection
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set 'SUMO_HOME' environment variable.")

from stable_baselines3 import SAC
import config as cfg
from sumo_env import SumoEnv
from wrappers import FullControlWrapper
from evaluate import find_model

# ---------------------------------------------------------------------------
# IEEE PAPER STYLE CONFIGURATION
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.format": "pdf", # PDF is preferred for LaTeX/IEEE
})

# Standard color palette for algorithms
COLORS = {
    "KD-SAC": "#D32F2F",       # Red
    "Curriculum": "#1976D2",   # Blue
    "Flat PPO": "#388E3C",     # Green
    "Rule-Based": "#FBC02D",   # Yellow
}

OUTPUT_DIR = "./reports/paper_figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. KL DIVERGENCE & ANNEALING SCHEDULE PLOT
# =============================================================================
def plot_kd_divergence():
    print("Generating KD Divergence plot...")
    # Find latest kd_metrics CSV
    files = glob.glob(os.path.join(cfg.KD_LOG_DIR, "kd_metrics_*.csv"))
    if not files:
        print("  [WARN] No KD metrics CSV found.")
        return
        
    latest_csv = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest_csv)
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    # Plot alpha/beta (Annealing) on left y-axis
    color_alpha = '#555555'
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel(r'KD Weights ($\alpha, \beta$)', color=color_alpha)
    line1 = ax1.plot(df['timestep'], df['alpha'], '--', color=color_alpha, label=r'Weight $\alpha, \beta$')
    ax1.tick_params(axis='y', labelcolor=color_alpha)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot KL divergence on right y-axis
    ax2 = ax1.twinx()
    color_kl = COLORS["KD-SAC"]
    ax2.set_ylabel('KL Divergence (Analytical)', color=color_kl)
    line2 = ax2.plot(df['timestep'], df['rolling_avg_kl_total'], '-', color=color_kl, label='Total KL Divergence')
    ax2.tick_params(axis='y', labelcolor=color_kl)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title("Multi-Teacher Knowledge Distillation over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_kd_divergence.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_kd_divergence.png"), dpi=300)
    plt.close()


# =============================================================================
# 2. RADAR CHART (Method Comparison)
# =============================================================================
def plot_radar_chart():
    print("Generating Radar Chart...")
    files = glob.glob(os.path.join(cfg.EVAL_OUTPUT_DIR, "eval_comparison_*.csv"))
    if not files:
        print("  [WARN] No evaluation comparison CSV found.")
        return
        
    latest_csv = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest_csv)
    
    # Extract means (handle "± std" strings if present)
    def extract_mean(val):
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        return float(val)

    # Prepare data
    methods = df['mode'].tolist()
    
    # Map raw names to clean names
    name_map = {
        "kd_sac": "KD-SAC",
        "curriculum_sac": "Curriculum",
        "flat_ppo": "Flat PPO",
        "rule_based": "Rule-Based"
    }
    methods_clean = [name_map.get(m, m) for m in methods]
    
    # Define metrics to plot (Must be higher = better)
    # We will normalize all to 0-1 scale.
    metrics = ["Success Rate", "Safety (1 - Collision)", "Speed", "Efficiency (-Energy/m)"]
    
    # Collect raw means
    raw_data = {
        "Success Rate": [extract_mean(x) for x in df["success_rate_%"]],
        "Safety (1 - Collision)": [100.0 - extract_mean(x) for x in df["collision_rate_%"]],
        "Speed": [extract_mean(x) for x in df["avg_speed_ms"]],
        "Efficiency (-Energy/m)": [-extract_mean(x) for x in df["avg_efficiency_wh_per_m"]] # Negative makes higher better
    }
    
    # Normalize 0 to 1 for Radar
    norm_data = []
    for m in methods_clean:
        norm_data.append([])
        
    for key in metrics:
        vals = np.array(raw_data[key])
        vmin, vmax = np.min(vals), np.max(vals)
        if vmax == vmin:
            norm_vals = np.ones_like(vals)
        else:
            norm_vals = (vals - vmin) / (vmax - vmin)
            
        for i, val in enumerate(norm_vals):
            # Scale slightly so center is 0.1 instead of 0 for aesthetic
            norm_data[i].append(0.1 + 0.9 * val)

    # Radar plot math
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the circle
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axes per variable and add labels
    plt.xticks(angles[:-1], metrics, color='black', size=11)
    
    # Remove radial ylabels completely
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    # Plot each method
    for i, method in enumerate(methods_clean):
        values = norm_data[i]
        values += values[:1] # Close the circle
        color = COLORS.get(method, "#333333")
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Multidimensional Performance Comparison", y=1.08)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_radar_chart.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_radar_chart.png"), dpi=300)
    plt.close()


# =============================================================================
# 3. ACTION DISTRIBUTION (Boxplot/Violin)
# =============================================================================
def collect_actions(model_path: str, episodes: int = 1) -> pd.DataFrame:
    """Runs a model for a few episodes and records its raw actions."""
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        return pd.DataFrame()
        
    print(f"  Collecting actions for {model_path}...")
    dummy_env = FullControlWrapper(SumoEnv(render=False))
    model = SAC.load(model_path, env=dummy_env, device="cpu") if "sac" in model_path.lower() else __import__('stable_baselines3').PPO.load(model_path, env=dummy_env, device="cpu")
    model.policy.set_training_mode(False)
    dummy_env.close()
    
    env = FullControlWrapper(SumoEnv(render=False))
    
    actions_steer = []
    actions_throttle = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_steer.append(action[0])    # 0 is steer in FullControl
            actions_throttle.append(action[1]) # 1 is throttle/brake
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            
    env.close()
    return pd.DataFrame({
        "Steering": actions_steer,
        "Throttle": actions_throttle
    })


def plot_action_distribution():
    print("Generating Action Distribution plots (Running inference)...")
    
    models_to_test = {
        "KD-SAC": getattr(cfg, "KD_MODEL_DIR", None),
        "Curriculum": cfg.STAGE3_MODEL_DIR,
        "Flat PPO": getattr(cfg, "FLAT_MODEL_DIR", None)
    }
    
    df_list = []
    for name, mdir in models_to_test.items():
        if mdir is None or not os.path.exists(mdir):
            continue
        try:
            if name == "KD-SAC":
                path = find_model(mdir, "kd_sac")
            elif name == "Curriculum":
                path = find_model(mdir, "finetune")
            else:
                path = find_model(mdir, "flat_baseline")
                
            df = collect_actions(path, episodes=2)
            if not df.empty:
                df["Method"] = name
                df_list.append(df)
        except FileNotFoundError:
            pass

    if not df_list:
        print("  [WARN] No models found to collect actions for.")
        return
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Melt dataframe for seaborn violin plot
    df_melt = full_df.melt(id_vars=["Method"], value_vars=["Steering", "Throttle"], 
                           var_name="Action Type", value_name="Action Value")

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df_melt, 
        x="Action Type", 
        y="Action Value", 
        hue="Method", 
        palette=COLORS,
        split=False,
        inner="quartile"
    )
    plt.title("Action Distribution (Smoothness Analysis)")
    plt.ylabel("Normalized Action Output [-1, 1]")
    plt.legend(title="Algorithm")
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_action_distribution.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_action_distribution.png"), dpi=300)
    plt.close()


# =============================================================================
# 4. GENERALIZATION PERFORMANCE
# =============================================================================
def plot_generalization():
    print("Generating Generalization Heatmaps/Bars...")
    files = glob.glob(os.path.join(cfg.EVAL_OUTPUT_DIR, "generalization_results_*.csv"))
    if not files:
        print("  [WARN] No generalization CSV found. Run evaluate_generalization.py first.")
        return
        
    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list, ignore_index=True)
    
    # Map model names to standard names
    df["Model"] = df["Model"].replace({"KD": "KD-SAC", "CURRICULUM": "Curriculum", "FLAT": "Flat PPO"})
    
    # Only pick success rate for plotting
    plt.figure(figsize=(8, 5))
    
    # Bar plot of Success Rate vs Traffic Scale, grouped by Model
    sns.barplot(
        data=df[df["Domain"] == "zero_shot_domain"],
        x="Traffic_Scale",
        y="Success_Rate_%",
        hue="Model",
        palette=COLORS,
        edgecolor="black"
    )
    
    plt.title("Zero-Shot Generalization: Success Rate vs Traffic Density")
    plt.xlabel("Traffic Density Scale (x)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)
    plt.legend(title="Algorithm")
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_generalization_zeroshot.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_generalization_zeroshot.png"), dpi=300)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  PHASE E: GENERATING IEEE PAPER VISUALIZATIONS")
    print("=" * 60)
    
    plot_kd_divergence()
    plot_radar_chart()
    plot_action_distribution()
    plot_generalization()
    
    print("\n" + "=" * 60)
    print(f"  ✅ ALL FIGURES SAVED TO: {OUTPUT_DIR}")
    print("  (Both high-res PNG and vector PDF formats generated)")
    print("=" * 60)


if __name__ == "__main__":
    main()
