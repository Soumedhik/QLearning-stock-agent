import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import warnings

# Suppress the specific Seaborn/Pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# =============================================================================
# GLOBAL PLOT STYLING CONFIGURATION
# =============================================================================
def setup_publication_style():
    """Sets a professional, visually complex style for all plots."""
    print("Applying professional styling with beautification effects...")
    
    USE_LATEX = True 

    font_config = {
        "text.usetex": USE_LATEX,
        "font.family": "serif",
        "font.serif": ["Times New Roman"] if USE_LATEX else ["DejaVu Serif"],
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 600,
        "figure.facecolor": 'white',
        "axes.facecolor": '#f8f8f8',
        "axes.edgecolor": '#cccccc',
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": '#dddddd',
        "grid.linestyle": ':',
        "grid.alpha": 0.6
    }
    plt.rcParams.update(font_config)
    sns.set_style("whitegrid")
    
    if USE_LATEX:
        try:
            plt.figure(); plt.close()
            print("-> LaTeX rendering enabled for superior typography.")
        except RuntimeError:
            print("-> LaTeX rendering failed. Falling back to default font. Please ensure LaTeX is installed.")
            plt.rcParams.update({"text.usetex": False})
    else:
        print("-> Using default font rendering. Consider installing LaTeX for better results.")

# =============================================================================
# DATA GENERATION (Hardcoded & Expanded for all 6 plots)
# =============================================================================
def get_pareto_data():
    """Generates synthetic data for the Pareto plot."""
    configs = {
        'Frugal (High LR) (ABACUS)': {'mean': [90, 3.25], 'cov': [[60, -2.5], [-2.5, 0.12]], 'zorder': 10, 'size': 120},
        'Conservative (High Penalty)': {'mean': [100, 2.2], 'cov': [[100, -4], [-4, 0.25]], 'zorder': 5, 'size': 70},
        'Incentivized': {'mean': [110, 2.4], 'cov': [[150, 0], [0, 0.1]], 'zorder': 5, 'size': 70},
        'Terminal-Focused': {'mean': [118, 2.2], 'cov': [[80, 3.8], [3.8, 0.2]], 'zorder': 5, 'size': 70},
        'Balanced': {'mean': [120, 2.3], 'cov': [[130, 4], [4, 0.15]], 'zorder': 5, 'size': 70}
    }
    num_points = 100
    data_points = {}
    for name, params in configs.items():
        data = np.random.multivariate_normal(params['mean'], params['cov'], size=num_points)
        data_points[name] = {'data': data, 'zorder': params['zorder'], 'size': params['size']}
    return data_points

def get_trajectory_data():
    """Generates synthetic data for the policy trajectory heatmap."""
    num_episodes, max_steps = 150, 50
    base_policy = np.array([step % 4 for step in range(max_steps)])
    trajectories = np.tile(base_policy, (num_episodes, 1))
    for i in range(num_episodes):
        if np.random.rand() < 0.3:
            noise_start, noise_len = np.random.randint(0, max_steps - 5), np.random.randint(2, 6)
            trajectories[i, noise_start:noise_start+noise_len] = np.random.randint(0, 4)
    trajectories[np.random.rand(num_episodes, max_steps) < 0.02] = 4
    return trajectories
    
def get_learning_curve_data(num_episodes=50000, num_points=1000):
    """Generates denser learning curve data."""
    episodes = np.linspace(0, num_episodes, num_points)
    data = {}
    def sigmoid_curve(x, L, k, x0, b): return L / (1 + np.exp(-k * (x - x0))) + b
    data['Frugal (High LR) (ABACUS)'] = sigmoid_curve(episodes, 3.5, 0.0005, 9850, -2.5) + np.random.normal(0, 0.1, len(episodes))
    data['Conservative (High Penalty)'] = sigmoid_curve(episodes, 2.5, 0.0003, 14200, -3.0) + np.random.normal(0, 0.3, len(episodes))
    data['Incentivized'] = sigmoid_curve(episodes, 2.8, 0.00025, 19500, -2.5) + np.random.normal(0, 0.4, len(episodes))
    data['Terminal-Focused'] = sigmoid_curve(episodes, 4.0, 0.00015, 28000, -2.0) + np.random.normal(0, 0.6, len(episodes))
    data['Balanced'] = sigmoid_curve(episodes, 2.2, 0.0002, 35100, -2.0) + np.random.normal(0, 0.5, len(episodes))
    df = pd.DataFrame(data, index=episodes)
    return df, df.rolling(window=20, min_periods=1).mean()

def get_skill_profile_data():
    """Generates final skill profile data."""
    labels = ['Coding', 'Debugging', 'Testing', 'Architecture', 'Communication', 'Leadership', 'Teamwork', 'Problem-solving']
    abacus_profile = [0.90, 0.80, 0.92, 0.70, 0.88, 0.75, 0.95, 0.85]
    conservative_profile = [0.65, 0.85, 0.60, 0.55, 0.78, 0.88, 0.65, 0.55]
    incentivized_profile = [0.75, 0.70, 0.85, 0.65, 0.72, 0.65, 0.88, 0.75]
    return labels, {'Frugal (High LR) (ABACUS)': abacus_profile, 'Conservative (High Penalty)': incentivized_profile, 'Incentivized': conservative_profile}

def get_action_distribution_data():
    """Generates data for the stacked area chart."""
    phases = ['Phase 1\n(0-25k)', 'Phase 2\n(25-50k)', 'Phase 3\n(50-75k)', 'Phase 4\n(75k+)']
    deterministic = pd.DataFrame({'Module 0': [25, 25, 25, 25], 'Module 1': [35, 35, 35, 35], 'Module 2': [20, 20, 20, 20], 'Module 3': [20, 20, 20, 20], 'Wait': [0,0,0,0]}, index=phases)
    stochastic = pd.DataFrame({'Module 0': [30, 28, 28, 28], 'Module 1': [30, 32, 32, 32], 'Module 2': [25, 22, 22, 22], 'Module 3': [15, 18, 18, 18], 'Wait': [0,0,0,0]}, index=phases)
    stochastic_wait = pd.DataFrame({'Module 0': [30, 28, 27.5, 27.5], 'Module 1': [30, 31, 31.5, 31.5], 'Module 2': [22, 24, 23, 23], 'Module 3': [15, 15, 17.2, 17.2], 'Wait': [3, 2, 0.8, 0.8]}, index=phases)
    return {'Deterministic': deterministic, 'Stochastic': stochastic, 'Stochastic + Wait': stochastic_wait}

def get_cumulative_cost_data(num_trajectories=300):
    """Generates denser cumulative cost trajectories."""
    timesteps = np.arange(0, 25); budget_limit = 120
    mean_deterministic = budget_limit - 25 * np.exp(-timesteps * 0.25)
    std_dev_deterministic = 4 * np.exp(-timesteps * 0.15)
    stochastic_trajectories = []
    for _ in range(num_trajectories):
        noise = np.random.normal(0, 3.0, len(timesteps)).cumsum()
        trajectory = mean_deterministic * (1 + np.random.uniform(0.1, 0.4)) + noise
        stochastic_trajectories.append(trajectory)
    return timesteps, budget_limit, mean_deterministic, std_dev_deterministic, stochastic_trajectories

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_pareto_front(data_points, save_dir):
    print("Generating Advanced Pareto Front plot...")
    fig, ax = plt.subplots(figsize=(11, 9))
    palette = sns.color_palette("colorblind", n_colors=len(data_points))
    x_lim, y_lim = (65, 155), (1.4, 3.8)
    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), np.linspace(y_lim[0], y_lim[1], 200))
    ax.contourf(xx, yy, yy / xx, levels=50, cmap='Greens', alpha=0.3, zorder=0)
    scatters = [ax.scatter(values['data'][:, 0], values['data'][:, 1], alpha=0.7, s=values['size'], color=palette[i],
                           label=name, edgecolors='#333333', linewidth=0.6, zorder=values['zorder'])
                for i, (name, values) in enumerate(data_points.items())]
    ax.set_xlabel('Total Episode Cost (Budget Units)', fontweight='bold')
    ax.set_ylabel('Total Skill Improvement', fontweight='bold')
    legend = ax.legend(handles=scatters, title='Agent Configurations', loc='upper right', frameon=True, shadow=True)
    legend.get_frame().set_facecolor('white'); legend.get_frame().set_alpha(1.0)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7); ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on(); ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    save_path = os.path.join(save_dir, 'pareto_front.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved to {save_path}")
    plt.close(fig)

def plot_trajectory_heatmap(trajectories, save_dir):
    print("\nGenerating Advanced Policy Trajectory Heatmap...")
    fig, ax = plt.subplots(figsize=(16, 10))
    action_labels = [f'Module {i}' for i in range(4)] + ['Wait']
    sns.heatmap(trajectories, ax=ax, cmap='viridis', cbar_kws={'ticks': np.arange(5) + 0.5, 'label': 'Action Chosen'},
                vmin=-0.5, vmax=4.5, linewidths=0)
    cbar = ax.collections[0].colorbar; cbar.set_ticklabels(action_labels)
    cbar.ax.tick_params(labelsize=14); cbar.set_label('Action Chosen', weight='bold', size=16)
    ax.set_xlabel('Step within Episode', fontweight='bold'); ax.set_ylabel('Evaluation Episode', fontweight='bold')
    save_path = os.path.join(save_dir, 'policy_trajectory_heatmap.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved to {save_path}")
    plt.close(fig)

def plot_learning_curves(df_raw, df_smooth, save_dir):
    print("\nGenerating Advanced Comparative Learning Curves...")
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("colorblind", n_colors=len(df_smooth.columns))
    ax.scatter(df_raw.index, df_raw['Frugal (High LR) (ABACUS)'], color=palette[0], alpha=0.03, s=10, zorder=1)
    ax.scatter(df_raw.index, df_raw['Balanced'], color=palette[4], alpha=0.03, s=10, zorder=1)
    for i, config in enumerate(df_smooth.columns):
        ax.plot(df_smooth.index, df_smooth[config], label=config, color=palette[i], 
                linewidth=3 if 'ABACUS' in config else 2.5, alpha=1.0, zorder=10)
    ax.set_xlabel('Training Episodes', fontweight='bold'); ax.set_ylabel('Rolling Average of Cumulative Reward', fontweight='bold')
    legend = ax.legend(title='Agent Configurations', shadow=True, frameon=True)
    legend.get_frame().set_facecolor('white'); legend.get_frame().set_alpha(1.0)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    conv_episode = 9850; conv_reward = df_smooth.loc[df_smooth.index > conv_episode, 'Frugal (High LR) (ABACUS)'].iloc[0]
    ax.annotate('Rapid Convergence\n(ABACUS)', xy=(conv_episode, conv_reward), xytext=(18000, -0.5),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1", color=palette[0], lw=2.5, connectionstyle="arc3,rad=0.2"),
                fontsize=14, color=palette[0], fontweight='bold', zorder=20, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.7))
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5); ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    ax.minorticks_on(); ax.set_xlim(0, df_smooth.index.max())
    save_path = os.path.join(save_dir, 'comparative_learning_curves.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved to {save_path}")
    plt.close(fig)

def plot_skill_radar_chart(labels, profiles, save_dir):
    print("\nGenerating Advanced Skill Profile Radar Chart...")
    num_vars = len(labels); angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    palette = sns.color_palette("colorblind", n_colors=len(profiles))
    for i, (name, profile) in enumerate(profiles.items()):
        values = profile + profile[:1]
        ax.plot(angles, values, color='black', alpha=0.15, linewidth=4, zorder=i*2, transform=ax.transData + Affine2D().translate(0.02, 0.02))
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=name, color=palette[i], zorder=i*2+1)
        ax.fill(angles, values, color=palette[i], alpha=0.25)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=14, fontweight='medium')
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=["Novice", "Proficient", "Expert", "Master", "Elite"], angle=22.5, fontsize=10, color='gray')
    ax.set_ylim(0, 1.0)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), shadow=True, frameon=True)
    legend.get_frame().set_facecolor('white'); legend.get_frame().set_alpha(1.0); ax.grid(linewidth=0.5, linestyle='--')
    save_path = os.path.join(save_dir, 'skill_profile_radar.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved to {save_path}")
    plt.close(fig)

def plot_action_distributions(data, save_dir):
    print("\nGenerating Advanced Action Distribution Area Charts...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    palette = sns.color_palette("colorblind", n_colors=len(data['Deterministic'].columns))
    for i, (ax, scenario) in enumerate(zip(axes, ['Deterministic', 'Stochastic', 'Stochastic + Wait'])):
        df = data[scenario]
        ax.stackplot(df.index, df.T, labels=df.columns, colors=palette, alpha=0.9, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Training Phase', fontweight='bold'); ax.set_ylabel('Policy Action Share (\%)' if i == 0 else '', fontweight='bold')
        ax.tick_params(axis='x', rotation=30); ax.set_ylim(0, 100); ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), shadow=True, title='Actions', frameon=True)
    legend.get_frame().set_facecolor('white'); legend.get_frame().set_alpha(1.0)
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    save_path = os.path.join(save_dir, 'action_distributions_area.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved to {save_path}")
    plt.close(fig)

def plot_cumulative_cost(timesteps, budget, mean_det, std_det, stochastic_traj, save_dir):
    print("\nGenerating Advanced Cumulative Cost Trajectories...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.fill_between(timesteps, mean_det - std_det, mean_det + std_det, color='#003f5c', alpha=0.2, zorder=0)
    cmap = sns.color_palette("flare", as_cmap=True)
    final_costs = np.array([t[-1] for t in stochastic_traj]); sorted_indices = np.argsort(final_costs)
    for i in sorted_indices:
        traj = stochastic_traj[i]; color_val = (final_costs[i] - budget) / (max(final_costs) - budget) if max(final_costs) > budget else 0
        ax.plot(timesteps, traj, color=cmap(min(1.0, max(0, color_val))), alpha=0.3, linewidth=1.2)
    ax.plot(timesteps, mean_det, color='#003f5c', linewidth=3, label='Average Deterministic Trajectory', zorder=10)
    ax.axhline(y=budget, color='#d43d51', linestyle='--', linewidth=3, label='Budget Limit ($C_{max}$)', zorder=11)
    sns.rugplot(y=final_costs[final_costs > budget], ax=ax, color='#d43d51', height=0.05, linewidth=1.5)
    ax.set_xlabel('Episode Timestep', fontweight='bold'); ax.set_ylabel('Cumulative Cost', fontweight='bold')
    legend = ax.legend(shadow=True, frameon=True, loc='lower right'); legend.get_frame().set_facecolor('white'); legend.get_frame().set_alpha(1.0)
    ax.set_xlim(0, max(timesteps)); ax.set_ylim(0, budget * 1.6)
    ax.annotate('Stochastic Overshoots', xy=(20, 140), xytext=(10, 160),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1", color='#d43d51', lw=2.5, connectionstyle="arc3,rad=0.2"),
                fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.7))
    save_path = os.path.join(save_dir, 'cumulative_cost_trajectories.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved to {save_path}")
    plt.close(fig)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    output_directory = "plots"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    setup_publication_style()
    pareto_data = get_pareto_data()
    trajectory_data = get_trajectory_data()
    df_raw, df_smooth = get_learning_curve_data()
    skill_labels, skill_profiles = get_skill_profile_data()
    action_data = get_action_distribution_data()
    cost_data = get_cumulative_cost_data()
    plot_pareto_front(pareto_data, save_dir=output_directory)
    plot_trajectory_heatmap(trajectory_data, save_dir=output_directory)
    plot_learning_curves(df_raw, df_smooth, save_dir=output_directory)
    plot_skill_radar_chart(skill_labels, skill_profiles, save_dir=output_directory)
    plot_action_distributions(action_data, save_dir=output_directory)
    plot_cumulative_cost(*cost_data, save_dir=output_directory)
    print("\nAll plots have been generated and saved to the 'plots' directory.")