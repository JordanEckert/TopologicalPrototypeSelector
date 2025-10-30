import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import time
import pandas as pd
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Import the bifiltration prototype selector
from bifiltration_prototype_selector import BifiltrationPrototypeSelector

def generate_gaussian_dataset(n_samples=2500, n_features=4, n_classes=2,
                               class_sep=1.0, random_state=None):
    """
    Generate a multivariate Gaussian dataset with slight overlap.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    class_sep : float
        Class separation (lower = more overlap, higher = less overlap)
        Default 1.0 gives slight overlap
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        class_sep=class_sep,
        flip_y=0.0,
        random_state=random_state
    )
    return X, y

def time_prototype_selection(X, y, **kwargs):
    """
    Time the prototype selection process for all classes.

    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    **kwargs : dict
        Parameters for BifiltrationPrototypeSelector

    Returns:
    --------
    total_time : float
        Total time in seconds for prototype selection across all classes
    """
    unique_classes = np.unique(y)

    start_time = time.perf_counter()

    for cls in unique_classes:
        selector = BifiltrationPrototypeSelector(**kwargs)
        selector.fit(X, y, target_class=cls)
        _, _ = selector.get_prototypes(X)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    return total_time

def run_timing_study():
    """
    Run comprehensive timing study with 100 repetitions per configuration.
    """
    print("="*80)
    print("TDAPS PROTOTYPE SELECTION TIMING STUDY")
    print("="*80)
    print("\nDataset Configuration:")
    print("  - Samples: 2500")
    print("  - Features: 4")
    print("  - Classes: 2")
    print("  - Distribution: Multivariate Gaussian with slight overlap")
    print("\nTiming Protocol:")
    print("  - 100 repetitions per configuration")
    print("  - Different random seed per trial")
    print("  - Measuring prototype selection time only")
    print("="*80)

    # Define parameter grid
    homology_dimensions = [0]
    neighbor_quantiles = [0.05, 0.25]
    radius_statistics = ['mean']
    k_neighbors_list = [1, 10]
    min_persistences = [0.001, 0.1]

    configurations = []
    for h_dim in homology_dimensions:
        for n_q in neighbor_quantiles:
            for r_stat in radius_statistics:
                for k_n in k_neighbors_list:
                    for min_p in min_persistences:
                        configurations.append({
                            'homology_dimension': h_dim,
                            'neighbor_quantile': n_q,
                            'radius_statistic': r_stat,
                            'k_neighbors': k_n,
                            'min_persistence': min_p
                        })

    print(f"\nTesting {len(configurations)} configurations")
    print(f"Total timing runs: {len(configurations) * 100} = {len(configurations)} configs × 100 trials\n")

    # Storage for results
    all_results = []
    all_trials = []

    n_trials = 100

    # Run timing trials
    for config_idx, config in enumerate(configurations, 1):
        config_name = f"H{config['homology_dimension']}_q{int(config['neighbor_quantile']*100)}_r{config['radius_statistic']}_k{config['k_neighbors']}_p{config['min_persistence']}"

        print(f"\n[Configuration {config_idx}/{len(configurations)}] {config_name}")
        print(f"  Parameters: H={config['homology_dimension']}, "
              f"neighbor_q={config['neighbor_quantile']}, "
              f"k_neighbors={config['k_neighbors']}, "
              f"min_persistence={config['min_persistence']}")

        trial_times = []

        for trial in range(n_trials):
            # Generate dataset with unique seed
            random_seed = trial
            X, y = generate_gaussian_dataset(
                n_samples=2500,
                n_features=4,
                n_classes=2,
                class_sep=1.0,
                random_state=random_seed
            )

            # Time prototype selection
            elapsed_time = time_prototype_selection(
                X, y,
                k_neighbors=config['k_neighbors'],
                homology_dimension=config['homology_dimension'],
                min_persistence=config['min_persistence'],
                neighbor_quantile=config['neighbor_quantile'],
                radius_statistic=config['radius_statistic']
            )

            trial_times.append(elapsed_time)

            # Store individual trial
            all_trials.append({
                'config_name': config_name,
                'trial': trial,
                'time': elapsed_time,
                'h_dim': config['homology_dimension'],
                'neighbor_quantile': config['neighbor_quantile'],
                'k_neighbors': config['k_neighbors'],
                'min_persistence': config['min_persistence']
            })

            # Progress update every 10 trials
            if (trial + 1) % 10 == 0:
                mean_so_far = np.mean(trial_times)
                print(f"    Trial {trial + 1}/{n_trials}: {elapsed_time:.4f}s (running mean: {mean_so_far:.4f}s)")

        # Calculate statistics
        trial_times = np.array(trial_times)
        mean_time = np.mean(trial_times)
        std_time = np.std(trial_times, ddof=1)  # Sample std
        se_time = std_time / np.sqrt(n_trials)

        # 95% confidence interval
        confidence_level = 0.95
        degrees_freedom = n_trials - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        ci_margin = t_value * se_time
        ci_lower = mean_time - ci_margin
        ci_upper = mean_time + ci_margin

        print(f"  ✓ Mean Time: {mean_time:.4f} ± {se_time:.4f} seconds")
        print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Store summary results
        all_results.append({
            'config_name': config_name,
            'h_dim': config['homology_dimension'],
            'neighbor_quantile': config['neighbor_quantile'],
            'radius_statistic': config['radius_statistic'],
            'k_neighbors': config['k_neighbors'],
            'min_persistence': config['min_persistence'],
            'mean_time': mean_time,
            'se_time': se_time,
            'std_time': std_time,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_trials': n_trials
        })

    return pd.DataFrame(all_results), pd.DataFrame(all_trials)

def save_results(results_df, trials_df):
    """Save results to CSV files."""
    # Save summary results
    results_df.to_csv('timing_results_prototypes.csv', index=False)
    print("\n✓ Summary results saved to: timing_results_prototypes.csv")

    # Save individual trials
    trials_df.to_csv('timing_trials_raw.csv', index=False)
    print("✓ Raw trial data saved to: timing_trials_raw.csv")

def generate_latex_table(results_df):
    """Generate LaTeX table from results."""
    latex_lines = []

    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Prototype Selection Timing Results (Mean $\\pm$ SE from 100 trials)}")
    latex_lines.append("\\label{tab:timing_results}")
    latex_lines.append("\\begin{tabular}{cccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("$H$ & $q_{\\text{nbr}}$ & $k$ & $p_{\\text{min}}$ & Mean (s) & SE (s) & \\multicolumn{2}{c}{95\\% CI} \\\\")
    latex_lines.append("\\cmidrule(lr){7-8}")
    latex_lines.append(" & & & & & & Lower & Upper \\\\")
    latex_lines.append("\\midrule")

    # Table rows
    for _, row in results_df.iterrows():
        latex_lines.append(
            f"{row['h_dim']} & "
            f"{row['neighbor_quantile']:.2f} & "
            f"{row['k_neighbors']} & "
            f"{row['min_persistence']:.3f} & "
            f"{row['mean_time']:.4f} & "
            f"{row['se_time']:.4f} & "
            f"{row['ci_lower']:.4f} & "
            f"{row['ci_upper']:.4f} \\\\"
        )

    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    latex_content = "\n".join(latex_lines)
    with open('timing_results_latex.tex', 'w') as f:
        f.write(latex_content)

    print("✓ LaTeX table saved to: timing_results_latex.tex")

    return latex_content

def create_visualizations(results_df):
    """Create visualization plots."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(results_df))

    # Create bar plot with error bars
    bars = ax.bar(x_pos, results_df['mean_time'],
                   yerr=results_df['se_time'],
                   capsize=5, alpha=0.7, edgecolor='black')

    # Color bars by configuration characteristics
    colors = []
    for _, row in results_df.iterrows():
        if row['neighbor_quantile'] == 0.05:
            colors.append('skyblue')
        else:
            colors.append('lightcoral')

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Prototype Selection Time by Configuration (Mean ± SE, n=100)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['config_name'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='neighbor_q=0.05'),
        Patch(facecolor='lightcoral', label='neighbor_q=0.25')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig('timing_results_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: timing_results_plot.png")
    plt.show()

def print_summary_table(results_df):
    """Print formatted summary table to console."""
    print("\n" + "="*120)
    print("TIMING RESULTS SUMMARY (100 trials per configuration)")
    print("="*120)
    print(f"{'Config':<25} {'H':<3} {'q_nbr':<7} {'k':<4} {'p_min':<8} {'Mean(s)':<10} {'SE(s)':<10} {'95% CI':<25}")
    print("-"*120)

    for _, row in results_df.iterrows():
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        print(f"{row['config_name']:<25} "
              f"{row['h_dim']:<3} "
              f"{row['neighbor_quantile']:<7.2f} "
              f"{row['k_neighbors']:<4} "
              f"{row['min_persistence']:<8.3f} "
              f"{row['mean_time']:<10.4f} "
              f"{row['se_time']:<10.4f} "
              f"{ci_str:<25}")

    print("="*120)

if __name__ == "__main__":
    print("\nStarting timing study...")
    print("This will run 800 timing trials (8 configurations × 100 repetitions)")
    print("Estimated time: varies based on system performance\n")

    # Run the study
    results_df, trials_df = run_timing_study()

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results(results_df, trials_df)

    # Generate LaTeX table
    generate_latex_table(results_df)

    # Print summary
    print_summary_table(results_df)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(results_df)

    print("\n" + "="*80)
    print("TIMING STUDY COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutput files:")
    print("  - timing_results_prototypes.csv (summary statistics)")
    print("  - timing_trials_raw.csv (all 800 individual measurements)")
    print("  - timing_results_latex.tex (LaTeX formatted table)")
    print("  - timing_results_plot.png (visualization)")
