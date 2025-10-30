import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Import the bifiltration prototype selector
from bifiltration_prototype_selector import BifiltrationPrototypeSelector

def calculate_gmean(y_true, y_pred, average='macro'):
    """
    Calculate G-Mean (Geometric Mean) for classification.

    G-Mean = (Product of class-wise recalls)^(1/n_classes)

    This metric is particularly useful for imbalanced datasets as it
    considers the performance on each class equally.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    recalls = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            correct = np.sum((y_true == cls) & (y_pred == cls))
            total = np.sum(mask)
            recall = correct / total
            recalls.append(recall)
        else:
            recalls.append(0.0)

    epsilon = 1e-10
    recalls = np.array(recalls)
    recalls = np.maximum(recalls, epsilon)

    if average == 'macro':
        gmean = np.prod(recalls) ** (1.0 / n_classes)
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == cls) for cls in classes])
        weights = weights / np.sum(weights)
        gmean = np.prod(recalls ** weights)
    else:
        gmean = recalls

    return gmean

def generate_imbalanced_dataset(n_samples=2500, n_features=4,
                                imbalance_ratio=0.2, random_state=42):
    """
    Generate an imbalanced binary classification dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    imbalance_ratio : float
        Ratio of minority to majority class (e.g., 0.2 for 80:20)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Class labels (0 and 1)
    """
    # Calculate class weights
    weights = [1 - imbalance_ratio, imbalance_ratio]

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        weights=weights,
        class_sep=1.0,
        flip_y=0.0,
        random_state=random_state
    )

    return X, y

def compute_prototypes_all_classes(X_train, y_train, **kwargs):
    """
    Compute prototypes for all classes using the bifiltration selector.

    Returns:
    --------
    all_prototype_indices : array
        Combined prototype indices from all classes
    prototypes_per_class : dict
        Dictionary mapping class label to number of prototypes
    selection_time : float
        Time taken for prototype selection
    """
    unique_classes = np.unique(y_train)
    all_prototype_indices = []
    prototypes_per_class = {}

    start_time = time.time()

    for cls in unique_classes:
        try:
            selector = BifiltrationPrototypeSelector(**kwargs)
            selector.fit(X_train, y_train, target_class=cls)
            _, prototype_indices = selector.get_prototypes(X_train)

            all_prototype_indices.extend(prototype_indices)
            prototypes_per_class[cls] = len(prototype_indices)

        except Exception as e:
            print(f"    Warning: Error processing class {cls}: {e}")
            prototypes_per_class[cls] = 0

    selection_time = time.time() - start_time
    all_prototype_indices = np.array(all_prototype_indices, dtype=int)

    return all_prototype_indices, prototypes_per_class, selection_time

def evaluate_single_fold(X_train, X_test, y_train, y_test, config):
    """
    Evaluate a single configuration on a single fold.

    Returns:
    --------
    fold_results : dict
        Dictionary containing all metrics for this fold
    """
    # Select prototypes
    all_prototype_indices, prototypes_per_class, selection_time = \
        compute_prototypes_all_classes(X_train, y_train, **config)

    # Handle case where no prototypes are selected
    if len(all_prototype_indices) == 0:
        return {
            'n_prototypes_total': 0,
            'n_prototypes_class_0': 0,
            'n_prototypes_class_1': 0,
            'gmean_full': 0.0,
            'gmean_prototypes': 0.0,
            'delta_gmean': 0.0,
            'selection_time': selection_time,
            'valid': False
        }

    # Train 1-NN on full training data (baseline)
    knn_full = KNeighborsClassifier(n_neighbors=1)
    knn_full.fit(X_train, y_train)
    y_pred_full = knn_full.predict(X_test)
    gmean_full = calculate_gmean(y_test, y_pred_full)

    # Train 1-NN on prototypes only
    X_prototypes = X_train[all_prototype_indices]
    y_prototypes = y_train[all_prototype_indices]

    knn_prototypes = KNeighborsClassifier(n_neighbors=1)
    knn_prototypes.fit(X_prototypes, y_prototypes)
    y_pred_prototypes = knn_prototypes.predict(X_test)
    gmean_prototypes = calculate_gmean(y_test, y_pred_prototypes)

    # Calculate delta g-mean
    delta_gmean = gmean_prototypes - gmean_full

    return {
        'n_prototypes_total': len(all_prototype_indices),
        'n_prototypes_class_0': prototypes_per_class.get(0, 0),
        'n_prototypes_class_1': prototypes_per_class.get(1, 0),
        'gmean_full': gmean_full,
        'gmean_prototypes': gmean_prototypes,
        'delta_gmean': delta_gmean,
        'selection_time': selection_time,
        'valid': True
    }

def run_cv_for_config(X, y, config, config_name, n_folds=10):
    """
    Run stratified k-fold cross-validation for a single configuration.

    Returns:
    --------
    aggregated_results : dict
        Mean and standard error for all metrics
    fold_results : list
        Individual results for each fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n  Configuration: {config_name}")
    print(f"    k_neighbors={config['k_neighbors']}, "
          f"min_persistence={config['min_persistence']}, "
          f"neighbor_quantile={config['neighbor_quantile']}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_result = evaluate_single_fold(X_train, X_test, y_train, y_test, config)
        fold_result['fold'] = fold_idx
        fold_result['config_name'] = config_name
        fold_results.append(fold_result)

        if fold_idx % 2 == 0 or fold_idx == n_folds:
            print(f"    Fold {fold_idx}/{n_folds}: "
                  f"n_proto={fold_result['n_prototypes_total']}, "
                  f"g-mean_full={fold_result['gmean_full']:.4f}, "
                  f"g-mean_proto={fold_result['gmean_prototypes']:.4f}, "
                  f"Delta_g-mean={fold_result['delta_gmean']:+.4f}")

    # Aggregate results across folds
    valid_folds = [r for r in fold_results if r['valid']]

    if len(valid_folds) == 0:
        return None, fold_results

    metrics = ['n_prototypes_total', 'n_prototypes_class_0', 'n_prototypes_class_1',
               'gmean_full', 'gmean_prototypes', 'delta_gmean', 'selection_time']

    aggregated = {'config_name': config_name}
    aggregated.update(config)

    for metric in metrics:
        values = np.array([r[metric] for r in valid_folds])
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values, ddof=1)
        aggregated[f'{metric}_se'] = np.std(values, ddof=1) / np.sqrt(len(values))

        # 95% confidence interval
        if len(values) > 1:
            confidence = 0.95
            df = len(values) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, df)
            margin = t_value * aggregated[f'{metric}_se']
            aggregated[f'{metric}_ci_lower'] = aggregated[f'{metric}_mean'] - margin
            aggregated[f'{metric}_ci_upper'] = aggregated[f'{metric}_mean'] + margin
        else:
            aggregated[f'{metric}_ci_lower'] = aggregated[f'{metric}_mean']
            aggregated[f'{metric}_ci_upper'] = aggregated[f'{metric}_mean']

    aggregated['n_valid_folds'] = len(valid_folds)

    # Calculate prototype class ratio
    class_0_mean = aggregated['n_prototypes_class_0_mean']
    class_1_mean = aggregated['n_prototypes_class_1_mean']
    total_mean = aggregated['n_prototypes_total_mean']

    if total_mean > 0:
        ratio_0 = (class_0_mean / total_mean) * 100
        ratio_1 = (class_1_mean / total_mean) * 100
        aggregated['prototype_ratio'] = f"{class_0_mean:.1f}:{class_1_mean:.1f} ({ratio_0:.1f}%:{ratio_1:.1f}%)"
    else:
        aggregated['prototype_ratio'] = "0:0"

    print(f"   Mean results: n_proto={aggregated['n_prototypes_total_mean']:.1f}�{aggregated['n_prototypes_total_se']:.1f}, "
          f"�g-mean={aggregated['delta_gmean_mean']:+.4f}�{aggregated['delta_gmean_se']:.4f}")

    return aggregated, fold_results

def create_parameter_configurations():
    """
    Create one-at-a-time parameter configurations.

    Baseline: k_neighbors=1, min_persistence=0.001, neighbor_quantile=0.05
    Fixed: homology_dimension=0, radius_statistic='mean'
    """
    baseline = {
        'k_neighbors': 1,
        'min_persistence': 0.001,
        'neighbor_quantile': 0.05,
        'homology_dimension': 0,
        'radius_statistic': 'mean'
    }

    configurations = []

    # Baseline configuration
    configurations.append({
        'config': baseline.copy(),
        'name': 'baseline',
        'varied_param': 'none',
        'varied_value': 'baseline'
    })

    # Vary k_neighbors: from 1 to 10% of dataset size (2500), 10 values evenly spaced
    dataset_size = 2500
    max_k = int(dataset_size * 0.10)  # 10% of 2500 = 250
    k_neighbors_values = np.linspace(1, max_k, 10, dtype=int)
    k_neighbors_values = np.unique(k_neighbors_values)  # Remove duplicates if any

    # Exclude baseline value (1) from variations
    k_neighbors_values = k_neighbors_values[k_neighbors_values != 1]

    for k in k_neighbors_values:
        config = baseline.copy()
        config['k_neighbors'] = int(k)
        configurations.append({
            'config': config,
            'name': f'k_neighbors_{k}',
            'varied_param': 'k_neighbors',
            'varied_value': int(k)
        })

    # Vary min_persistence
    min_persistence_values = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 10.0]
    for p in min_persistence_values:
        config = baseline.copy()
        config['min_persistence'] = p
        configurations.append({
            'config': config,
            'name': f'min_persistence_{p}',
            'varied_param': 'min_persistence',
            'varied_value': p
        })

    # Vary neighbor_quantile
    neighbor_quantile_values = [0.10, 0.15, 0.20, 0.25, 0.50]
    for q in neighbor_quantile_values:
        config = baseline.copy()
        config['neighbor_quantile'] = q
        configurations.append({
            'config': config,
            'name': f'neighbor_quantile_{q}',
            'varied_param': 'neighbor_quantile',
            'varied_value': q
        })

    return configurations

def run_parameter_study():
    """
    Run the complete parameter study with stratified k-fold cross-validation.
    """
    print("="*80)
    print("PARAMETER STUDY: BifiltrationPrototypeSelector")
    print("="*80)
    print("\nStudy Design:")
    print("  - One-at-a-time sensitivity analysis")
    print("  - Baseline: k_neighbors=1, min_persistence=0.001, neighbor_quantile=0.05")
    print("  - Fixed: homology_dimension=0, radius_statistic='mean'")
    print("\nDataset:")
    print("  - Samples: 2500")
    print("  - Features: 4")
    print("  - Classes: 2 (binary)")
    print("  - Imbalance: 80:20")
    print("\nEvaluation:")
    print("  - 10-fold stratified cross-validation")
    print("  - Baseline: 1-NN on full training data")
    print("  - Comparison: 1-NN on prototypes only")
    print("  - Metric: G-mean (geometric mean of per-class recalls)")
    print("="*80)

    # Generate dataset
    print("\nGenerating imbalanced dataset...")
    X, y = generate_imbalanced_dataset(n_samples=2500, n_features=4,
                                       imbalance_ratio=0.2, random_state=42)

    unique, counts = np.unique(y, return_counts=True)
    print(f"Dataset generated: {len(X)} samples")
    print(f"Class distribution: Class 0: {counts[0]}, Class 1: {counts[1]}")
    print(f"Imbalance ratio: {counts[0]/counts[1]:.1f}:1")

    # Create configurations
    configurations = create_parameter_configurations()
    print(f"\nTotal configurations to test: {len(configurations)}")
    print(f"Total CV runs: {len(configurations)} � 10 folds = {len(configurations) * 10}")

    # Run parameter study
    all_aggregated_results = []
    all_fold_results = []

    print("\n" + "="*80)
    print("RUNNING PARAMETER STUDY")
    print("="*80)

    for idx, config_info in enumerate(configurations, 1):
        print(f"\n[{idx}/{len(configurations)}] Testing: {config_info['name']}")
        print(f"  Varying: {config_info['varied_param']} = {config_info['varied_value']}")

        aggregated, fold_results = run_cv_for_config(
            X, y,
            config_info['config'],
            config_info['name'],
            n_folds=10
        )

        if aggregated is not None:
            aggregated['varied_param'] = config_info['varied_param']
            aggregated['varied_value'] = str(config_info['varied_value'])
            all_aggregated_results.append(aggregated)

        all_fold_results.extend(fold_results)

    return pd.DataFrame(all_aggregated_results), pd.DataFrame(all_fold_results)

def save_results(summary_df, detailed_df):
    """Save results to CSV files."""
    # Save summary results
    import os
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/parameter_study_summary.csv', index=False)
    print("\n Summary results saved to: results/parameter_study_summary.csv")

    # Save detailed fold results
    detailed_df.to_csv('results/parameter_study_detailed.csv', index=False)
    print(" Detailed fold results saved to: results/parameter_study_detailed.csv")

def generate_latex_table(summary_df):
    """Generate LaTeX table from summary results."""
    latex_lines = []

    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Parameter Study Results for BifiltrationPrototypeSelector (Mean $\\pm$ SE from 10-fold CV)}")
    latex_lines.append("\\label{tab:parameter_study}")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{llcccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Param & Value & \\makecell{$n_{\\text{proto}}$ \\\\ (Mean $\\pm$ SE)} & "
                      "\\makecell{Class 0:1 \\\\ Ratio} & \\makecell{G-Mean \\\\ (Full)} & "
                      "\\makecell{G-Mean \\\\ (Proto)} & \\makecell{$\\Delta$G-Mean \\\\ (Mean $\\pm$ SE)} & "
                      "\\makecell{Time \\\\ (s)} \\\\")
    latex_lines.append("\\hline")

    # Baseline row
    baseline = summary_df[summary_df['config_name'] == 'baseline'].iloc[0]
    latex_lines.append(
        f"Baseline & - & "
        f"{baseline['n_prototypes_total_mean']:.1f} $\\pm$ {baseline['n_prototypes_total_se']:.1f} & "
        f"{baseline['n_prototypes_class_0_mean']:.0f}:{baseline['n_prototypes_class_1_mean']:.0f} & "
        f"{baseline['gmean_full_mean']:.3f} & "
        f"{baseline['gmean_prototypes_mean']:.3f} & "
        f"{baseline['delta_gmean_mean']:+.3f} $\\pm$ {baseline['delta_gmean_se']:.3f} & "
        f"{baseline['selection_time_mean']:.2f} \\\\"
    )
    latex_lines.append("\\hline")

    # Group by parameter
    for param in ['k_neighbors', 'min_persistence', 'neighbor_quantile']:
        param_df = summary_df[summary_df['varied_param'] == param]

        if len(param_df) > 0:
            param_label = param.replace('_', '\\_')
            latex_lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{param_label}}}}} \\\\")

            for _, row in param_df.iterrows():
                value_str = row['varied_value']
                latex_lines.append(
                    f"  & {value_str} & "
                    f"{row['n_prototypes_total_mean']:.1f} $\\pm$ {row['n_prototypes_total_se']:.1f} & "
                    f"{row['n_prototypes_class_0_mean']:.0f}:{row['n_prototypes_class_1_mean']:.0f} & "
                    f"{row['gmean_full_mean']:.3f} & "
                    f"{row['gmean_prototypes_mean']:.3f} & "
                    f"{row['delta_gmean_mean']:+.3f} $\\pm$ {row['delta_gmean_se']:.3f} & "
                    f"{row['selection_time_mean']:.2f} \\\\"
                )
            latex_lines.append("\\hline")

    # Remove last hline
    latex_lines[-1] = latex_lines[-1].replace("\\hline", "")

    # Table footer
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    import os
    os.makedirs('results', exist_ok=True)

    latex_content = "\n".join(latex_lines)
    with open('results/parameter_study_latex.tex', 'w') as f:
        f.write(latex_content)

    print(" LaTeX table saved to: results/parameter_study_latex.tex")

    return latex_content

def create_visualizations(summary_df):
    """Create comprehensive visualization plots."""

    # Extract baseline values for reference
    baseline = summary_df[summary_df['config_name'] == 'baseline'].iloc[0]
    baseline_gmean_proto = baseline['gmean_prototypes_mean']
    baseline_n_proto = baseline['n_prototypes_total_mean']

    import os
    os.makedirs('results', exist_ok=True)

    # k_neighbors values: 1 to 250 (10% of 2500), 10 evenly spaced values
    dataset_size = 2500
    max_k = int(dataset_size * 0.10)
    k_values_for_viz = list(np.linspace(1, max_k, 10, dtype=int))

    params_info = [
        ('k_neighbors', 'k-neighbors', k_values_for_viz),
        ('min_persistence', 'Minimum Persistence', [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 10.0]),
        ('neighbor_quantile', 'Neighbor Quantile', [0.05, 0.10, 0.15, 0.20, 0.25, 0.50])
    ]

    # Create separate figure for each parameter
    for param, param_label, _ in params_info:
        # Get data for this parameter
        param_df = summary_df[summary_df['varied_param'] == param].copy()
        if len(param_df) == 0:
            continue

        # Convert varied_value to numeric and sort
        if param == 'k_neighbors':
            param_df['varied_value_numeric'] = param_df['varied_value'].astype(int)
        else:
            param_df['varied_value_numeric'] = param_df['varied_value'].astype(float)

        param_df = param_df.sort_values(by='varied_value_numeric')

        # Extract x values
        x_values = param_df['varied_value_numeric'].values

        # Create figure with 2 rows (one column)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Row 1: Number of Prototypes
        ax1 = axes[0]
        y_mean = param_df['n_prototypes_total_mean'].values
        y_se = param_df['n_prototypes_total_se'].values

        ax1.errorbar(x_values, y_mean, yerr=y_se, marker='o', capsize=5,
                     linewidth=2, markersize=8, label='Total Prototypes')
        ax1.axhline(y=baseline_n_proto, color='red', linestyle='--',
                   linewidth=1.5, label='Baseline', alpha=0.7)
        ax1.set_xlabel(param_label, fontsize=11)
        ax1.set_ylabel('Number of Prototypes', fontsize=11)
        ax1.set_title(f'Effect of {param_label} on Prototype Count', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # Row 2: G-mean Comparison
        ax2 = axes[1]

        gmean_full = param_df['gmean_full_mean'].values
        gmean_proto = param_df['gmean_prototypes_mean'].values
        gmean_proto_se = param_df['gmean_prototypes_se'].values

        ax2.errorbar(x_values, gmean_proto, yerr=gmean_proto_se, marker='o',
                    capsize=5, linewidth=2, markersize=8, label='Proto 1-NN', color='green')
        ax2.plot(x_values, gmean_full, marker='s', linewidth=2, markersize=8,
                label='Full 1-NN', color='blue', alpha=0.7)
        ax2.axhline(y=baseline_gmean_proto, color='red', linestyle='--',
                   linewidth=1.5, label='Baseline Proto', alpha=0.7)
        ax2.set_xlabel(param_label, fontsize=11)
        ax2.set_ylabel('G-Mean', fontsize=11)
        ax2.set_title(f'Effect of {param_label} on G-Mean', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        plt.suptitle(f'Parameter Study: Effect of {param_label}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save individual parameter plot
        filename = f'results/parameter_study_{param}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Parameter plot saved to: {filename}")
        plt.show()

    # Create additional plot: Prototype class ratios
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (param, param_label, _) in enumerate(params_info):
        ax = axes[idx]
        param_df = summary_df[summary_df['varied_param'] == param].copy()

        if len(param_df) == 0:
            continue

        # Convert varied_value to numeric and sort
        if param == 'k_neighbors':
            param_df['varied_value_numeric'] = param_df['varied_value'].astype(int)
        else:
            param_df['varied_value_numeric'] = param_df['varied_value'].astype(float)

        param_df = param_df.sort_values(by='varied_value_numeric')
        x_values = param_df['varied_value_numeric'].values

        class_0 = param_df['n_prototypes_class_0_mean'].values
        class_1 = param_df['n_prototypes_class_1_mean'].values

        x_pos = np.arange(len(x_values))
        width = 0.35

        ax.bar(x_pos - width/2, class_0, width, label='Class 0 (Majority)', alpha=0.8, color='skyblue')
        ax.bar(x_pos + width/2, class_1, width, label='Class 1 (Minority)', alpha=0.8, color='lightcoral')

        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel('Number of Prototypes', fontsize=11)
        ax.set_title(f'Prototype Distribution: {param_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{v}' for v in x_values], rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Prototype Class Distribution by Parameter',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/parameter_study_class_distribution.png', dpi=300, bbox_inches='tight')
    print(" Class distribution plot saved to: results/parameter_study_class_distribution.png")
    plt.show()

def print_summary_statistics(summary_df):
    """Print comprehensive summary statistics."""
    print("\n" + "="*120)
    print("PARAMETER STUDY SUMMARY")
    print("="*120)

    # Baseline
    baseline = summary_df[summary_df['config_name'] == 'baseline'].iloc[0]
    print("\nBaseline Configuration:")
    print(f"  k_neighbors=1, min_persistence=0.001, neighbor_quantile=0.05")
    print(f"  Prototypes: {baseline['n_prototypes_total_mean']:.1f} � {baseline['n_prototypes_total_se']:.1f}")
    print(f"  Class 0: {baseline['n_prototypes_class_0_mean']:.1f}, Class 1: {baseline['n_prototypes_class_1_mean']:.1f}")
    print(f"  G-Mean (Full): {baseline['gmean_full_mean']:.4f}")
    print(f"  G-Mean (Proto): {baseline['gmean_prototypes_mean']:.4f}")
    print(f"  �G-Mean: {baseline['delta_gmean_mean']:+.4f} � {baseline['delta_gmean_se']:.4f}")
    print(f"  Selection Time: {baseline['selection_time_mean']:.2f}s")

    # Best and worst configurations by delta g-mean
    print("\n" + "-"*120)
    print("Best Configuration (Highest �G-Mean):")
    best = summary_df.loc[summary_df['delta_gmean_mean'].idxmax()]
    print(f"  {best['config_name']}")
    print(f"  Prototypes: {best['n_prototypes_total_mean']:.1f} � {best['n_prototypes_total_se']:.1f}")
    print(f"  G-Mean (Proto): {best['gmean_prototypes_mean']:.4f}")
    print(f"  �G-Mean: {best['delta_gmean_mean']:+.4f} � {best['delta_gmean_se']:.4f}")

    print("\nWorst Configuration (Lowest �G-Mean):")
    worst = summary_df.loc[summary_df['delta_gmean_mean'].idxmin()]
    print(f"  {worst['config_name']}")
    print(f"  Prototypes: {worst['n_prototypes_total_mean']:.1f} � {worst['n_prototypes_total_se']:.1f}")
    print(f"  G-Mean (Proto): {worst['gmean_prototypes_mean']:.4f}")
    print(f"  �G-Mean: {worst['delta_gmean_mean']:+.4f} � {worst['delta_gmean_se']:.4f}")

    # Configuration with fewest prototypes
    print("\nConfiguration with Fewest Prototypes:")
    fewest = summary_df.loc[summary_df['n_prototypes_total_mean'].idxmin()]
    print(f"  {fewest['config_name']}")
    print(f"  Prototypes: {fewest['n_prototypes_total_mean']:.1f} � {fewest['n_prototypes_total_se']:.1f}")
    print(f"  �G-Mean: {fewest['delta_gmean_mean']:+.4f} � {fewest['delta_gmean_se']:.4f}")

    # Configuration with most prototypes
    print("\nConfiguration with Most Prototypes:")
    most = summary_df.loc[summary_df['n_prototypes_total_mean'].idxmax()]
    print(f"  {most['config_name']}")
    print(f"  Prototypes: {most['n_prototypes_total_mean']:.1f} � {most['n_prototypes_total_se']:.1f}")
    print(f"  �G-Mean: {most['delta_gmean_mean']:+.4f} � {most['delta_gmean_se']:.4f}")

    print("\n" + "-"*120)
    print("Parameter Effects:")

    for param in ['k_neighbors', 'min_persistence', 'neighbor_quantile']:
        param_df = summary_df[summary_df['varied_param'] == param]
        if len(param_df) > 0:
            print(f"\n{param}:")
            print(f"  Range of �G-Mean: [{param_df['delta_gmean_mean'].min():+.4f}, {param_df['delta_gmean_mean'].max():+.4f}]")
            print(f"  Range of Prototypes: [{param_df['n_prototypes_total_mean'].min():.1f}, {param_df['n_prototypes_total_mean'].max():.1f}]")

            # Find best value for this parameter
            best_idx = param_df['delta_gmean_mean'].idxmax()
            best_value = param_df.loc[best_idx, 'varied_value']
            best_delta = param_df.loc[best_idx, 'delta_gmean_mean']
            print(f"  Best value: {best_value} (�G-Mean = {best_delta:+.4f})")

    print("\n" + "="*120)

if __name__ == "__main__":
    print("\nStarting parameter study...")
    print("This will run 220 CV iterations (22 configurations × 10 folds)")
    print("Estimated time: 25-40 minutes depending on system performance\n")

    # Run the parameter study
    summary_df, detailed_df = run_parameter_study()

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results(summary_df, detailed_df)

    # Generate LaTeX table
    generate_latex_table(summary_df)

    # Print summary statistics
    print_summary_statistics(summary_df)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(summary_df)

    print("\n" + "="*80)
    print("PARAMETER STUDY COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutput files saved in 'results/' directory:")
    print("  - results/parameter_study_summary.csv (aggregated results)")
    print("  - results/parameter_study_detailed.csv (per-fold results)")
    print("  - results/parameter_study_latex.tex (LaTeX table)")
    print("  - results/parameter_study_k_neighbors.png (k-neighbors sensitivity plot)")
    print("  - results/parameter_study_min_persistence.png (min_persistence sensitivity plot)")
    print("  - results/parameter_study_neighbor_quantile.png (neighbor_quantile sensitivity plot)")
    print("  - results/parameter_study_class_distribution.png (class distribution)")
    print("="*80)
