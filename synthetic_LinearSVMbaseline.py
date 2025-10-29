import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import seaborn as sns
import time
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Import the bifiltration selector
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

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics including G-Mean."""
    accuracy = accuracy_score(y_true, y_pred)
    gmean = calculate_gmean(y_true, y_pred, average='macro')

    classes = np.unique(y_true)
    per_class_recalls = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            recall = recall_score(y_true[mask], y_pred[mask], average='micro')
            per_class_recalls.append(recall)
        else:
            per_class_recalls.append(0.0)

    return {
        'accuracy': accuracy,
        'gmean': gmean,
        'per_class_recalls': per_class_recalls
    }

def compute_prototypes_all_classes(X, y, **kwargs):
    """
    Compute prototypes for all classes using the bifiltration selector.
    """
    unique_classes = np.unique(y)
    prototypes_dict = {}
    selectors_dict = {}

    print("\nComputing prototypes for all classes using bifiltration...")
    print("="*60)

    for cls in unique_classes:
        print(f"\nProcessing class {cls}:")
        print("-" * 40)

        try:
            selector = BifiltrationPrototypeSelector(**kwargs)
            selector.fit(X, y, target_class=cls)

            _, prototype_indices = selector.get_prototypes(X)
            prototypes_dict[cls] = prototype_indices
            selectors_dict[cls] = selector

            print(f"Successfully selected {len(prototype_indices)} prototypes for class {cls}")

        except Exception as e:
            print(f"Error processing class {cls}: {e}")
            import traceback
            traceback.print_exc()
            prototypes_dict[cls] = np.array([])
            selectors_dict[cls] = None

    return prototypes_dict, selectors_dict

def generate_datasets():
    """Generate multiple test datasets with different characteristics."""
    datasets = {}

    # Dataset 1: Well-separated blobs (3 classes)
    X1, y1 = make_blobs(n_samples=600, n_features=2, centers=3,
                       cluster_std=0.5, center_box=(-10, 10), random_state=42)
    datasets['blobs_well_separated'] = (X1, y1, "Well-Separated Blobs (3 classes)")

    # Dataset 2: Overlapping blobs (4 classes)
    X2, y2 = make_blobs(n_samples=800, n_features=2, centers=4,
                       cluster_std=2.5, center_box=(-8, 8), random_state=42)
    datasets['blobs_overlapping'] = (X2, y2, "Overlapping Blobs (4 classes)")

    # Dataset 3: Overlapping clusters (3 classes)
    X4, y4 = make_classification(n_samples=600, n_features=2, n_redundant=0,
                                n_informative=2, n_classes=3, n_clusters_per_class=1,
                                class_sep=0.8, flip_y=0.1, random_state=42)
    datasets['clusters_overlapping'] = (X4, y4, "Overlapping Clusters (3 classes)")

    # Dataset 4: Two Moons (Moderate Noise)
    X5, y5 = make_moons(n_samples=500, noise=0.15, random_state=42)
    datasets['moons'] = (X5, y5, "Two Moons (Moderate Noise)")

    # Dataset 5: Noisy Moons
    X6, y6 = make_moons(n_samples=500, noise=0.3, random_state=42)
    datasets['moons_noisy'] = (X6, y6, "Two Moons (High Noise)")

    # Dataset 6: Concentric Circles (Moderate Noise)
    X7, y7 = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    datasets['circles'] = (X7, y7, "Concentric Circles (Moderate Noise)")

    # Dataset 7: Noisy Circles
    X8, y8 = make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42)
    datasets['circles_noisy'] = (X8, y8, "Concentric Circles (High Noise)")

    # Dataset 8: Imbalanced classes (binary)
    X9, y9 = make_blobs(n_samples=[400, 100],
                       cluster_std=5.0, random_state=42)
    datasets['imbalanced'] = (X9, y9, "Imbalanced Classes (80/20%)")

    # Dataset 9: Multi-class with mixed characteristics (3 classes)
    centers = [[-5, -5], [5, -5], [0, 5]]
    X10a, y10a = make_blobs(n_samples=400, centers=centers[0:1],
                           cluster_std=4.0, random_state=42)
    X10b, y10b = make_blobs(n_samples=300, centers=centers[1:2],
                           cluster_std=3.5, random_state=42)
    X10c, y10c = make_blobs(n_samples=100, centers=centers[2:3],
                           cluster_std=0.5, random_state=42)

    X10 = np.vstack([X10a, X10b, X10c])
    y10 = np.hstack([np.zeros(400, dtype=int),
                     np.ones(300, dtype=int),
                     np.full(100, 2, dtype=int)])

    shuffle_idx = np.random.RandomState(42).permutation(len(X10))
    X10, y10 = X10[shuffle_idx], y10[shuffle_idx]

    datasets['mixed_multiclass'] = (X10, y10, "Mixed Multi-class (3 classes, 40/30/10%)")

    return datasets

def format_config_name(config_name):
    """Format configuration name for better readability."""
    parts = config_name.split('_')
    formatted = []
    for part in parts:
        if part.startswith('q') and part[1:].isdigit():
            formatted.append(f"n_q={part[1:]}%")
        elif part.startswith('r') and not part[1:].isdigit():
            formatted.append(f"r_{part[1:]}")
        elif part.startswith('k') and part[1:].isdigit():
            formatted.append(f"k={part[1:]}")
        elif part.startswith('p'):
            formatted.append(f"min_p={part[1:]}")
        else:
            formatted.append(part)
    return ", ".join(formatted)

def compute_prototype_class_ratio(prototypes_dict, y_train):
    """Compute the ratio of prototypes for each class."""
    class_counts = {}
    total_prototypes = 0

    for cls, indices in prototypes_dict.items():
        class_counts[cls] = len(indices)
        total_prototypes += len(indices)

    sorted_classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in sorted_classes]

    if total_prototypes == 0:
        return "0:0" if len(sorted_classes) == 2 else ":".join(["0"] * len(sorted_classes))

    ratio_str = ":".join(map(str, counts))
    percentages = [f"{(count/total_prototypes)*100:.1f}%" for count in counts]
    percentage_str = " (" + ", ".join(percentages) + ")"

    return ratio_str + percentage_str

def test_all_configurations(X, y, dataset_name):
    """
    Test multiple TDAPS configurations on a dataset using bifiltration.
    """
    print(f"\n{'='*80}")
    print(f"Testing Dataset: {dataset_name}")
    print(f"{'='*80}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    imbalance_ratio = max(class_counts) / min(class_counts)

    print(f"Dataset size: {len(X_train)} training, {len(X_test)} testing")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Class distribution in training: {dict(zip(unique_classes, class_counts))}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    # Baseline: Linear SVM with all training data
    print("\n--- Baseline (All training data, Linear SVM) ---")
    svm_all = LinearSVC(max_iter=10000, dual='auto', random_state=42)
    svm_all.fit(X_train, y_train)
    y_pred_all = svm_all.predict(X_test)

    baseline_metrics = calculate_metrics(y_test, y_pred_all)
    print(f"Accuracy: {baseline_metrics['accuracy']*100:.2f}%")
    print(f"G-Mean: {baseline_metrics['gmean']*100:.2f}%")

    baseline_gmean = baseline_metrics['gmean']

    # Define configurations to test
    # Expanded parameter grid (H0 only)
    configurations = []

    homology_dimensions = [0]
    neighbor_quantiles = [0.05, 0.10, 0.15, 0.20, 0.25]
    radius_statistics = ['mean']
    k_neighbors_list = [1, 3, 5, 10]
    min_persistences = [0.001, 0.01, 0.1]

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

    print(f"\nTesting {len(configurations)} configurations total")

    results = []

    for config in configurations:
        config_name = f"H{config['homology_dimension']}_q{int(config['neighbor_quantile']*100)}_r{config['radius_statistic']}_k{config['k_neighbors']}_p{config['min_persistence']}"
        print(f"\n--- Configuration: H{config['homology_dimension']}, "
              f"neighbor_q={config['neighbor_quantile']}, "
              f"radius_stat={config['radius_statistic']}, "
              f"k_neighbors={config['k_neighbors']}, "
              f"min_persistence={config['min_persistence']} (bifiltration ) ---")

        start_time = time.time()

        try:
            prototypes_dict, selectors_dict = compute_prototypes_all_classes(
                X_train, y_train,
                k_neighbors=config['k_neighbors'],
                homology_dimension=config['homology_dimension'],
                min_persistence=config['min_persistence'],
                neighbor_quantile=config['neighbor_quantile'],
                radius_statistic=config['radius_statistic']
            )

            selection_time = time.time() - start_time

            all_prototype_indices = []
            for cls in unique_classes:
                if cls in prototypes_dict:
                    all_prototype_indices.extend(prototypes_dict[cls])

            all_prototype_indices = np.array(all_prototype_indices)

            if len(all_prototype_indices) == 0:
                print("Warning: No prototypes selected, skipping configuration")
                results.append({
                    'config_name': config_name,
                    'homology_dim': config['homology_dimension'],
                    'neighbor_quantile': config['neighbor_quantile'],
                    'radius_statistic': config['radius_statistic'],
                    'k_neighbors': config['k_neighbors'],
                    'min_persistence': config['min_persistence'],
                    'n_prototypes': 0,
                    'reduction_pct': 100.0,
                    'accuracy': 0,
                    'gmean': 0,
                    'gmean_diff': -baseline_gmean,
                    'selection_time': selection_time,
                    'prototype_ratio': "0:0",
                    'per_class_recalls': [0] * len(unique_classes)
                })
                continue

            # Train Linear SVM with prototypes
            X_prototypes = X_train[all_prototype_indices]
            y_prototypes = y_train[all_prototype_indices]

            svm_prototypes = LinearSVC(max_iter=10000, dual='auto', random_state=42)
            svm_prototypes.fit(X_prototypes, y_prototypes)
            y_pred_prototypes = svm_prototypes.predict(X_test)

            proto_metrics = calculate_metrics(y_test, y_pred_prototypes)
            reduction_pct = (1 - len(all_prototype_indices) / len(X_train)) * 100
            prototype_ratio = compute_prototype_class_ratio(prototypes_dict, y_train)

            print(f"Prototypes selected: {len(all_prototype_indices)} ({reduction_pct:.1f}% reduction)")
            print(f"Prototype class ratio: {prototype_ratio}")
            print(f"Accuracy: {proto_metrics['accuracy']*100:.2f}%")
            print(f"G-Mean: {proto_metrics['gmean']*100:.2f}%")
            print(f"G-Mean difference: {(proto_metrics['gmean'] - baseline_gmean)*100:+.2f}%")
            print(f"Selection time: {selection_time:.2f}s")

            results.append({
                'config_name': config_name,
                'homology_dim': config['homology_dimension'],
                'neighbor_quantile': config['neighbor_quantile'],
                'radius_statistic': config['radius_statistic'],
                'k_neighbors': config['k_neighbors'],
                'min_persistence': config['min_persistence'],
                'n_prototypes': len(all_prototype_indices),
                'reduction_pct': reduction_pct,
                'accuracy': proto_metrics['accuracy'],
                'gmean': proto_metrics['gmean'],
                'gmean_diff': proto_metrics['gmean'] - baseline_gmean,
                'selection_time': selection_time,
                'prototype_ratio': prototype_ratio,
                'per_class_recalls': proto_metrics['per_class_recalls'],
                'prototypes_dict': prototypes_dict,
                'selectors_dict': selectors_dict,
                'y_pred': y_pred_prototypes
            })

        except Exception as e:
            print(f"Error in configuration {config_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config_name': config_name,
                'homology_dim': config['homology_dimension'],
                'neighbor_quantile': config.get('neighbor_quantile', 0.25),
                'radius_statistic': config.get('radius_statistic', 'mean'),
                'k_neighbors': config.get('k_neighbors', 3),
                'min_persistence': config.get('min_persistence', 0.001),
                'n_prototypes': 0,
                'reduction_pct': 100.0,
                'accuracy': 0,
                'gmean': 0,
                'gmean_diff': -baseline_gmean,
                'selection_time': 0,
                'prototype_ratio': "0:0",
                'per_class_recalls': [0] * len(unique_classes)
            })

    return {
        'dataset_name': dataset_name,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'baseline_accuracy': baseline_metrics['accuracy'],
        'baseline_gmean': baseline_gmean,
        'baseline_pred': y_pred_all,
        'configurations': results,
        'imbalance_ratio': imbalance_ratio,
        'n_classes': len(unique_classes)
    }

def visualize_dataset_and_prototypes(result):
    """Visualize the dataset and selected prototypes for the best configuration."""
    X_train = result['X_train']
    y_train = result['y_train']

    valid_configs = [c for c in result['configurations'] if c['n_prototypes'] > 0]
    if not valid_configs:
        print(f"No valid configurations for {result['dataset_name']}")
        return result

    best_config = max(valid_configs, key=lambda x: x['gmean'])
    result['best_config'] = best_config

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Original training data
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                           cmap='tab10', alpha=0.6, s=30)
    ax1.set_title('Original Training Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=ax1)

    # Plot 2: Selected prototypes showing filtration vertices
    ax2 = axes[1]
    ax2.scatter(X_train[:, 0], X_train[:, 1], c='lightgray', alpha=0.3, s=20)

    unique_classes = np.unique(y_train)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    # Show vertices from each filtration if selector is available
    for idx, cls in enumerate(unique_classes):
        if cls in best_config.get('selectors_dict', {}):
            selector = best_config['selectors_dict'][cls]
            if selector is not None and selector.bifiltration_data_ is not None:
                target_mask = y_train == cls
                target_indices = np.where(target_mask)[0]

                # Show radius and neighbor vertices
                radius_vertices = selector.bifiltration_data_.get('radius_vertices', [])
                neighbor_vertices = selector.bifiltration_data_.get('neighbor_vertices', [])

                if len(radius_vertices) > 0:
                    radius_global = target_indices[radius_vertices]
                    ax2.scatter(X_train[radius_global, 0], X_train[radius_global, 1],
                               c=[colors[idx]], s=60, marker='s', alpha=0.4,
                               edgecolors='black', linewidths=0.5)

                if len(neighbor_vertices) > 0:
                    neighbor_global = target_indices[neighbor_vertices]
                    ax2.scatter(X_train[neighbor_global, 0], X_train[neighbor_global, 1],
                               c=[colors[idx]], s=60, marker='^', alpha=0.4,
                               edgecolors='black', linewidths=0.5)

    # Plot final prototypes on top
    for idx, cls in enumerate(unique_classes):
        if cls in best_config['prototypes_dict']:
            proto_indices = best_config['prototypes_dict'][cls]
            if len(proto_indices) > 0:
                ax2.scatter(X_train[proto_indices, 0], X_train[proto_indices, 1],
                           c=[colors[idx]], s=120, marker='*',
                           edgecolors='black', linewidths=1.5,
                           label=f'Class {cls} ({len(proto_indices)} protos)', zorder=5)

    ax2.set_title('Selected Prototypes')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend(loc='best', fontsize=8)

    # Plot 3: Performance comparison
    ax3 = axes[2]
    metrics = ['Baseline\nG-Mean', 'TPS\nG-Mean', 'Data\nReduction']
    values = [
        result['baseline_gmean'] * 100,
        best_config['gmean'] * 100,
        best_config['reduction_pct']
    ]
    colors_bar = ['blue', 'green', 'orange']

    bars = ax3.bar(metrics, values, color=colors_bar, alpha=0.7)
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Performance Metrics')
    ax3.set_ylim(0, 105)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')

    plt.suptitle(f'{result["dataset_name"]} (Linear SVM)\n'
                f'Best Config G-Mean Diff: {best_config["gmean_diff"]*100:+.2f}%',
                fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure to results folder
    os.makedirs('results', exist_ok=True)
    # Create sanitized filename from dataset name
    safe_name = result["dataset_name"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("%", "pct").replace(",", "")
    filename = f'results/{safe_name}_LinearSVM.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")

    plt.show()

    return result

def plot_confusion_matrix(result):
    """Plot confusion matrix for the best configuration."""
    best_config = result.get('best_config', None)
    if best_config is None or best_config['n_prototypes'] == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_baseline = confusion_matrix(result['y_test'], result['baseline_pred'])
    ax1 = axes[0]
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Baseline (All Data, Linear SVM)\nG-Mean: {result["baseline_gmean"]*100:.2f}%')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    cm_tdaps = confusion_matrix(result['y_test'], best_config['y_pred'])
    ax2 = axes[1]
    sns.heatmap(cm_tdaps, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title(f'TPS ({format_config_name(best_config["config_name"])}, Linear SVM)\n'
                 f'G-Mean: {best_config["gmean"]*100:.2f}%')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.suptitle(f'{result["dataset_name"]} - Confusion Matrices (Linear SVM)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_summary_statistics(all_results):
    """Create comprehensive summary statistics with G-Mean focus."""
    best_config_summary = []

    for result in all_results:
        best_config = result.get('best_config', None)
        if best_config is None:
            continue

        best_config_summary.append({
            'Dataset': result['dataset_name'],
            'Classes': result['n_classes'],
            'Imbalance_Ratio': result['imbalance_ratio'],
            'Best_Config': best_config['config_name'],
            'H_dim': best_config['homology_dim'],
            'Neighbor_Quantile': best_config['neighbor_quantile'],
            'Radius_Statistic': best_config['radius_statistic'],
            'K_Neighbors': best_config['k_neighbors'],
            'Min_Persistence': best_config['min_persistence'],
            'Baseline_GMean': result['baseline_gmean'],
            'TDAPS_GMean': best_config['gmean'],
            'GMean_Diff': best_config['gmean_diff'],
            'Prototypes': best_config['n_prototypes'],
            'Reduction_%': best_config['reduction_pct'],
            'TDA_Time': best_config['selection_time'],
            'Prototype_Ratio': best_config['prototype_ratio']
        })

    if not best_config_summary:
        print("No valid results to summarize")
        return None

    best_df = pd.DataFrame(best_config_summary)

    print("\n" + "="*150)
    print("BEST CONFIGURATION FOR EACH DATASET (OPTIMIZED FOR G-MEAN) - bifiltration  (Linear SVM)")
    print("="*150)
    display_df = best_df.copy()
    for col in ['Baseline_GMean', 'TDAPS_GMean', 'GMean_Diff']:
        display_df[col] = display_df[col] * 100
    print(display_df.to_string(index=False, float_format=lambda x: f'{x:.3f}' if abs(x) < 1000 else f'{x:.0f}'))

    print("\n" + "="*60)
    print("CONFIGURATION POPULARITY ANALYSIS (BASED ON G-MEAN)")
    print("="*60)

    config_counts = best_df['Best_Config'].value_counts()
    print("\nHow often each configuration was best:")
    for config, count in config_counts.items():
        percentage = (count / len(best_df)) * 100
        print(f"{config}: {count} datasets ({percentage:.1f}%)")

    h0_count = sum(best_df['H_dim'] == 0)
    h1_count = sum(best_df['H_dim'] == 1)
    print(f"\nHomology dimension preference:")
    print(f"H0 (Connected Components): {h0_count} datasets ({h0_count/len(best_df)*100:.1f}%)")
    print(f"H1 (Loops/Cycles): {h1_count} datasets ({h1_count/len(best_df)*100:.1f}%)")

    radius_stat_counts = best_df['Radius_Statistic'].value_counts()
    print(f"\nRadius statistic preference:")
    for stat, count in radius_stat_counts.items():
        print(f"{stat}: {count} datasets ({count/len(best_df)*100:.1f}%)")

    k_neighbors_counts = best_df['K_Neighbors'].value_counts().sort_index()
    print(f"\nK-neighbors preference:")
    for k, count in k_neighbors_counts.items():
        print(f"k={k}: {count} datasets ({count/len(best_df)*100:.1f}%)")

    min_persistence_counts = best_df['Min_Persistence'].value_counts().sort_index()
    print(f"\nMin persistence preference:")
    for p, count in min_persistence_counts.items():
        print(f"p={p}: {count} datasets ({count/len(best_df)*100:.1f}%)")

    print("\n" + "="*60)
    print("OVERALL PERFORMANCE STATISTICS (bifiltration , Linear SVM)")
    print("="*60)
    print(f"Average G-Mean improvement: {best_df['GMean_Diff'].mean()*100:+.3f}%")
    print(f"Average data reduction: {best_df['Reduction_%'].mean():.1f}%")
    print(f"Average TDA computation time: {best_df['TDA_Time'].mean():.2f}s")
    print(f"Datasets with improved G-Mean: {sum(best_df['GMean_Diff'] > 0)}")
    print(f"Datasets with maintained G-Mean (±1%): {sum(abs(best_df['GMean_Diff']) <= 0.01)}")
    print(f"Datasets with degraded G-Mean: {sum(best_df['GMean_Diff'] < -0.01)}")

    return best_df

def run_comprehensive_synthetic_study():
    """Run the complete study on all synthetic datasets using bifiltration with Linear SVM."""
    print("="*80)
    print("COMPREHENSIVE TDAPS STUDY - bifiltration - EXPANDED (Linear SVM)")
    print("="*80)
    print("Using bifiltration approach:")
    print("- SEQUENTIAL two-step filtration process:")
    print("  * STEP 1 - Neighbor filtration: inter-class separation")
    print("    Selects feature closest to quantile threshold")
    print("  * STEP 2 - Radius filtration: intra-class structure (LEVEL SET)")
    print("    Runs ONLY on vertices from Step 1")
    print("    Selects feature closest to mean threshold")
    print("- Uses sum of ALL distances for radius parameter ")
    print("- Uses sum of distances to k_neighbors for neighbor parameter")
    print("- Prototypes: Final vertices from radius filtration (already restricted)")
    print("\nBaseline and Comparison: Linear Support Vector Machine (SVM)")
    print("\nExpanded Parameter Grid (H0 only):")
    print("  * Homology dimension: H0")
    print("  * Neighbor quantiles: 0.05, 0.10, 0.15, 0.20, 0.25")
    print("  * Radius statistic: mean")
    print("  * K-neighbors: 1, 3, 5, 10")
    print("  * Min persistence: 0.001, 0.01, 0.1")
    print("  * Total configurations: 60 (1 × 5 × 1 × 4 × 3)")
    print("="*80)

    datasets = generate_datasets()
    all_results = []

    for key, (X, y, name) in datasets.items():
        result = test_all_configurations(X, y, name)
        result = visualize_dataset_and_prototypes(result)
        plot_confusion_matrix(result)
        all_results.append(result)

    best_df = create_summary_statistics(all_results)

    if best_df is not None:
        os.makedirs('results', exist_ok=True)
        best_df.to_csv('results/results_synthetic_LinearSVMbaseline.csv', index=False)
        print("\n" + "="*60)
        print("Results saved to: 'results/results_synthetic_LinearSVMbaseline.csv'")
        print("="*60)

    return all_results, best_df

if __name__ == "__main__":
    print("\nStarting EXPANDED synthetic study with bifiltration implementation ()...")
    print("Using Linear Support Vector Machine (SVM) for baseline and comparison")
    print("This version runs SEQUENTIAL filtrations: neighbor first, then radius on subset")
    print("Feature selection: selects features closest to threshold (true level set)")
    print("Radius calculation: Uses sum of ALL distances (not k-NN)")
    print("Testing 60 configurations across 9 synthetic datasets = 540 total experiments\n")

    all_results, best_df = run_comprehensive_synthetic_study()

    print("\n" + "="*80)
    print("STUDY COMPLETED SUCCESSFULLY!")
    print("="*80)
