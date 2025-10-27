import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import issparse
from ripser import ripser
import warnings
warnings.filterwarnings('ignore')

class StandardBifiltrationPrototypeSelector:
    def __init__(self, k_neighbors=5, homology_dimension=0, min_persistence=0.01,
                 neighbor_quantile=0.15, radius_statistic='mean',
                 metric='euclidean', metric_params=None):
        """
        Bifiltration prototype selector using Rips filtrations.

        This implements a bifiltration approach using Rips filtrations where:
        1. We construct two different distance matrices for same-class points:
           - neighbor_distances: measures inter-class separation (how far pairs are from other class)
           - radius_distances: measures intra-class structure (standard pairwise distances)
        2. We run a SEQUENTIAL two-step filtration process:
           - STEP 1 - Neighbor filtration: standard Rips on neighbor_distances
             Select the feature whose lifetime minimizes |lifetime - quantile(lifetimes)|
           - STEP 2 - Radius filtration: standard Rips on radius_distances (level set)
             Run ONLY on vertices from neighbor filtration
             Select the feature whose lifetime minimizes |lifetime - mean/median(lifetimes)|
        3. Final prototypes are vertices from the radius filtration

        Parameters:
        -----------
        k_neighbors : int
            Number of neighbors for computing neighbor distances (inter-class only)
        homology_dimension : int
            Dimension of homology to compute (0 or 1)
        min_persistence : float
            Minimum persistence threshold for features
        neighbor_quantile : float
            Quantile for neighbor filtration lifetime threshold (0.0 to 1.0)
            Example: 0.25 means features in bottom 25% of lifetimes
        radius_statistic : str
            Statistic for radius filtration threshold: 'mean' or 'median'
        metric : str or callable
            Distance metric to use
        metric_params : dict
            Additional parameters for the metric
        """
        self.k_neighbors = k_neighbors
        self.homology_dimension = homology_dimension
        self.min_persistence = min_persistence
        self.neighbor_quantile = neighbor_quantile
        self.radius_statistic = radius_statistic
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}
        self.prototypes_ = None
        self.bifiltration_data_ = None

        # Map common metric names
        self._metric_aliases = {
            'l1': 'cityblock',
            'l2': 'euclidean',
            'manhattan': 'cityblock',
            'cosine_distance': 'cosine',
        }

    def _ensure_dense(self, arr):
        """Helper function to convert sparse arrays to dense if needed."""
        if issparse(arr):
            return arr.toarray()
        else:
            return arr

    def _compute_pairwise_distances(self, X):
        """Compute pairwise distances using the specified metric."""
        X = self._ensure_dense(X)

        if callable(self.metric):
            distance_matrix = self.metric(X, X, **self.metric_params)
            if distance_matrix.shape != (len(X), len(X)):
                raise ValueError(f"Custom metric must return square matrix, got shape {distance_matrix.shape}")
            return distance_matrix
        else:
            metric_name = self._metric_aliases.get(self.metric, self.metric)

            try:
                if metric_name in ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                  'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                  'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                                  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
                    distances = pdist(X, metric=metric_name, **self.metric_params)
                    return squareform(distances)
                else:
                    return cdist(X, X, metric=metric_name, **self.metric_params)
            except Exception as e:
                try:
                    return cdist(X, X, metric=metric_name, **self.metric_params)
                except Exception as e2:
                    raise ValueError(f"Could not compute distances with metric '{metric_name}': {e2}")

    def _get_sklearn_metric(self):
        """Get the appropriate metric name for sklearn's NearestNeighbors."""
        if callable(self.metric):
            return 'precomputed'

        metric_name = self._metric_aliases.get(self.metric, self.metric)

        sklearn_metric_map = {
            'cityblock': 'manhattan',
            'euclidean': 'euclidean',
            'sqeuclidean': 'sqeuclidean',
            'cosine': 'cosine',
            'correlation': 'correlation',
            'hamming': 'hamming',
            'jaccard': 'jaccard',
            'chebyshev': 'chebyshev',
            'minkowski': 'minkowski',
        }

        if metric_name in sklearn_metric_map:
            return sklearn_metric_map[metric_name]
        else:
            return metric_name

    def fit(self, X, y, target_class=1):
        """
        Fit the bifiltration prototype selector using standard Rips filtrations.

        Algorithm:
        1. Compute two distance matrices for same-class points:
           - neighbor_distances: based on separation from other class
           - radius_distances: standard pairwise distances
        2. Run persistence on neighbor_distances (STANDARD RIPS, FIRST)
           - Extract features with significant persistence
           - Select feature whose lifetime is closest to quantile threshold
           - Extract participating vertices
        3. Run persistence on radius_distances (STANDARD RIPS, SECOND, on subset)
           - Restrict to vertices from neighbor filtration (level set)
           - Extract features with significant persistence
           - Select feature whose lifetime is closest to mean/median threshold
           - Extract participating vertices (within the subset)
        4. Final prototypes are vertices from the radius filtration

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Class labels
        target_class : int
            The class to select prototypes for
        """
        X = self._ensure_dense(X)

        # Separate target class from others
        target_mask = (y == target_class)
        other_mask = ~target_mask

        if not np.any(target_mask):
            raise ValueError(f"No samples found for target class {target_class}")
        if not np.any(other_mask):
            raise ValueError("No samples found for non-target classes")

        X_target = X[target_mask]
        X_other = X[other_mask]
        target_indices = np.where(target_mask)[0]

        print(f"Processing {len(X_target)} target samples and {len(X_other)} other samples")

        # STEP 1: Compute distance matrices for bifiltration
        print("\n=== Step 1: Computing Distance Matrices ===")
        neighbor_distances, radius_distances = self._compute_bifiltration_distances(X_target, X_other)

        # STEP 2: Run persistence on neighbor_distances (Rips on neighbor filtration first)
        print("\n=== Step 2: Neighbor Filtration (Inter-class, Standard Rips) ===")
        neighbor_features, neighbor_threshold = self._compute_filtration_persistence(
            neighbor_distances, filtration_name="neighbor"
        )

        # STEP 3: Extract vertices from neighbor filtration
        print("\n=== Step 3: Extracting Neighbor Vertices ===")
        neighbor_vertices = self._extract_vertices_from_features(
            neighbor_features, neighbor_distances
        )
        print(f"Neighbor filtration vertices: {len(neighbor_vertices)}")

        if len(neighbor_vertices) == 0:
            raise ValueError(
                "No vertices found in neighbor filtration."
                "Try adjusting min_persistence or neighbor_quantile parameters."
            )

        # STEP 4: Run persistence on radius_distances (Runs radius filtration on neighbor level set)
        print("\n=== Step 4: Radius Filtration on Neighbor Vertices (Level Set, Standard Rips) ===")
        # Subset the data to neighbor vertices only
        radius_distances_subset = radius_distances[np.ix_(neighbor_vertices, neighbor_vertices)]

        radius_features, radius_threshold = self._compute_filtration_persistence(
            radius_distances_subset, filtration_name="radius"
        )

        # STEP 5: Extract vertices from radius filtration (indices are local to subset)
        print("\n=== Step 5: Extracting Radius Vertices from Subset ===")
        radius_vertices_local = self._extract_vertices_from_features(
            radius_features, radius_distances_subset
        )
        print(f"Radius filtration vertices (within neighbor subset): {len(radius_vertices_local)}")

        if len(radius_vertices_local) == 0:
            print("Warning: No vertices found in radius filtration!")
            print("Using all neighbor vertices instead")
            radius_vertices_local = np.arange(len(neighbor_vertices))

        # Ensure integer dtype for indexing
        radius_vertices_local = np.asarray(radius_vertices_local, dtype=int)

        # Map local radius indices back to original target class indices
        final_vertices = neighbor_vertices[radius_vertices_local]

        # Map back to original dataset indices
        prototype_indices = target_indices[final_vertices]

        self.prototypes_ = prototype_indices
        self.bifiltration_data_ = {
            'neighbor_distances': neighbor_distances,
            'radius_distances': radius_distances,
            'radius_features': radius_features,
            'neighbor_features': neighbor_features,
            'radius_threshold': radius_threshold,
            'neighbor_threshold': neighbor_threshold,
            'neighbor_vertices': neighbor_vertices,
            'radius_vertices_local': radius_vertices_local,
            'final_vertices': final_vertices,
            'radius_distances_subset': radius_distances_subset,
        }

        print(f"\nFinal prototypes: {len(prototype_indices)} vertices")

        return self

    def _compute_bifiltration_distances(self, X_target, X_other):
        """
        Compute the two distance matrices for bifiltration using standard Rips.

        Returns:
        --------
        neighbor_distances : array, shape (n_target, n_target)
            Distance matrix for neighbor filtration (inter-class separation)
            For edge (i,j), distance = max separation of i and j from other class
        radius_distances : array, shape (n_target, n_target)
            Distance matrix for radius filtration (intra-class structure)
            Standard pairwise distances between same-class points
        """
        n_target = len(X_target)
        sklearn_metric = self._get_sklearn_metric()

        # Compute radius distances (intra-class structure) - STANDARD PAIRWISE DISTANCES
        print("Computing radius distances (standard pairwise distances)...")
        radius_distances = self._compute_pairwise_distances(X_target)

        # Compute neighbor distances (inter-class separation)
        print("Computing neighbor distances (inter-class separation)...")

        # For each target point, compute its minimum distance to other class
        if sklearn_metric == 'precomputed':
            cross_distances = self.metric(X_target, X_other, **self.metric_params)
        else:
            cross_distances = cdist(X_target, X_other,
                                   metric=self._metric_aliases.get(self.metric, self.metric),
                                   **self.metric_params)

        # For each target point, compute mean distance to k nearest other-class points
        neighbor_separation = np.zeros(n_target)
        for i in range(n_target):
            distances = cross_distances[i]
            k_nearest = np.partition(distances, min(self.k_neighbors-1, len(distances)-1))
            neighbor_separation[i] = np.mean(k_nearest[:min(self.k_neighbors, len(distances))])

        # Construct neighbor distance matrix
        # For edge (i,j), use the MINIMUM of their separations
        # This means edges appear when BOTH points are well-separated
        neighbor_distances = np.zeros((n_target, n_target))
        for i in range(n_target):
            for j in range(n_target):
                if i == j:
                    neighbor_distances[i, j] = 0
                else:
                    # Edge appears when both vertices are separated from other class
                    # Using min means edge appears when the LESS separated point reaches threshold
                    neighbor_distances[i, j] = min(neighbor_separation[i], neighbor_separation[j])

        print(f"Radius distances (intra-class): [{np.min(radius_distances):.4f}, {np.max(radius_distances):.4f}]")
        print(f"Neighbor distances (inter-class): [{np.min(neighbor_distances):.4f}, {np.max(neighbor_distances):.4f}]")

        return neighbor_distances, radius_distances

    def _compute_filtration_persistence(self, distance_matrix, filtration_name):
        """
        Run persistence homology on a standard Rips filtration.

        Parameters:
        -----------
        distance_matrix : array, shape (n, n)
            Pairwise distance matrix
        filtration_name : str
            Name of the filtration (for logging)

        Returns:
        --------
        selected_features : list of (birth, death, lifetime)
            Features that pass the persistence threshold
        lifetime_threshold : float
            The threshold used for selection
        """
        # Compute persistence using standard Rips. Ensure full coverage using 1.5
        max_thresh = np.max(distance_matrix) * 1.5

        try:
            result = ripser(distance_matrix, distance_matrix=True,
                          maxdim=self.homology_dimension, thresh=max_thresh)
            diagrams = result['dgms']

            if self.homology_dimension >= len(diagrams):
                print(f"No H{self.homology_dimension} diagram available")
                return [], 0.0

            persistence_diagram = diagrams[self.homology_dimension]

            # Extract finite features
            finite_features = []
            for birth, death in persistence_diagram:
                if death != np.inf:
                    lifetime = death - birth
                    if lifetime >= self.min_persistence:
                        finite_features.append((birth, death, lifetime))

            print(f"{filtration_name.capitalize()} filtration: {len(finite_features)} features with persistence >= {self.min_persistence}")

            if len(finite_features) == 0:
                print(f"No significant features in {filtration_name} filtration")
                return [], 0.0

            # Compute lifetime threshold based on filtration type
            lifetimes = np.array([f[2] for f in finite_features])

            if filtration_name == "radius":
                # Use mean or median for radius filtration
                if self.radius_statistic == 'mean':
                    lifetime_threshold = np.mean(lifetimes)
                elif self.radius_statistic == 'median':
                    lifetime_threshold = np.median(lifetimes)
                else:
                    lifetime_threshold = np.mean(lifetimes)
                print(f"Radius lifetime threshold ({self.radius_statistic}): {lifetime_threshold:.4f}")
            else:
                # Use quantile for neighbor filtration
                lifetime_threshold = np.quantile(lifetimes, self.neighbor_quantile)
                print(f"Neighbor lifetime threshold (quantile={self.neighbor_quantile}): {lifetime_threshold:.4f}")

            # Select feature(s) whose lifetime minimizes |lifetime - threshold| (true level set)
            lifetimes_array = np.array([f[2] for f in finite_features])
            distances_to_threshold = np.abs(lifetimes_array - lifetime_threshold)

            # Find the feature that minimizes the distance to threshold
            closest_idx = np.argmin(distances_to_threshold)
            closest_distance = distances_to_threshold[closest_idx]

            # Take all features with the same minimum distance (handles ties)
            selected_indices = np.where(distances_to_threshold == closest_distance)[0]
            selected_features = [finite_features[i] for i in selected_indices]

            print(f"Selected {len(selected_features)} feature(s) from {filtration_name} filtration")
            print(f"  Target threshold: {lifetime_threshold:.4f}, Actual lifetime: {lifetimes_array[closest_idx]:.4f}")

            return selected_features, lifetime_threshold

        except Exception as e:
            print(f"Error computing persistence for {filtration_name} filtration: {e}")
            return [], 0.0

    def _extract_vertices_from_features(self, features, distance_matrix):
        """
        Identify which vertices participate in the given persistent features.

        For standard Rips filtration, we select vertices that are involved
        in edges at the birth and death times of features, using a minimization
        approach to find the edges closest to these critical values.

        Parameters:
        -----------
        features : list of (birth, death, lifetime)
            Selected persistent features
        distance_matrix : array, shape (n, n)
            Distance matrix

        Returns:
        --------
        participating_vertices : array
            Indices of vertices participating in features
        """
        if len(features) == 0:
            return np.array([], dtype=int)

        participating_set = set()
        n_vertices = len(distance_matrix)

        for birth, death, _ in features:
            if self.homology_dimension == 0:
                # H0: vertices on edges at birth (create component) and death (merge components)

                # Find edges that minimize |distance - birth|
                min_birth_distance = float('inf')
                birth_edges = []
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        dist_to_birth = abs(distance_matrix[i, j] - birth)
                        if dist_to_birth < min_birth_distance:
                            min_birth_distance = dist_to_birth
                            birth_edges = [(i, j)]
                        elif dist_to_birth == min_birth_distance:
                            birth_edges.append((i, j))

                # Add vertices from edges at birth
                for i, j in birth_edges:
                    participating_set.add(i)
                    participating_set.add(j)

                # Find edges that minimize |distance - death|
                min_death_distance = float('inf')
                death_edges = []
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        dist_to_death = abs(distance_matrix[i, j] - death)
                        if dist_to_death < min_death_distance:
                            min_death_distance = dist_to_death
                            death_edges = [(i, j)]
                        elif dist_to_death == min_death_distance:
                            death_edges.append((i, j))

                # Add vertices from edges at death
                for i, j in death_edges:
                    participating_set.add(i)
                    participating_set.add(j)

            elif self.homology_dimension == 1:
                # H1: vertices forming loops at birth time

                # Find edges that minimize |distance - birth|
                min_birth_distance = float('inf')
                birth_edges = []
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        dist_to_birth = abs(distance_matrix[i, j] - birth)
                        if dist_to_birth < min_birth_distance:
                            min_birth_distance = dist_to_birth
                            birth_edges = [(i, j)]
                        elif dist_to_birth == min_birth_distance:
                            birth_edges.append((i, j))

                # Add vertices from edges at birth
                for i, j in birth_edges:
                    participating_set.add(i)
                    participating_set.add(j)

            else:
                # Higher dimensional homology - use edges at birth
                min_birth_distance = float('inf')
                birth_edges = []
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        dist_to_birth = abs(distance_matrix[i, j] - birth)
                        if dist_to_birth < min_birth_distance:
                            min_birth_distance = dist_to_birth
                            birth_edges = [(i, j)]
                        elif dist_to_birth == min_birth_distance:
                            birth_edges.append((i, j))

                # Add vertices from edges at birth
                for i, j in birth_edges:
                    participating_set.add(i)
                    participating_set.add(j)

        # If no vertices found, add vertices on shortest edges
        if len(participating_set) == 0 and len(features) > 0:
            # Find edges with smallest distances
            distances_flat = []
            for i in range(n_vertices):
                for j in range(i+1, n_vertices):
                    distances_flat.append((distance_matrix[i, j], i, j))
            distances_flat.sort()

            # Add vertices from 5 shortest edges
            for k in range(min(5, len(distances_flat))):
                _, i, j = distances_flat[k]
                participating_set.add(i)
                participating_set.add(j)

        return np.array(sorted(participating_set), dtype=int)

    def get_prototypes(self, X):
        """
        Get the selected prototypes from the original dataset.

        Returns:
        --------
        X_prototypes : array of prototype samples
        prototype_indices : array of prototype indices in original dataset
        """
        if self.prototypes_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        prototype_indices = self.prototypes_
        if prototype_indices is not None and len(prototype_indices) > 0:
            X_dense = self._ensure_dense(X)
            return X_dense[prototype_indices], prototype_indices
        else:
            return np.array([]), np.array([])

    def get_summary_statistics(self):
        """Get summary statistics about the prototype selection."""
        if self.prototypes_ is None:
            return None

        stats = {
            'n_prototypes': len(self.prototypes_),
            'min_persistence': self.min_persistence,
            'homology_dimension': self.homology_dimension,
            'k_neighbors': self.k_neighbors,
            'neighbor_quantile': self.neighbor_quantile,
            'radius_statistic': self.radius_statistic,
            'metric': self.metric if not callable(self.metric) else 'custom',
        }

        if self.bifiltration_data_ is not None:
            stats['n_radius_features'] = len(self.bifiltration_data_['radius_features'])
            stats['n_neighbor_features'] = len(self.bifiltration_data_['neighbor_features'])
            stats['n_neighbor_vertices'] = len(self.bifiltration_data_['neighbor_vertices'])
            stats['n_radius_vertices_local'] = len(self.bifiltration_data_['radius_vertices_local'])
            stats['n_final_vertices'] = len(self.bifiltration_data_['final_vertices'])
            stats['radius_threshold'] = self.bifiltration_data_['radius_threshold']
            stats['neighbor_threshold'] = self.bifiltration_data_['neighbor_threshold']

            rd = self.bifiltration_data_['radius_distances']
            nd = self.bifiltration_data_['neighbor_distances']
            # Get non-diagonal elements for statistics
            rd_nondiag = rd[~np.eye(rd.shape[0], dtype=bool)]
            nd_nondiag = nd[~np.eye(nd.shape[0], dtype=bool)]
            stats['radius_distance_range'] = (np.min(rd_nondiag), np.max(rd_nondiag))
            stats['neighbor_distance_range'] = (np.min(nd_nondiag), np.max(nd_nondiag))

            rd_subset = self.bifiltration_data_['radius_distances_subset']
            if rd_subset.size > 0:
                rd_subset_nondiag = rd_subset[~np.eye(rd_subset.shape[0], dtype=bool)] if rd_subset.shape[0] > 0 else rd_subset
                if len(rd_subset_nondiag) > 0:
                    stats['radius_subset_distance_range'] = (np.min(rd_subset_nondiag), np.max(rd_subset_nondiag))

        return stats

    def plot_prototypes(self, X, y, target_class=1, figsize=(18, 5)):
        """Visualize the selected prototypes and bifiltration structure (2D data only)."""
        if X.shape[1] != 2:
            print("Visualization only available for 2D data")
            return

        if self.prototypes_ is None:
            print("No prototypes to plot. Fit the model first.")
            return

        X = self._ensure_dense(X)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        target_mask = (y == target_class)
        other_mask = ~target_mask

        # Plot 1: Original data with prototypes and filtration vertices
        ax1 = axes[0]
        ax1.scatter(X[other_mask, 0], X[other_mask, 1], c='lightgray',
                   alpha=0.5, s=30, label='Other classes')
        ax1.scatter(X[target_mask, 0], X[target_mask, 1], c='lightblue',
                   alpha=0.5, s=30, label=f'Class {target_class}')

        # Show vertices from each filtration
        target_indices = np.where(target_mask)[0]
        if self.bifiltration_data_ is not None:
            neighbor_global = target_indices[self.bifiltration_data_['neighbor_vertices']]
            final_global = target_indices[self.bifiltration_data_['final_vertices']]

            ax1.scatter(X[neighbor_global, 0], X[neighbor_global, 1],
                       c='lightgreen', s=70, marker='o', alpha=0.4,
                       label='Neighbor vertices', edgecolors='black', linewidths=0.5)
            ax1.scatter(X[final_global, 0], X[final_global, 1],
                       c='orange', s=90, marker='s', alpha=0.7,
                       label='Radius vertices (level set)', edgecolors='black', linewidths=0.8)

        if len(self.prototypes_) > 0:
            ax1.scatter(X[self.prototypes_, 0], X[self.prototypes_, 1],
                       c='red', s=150, marker='*', edgecolors='black',
                       linewidths=1.5, label='Final Prototypes', zorder=5)

        ax1.set_title(f'Standard Rips Bifiltration: Prototypes ({len(self.prototypes_)} points)')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distance matrix comparison
        ax2 = axes[1]
        if self.bifiltration_data_ is not None:
            # Show heatmap comparison of distance matrices
            rd = self.bifiltration_data_['radius_distances']
            nd = self.bifiltration_data_['neighbor_distances']

            # Compute correlation or difference
            rd_flat = rd[np.triu_indices_from(rd, k=1)]
            nd_flat = nd[np.triu_indices_from(nd, k=1)]

            ax2.scatter(rd_flat, nd_flat, alpha=0.3, s=10)
            ax2.set_xlabel('Radius distances (intra-class)')
            ax2.set_ylabel('Neighbor distances (inter-class)')
            ax2.set_title('Distance Matrix Comparison')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Summary statistics
        ax3 = axes[2]
        ax3.axis('off')

        stats_text = "Standard Rips Bifiltration (V7)\n" + "="*35 + "\n"
        stats = self.get_summary_statistics()
        if stats:
            stats_text += f"Final prototypes: {stats['n_prototypes']}\n"
            stats_text += f"Homology dim: H{stats['homology_dimension']}\n"
            stats_text += f"k-neighbors: {stats['k_neighbors']}\n\n"

            stats_text += "Step 1 - Neighbor Filtration:\n"
            stats_text += f"  Type: Standard Rips\n"
            stats_text += f"  Features: {stats.get('n_neighbor_features', 0)}\n"
            stats_text += f"  Vertices: {stats.get('n_neighbor_vertices', 0)}\n"
            stats_text += f"  Threshold: {stats.get('neighbor_threshold', 0):.4f}\n"
            stats_text += f"  Quantile: {stats['neighbor_quantile']}\n\n"

            stats_text += "Step 2 - Radius (Level Set):\n"
            stats_text += f"  Type: Standard Rips\n"
            stats_text += f"  Features: {stats.get('n_radius_features', 0)}\n"
            stats_text += f"  Local vertices: {stats.get('n_radius_vertices_local', 0)}\n"
            stats_text += f"  Threshold: {stats.get('radius_threshold', 0):.4f}\n"
            stats_text += f"  Statistic: {stats['radius_statistic']}\n\n"

            stats_text += f"Final: {stats.get('n_final_vertices', 0)} vertices"

        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f'Standard Rips Bifiltration Analysis (Class {target_class}) - V7',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Testing function
def test_standard_rips_bifiltration():
    """Test the standard Rips bifiltration implementation"""
    import time

    print("Testing Standard Rips Bifiltration Implementation")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=2,
                              class_sep=1.0, random_state=42)

    print("\nStandard Rips Bifiltration Approach:")
    print("- Step 1: Run neighbor filtration (Filtration on inter-class distances)")
    print("- Step 2: Run radius filtration on neighbor vertices (Filtration on intra-class distances)")
    print("- Uses standard pairwise distances for radius filtration")
    print("- Uses inter-class separation distances for neighbor filtration")
    print("- Selects features closest to threshold (true level set)")
    print("-" * 70)

    selector = StandardBifiltrationPrototypeSelector(
        k_neighbors=10,
        homology_dimension=0,
        min_persistence=0.01,
        neighbor_quantile=0.25,  # Top 75% of lifetimes in neighbor filtration (Step 1)
        radius_statistic='mean'  # Features closest to mean lifetime in radius filtration (Step 2)
    )

    start_time = time.time()
    selector.fit(X, y, target_class=1)
    elapsed_time = time.time() - start_time

    prototypes, prototype_indices = selector.get_prototypes(X)

    print(f"\nResults:")
    print(f"Time taken: {elapsed_time:.3f} seconds")
    print(f"Prototypes selected: {len(prototype_indices)}")

    # Visualize results
    selector.plot_prototypes(X, y, target_class=1)

    return selector


if __name__ == "__main__":
    # Test the standard Rips bifiltration implementation
    selector = test_standard_rips_bifiltration()

    # Get statistics
    stats = selector.get_summary_statistics()
    print("\n" + "="*70)
    print("Detailed Statistics:")
    print("="*70)
    for key, value in stats.items():
        print(f"  {key}: {value}")
