import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import issparse
from ripser import ripser
import warnings
warnings.filterwarnings('ignore')

class BifiltrationPrototypeSelector:
    def __init__(self, k_neighbors=5, homology_dimension=0, min_persistence=0.01,
                 neighbor_quantile=0.15, radius_statistic='mean',
                 metric='euclidean', metric_params=None):
        """
        Bifiltration prototype selector.

        This implements a bifiltration level set approach where:
        1. Each vertex has two parameters: radius (r) and neighbor (n)
           - radius = sum of distances to ALL same-class neighbors
           - neighbor = sum of distances to k nearest other-class neighbors
        2. We run a sequential two-step filtration process:
           - STEP 1 - Neighbor filtration: based on inter-class separation
             Select the feature whose lifetime minimizes |lifetime - quantile(lifetimes)|
           - STEP 2 - Radius filtration: based on intra-class structure
             Run ONLY on vertices from neighbor filtration (level set)
             Select the feature whose lifetime minimizes |lifetime - mean(lifetimes)| (or median)
        3. Final prototypes are vertices from the selected subcomplex of the radius filtration
           (which are already restricted by the neighbor filtration vertices)

        Parameters:
        -----------
        k_neighbors : int
            Number of neighbors for computing neighbor values (inter-class only)
        homology_dimension : int
            Dimension of homology to compute (0 or 1)
        min_persistence : float
            Minimum persistence threshold for features
        neighbor_quantile : float
            Quantile for neighbor filtration lifetime threshold (0.0 to 1.0)
            Example: 0.25 selects features near the 25th percentile (shorter lifetimes)
                     0.75 selects features near the 75th percentile (longer lifetimes)
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
        Fit the bifiltration level set prototype selector.

        Algorithm:
        1. Compute two parameters for each vertex:
           - radius_values: sum of distances to ALL same-class neighbors
           - neighbor_values: sum of distances to k nearest other-class neighbors
        2. Run persistence on NEIGHBOR filtration 
           - Create vertex-weighted filtration using neighbor_values
           - Extract features with significant persistence
           - Select feature whose lifetime is closest to quantile threshold
           - Extract participating vertices
        3. Run persistence on RADIUS filtration
           - Restrict to vertices from neighbor filtration (level set)
           - Create vertex-weighted filtration using radius_values
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

        # STEP 1: Compute bifiltration parameters
        print("\n=== Step 1: Computing Bifiltration Parameters ===")
        radius_values, neighbor_values = self._compute_bifiltration_values(X_target, X_other)

        # Compute base distance matrix (for geometric structure)
        base_distances = self._compute_pairwise_distances(X_target)

        # STEP 2: Run persistence on NEIGHBOR filtration (FIRST)
        print("\n=== Step 2: Neighbor Filtration (Inter-class) ===")
        neighbor_features, neighbor_threshold = self._compute_filtration_persistence(
            base_distances, neighbor_values, filtration_name="neighbor"
        )

        # STEP 3: Extract vertices from neighbor filtration
        print("\n=== Step 3: Extracting Neighbor Vertices ===")
        neighbor_vertices = self._extract_vertices_from_features(
            neighbor_features, neighbor_values, base_distances
        )
        print(f"Neighbor filtration vertices: {len(neighbor_vertices)}")

        if len(neighbor_vertices) == 0:
            print("Warning: No vertices found in neighbor filtration!")
            print("Using all vertices instead")
            neighbor_vertices = np.arange(len(X_target))

        # STEP 4: Run persistence on RADIUS filtration (SECOND, on neighbor vertices subset)
        print("\n=== Step 4: Radius Filtration on Neighbor Vertices (Level Set) ===")
        # Subset the data to neighbor vertices only
        radius_values_subset = radius_values[neighbor_vertices]
        base_distances_subset = base_distances[np.ix_(neighbor_vertices, neighbor_vertices)]

        radius_features, radius_threshold = self._compute_filtration_persistence(
            base_distances_subset, radius_values_subset, filtration_name="radius"
        )

        # STEP 5: Extract vertices from radius filtration (indices are local to subset)
        print("\n=== Step 5: Extracting Radius Vertices from Subset ===")
        radius_vertices_local = self._extract_vertices_from_features(
            radius_features, radius_values_subset, base_distances_subset
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
            'radius_values': radius_values,
            'neighbor_values': neighbor_values,
            'radius_features': radius_features,
            'neighbor_features': neighbor_features,
            'radius_threshold': radius_threshold,
            'neighbor_threshold': neighbor_threshold,
            'neighbor_vertices': neighbor_vertices,
            'radius_vertices_local': radius_vertices_local,
            'final_vertices': final_vertices,
            'base_distances': base_distances,
            'base_distances_subset': base_distances_subset,
            'radius_values_subset': radius_values_subset
        }

        print(f"\nFinal prototypes: {len(prototype_indices)} vertices")

        return self

    def _compute_bifiltration_values(self, X_target, X_other):
        """
        Compute the two filtration parameters for each vertex.

        Returns:
        --------
        radius_values : array, shape (n_target,)
            Intra-class filtration values (sum of distances to ALL same-class neighbors)
        neighbor_values : array, shape (n_target,)
            Inter-class filtration values (sum of distances to k nearest other-class neighbors)
        """
        n_target = len(X_target)
        sklearn_metric = self._get_sklearn_metric()

        # Compute radius values (intra-class structure) - SUM OF ALL DISTANCES
        print("Computing radius values using SUM OF ALL DISTANCES to same-class neighbors...")
        distance_matrix = self._compute_pairwise_distances(X_target)

        # For each point, sum all distances to other same-class points (excluding self)
        radius_values = np.zeros(n_target)
        for i in range(n_target):
            # Sum all distances except self-distance (which is 0)
            radius_values[i] = np.sum(distance_matrix[i]) - distance_matrix[i, i]

        # Compute neighbor values (inter-class separation) - STILL USES k-NN
        if sklearn_metric == 'precomputed':
            distance_matrix = self.metric(X_target, X_other, **self.metric_params)
            neighbor_values = np.zeros(n_target)
            for i in range(n_target):
                distances = distance_matrix[i]
                k_nearest = np.partition(distances, min(self.k_neighbors-1, len(distances)-1))
                # Sum of distances to k nearest neighbors
                neighbor_values[i] = np.sum(k_nearest[:min(self.k_neighbors, len(distances))])
        else:
            nn_other = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X_other)),
                                       metric=sklearn_metric,
                                       metric_params=self.metric_params)
            nn_other.fit(X_other)
            distances, _ = nn_other.kneighbors(X_target)
            # Sum of distances to k nearest neighbors
            neighbor_values = np.sum(distances, axis=1)

        print(f"Radius values (intra-class, ALL distances): [{np.min(radius_values):.4f}, {np.max(radius_values):.4f}]")
        print(f"Neighbor values (inter-class, k-NN): [{np.min(neighbor_values):.4f}, {np.max(neighbor_values):.4f}]")

        return radius_values, neighbor_values

    def _create_weighted_distance_matrix(self, base_distances, vertex_weights):
        """
        Create a distance matrix for vertex-weighted Rips filtration.

        In a vertex-weighted Rips filtration:
        - Vertex i "appears" at time vertex_weights[i]
        - Edge (i,j) appears at time max(vertex_weights[i], vertex_weights[j], base_distances[i,j])

        We encode this by modifying the distance matrix so ripser sees the correct
        filtration values.

        Parameters:
        -----------
        base_distances : array, shape (n, n)
            Original pairwise distances
        vertex_weights : array, shape (n,)
            Weight (birth time) for each vertex

        Returns:
        --------
        weighted_distances : array, shape (n, n)
            Modified distance matrix encoding vertex weights
        """
        n = len(vertex_weights)
        weighted_distances = base_distances.copy()

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Edge (i,j) appears at max of vertex weights and distance
                    weighted_distances[i, j] = max(
                        base_distances[i, j],
                        vertex_weights[i],
                        vertex_weights[j]
                    )

        return weighted_distances

    def _compute_filtration_persistence(self, base_distances, vertex_weights, filtration_name):
        """
        Run persistence homology on a vertex-weighted filtration.

        Parameters:
        -----------
        base_distances : array, shape (n, n)
            Original pairwise distances
        vertex_weights : array, shape (n,)
            Vertex weights for this filtration
        filtration_name : str
            Name of the filtration (for logging)

        Returns:
        --------
        selected_features : list of (birth, death, lifetime)
            Features that pass the persistence threshold
        lifetime_threshold : float
            The threshold used for selection
        """
        # Create vertex-weighted distance matrix
        weighted_distances = self._create_weighted_distance_matrix(base_distances, vertex_weights)

        # Compute persistence
        max_thresh = np.max(weighted_distances) * 1.5

        try:
            result = ripser(weighted_distances, distance_matrix=True,
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

    def _extract_vertices_from_features(self, features, vertex_weights, distance_matrix):
        """
        Identify which vertices participate in the given persistent features.

        For a level set approach, we select vertices that are directly involved
        in the feature: vertices near the birth time (component creation) and
        vertices on edges near the death time (component merging).

        Parameters:
        -----------
        features : list of (birth, death, lifetime)
            Selected persistent features
        vertex_weights : array, shape (n,)
            Vertex weights for this filtration
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
        n_vertices = len(vertex_weights)

        for birth, death, lifetime in features:
            # Use 10% of the feature's lifetime as tolerance
            birth_tolerance = lifetime * 0.1
            death_tolerance = lifetime * 0.1

            if self.homology_dimension == 0:
                # H0: vertices that create/merge connected components

                # Vertices at/near birth time (create new component)
                birth_mask = np.abs(vertex_weights - birth) <= birth_tolerance
                participating_set.update(np.where(birth_mask)[0])

                # Vertices on edges at/near death time (merge components)
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        edge_value = max(distance_matrix[i, j], vertex_weights[i], vertex_weights[j])
                        if abs(edge_value - death) <= death_tolerance:
                            participating_set.add(i)
                            participating_set.add(j)

            elif self.homology_dimension == 1:
                # H1: vertices that form loops

                # Vertices at/near birth time (loop appears)
                birth_mask = np.abs(vertex_weights - birth) <= birth_tolerance
                participating_set.update(np.where(birth_mask)[0])

                # Vertices on edges at/near birth time (forming the loop)
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        edge_value = max(distance_matrix[i, j], vertex_weights[i], vertex_weights[j])
                        if abs(edge_value - birth) <= birth_tolerance:
                            participating_set.add(i)
                            participating_set.add(j)

            else:
                # Higher dimensional homology - just use vertices near birth
                birth_mask = np.abs(vertex_weights - birth) <= birth_tolerance
                participating_set.update(np.where(birth_mask)[0])

        # If no vertices found, add vertices with lowest weights
        if len(participating_set) == 0 and len(features) > 0:
            n_to_add = min(5, n_vertices)
            lowest_indices = np.argsort(vertex_weights)[:n_to_add]
            participating_set.update(lowest_indices)

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

            rv = self.bifiltration_data_['radius_values']
            nv = self.bifiltration_data_['neighbor_values']
            stats['radius_value_range'] = (np.min(rv), np.max(rv))
            stats['neighbor_value_range'] = (np.min(nv), np.max(nv))

            rv_subset = self.bifiltration_data_['radius_values_subset']
            stats['radius_subset_value_range'] = (np.min(rv_subset), np.max(rv_subset))

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

        ax1.set_title(f'Bifiltration Level Set: Prototypes ({len(self.prototypes_)} points)')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Bifiltration parameter space
        ax2 = axes[1]
        if self.bifiltration_data_ is not None:
            rv = self.bifiltration_data_['radius_values']
            nv = self.bifiltration_data_['neighbor_values']

            # Plot all vertices
            ax2.scatter(rv, nv, c='lightblue', s=50, alpha=0.5, label='All vertices')

            # Highlight vertices from neighbor filtration
            neighbor_local = self.bifiltration_data_['neighbor_vertices']
            final_local = self.bifiltration_data_['final_vertices']

            ax2.scatter(rv[neighbor_local], nv[neighbor_local],
                       c='lightgreen', s=80, marker='o', alpha=0.4,
                       label='Neighbor filtration', edgecolors='black', linewidths=0.5)
            ax2.scatter(rv[final_local], nv[final_local],
                       c='orange', s=100, marker='s', alpha=0.7,
                       label='Radius (level set)', edgecolors='black', linewidths=0.8)
            ax2.scatter(rv[final_local], nv[final_local],
                       c='red', s=120, marker='*', alpha=0.9,
                       label='Final prototypes', edgecolors='black', linewidths=1, zorder=5)

            ax2.set_xlabel('Radius (intra-class, ALL distances)')
            ax2.set_ylabel('Neighbor (inter-class, k-NN)')
            ax2.set_title('Bifiltration Parameter Space')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Summary statistics
        ax3 = axes[2]
        ax3.axis('off')

        stats_text = "Bifiltration Level Set Statistics\n" + "="*35 + "\n"
        stats = self.get_summary_statistics()
        if stats:
            stats_text += f"Final prototypes: {stats['n_prototypes']}\n"
            stats_text += f"Homology dim: H{stats['homology_dimension']}\n"
            stats_text += f"k-neighbors: {stats['k_neighbors']}\n\n"

            stats_text += "Step 1 - Neighbor Filtration:\n"
            stats_text += f"  Features: {stats.get('n_neighbor_features', 0)}\n"
            stats_text += f"  Vertices: {stats.get('n_neighbor_vertices', 0)}\n"
            stats_text += f"  Threshold: {stats.get('neighbor_threshold', 0):.4f}\n"
            stats_text += f"  Quantile: {stats['neighbor_quantile']}\n\n"

            stats_text += "Step 2 - Radius Filtration:\n"
            stats_text += f"  Features: {stats.get('n_radius_features', 0)}\n"
            stats_text += f"  Local vertices: {stats.get('n_radius_vertices_local', 0)}\n"
            stats_text += f"  Threshold: {stats.get('radius_threshold', 0):.4f}\n"
            stats_text += f"  Statistic: {stats['radius_statistic']}\n"
            stats_text += f"  Type: ALL distances\n\n"

            stats_text += f"Final: {stats.get('n_final_vertices', 0)} vertices"

        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f'Bifiltration Level Set Analysis (Class {target_class}) - V6',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Testing function
def test_level_set_bifiltration():
    """Test the bifiltration level set implementation"""
    import time

    print("Testing Bifiltration Level Set Implementation")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=2,
                              class_sep=1.0, random_state=42)

    print("\nBifiltration Approach:")
    print("- Step 1: Run neighbor filtration (inter-class separation)")
    print("- Step 2: Run radius filtration on neighbor vertices (level set)")
    print("- Uses sum of distances to k_neighbors for neighbor vertex weights")
    print("- Uses sum of ALL distances for calculation of radius vertex weights")
    print("- Selects features closest to thresholds in each step")
    print("-" * 70)

    selector = BifiltrationPrototypeSelector(
        k_neighbors=3,
        homology_dimension=0,
        min_persistence=0.001,
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
    # Test the bifiltration level set implementation
    selector = test_level_set_bifiltration()

    # Get statistics
    stats = selector.get_summary_statistics()
    print("\n" + "="*70)
    print("Detailed Statistics:")
    print("="*70)
    for key, value in stats.items():
        print(f"  {key}: {value}")
