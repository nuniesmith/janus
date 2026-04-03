//! UMAP dimensionality reduction
//!
//! Part of the Visual Cortex region
//! Component: visualization
//!
//! Implements a simplified UMAP (Uniform Manifold Approximation and Projection)
//! for dimensionality reduction of market data, enabling:
//! - Visualization of high-dimensional feature spaces
//! - Clustering of market regimes
//! - Anomaly detection through outlier identification
//! - Pattern discovery in trading signals

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Configuration for UMAP algorithm
#[derive(Debug, Clone)]
pub struct UmapConfig {
    /// Number of neighbors to consider for each point
    pub n_neighbors: usize,
    /// Target dimensionality (usually 2 or 3 for visualization)
    pub n_components: usize,
    /// Minimum distance between points in low-dimensional space
    pub min_dist: f64,
    /// Controls how tightly UMAP packs points together
    pub spread: f64,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Number of epochs for optimization
    pub n_epochs: usize,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Metric for distance calculation
    pub metric: DistanceMetric,
    /// Negative sample rate
    pub negative_sample_rate: usize,
    /// Local connectivity parameter
    pub local_connectivity: f64,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            learning_rate: 1.0,
            n_epochs: 200,
            random_seed: 42,
            metric: DistanceMetric::Euclidean,
            negative_sample_rate: 5,
            local_connectivity: 1.0,
        }
    }
}

/// Distance metrics for UMAP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Standard Euclidean distance
    Euclidean,
    /// Manhattan/L1 distance
    Manhattan,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Correlation distance
    Correlation,
    /// Chebyshev/L-infinity distance
    Chebyshev,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    pub fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }

        match self {
            Self::Euclidean => {
                let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                sum.sqrt()
            }
            Self::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            Self::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_a < 1e-10 || norm_b < 1e-10 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            Self::Correlation => {
                let n = a.len() as f64;
                let mean_a = a.iter().sum::<f64>() / n;
                let mean_b = b.iter().sum::<f64>() / n;

                let mut cov = 0.0;
                let mut var_a = 0.0;
                let mut var_b = 0.0;

                for (x, y) in a.iter().zip(b.iter()) {
                    let da = x - mean_a;
                    let db = y - mean_b;
                    cov += da * db;
                    var_a += da * da;
                    var_b += db * db;
                }

                let std_a = var_a.sqrt();
                let std_b = var_b.sqrt();

                if std_a < 1e-10 || std_b < 1e-10 {
                    1.0
                } else {
                    1.0 - (cov / (std_a * std_b))
                }
            }
            Self::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max),
        }
    }
}

/// A neighbor with its distance
#[derive(Debug, Clone)]
struct Neighbor {
    index: usize,
    distance: f64,
}

/// Fuzzy simplicial set edge
#[derive(Debug, Clone)]
struct FuzzyEdge {
    source: usize,
    target: usize,
    weight: f64,
}

/// Result of UMAP embedding
#[derive(Debug, Clone)]
pub struct UmapEmbedding {
    /// Low-dimensional coordinates (n_samples x n_components)
    pub coordinates: Vec<Vec<f64>>,
    /// Number of samples
    pub n_samples: usize,
    /// Number of components
    pub n_components: usize,
    /// Final optimization loss
    pub final_loss: f64,
    /// Number of epochs run
    pub epochs_run: usize,
}

impl UmapEmbedding {
    /// Get coordinates for a specific sample
    pub fn get(&self, index: usize) -> Option<&[f64]> {
        self.coordinates.get(index).map(|v| v.as_slice())
    }

    /// Get all x coordinates (for 2D plotting)
    pub fn x_coords(&self) -> Vec<f64> {
        self.coordinates.iter().map(|c| c[0]).collect()
    }

    /// Get all y coordinates (for 2D plotting)
    pub fn y_coords(&self) -> Vec<f64> {
        if self.n_components < 2 {
            vec![0.0; self.n_samples]
        } else {
            self.coordinates.iter().map(|c| c[1]).collect()
        }
    }

    /// Get all z coordinates (for 3D plotting)
    pub fn z_coords(&self) -> Vec<f64> {
        if self.n_components < 3 {
            vec![0.0; self.n_samples]
        } else {
            self.coordinates.iter().map(|c| c[2]).collect()
        }
    }

    /// Find nearest neighbors in embedding space
    pub fn find_neighbors(&self, point: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = self
            .coordinates
            .iter()
            .enumerate()
            .map(|(i, coord)| {
                let dist = coord
                    .iter()
                    .zip(point.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
}

/// Statistics about the UMAP fit
#[derive(Debug, Clone, Default)]
pub struct UmapStats {
    /// Number of samples processed
    pub n_samples: usize,
    /// Input dimensionality
    pub input_dim: usize,
    /// Output dimensionality
    pub output_dim: usize,
    /// Average number of neighbors found
    pub avg_neighbors: f64,
    /// Total edges in fuzzy graph
    pub total_edges: usize,
    /// Optimization epochs completed
    pub epochs_completed: usize,
}

/// UMAP dimensionality reduction
///
/// Implements a simplified version of UMAP for reducing high-dimensional
/// market data to 2D/3D for visualization and analysis.
pub struct Umap {
    /// Configuration
    config: UmapConfig,
    /// High-dimensional data points
    data: Vec<Vec<f64>>,
    /// k-nearest neighbors for each point
    knn_indices: Vec<Vec<usize>>,
    /// Distances to k-nearest neighbors
    knn_distances: Vec<Vec<f64>>,
    /// Fuzzy simplicial set (graph edges)
    fuzzy_set: Vec<FuzzyEdge>,
    /// Current embedding
    embedding: Option<UmapEmbedding>,
    /// Statistics
    stats: UmapStats,
    /// RNG state
    rng_state: u64,
}

impl Default for Umap {
    fn default() -> Self {
        Self::new()
    }
}

impl Umap {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(UmapConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: UmapConfig) -> Self {
        Self {
            rng_state: config.random_seed,
            config,
            data: Vec::new(),
            knn_indices: Vec::new(),
            knn_distances: Vec::new(),
            fuzzy_set: Vec::new(),
            embedding: None,
            stats: UmapStats::default(),
        }
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Fit UMAP to data and return embedding
    pub fn fit_transform(&mut self, data: &[Vec<f64>]) -> Result<UmapEmbedding> {
        if data.is_empty() {
            return Err(Error::InvalidInput("Empty data".into()));
        }

        let n_samples = data.len();
        let input_dim = data[0].len();

        if n_samples < self.config.n_neighbors {
            return Err(Error::InvalidInput(format!(
                "Need at least {} samples for {} neighbors",
                self.config.n_neighbors, self.config.n_neighbors
            )));
        }

        // Store data
        self.data = data.to_vec();

        // Step 1: Compute k-nearest neighbors
        self.compute_knn();

        // Step 2: Compute fuzzy simplicial set
        self.compute_fuzzy_simplicial_set();

        // Step 3: Initialize embedding
        let mut embedding = self.initialize_embedding(n_samples);

        // Step 4: Optimize embedding
        let final_loss = self.optimize_embedding(&mut embedding);

        self.stats = UmapStats {
            n_samples,
            input_dim,
            output_dim: self.config.n_components,
            avg_neighbors: self.knn_indices.iter().map(|v| v.len()).sum::<usize>() as f64
                / n_samples as f64,
            total_edges: self.fuzzy_set.len(),
            epochs_completed: self.config.n_epochs,
        };

        let result = UmapEmbedding {
            coordinates: embedding,
            n_samples,
            n_components: self.config.n_components,
            final_loss,
            epochs_run: self.config.n_epochs,
        };

        self.embedding = Some(result.clone());
        Ok(result)
    }

    /// Transform new data points using the fitted model
    pub fn transform(&self, new_data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.data.is_empty() {
            return Err(Error::InvalidInput("Model not fitted".into()));
        }

        let mut result = Vec::with_capacity(new_data.len());

        for point in new_data {
            // Find nearest neighbors in training data
            let neighbors = self.find_nearest_neighbors(point, self.config.n_neighbors);

            // Interpolate position from neighbors
            let mut coord = vec![0.0; self.config.n_components];

            if let Some(ref embedding) = self.embedding {
                let mut weight_sum = 0.0;
                for (idx, dist) in neighbors {
                    let weight = 1.0 / (dist + 1e-10);
                    weight_sum += weight;
                    for (c, &emb_val) in coord.iter_mut().zip(embedding.coordinates[idx].iter()) {
                        *c += weight * emb_val;
                    }
                }
                for c in &mut coord {
                    *c /= weight_sum;
                }
            }

            result.push(coord);
        }

        Ok(result)
    }

    /// Get the current embedding
    pub fn embedding(&self) -> Option<&UmapEmbedding> {
        self.embedding.as_ref()
    }

    /// Get statistics
    pub fn stats(&self) -> &UmapStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &UmapConfig {
        &self.config
    }

    /// Reset the model
    pub fn reset(&mut self) {
        self.data.clear();
        self.knn_indices.clear();
        self.knn_distances.clear();
        self.fuzzy_set.clear();
        self.embedding = None;
        self.stats = UmapStats::default();
    }

    // --- Private methods ---

    /// Compute k-nearest neighbors for all points
    fn compute_knn(&mut self) {
        let n_samples = self.data.len();
        let k = self.config.n_neighbors;

        self.knn_indices = Vec::with_capacity(n_samples);
        self.knn_distances = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut neighbors: Vec<Neighbor> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = self.config.metric.distance(&self.data[i], &self.data[j]);
                    Neighbor {
                        index: j,
                        distance: dist,
                    }
                })
                .collect();

            // Sort by distance and take k nearest
            neighbors.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            neighbors.truncate(k);

            let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
            let distances: Vec<f64> = neighbors.iter().map(|n| n.distance).collect();

            self.knn_indices.push(indices);
            self.knn_distances.push(distances);
        }
    }

    /// Compute fuzzy simplicial set from k-NN graph
    fn compute_fuzzy_simplicial_set(&mut self) {
        let n_samples = self.data.len();
        self.fuzzy_set.clear();

        // Compute local connectivity (rho) and sigma for each point
        let mut rhos = vec![0.0; n_samples];
        let mut sigmas = vec![1.0; n_samples];

        for i in 0..n_samples {
            if !self.knn_distances[i].is_empty() {
                // rho is the distance to the nearest neighbor
                rhos[i] = self.knn_distances[i][0];

                // sigma is computed via binary search to achieve target sum
                sigmas[i] = self.compute_sigma(i, rhos[i]);
            }
        }

        // Build fuzzy set with membership strengths
        for i in 0..n_samples {
            for (j_idx, &j) in self.knn_indices[i].iter().enumerate() {
                let dist = self.knn_distances[i][j_idx];
                let rho = rhos[i];
                let sigma = sigmas[i];

                // Compute membership strength
                let weight = if dist <= rho {
                    1.0
                } else {
                    (-(dist - rho) / sigma).exp()
                };

                if weight > 1e-10 {
                    self.fuzzy_set.push(FuzzyEdge {
                        source: i,
                        target: j,
                        weight,
                    });
                }
            }
        }

        // Symmetrize: combine edges (a,b) and (b,a)
        let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();

        for edge in &self.fuzzy_set {
            let key = if edge.source < edge.target {
                (edge.source, edge.target)
            } else {
                (edge.target, edge.source)
            };

            let current = edge_map.entry(key).or_insert(0.0);
            // Fuzzy union: a + b - a*b
            *current = *current + edge.weight - *current * edge.weight;
        }

        // Rebuild fuzzy set with symmetrized weights
        self.fuzzy_set.clear();
        for ((i, j), weight) in edge_map {
            self.fuzzy_set.push(FuzzyEdge {
                source: i,
                target: j,
                weight,
            });
            self.fuzzy_set.push(FuzzyEdge {
                source: j,
                target: i,
                weight,
            });
        }
    }

    /// Compute sigma via binary search to achieve target neighbor sum
    fn compute_sigma(&self, point_idx: usize, rho: f64) -> f64 {
        let target = (self.config.n_neighbors as f64).ln();
        let distances = &self.knn_distances[point_idx];

        let mut lo = 1e-10;
        let mut hi = 1e10;
        let mut mid = 1.0;

        for _ in 0..64 {
            mid = (lo + hi) / 2.0;

            let sum: f64 = distances
                .iter()
                .map(|&d| {
                    if d <= rho {
                        1.0
                    } else {
                        (-(d - rho) / mid).exp()
                    }
                })
                .sum();

            if (sum - target).abs() < 1e-5 {
                break;
            }

            if sum < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        mid
    }

    /// Initialize embedding with spectral or random initialization
    fn initialize_embedding(&mut self, n_samples: usize) -> Vec<Vec<f64>> {
        let n_components = self.config.n_components;

        // Use random initialization with spectral-like structure
        let mut embedding = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut coords = Vec::with_capacity(n_components);
            for _ in 0..n_components {
                // Box-Muller transform for normal distribution
                let u1 = self.next_random().max(1e-10);
                let u2 = self.next_random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                coords.push(z * 0.0001); // Small initial spread
            }
            embedding.push(coords);
        }

        embedding
    }

    /// Optimize embedding using stochastic gradient descent
    fn optimize_embedding(&mut self, embedding: &mut [Vec<f64>]) -> f64 {
        let n_epochs = self.config.n_epochs;
        let n_samples = embedding.len();
        let n_components = self.config.n_components;

        // Precompute a and b parameters for the low-dimensional similarity
        let (a, b) = self.find_ab_params();

        let mut loss = 0.0;

        for epoch in 0..n_epochs {
            // Adaptive learning rate
            let alpha = self.config.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);

            loss = 0.0;

            // Process positive edges (attractive forces)
            for edge in &self.fuzzy_set {
                let i = edge.source;
                let j = edge.target;
                let w = edge.weight;

                // Compute distance in embedding space
                let mut dist_sq = 0.0;
                for c in 0..n_components {
                    let diff = embedding[i][c] - embedding[j][c];
                    dist_sq += diff * diff;
                }
                let dist_sq = dist_sq.max(1e-10);

                // Compute gradient
                let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));

                let grad_coeff = grad_coeff * w * alpha;

                // Update positions
                for c in 0..n_components {
                    let diff = embedding[i][c] - embedding[j][c];
                    let grad = grad_coeff * diff;

                    embedding[i][c] -= grad;
                    embedding[j][c] += grad;
                }

                // Accumulate loss
                let q = 1.0 / (1.0 + a * dist_sq.powf(b));
                loss -= w * q.max(1e-10).ln();
            }

            // Apply negative sampling (repulsive forces)
            for _ in 0..self.config.negative_sample_rate {
                for i in 0..n_samples {
                    let j = (self.next_random() * n_samples as f64) as usize % n_samples;
                    if i == j {
                        continue;
                    }

                    let mut dist_sq = 0.0;
                    for c in 0..n_components {
                        let diff = embedding[i][c] - embedding[j][c];
                        dist_sq += diff * diff;
                    }
                    let dist_sq = dist_sq.max(0.01); // Prevent division by zero

                    // Repulsive gradient
                    let grad_coeff = 2.0 * b / (0.001 + dist_sq) / (1.0 + a * dist_sq.powf(b));
                    let grad_coeff = grad_coeff * alpha / self.config.negative_sample_rate as f64;

                    for c in 0..n_components {
                        let diff = embedding[i][c] - embedding[j][c];
                        let grad = grad_coeff * diff;

                        // Clip gradient
                        let grad = grad.clamp(-4.0, 4.0);

                        embedding[i][c] += grad;
                    }
                }
            }
        }

        loss
    }

    /// Find a and b parameters for the low-dimensional distribution
    fn find_ab_params(&self) -> (f64, f64) {
        // These are approximations for the t-distribution-like curve
        // that UMAP uses in the low-dimensional space
        let min_dist = self.config.min_dist;
        let spread = self.config.spread;

        // Approximate a and b via curve fitting (simplified)
        let b = 1.0;
        let a = 1.0 / (spread - min_dist);

        (a, b)
    }

    /// Find nearest neighbors in original high-dimensional space
    fn find_nearest_neighbors(&self, point: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut neighbors: Vec<(usize, f64)> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, data_point)| {
                let dist = self.config.metric.distance(point, data_point);
                (i, dist)
            })
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);
        neighbors
    }

    /// Simple xorshift64 PRNG
    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
        let mut data = Vec::with_capacity(n_samples);
        let mut seed: u64 = 12345;

        for i in 0..n_samples {
            let mut point = Vec::with_capacity(n_features);
            // Create clusters
            let cluster = i % 3;
            let offset = cluster as f64 * 5.0;

            for _ in 0..n_features {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let rand = (seed as f64) / (u64::MAX as f64);
                point.push(offset + rand);
            }
            data.push(point);
        }

        data
    }

    #[test]
    fn test_basic() {
        let instance = Umap::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        assert!((DistanceMetric::Euclidean.distance(&a, &b) - 5.0).abs() < 0.01);
        assert!((DistanceMetric::Manhattan.distance(&a, &b) - 7.0).abs() < 0.01);
        assert!((DistanceMetric::Chebyshev.distance(&a, &b) - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        // Perpendicular vectors have cosine distance of 1
        let dist = DistanceMetric::Cosine.distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.01);

        // Parallel vectors have cosine distance of 0
        let c = vec![2.0, 0.0];
        let dist = DistanceMetric::Cosine.distance(&a, &c);
        assert!(dist.abs() < 0.01);
    }

    #[test]
    fn test_fit_transform() {
        let config = UmapConfig {
            n_neighbors: 5,
            n_components: 2,
            n_epochs: 50,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(30, 10);
        let result = umap.fit_transform(&data);

        assert!(result.is_ok());
        let embedding = result.unwrap();

        assert_eq!(embedding.n_samples, 30);
        assert_eq!(embedding.n_components, 2);
        assert_eq!(embedding.coordinates.len(), 30);
        assert_eq!(embedding.coordinates[0].len(), 2);
    }

    #[test]
    fn test_insufficient_data() {
        let config = UmapConfig {
            n_neighbors: 15,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(5, 10); // Only 5 samples
        let result = umap.fit_transform(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_coords() {
        let config = UmapConfig {
            n_neighbors: 5,
            n_components: 3,
            n_epochs: 20,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(20, 5);
        let embedding = umap.fit_transform(&data).unwrap();

        let x = embedding.x_coords();
        let y = embedding.y_coords();
        let z = embedding.z_coords();

        assert_eq!(x.len(), 20);
        assert_eq!(y.len(), 20);
        assert_eq!(z.len(), 20);
    }

    #[test]
    fn test_transform_new_points() {
        let config = UmapConfig {
            n_neighbors: 5,
            n_components: 2,
            n_epochs: 30,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(30, 5);
        umap.fit_transform(&data).unwrap();

        let new_data = generate_test_data(5, 5);
        let transformed = umap.transform(&new_data);

        assert!(transformed.is_ok());
        let coords = transformed.unwrap();
        assert_eq!(coords.len(), 5);
        assert_eq!(coords[0].len(), 2);
    }

    #[test]
    fn test_find_neighbors_in_embedding() {
        let config = UmapConfig {
            n_neighbors: 5,
            n_components: 2,
            n_epochs: 20,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(20, 5);
        let embedding = umap.fit_transform(&data).unwrap();

        let query = embedding.coordinates[0].clone();
        let neighbors = embedding.find_neighbors(&query, 3);

        assert_eq!(neighbors.len(), 3);
        // First neighbor should be the point itself (distance 0)
        assert_eq!(neighbors[0].0, 0);
        assert!(neighbors[0].1 < 0.01);
    }

    #[test]
    fn test_stats() {
        let config = UmapConfig {
            n_neighbors: 5,
            n_components: 2,
            n_epochs: 20,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(20, 8);
        umap.fit_transform(&data).unwrap();

        let stats = umap.stats();
        assert_eq!(stats.n_samples, 20);
        assert_eq!(stats.input_dim, 8);
        assert_eq!(stats.output_dim, 2);
        assert!(stats.avg_neighbors > 0.0);
        assert!(stats.total_edges > 0);
    }

    #[test]
    fn test_reset() {
        let config = UmapConfig {
            n_neighbors: 5,
            n_components: 2,
            n_epochs: 20,
            ..Default::default()
        };
        let mut umap = Umap::with_config(config);

        let data = generate_test_data(20, 5);
        umap.fit_transform(&data).unwrap();

        assert!(umap.embedding().is_some());

        umap.reset();

        assert!(umap.embedding().is_none());
        assert_eq!(umap.stats().n_samples, 0);
    }

    #[test]
    fn test_different_metrics() {
        let data = generate_test_data(20, 5);

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Cosine,
        ] {
            let config = UmapConfig {
                n_neighbors: 5,
                n_components: 2,
                n_epochs: 20,
                metric,
                ..Default::default()
            };
            let mut umap = Umap::with_config(config);

            let result = umap.fit_transform(&data);
            assert!(result.is_ok(), "Failed with metric {:?}", metric);
        }
    }

    #[test]
    fn test_transform_without_fit() {
        let umap = Umap::new();
        let new_data = generate_test_data(5, 5);

        let result = umap.transform(&new_data);
        assert!(result.is_err());
    }
}
