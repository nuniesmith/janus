//! Parametric UMAP — Neural Network Dimensionality Reduction
//!
//! Implements Parametric UMAP (Sainburg et al., 2021) using Candle for the
//! encoder network. Unlike standard UMAP which produces a fixed embedding,
//! Parametric UMAP trains a neural network to learn the mapping function,
//! enabling real-time projection of new data points without re-fitting.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Parametric UMAP Pipeline                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐     ┌───────────────┐     ┌────────────┐ │
//! │  │ High-dim     │     │  Fuzzy        │     │ Parametric │ │
//! │  │ Input        │────▶│  Simplicial   │────▶│ Encoder    │ │
//! │  │ (768-d BERT) │     │  Set (k-NN)   │     │ (MLP)      │ │
//! │  └──────────────┘     └───────────────┘     └─────┬──────┘ │
//! │                                                    │        │
//! │         ┌──────────────────────────────────────────┘        │
//! │         ▼                                                   │
//! │  ┌──────────────┐     ┌───────────────┐                    │
//! │  │ Low-dim      │     │   Qdrant      │                    │
//! │  │ Embedding    │────▶│   Storage     │                    │
//! │  │ (2-d / 3-d)  │     │   + Monitoring│                    │
//! │  └──────────────┘     └───────────────┘                    │
//! │                                                              │
//! │  Use cases:                                                  │
//! │  • Regime cluster visualization                             │
//! │  • Drift detection via embedding shift                      │
//! │  • Real-time projection of new BERT embeddings              │
//! │  • Similarity search in reduced space                       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::visual_cortex::parametric_umap::*;
//!
//! // Configure and build
//! let config = UmapConfig::default();
//! let mut umap = ParametricUmap::new(config)?;
//!
//! // Fit on high-dimensional data (e.g., BERT [CLS] embeddings)
//! let data: Vec<Vec<f32>> = load_embeddings();
//! let training_stats = umap.fit(&data)?;
//!
//! // Project new points in real-time
//! let new_embedding = vec![0.1f32; 768];
//! let projected = umap.transform(&[new_embedding])?;
//!
//! // Store in Qdrant for monitoring
//! let bridge = UmapQdrantBridge::new(qdrant_client, UmapQdrantConfig::default());
//! bridge.store_projection(&projected[0], metadata).await?;
//! ```

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, linear};
use memory::qdrant_client::PayloadValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// UMAP hyperparameters and encoder architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapConfig {
    /// Number of nearest neighbors for graph construction.
    /// Higher values capture more global structure; lower values preserve
    /// more local detail. Typical range: 5–50.
    pub n_neighbors: usize,

    /// Output dimensionality (2 or 3 for visualization, higher for downstream ML).
    pub n_components: usize,

    /// Minimum distance between embedded points. Controls cluster tightness.
    /// Range: 0.0–1.0. Lower = tighter clusters.
    pub min_dist: f32,

    /// Spread of the embedded points. Together with `min_dist`, controls
    /// the clumpiness of the embedding.
    pub spread: f32,

    /// Number of negative samples per positive edge in the cross-entropy loss.
    pub negative_sample_rate: usize,

    /// Learning rate for the encoder optimizer.
    pub learning_rate: f64,

    /// Weight decay (L2 regularization) for the encoder optimizer.
    pub weight_decay: f64,

    /// Number of training epochs over the edge set.
    pub n_epochs: usize,

    /// Mini-batch size (number of edges per batch).
    pub batch_size: usize,

    /// Hidden layer dimensions for the encoder MLP.
    /// E.g. `[256, 128, 64]` for a 3-hidden-layer network.
    pub encoder_hidden_dims: Vec<usize>,

    /// Input dimensionality (e.g. 768 for BERT-base, 256 for BERT-tiny).
    pub input_dim: usize,

    /// Dropout rate for encoder hidden layers (0.0 = no dropout).
    pub dropout_rate: f32,

    /// Whether to apply batch normalization in the encoder.
    pub use_batch_norm: bool,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Device to run on.
    pub device: InferenceDevice,

    /// Local connectivity parameter for fuzzy set construction.
    /// Number of iterations for smooth kNN distance estimation.
    pub local_connectivity: usize,

    /// Bandwidth for the kernel in high-dimensional space.
    /// Usually found via binary search per point.
    pub bandwidth: f32,

    /// Number of epochs between logging training metrics.
    pub log_interval: usize,

    /// Whether to use the approximate (random projection tree) kNN.
    /// When false, uses exact brute-force kNN.
    pub approximate_knn: bool,

    /// Metric for computing distances in the input space.
    pub metric: DistanceMetric,
}

/// Distance metric for the input space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance (1 - cosine_similarity).
    Cosine,
    /// Manhattan (L1) distance.
    Manhattan,
}

/// Device selection for inference / training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "metal")]
    Metal,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            negative_sample_rate: 5,
            learning_rate: 1e-3,
            weight_decay: 1e-5,
            n_epochs: 200,
            batch_size: 512,
            encoder_hidden_dims: vec![256, 128, 64],
            input_dim: 768,
            dropout_rate: 0.0,
            use_batch_norm: false,
            seed: 42,
            device: InferenceDevice::Cpu,
            local_connectivity: 1,
            bandwidth: 1.0,
            log_interval: 20,
            approximate_knn: false,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl UmapConfig {
    /// Configuration preset for BERT-base embeddings (768-d → 2-d).
    pub fn bert_base() -> Self {
        Self {
            input_dim: 768,
            n_components: 2,
            encoder_hidden_dims: vec![512, 256, 128],
            n_epochs: 200,
            n_neighbors: 15,
            min_dist: 0.1,
            ..Default::default()
        }
    }

    /// Configuration preset for BERT-tiny embeddings (256-d → 2-d).
    pub fn bert_tiny() -> Self {
        Self {
            input_dim: 256,
            n_components: 2,
            encoder_hidden_dims: vec![128, 64],
            n_epochs: 150,
            n_neighbors: 10,
            ..Default::default()
        }
    }

    /// Configuration for regime monitoring with 3-d output.
    pub fn regime_3d() -> Self {
        Self {
            input_dim: 768,
            n_components: 3,
            encoder_hidden_dims: vec![512, 256, 128],
            n_epochs: 300,
            n_neighbors: 20,
            min_dist: 0.15,
            ..Default::default()
        }
    }

    /// Builder: set input dimensionality.
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Builder: set output dimensionality.
    pub fn with_n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Builder: set number of neighbors.
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Builder: set number of training epochs.
    pub fn with_n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = n;
        self
    }

    /// Builder: set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set encoder hidden layer dimensions.
    pub fn with_encoder_dims(mut self, dims: Vec<usize>) -> Self {
        self.encoder_hidden_dims = dims;
        self
    }

    /// Builder: set distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Resolve the Candle device from config.
    fn candle_device(&self) -> CandleResult<Device> {
        match self.device {
            InferenceDevice::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            InferenceDevice::Cuda(ordinal) => Device::new_cuda(ordinal),
            #[cfg(feature = "metal")]
            InferenceDevice::Metal => Device::new_metal(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Smooth kNN Distance (sigma search)
// ---------------------------------------------------------------------------

/// Find the bandwidth (sigma) for each point such that the perplexity of the
/// conditional probability distribution matches `log2(n_neighbors)`.
///
/// Uses binary search per point (up to 64 iterations).
fn smooth_knn_dists(
    distances: &[Vec<f32>],
    k: usize,
    local_connectivity: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = distances.len();
    let target = (k as f32).ln();
    let mut sigmas = vec![1.0f32; n];
    let mut rhos = vec![0.0f32; n];

    for i in 0..n {
        let dists = &distances[i];
        if dists.is_empty() {
            continue;
        }

        // rho = distance to the nearest neighbor (index `local_connectivity - 1`)
        let rho_idx = (local_connectivity.saturating_sub(1)).min(dists.len() - 1);
        rhos[i] = dists[rho_idx].max(0.0);

        let mut lo: f32 = 1e-8;
        let mut hi: f32 = f32::INFINITY;
        let mut mid: f32 = 1.0;

        for _ in 0..64 {
            let mut val = 0.0f32;
            for d in dists.iter() {
                let shifted = (*d - rhos[i]).max(0.0);
                val += (-shifted / mid).exp();
            }

            if (val - target).abs() < 1e-5 {
                break;
            }

            if val > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi.is_infinite() {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        sigmas[i] = mid;
    }

    (sigmas, rhos)
}

// ---------------------------------------------------------------------------
// Fuzzy Simplicial Set (k-NN graph construction)
// ---------------------------------------------------------------------------

/// An edge in the fuzzy simplicial set graph.
#[derive(Debug, Clone, Copy)]
pub struct GraphEdge {
    /// Source node index.
    pub from: usize,
    /// Target node index.
    pub to: usize,
    /// Edge weight (membership strength), in [0, 1].
    pub weight: f32,
}

/// The fuzzy simplicial set — a weighted k-NN graph that encodes the
/// topological structure of the high-dimensional data.
#[derive(Debug, Clone)]
pub struct FuzzySimplicialSet {
    /// Number of data points.
    pub n_points: usize,
    /// All directed edges with membership strengths.
    pub edges: Vec<GraphEdge>,
    /// Per-point neighbor indices.
    pub neighbor_indices: Vec<Vec<usize>>,
    /// Per-point neighbor distances.
    pub neighbor_distances: Vec<Vec<f32>>,
}

impl FuzzySimplicialSet {
    /// Build the fuzzy simplicial set from raw data.
    ///
    /// Steps:
    /// 1. Compute pairwise distances (brute-force or approximate).
    /// 2. For each point, find the k nearest neighbors.
    /// 3. Compute smooth kNN distances (sigma, rho per point).
    /// 4. Construct directed edges with membership strengths.
    /// 5. Symmetrize: P(a→b) ∪ P(b→a) = P(a→b) + P(b→a) - P(a→b)·P(b→a).
    pub fn build(data: &[Vec<f32>], config: &UmapConfig) -> Self {
        let n = data.len();
        let k = config.n_neighbors.min(n - 1);

        // Step 1 & 2: find k-nearest neighbors
        let (neighbor_indices, neighbor_distances) = Self::compute_knn(data, k, config.metric);

        // Step 3: smooth kNN distances
        let (sigmas, rhos) = smooth_knn_dists(&neighbor_distances, k, config.local_connectivity);

        // Step 4: construct directed membership graph
        let mut directed: HashMap<(usize, usize), f32> = HashMap::new();

        for i in 0..n {
            for (j_idx, &j) in neighbor_indices[i].iter().enumerate() {
                let dist = neighbor_distances[i][j_idx];
                let shifted = (dist - rhos[i]).max(0.0);
                let weight = (-shifted / sigmas[i].max(1e-10)).exp();
                directed.insert((i, j), weight);
            }
        }

        // Step 5: symmetrize
        let mut symmetric: HashMap<(usize, usize), f32> = HashMap::new();

        for (&(a, b), &w_ab) in &directed {
            let w_ba = directed.get(&(b, a)).copied().unwrap_or(0.0);
            let combined = w_ab + w_ba - w_ab * w_ba;

            // Store only one direction (a < b) to avoid duplicates, but
            // keep both for sampling purposes.
            let key = (a, b);
            symmetric
                .entry(key)
                .and_modify(|w| *w = w.max(combined))
                .or_insert(combined);
        }

        let edges: Vec<GraphEdge> = symmetric
            .into_iter()
            .filter(|(_, w)| *w > 1e-8)
            .map(|((from, to), weight)| GraphEdge { from, to, weight })
            .collect();

        info!(
            "Fuzzy simplicial set: {} points, {} edges, k={}",
            n,
            edges.len(),
            k
        );

        Self {
            n_points: n,
            edges,
            neighbor_indices,
            neighbor_distances,
        }
    }

    /// Brute-force k-nearest neighbor search.
    fn compute_knn(
        data: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n = data.len();
        let mut indices = vec![Vec::new(); n];
        let mut distances = vec![Vec::new(); n];

        // Pre-compute norms for cosine distance
        let norms: Vec<f32> = if metric == DistanceMetric::Cosine {
            data.iter()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10))
                .collect()
        } else {
            vec![]
        };

        for i in 0..n {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = match metric {
                        DistanceMetric::Euclidean => data[i]
                            .iter()
                            .zip(data[j].iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            .sqrt(),
                        DistanceMetric::Cosine => {
                            let dot: f32 =
                                data[i].iter().zip(data[j].iter()).map(|(a, b)| a * b).sum();
                            1.0 - (dot / (norms[i] * norms[j]))
                        }
                        DistanceMetric::Manhattan => data[i]
                            .iter()
                            .zip(data[j].iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f32>(),
                    };
                    (j, d)
                })
                .collect();

            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);

            indices[i] = dists.iter().map(|(j, _)| *j).collect();
            distances[i] = dists.iter().map(|(_, d)| *d).collect();
        }

        (indices, distances)
    }

    /// Sample edges for training, weighted by membership strength.
    /// Returns `(positive_pairs, epochs_per_sample)`.
    pub fn compute_epoch_edges(&self, n_epochs: usize) -> (Vec<(usize, usize)>, Vec<f32>) {
        let max_weight = self.edges.iter().map(|e| e.weight).fold(0.0f32, f32::max);

        if max_weight <= 0.0 {
            return (vec![], vec![]);
        }

        let mut pairs = Vec::new();
        let mut epochs_per_sample = Vec::new();

        for edge in &self.edges {
            let eps = max_weight / edge.weight.max(1e-10);
            if eps <= n_epochs as f32 {
                pairs.push((edge.from, edge.to));
                epochs_per_sample.push(eps);
            }
        }

        (pairs, epochs_per_sample)
    }
}

// ---------------------------------------------------------------------------
// UMAP Curve Parameters (a, b from min_dist / spread)
// ---------------------------------------------------------------------------

/// Compute the `a` and `b` parameters for the low-dimensional membership
/// function: `1 / (1 + a * d^(2b))`.
///
/// These are found by least-squares fitting to a piecewise-linear target
/// derived from `min_dist` and `spread`.
fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    // Approximate closed-form (good enough for most configurations).
    // The exact solution uses scipy.optimize.curve_fit in the Python impl.
    //
    // For spread=1.0, min_dist=0.1:  a ≈ 1.929, b ≈ 0.7915
    // For spread=1.0, min_dist=0.25: a ≈ 1.597, b ≈ 0.8951
    // For spread=1.0, min_dist=0.5:  a ≈ 1.177, b ≈ 1.0474

    // Use a simple parametric fit that matches reference values well.
    let b = if min_dist < 1e-6 {
        1.0f32
    } else {
        // Approximate: b ≈ 1 / (1 + ln(min_dist / spread + 1))
        // Better approximation derived from reference:
        let ratio = min_dist / spread.max(1e-6);
        let b_raw = 0.8 + 0.3 * ratio;
        b_raw.clamp(0.5, 2.0)
    };

    // a is derived from b and min_dist:
    // at d = min_dist: 1 / (1 + a * min_dist^(2b)) ≈ 1.0
    // at d = spread:   1 / (1 + a * spread^(2b)) ≈ 0.5
    // From the second condition: a = 1 / spread^(2b)
    let a = 1.0 / spread.powf(2.0 * b).max(1e-10);

    (a, b)
}

// ---------------------------------------------------------------------------
// Parametric Encoder (MLP)
// ---------------------------------------------------------------------------

/// Multi-layer perceptron encoder that maps high-dimensional inputs
/// to the low-dimensional UMAP embedding space.
struct ParametricEncoder {
    layers: Vec<Linear>,
    #[allow(dead_code)]
    n_components: usize,
}

impl ParametricEncoder {
    /// Build the encoder from the VarBuilder.
    fn new(config: &UmapConfig, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut in_dim = config.input_dim;

        // Hidden layers
        for (i, &hidden_dim) in config.encoder_hidden_dims.iter().enumerate() {
            let layer = linear(in_dim, hidden_dim, vb.pp(format!("hidden_{}", i)))?;
            layers.push(layer);
            in_dim = hidden_dim;
        }

        // Output projection layer
        let output_layer = linear(in_dim, config.n_components, vb.pp("output"))?;
        layers.push(output_layer);

        Ok(Self {
            layers,
            n_components: config.n_components,
        })
    }

    /// Forward pass through the encoder.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let n_layers = self.layers.len();
        let mut out = x.clone();

        // Hidden layers with ReLU activation
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;
            // Apply ReLU to all but the last layer
            if i < n_layers - 1 {
                out = out.relu()?;
            }
        }

        Ok(out)
    }

    /// Get the number of output dimensions.
    #[allow(dead_code)]
    fn output_dim(&self) -> usize {
        self.n_components
    }
}

// ---------------------------------------------------------------------------
// UMAP Cross-Entropy Loss
// ---------------------------------------------------------------------------

/// Compute the UMAP cross-entropy loss for a batch of positive pairs
/// and negative samples.
///
/// The loss has two terms:
/// - **Attractive**: for positive edges (i, j), minimize `-w_ij * log(q_ij)`
/// - **Repulsive**: for negative samples (i, k), minimize `-(1 - w_ik) * log(1 - q_ik)`
///
/// where `q_ij = 1 / (1 + a * ||y_i - y_j||^(2b))`.
fn umap_loss(
    embeddings_from: &Tensor,
    embeddings_to: &Tensor,
    embeddings_neg: &Tensor,
    a: f64,
    _b: f64,
) -> CandleResult<Tensor> {
    let device = embeddings_from.device();
    let one = Tensor::new(1.0f32, device)?;
    // --- Attractive loss ---
    // Squared distances between positive pairs
    let diff_pos = embeddings_from.sub(embeddings_to)?;
    let dist_sq_pos = diff_pos.sqr()?.sum(1)?; // [batch]

    // q_ij = 1 / (1 + a * d^(2b))
    // For b ≈ 1, d^(2b) ≈ d^2, so q = 1 / (1 + a * dist_sq)
    let a_tensor = Tensor::new(a as f32, device)?;
    let q_pos = one
        .broadcast_add(&a_tensor.broadcast_mul(&dist_sq_pos)?)?
        .recip()?;

    // Clamp for log stability
    let q_pos_clamped = q_pos.clamp(1e-6, 1.0 - 1e-6)?;
    let attractive_loss = q_pos_clamped.log()?.neg()?.mean(0)?;

    // --- Repulsive loss ---
    // For each positive sample, we have negative samples
    let diff_neg = embeddings_from.sub(embeddings_neg)?;
    let dist_sq_neg = diff_neg.sqr()?.sum(1)?;

    let q_neg = one
        .broadcast_add(&a_tensor.broadcast_mul(&dist_sq_neg)?)?
        .recip()?;

    let one_minus_q = one.broadcast_sub(&q_neg)?;
    let one_minus_q_clamped = one_minus_q.clamp(1e-6, 1.0 - 1e-6)?;
    let repulsive_loss = one_minus_q_clamped.log()?.neg()?.mean(0)?;

    // Combined loss
    let total = attractive_loss.add(&repulsive_loss)?;
    Ok(total)
}

// ---------------------------------------------------------------------------
// Training Statistics
// ---------------------------------------------------------------------------

/// Statistics from a UMAP training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapTrainingStats {
    /// Number of training epochs completed.
    pub epochs_completed: usize,

    /// Final training loss.
    pub final_loss: f32,

    /// Loss history (per epoch).
    pub loss_history: Vec<f32>,

    /// Total training wall-clock time.
    pub training_duration: Duration,

    /// Number of edges in the fuzzy simplicial set.
    pub n_edges: usize,

    /// Number of data points fitted.
    pub n_points: usize,

    /// Input dimensionality.
    pub input_dim: usize,

    /// Output dimensionality.
    pub output_dim: usize,

    /// Average time per epoch.
    pub avg_epoch_time: Duration,
}

// ---------------------------------------------------------------------------
// Projected Point
// ---------------------------------------------------------------------------

/// A single projected point in the low-dimensional UMAP space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedPoint {
    /// Low-dimensional coordinates (length = n_components).
    pub coordinates: Vec<f32>,

    /// Optional: original point index in the training set.
    pub source_index: Option<usize>,

    /// Optional: regime label associated with this point.
    pub regime: Option<String>,

    /// Optional: timestamp of the original data.
    pub timestamp: Option<f64>,

    /// Optional: arbitrary metadata tags.
    pub metadata: HashMap<String, String>,
}

impl ProjectedPoint {
    /// Create from raw coordinates.
    pub fn from_coords(coords: Vec<f32>) -> Self {
        Self {
            coordinates: coords,
            source_index: None,
            regime: None,
            timestamp: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder: attach a regime label.
    pub fn with_regime(mut self, regime: impl Into<String>) -> Self {
        self.regime = Some(regime.into());
        self
    }

    /// Builder: attach a timestamp.
    pub fn with_timestamp(mut self, ts: f64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Builder: attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Euclidean distance to another projected point.
    pub fn distance_to(&self, other: &ProjectedPoint) -> f32 {
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }

    /// Number of dimensions.
    pub fn dim(&self) -> usize {
        self.coordinates.len()
    }
}

// ---------------------------------------------------------------------------
// Parametric UMAP (main struct)
// ---------------------------------------------------------------------------

/// State of the Parametric UMAP model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UmapState {
    /// Model has been created but not yet fitted.
    Initialized,
    /// Model has been fitted on training data.
    Fitted,
    /// Model fitting failed.
    Failed,
}

/// Parametric UMAP — trains a neural network encoder to learn the
/// UMAP embedding function, enabling real-time projection of new points.
pub struct ParametricUmap {
    config: UmapConfig,
    var_map: VarMap,
    state: UmapState,
    device: Device,
    training_stats: Option<UmapTrainingStats>,

    // Curve parameters derived from min_dist / spread
    a_param: f32,
    b_param: f32,

    // Fitted graph (kept for reference / retraining)
    graph: Option<FuzzySimplicialSet>,

    // Training data centroid for normalization reference
    centroid: Option<Vec<f32>>,
    scale: Option<f32>,
}

impl ParametricUmap {
    /// Create a new Parametric UMAP model.
    pub fn new(config: UmapConfig) -> Result<Self, UmapError> {
        let device = config
            .candle_device()
            .map_err(|e| UmapError::DeviceError(e.to_string()))?;

        let (a, b) = find_ab_params(config.spread, config.min_dist);
        debug!("UMAP curve params: a={:.4}, b={:.4}", a, b);

        let var_map = VarMap::new();

        Ok(Self {
            config,
            var_map,
            state: UmapState::Initialized,
            device,
            training_stats: None,
            a_param: a,
            b_param: b,
            graph: None,
            centroid: None,
            scale: None,
        })
    }

    /// Get the current model state.
    pub fn state(&self) -> UmapState {
        self.state
    }

    /// Get the config.
    pub fn config(&self) -> &UmapConfig {
        &self.config
    }

    /// Get training statistics (available after `fit`).
    pub fn training_stats(&self) -> Option<&UmapTrainingStats> {
        self.training_stats.as_ref()
    }

    /// Get the fitted fuzzy simplicial set graph.
    pub fn graph(&self) -> Option<&FuzzySimplicialSet> {
        self.graph.as_ref()
    }

    /// Fit the parametric UMAP on a set of high-dimensional vectors.
    ///
    /// # Arguments
    /// * `data` - Slice of high-dimensional vectors. Each vector must have
    ///   length equal to `config.input_dim`.
    ///
    /// # Returns
    /// Training statistics on success.
    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<UmapTrainingStats, UmapError> {
        let n = data.len();
        if n < 2 {
            return Err(UmapError::InsufficientData { got: n, min: 2 });
        }

        // Validate input dimensions
        for (i, vec) in data.iter().enumerate() {
            if vec.len() != self.config.input_dim {
                return Err(UmapError::DimensionMismatch {
                    expected: self.config.input_dim,
                    got: vec.len(),
                    index: i,
                });
            }
        }

        let start = Instant::now();
        info!(
            "Starting Parametric UMAP fit: {} points, {}-d → {}-d",
            n, self.config.input_dim, self.config.n_components
        );

        // Compute centroid and scale for normalization
        self.compute_normalization(data);

        // Build the fuzzy simplicial set (k-NN graph)
        let graph = FuzzySimplicialSet::build(data, &self.config);
        let n_edges = graph.edges.len();

        // Compute epoch-based edge sampling schedule
        let (positive_pairs, epochs_per_sample) = graph.compute_epoch_edges(self.config.n_epochs);

        if positive_pairs.is_empty() {
            return Err(UmapError::NoEdges);
        }

        info!(
            "Graph built: {} edges, {} training pairs",
            n_edges,
            positive_pairs.len()
        );

        // Initialize fresh VarMap and encoder
        self.var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&self.var_map, DType::F32, &self.device);
        let encoder = ParametricEncoder::new(&self.config, vb)
            .map_err(|e| UmapError::ModelError(e.to_string()))?;

        // Pre-load all data into a tensor
        let flat_data: Vec<f32> = data.iter().flatten().copied().collect();
        let data_tensor = Tensor::from_vec(flat_data, (n, self.config.input_dim), &self.device)
            .map_err(|e| UmapError::TensorError(e.to_string()))?;

        // Create optimizer
        let params = candle_nn::ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: self.config.weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut optimizer = candle_nn::AdamW::new(self.var_map.all_vars(), params)
            .map_err(|e| UmapError::OptimizerError(e.to_string()))?;

        // Training loop
        let mut loss_history = Vec::with_capacity(self.config.n_epochs);
        let mut epoch_next: Vec<f32> = vec![0.0; positive_pairs.len()];
        let n_pairs = positive_pairs.len();
        let a = self.a_param as f64;
        let b = self.b_param as f64;

        for epoch in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_batches = 0u32;

            // Collect edges active in this epoch
            let mut active_edges: Vec<usize> = Vec::new();
            for (idx, next) in epoch_next.iter_mut().enumerate() {
                if *next <= epoch as f32 {
                    active_edges.push(idx);
                    *next += epochs_per_sample[idx];
                }
            }

            if active_edges.is_empty() {
                loss_history.push(0.0);
                continue;
            }

            // Process in mini-batches
            for batch_start in (0..active_edges.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(active_edges.len());
                let batch_indices = &active_edges[batch_start..batch_end];
                let batch_size = batch_indices.len();

                // Gather source and target indices
                let from_indices: Vec<u32> = batch_indices
                    .iter()
                    .map(|&idx| positive_pairs[idx].0 as u32)
                    .collect();
                let to_indices: Vec<u32> = batch_indices
                    .iter()
                    .map(|&idx| positive_pairs[idx].1 as u32)
                    .collect();

                // Generate negative samples (random points != source)
                let neg_indices: Vec<u32> = from_indices
                    .iter()
                    .map(|&from| {
                        let mut neg = from;
                        // Simple rejection sampling
                        let mut attempts = 0;
                        while neg == from && attempts < 20 {
                            neg = (pseudo_random(epoch * n_pairs + batch_start + attempts)
                                % n as u64) as u32;
                            attempts += 1;
                        }
                        neg
                    })
                    .collect();

                // Create index tensors
                let from_idx_tensor = Tensor::from_vec(from_indices, (batch_size,), &self.device)
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;
                let to_idx_tensor = Tensor::from_vec(to_indices, (batch_size,), &self.device)
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;
                let neg_idx_tensor = Tensor::from_vec(neg_indices, (batch_size,), &self.device)
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;

                // Gather input vectors
                let x_from = data_tensor
                    .index_select(&from_idx_tensor, 0)
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;
                let x_to = data_tensor
                    .index_select(&to_idx_tensor, 0)
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;
                let x_neg = data_tensor
                    .index_select(&neg_idx_tensor, 0)
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;

                // Forward pass through encoder
                let y_from = encoder
                    .forward(&x_from)
                    .map_err(|e| UmapError::ModelError(e.to_string()))?;
                let y_to = encoder
                    .forward(&x_to)
                    .map_err(|e| UmapError::ModelError(e.to_string()))?;
                let y_neg = encoder
                    .forward(&x_neg)
                    .map_err(|e| UmapError::ModelError(e.to_string()))?;

                // Compute UMAP loss
                let loss = umap_loss(&y_from, &y_to, &y_neg, a, b)
                    .map_err(|e| UmapError::LossError(e.to_string()))?;

                // Backward pass and optimizer step
                optimizer
                    .backward_step(&loss)
                    .map_err(|e| UmapError::OptimizerError(e.to_string()))?;

                let loss_val = loss
                    .to_scalar::<f32>()
                    .map_err(|e| UmapError::TensorError(e.to_string()))?;
                epoch_loss += loss_val;
                n_batches += 1;
            }

            let avg_loss = if n_batches > 0 {
                epoch_loss / n_batches as f32
            } else {
                0.0
            };
            loss_history.push(avg_loss);

            if (epoch + 1) % self.config.log_interval == 0 || epoch == 0 {
                info!(
                    "Epoch {}/{}: loss={:.6}, active_edges={}",
                    epoch + 1,
                    self.config.n_epochs,
                    avg_loss,
                    active_edges.len()
                );
            }
        }

        let duration = start.elapsed();
        let avg_epoch = duration / self.config.n_epochs as u32;

        let stats = UmapTrainingStats {
            epochs_completed: self.config.n_epochs,
            final_loss: *loss_history.last().unwrap_or(&0.0),
            loss_history,
            training_duration: duration,
            n_edges,
            n_points: n,
            input_dim: self.config.input_dim,
            output_dim: self.config.n_components,
            avg_epoch_time: avg_epoch,
        };

        self.graph = Some(graph);
        self.training_stats = Some(stats.clone());
        self.state = UmapState::Fitted;

        info!(
            "UMAP fit complete: final_loss={:.6}, duration={:.1}s",
            stats.final_loss,
            stats.training_duration.as_secs_f64()
        );

        Ok(stats)
    }

    /// Project new high-dimensional data points into the UMAP embedding space.
    ///
    /// The model must have been fitted first via `fit()`.
    ///
    /// # Arguments
    /// * `data` - Slice of high-dimensional vectors to project.
    ///
    /// # Returns
    /// Vector of `ProjectedPoint`s with the low-dimensional coordinates.
    pub fn transform(&self, data: &[Vec<f32>]) -> Result<Vec<ProjectedPoint>, UmapError> {
        if self.state != UmapState::Fitted {
            return Err(UmapError::NotFitted);
        }

        if data.is_empty() {
            return Ok(vec![]);
        }

        // Validate dimensions
        for (i, vec) in data.iter().enumerate() {
            if vec.len() != self.config.input_dim {
                return Err(UmapError::DimensionMismatch {
                    expected: self.config.input_dim,
                    got: vec.len(),
                    index: i,
                });
            }
        }

        let n = data.len();
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let input = Tensor::from_vec(flat, (n, self.config.input_dim), &self.device)
            .map_err(|e| UmapError::TensorError(e.to_string()))?;

        // Rebuild encoder from VarMap (same weights)
        let vb = VarBuilder::from_varmap(&self.var_map, DType::F32, &self.device);
        let encoder = ParametricEncoder::new(&self.config, vb)
            .map_err(|e| UmapError::ModelError(e.to_string()))?;

        let output = encoder
            .forward(&input)
            .map_err(|e| UmapError::ModelError(e.to_string()))?;

        // Extract coordinates
        let output_data: Vec<f32> = output
            .flatten_all()
            .map_err(|e| UmapError::TensorError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| UmapError::TensorError(e.to_string()))?;

        let points: Vec<ProjectedPoint> = output_data
            .chunks(self.config.n_components)
            .enumerate()
            .map(|(i, chunk)| {
                let mut p = ProjectedPoint::from_coords(chunk.to_vec());
                p.source_index = Some(i);
                p
            })
            .collect();

        Ok(points)
    }

    /// Fit and transform in one step: fits on the data, then returns the
    /// embeddings for the same data.
    pub fn fit_transform(
        &mut self,
        data: &[Vec<f32>],
    ) -> Result<(UmapTrainingStats, Vec<ProjectedPoint>), UmapError> {
        let stats = self.fit(data)?;
        let points = self.transform(data)?;
        Ok((stats, points))
    }

    /// Transform a single point (convenience wrapper).
    pub fn transform_one(&self, point: &[f32]) -> Result<ProjectedPoint, UmapError> {
        let data = vec![point.to_vec()];
        let mut results = self.transform(&data)?;
        results
            .pop()
            .ok_or(UmapError::ModelError("Empty transform output".to_string()))
    }

    /// Compute centroid and scale from training data for normalization reference.
    fn compute_normalization(&mut self, data: &[Vec<f32>]) {
        let n = data.len();
        let dim = self.config.input_dim;

        let mut centroid = vec![0.0f32; dim];
        for vec in data {
            for (j, &v) in vec.iter().enumerate() {
                centroid[j] += v;
            }
        }
        for c in &mut centroid {
            *c /= n as f32;
        }

        // Compute average distance from centroid
        let mut total_dist = 0.0f32;
        for vec in data {
            let dist: f32 = vec
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            total_dist += dist;
        }
        let scale = total_dist / n as f32;

        self.centroid = Some(centroid);
        self.scale = Some(scale.max(1e-6));
    }

    /// Compute drift: average distance of new points from the training centroid
    /// in the embedding space, relative to the training scale.
    ///
    /// Returns a drift score where:
    /// - < 1.0: new data is within the training distribution
    /// - > 1.0: new data is drifting away from the training distribution
    /// - > 2.0: significant distribution shift detected
    pub fn compute_drift(&self, new_data: &[Vec<f32>]) -> Result<DriftReport, UmapError> {
        if self.state != UmapState::Fitted {
            return Err(UmapError::NotFitted);
        }

        let centroid = self
            .centroid
            .as_ref()
            .ok_or(UmapError::ModelError("No centroid computed".to_string()))?;
        let scale = self
            .scale
            .ok_or(UmapError::ModelError("No scale computed".to_string()))?;

        // Project new data
        let projected = self.transform(new_data)?;

        // Also project the centroid
        let centroid_proj = self.transform_one(centroid)?;

        // Compute distances from centroid in embedding space
        let distances: Vec<f32> = projected
            .iter()
            .map(|p| p.distance_to(&centroid_proj))
            .collect();

        let mean_dist = distances.iter().sum::<f32>() / distances.len().max(1) as f32;
        let max_dist = distances.iter().cloned().fold(0.0f32, f32::max);

        // Also compute input-space drift
        let mut input_dists = Vec::with_capacity(new_data.len());
        for vec in new_data {
            let d: f32 = vec
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            input_dists.push(d);
        }
        let mean_input_dist = input_dists.iter().sum::<f32>() / input_dists.len().max(1) as f32;
        let drift_score = mean_input_dist / scale;

        Ok(DriftReport {
            drift_score,
            mean_embedding_distance: mean_dist,
            max_embedding_distance: max_dist,
            mean_input_distance: mean_input_dist,
            reference_scale: scale,
            n_points: new_data.len(),
            is_drifting: drift_score > 1.5,
            severity: if drift_score > 3.0 {
                DriftSeverity::Critical
            } else if drift_score > 2.0 {
                DriftSeverity::High
            } else if drift_score > 1.5 {
                DriftSeverity::Moderate
            } else {
                DriftSeverity::Low
            },
        })
    }

    /// Save the encoder weights to a file (JSON-serialized VarMap).
    pub fn save_weights(&self, path: &str) -> Result<(), UmapError> {
        self.var_map
            .save(path)
            .map_err(|e| UmapError::IoError(e.to_string()))
    }

    /// Load encoder weights from a file.
    pub fn load_weights(&mut self, path: &str) -> Result<(), UmapError> {
        self.var_map
            .load(path)
            .map_err(|e| UmapError::IoError(e.to_string()))?;
        self.state = UmapState::Fitted;
        Ok(())
    }
}

/// Simple deterministic pseudo-random for negative sampling
/// (avoids pulling in the full rand crate just for this).
fn pseudo_random(seed: usize) -> u64 {
    let mut x = seed as u64;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

// ---------------------------------------------------------------------------
// Drift Detection
// ---------------------------------------------------------------------------

/// Severity levels for distribution drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Drift score < 1.5. Within expected variation.
    Low,
    /// Drift score 1.5–2.0. Noticeable shift.
    Moderate,
    /// Drift score 2.0–3.0. Significant distribution shift.
    High,
    /// Drift score > 3.0. Major regime change or data quality issue.
    Critical,
}

/// Report from drift detection analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Normalized drift score (mean input distance / training scale).
    pub drift_score: f32,

    /// Mean distance from centroid in the embedding space.
    pub mean_embedding_distance: f32,

    /// Maximum distance from centroid in the embedding space.
    pub max_embedding_distance: f32,

    /// Mean distance from centroid in the input space.
    pub mean_input_distance: f32,

    /// Reference scale from training data.
    pub reference_scale: f32,

    /// Number of points analyzed.
    pub n_points: usize,

    /// Whether the drift exceeds the moderate threshold (score > 1.5).
    pub is_drifting: bool,

    /// Severity classification.
    pub severity: DriftSeverity,
}

// ---------------------------------------------------------------------------
// Qdrant Integration — UMAP Embedding Storage
// ---------------------------------------------------------------------------

/// Configuration for storing UMAP embeddings in Qdrant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapQdrantConfig {
    /// Qdrant collection name for UMAP embeddings.
    pub collection_name: String,

    /// Number of UMAP dimensions stored as the vector.
    pub n_components: usize,

    /// Maximum batch size for upserts.
    pub max_batch_size: usize,

    /// Whether to store the original high-dimensional vector in the payload.
    pub store_original_vector: bool,
}

impl Default for UmapQdrantConfig {
    fn default() -> Self {
        Self {
            collection_name: "umap_regime_embeddings".to_string(),
            n_components: 2,
            max_batch_size: 100,
            store_original_vector: false,
        }
    }
}

impl UmapQdrantConfig {
    /// Configuration for 3-D regime monitoring.
    pub fn regime_3d() -> Self {
        Self {
            collection_name: "umap_regime_3d".to_string(),
            n_components: 3,
            ..Default::default()
        }
    }
}

/// Metadata for a stored UMAP projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapProjectionMeta {
    /// Regime label (e.g., "trending_bullish", "volatile").
    pub regime: Option<String>,

    /// Source identifier (e.g., "finbert", "news_sentiment").
    pub source: Option<String>,

    /// Symbol/asset (e.g., "BTCUSDT").
    pub symbol: Option<String>,

    /// Timestamp as Unix epoch seconds.
    pub timestamp: f64,

    /// Drift score at time of projection (if computed).
    pub drift_score: Option<f32>,

    /// Drift severity classification.
    pub drift_severity: Option<String>,

    /// Confidence score from the upstream model.
    pub confidence: Option<f32>,

    /// Sentiment score from the upstream model.
    pub sentiment_score: Option<f32>,

    /// Arbitrary tags.
    pub tags: HashMap<String, String>,
}

impl UmapProjectionMeta {
    /// Create with just a timestamp.
    pub fn now() -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Self {
            regime: None,
            source: None,
            symbol: None,
            timestamp: ts,
            drift_score: None,
            drift_severity: None,
            confidence: None,
            sentiment_score: None,
            tags: HashMap::new(),
        }
    }

    /// Builder: set regime.
    pub fn with_regime(mut self, regime: impl Into<String>) -> Self {
        self.regime = Some(regime.into());
        self
    }

    /// Builder: set source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Builder: set symbol.
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Builder: set drift info.
    pub fn with_drift(mut self, report: &DriftReport) -> Self {
        self.drift_score = Some(report.drift_score);
        self.drift_severity = Some(format!("{:?}", report.severity));
        self
    }

    /// Builder: set confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Builder: set sentiment score.
    pub fn with_sentiment(mut self, score: f32) -> Self {
        self.sentiment_score = Some(score);
        self
    }
}

/// Bridge connecting Parametric UMAP projections to Qdrant for
/// regime monitoring and similarity search.
///
/// Stores low-dimensional UMAP embeddings as Qdrant vectors with
/// rich metadata payloads for filtering and analysis.
pub struct UmapQdrantBridge {
    client: memory::qdrant_client::QdrantProductionClient,
    config: UmapQdrantConfig,
    stats: std::sync::Mutex<BridgeStatsInner>,
}

#[derive(Debug, Default)]
struct BridgeStatsInner {
    points_stored: u64,
    points_searched: u64,
    errors: u64,
    last_store_time: Option<Instant>,
}

/// Public statistics for the UMAP-Qdrant bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapBridgeStats {
    pub points_stored: u64,
    pub points_searched: u64,
    pub errors: u64,
    pub collection_name: String,
}

impl UmapQdrantBridge {
    /// Create a new bridge with the given Qdrant client and config.
    pub fn new(
        client: memory::qdrant_client::QdrantProductionClient,
        config: UmapQdrantConfig,
    ) -> Self {
        Self {
            client,
            config,
            stats: std::sync::Mutex::new(BridgeStatsInner::default()),
        }
    }

    /// Ensure the Qdrant collection exists with the correct schema.
    pub async fn ensure_collection(&self) -> Result<(), UmapError> {
        use memory::qdrant_client::{CollectionSpec, DistanceMetric};

        let spec = CollectionSpec {
            name: self.config.collection_name.clone(),
            vector_dim: self.config.n_components as u64,
            distance: DistanceMetric::Cosine,
            payload_index_fields: vec![
                "regime".to_string(),
                "timestamp".to_string(),
                "source".to_string(),
                "symbol".to_string(),
            ],
        };

        self.client
            .ensure_collection(&spec)
            .await
            .map_err(|e| UmapError::QdrantError(e.to_string()))
    }

    /// Build a payload HashMap from projection metadata.
    fn build_payload(meta: &UmapProjectionMeta) -> HashMap<String, PayloadValue> {
        let mut payload = HashMap::new();
        payload.insert("timestamp".to_string(), PayloadValue::Float(meta.timestamp));

        if let Some(ref regime) = meta.regime {
            payload.insert("regime".to_string(), PayloadValue::String(regime.clone()));
        }
        if let Some(ref source) = meta.source {
            payload.insert("source".to_string(), PayloadValue::String(source.clone()));
        }
        if let Some(ref symbol) = meta.symbol {
            payload.insert("symbol".to_string(), PayloadValue::String(symbol.clone()));
        }
        if let Some(drift_score) = meta.drift_score {
            payload.insert(
                "drift_score".to_string(),
                PayloadValue::Float(drift_score as f64),
            );
        }
        if let Some(ref severity) = meta.drift_severity {
            payload.insert(
                "drift_severity".to_string(),
                PayloadValue::String(severity.clone()),
            );
        }
        if let Some(confidence) = meta.confidence {
            payload.insert(
                "confidence".to_string(),
                PayloadValue::Float(confidence as f64),
            );
        }
        if let Some(sentiment) = meta.sentiment_score {
            payload.insert(
                "sentiment_score".to_string(),
                PayloadValue::Float(sentiment as f64),
            );
        }

        payload
    }

    /// Store a single UMAP projection in Qdrant.
    pub async fn store_projection(
        &self,
        point: &ProjectedPoint,
        meta: &UmapProjectionMeta,
    ) -> Result<String, UmapError> {
        let id = uuid::Uuid::new_v4().to_string();
        let payload = Self::build_payload(meta);

        // Store coordinates as vector (convert f32 → f64 for Qdrant)
        let upsert = memory::qdrant_client::UpsertPoint {
            id: id.clone(),
            vector: point.coordinates.iter().map(|&v| v as f64).collect(),
            payload,
        };

        self.client
            .upsert(&self.config.collection_name, &[upsert])
            .await
            .map_err(|e| {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.errors += 1;
                }
                UmapError::QdrantError(e.to_string())
            })?;

        if let Ok(mut stats) = self.stats.lock() {
            stats.points_stored += 1;
            stats.last_store_time = Some(Instant::now());
        }

        Ok(id)
    }

    /// Store a batch of UMAP projections.
    pub async fn store_batch(
        &self,
        points: &[ProjectedPoint],
        metas: &[UmapProjectionMeta],
    ) -> Result<Vec<String>, UmapError> {
        if points.len() != metas.len() {
            return Err(UmapError::BatchSizeMismatch {
                points: points.len(),
                metas: metas.len(),
            });
        }

        let mut ids = Vec::with_capacity(points.len());

        for chunk_start in (0..points.len()).step_by(self.config.max_batch_size) {
            let chunk_end = (chunk_start + self.config.max_batch_size).min(points.len());
            let mut upserts = Vec::with_capacity(chunk_end - chunk_start);

            for i in chunk_start..chunk_end {
                let id = uuid::Uuid::new_v4().to_string();
                let payload = Self::build_payload(&metas[i]);

                upserts.push(memory::qdrant_client::UpsertPoint {
                    id: id.clone(),
                    vector: points[i].coordinates.iter().map(|&v| v as f64).collect(),
                    payload,
                });
                ids.push(id);
            }

            self.client
                .upsert(&self.config.collection_name, &upserts)
                .await
                .map_err(|e| {
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.errors += 1;
                    }
                    UmapError::QdrantError(e.to_string())
                })?;

            if let Ok(mut stats) = self.stats.lock() {
                stats.points_stored += upserts.len() as u64;
                stats.last_store_time = Some(Instant::now());
            }
        }

        Ok(ids)
    }

    /// Extract an optional string from a PayloadValue.
    fn payload_string(payload: &HashMap<String, PayloadValue>, key: &str) -> Option<String> {
        match payload.get(key) {
            Some(PayloadValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Extract an optional f64 from a PayloadValue.
    fn payload_f64(payload: &HashMap<String, PayloadValue>, key: &str) -> Option<f64> {
        match payload.get(key) {
            Some(PayloadValue::Float(f)) => Some(*f),
            Some(PayloadValue::Integer(i)) => Some(*i as f64),
            _ => None,
        }
    }

    /// Search for similar UMAP projections (nearest neighbors in embedding space).
    pub async fn search_similar(
        &self,
        query_point: &ProjectedPoint,
        limit: usize,
    ) -> Result<Vec<RetrievedUmapPoint>, UmapError> {
        // Convert f32 coordinates to f64 for the Qdrant search API
        let query_vec: Vec<f64> = query_point.coordinates.iter().map(|&v| v as f64).collect();

        let scored = self
            .client
            .search(&self.config.collection_name, &query_vec, limit as u64, None)
            .await
            .map_err(|e| {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.errors += 1;
                }
                UmapError::QdrantError(e.to_string())
            })?;

        if let Ok(mut stats) = self.stats.lock() {
            stats.points_searched += 1;
        }

        let results: Vec<RetrievedUmapPoint> = scored
            .into_iter()
            .map(|sp| RetrievedUmapPoint {
                id: sp.id,
                coordinates: sp
                    .vector
                    .unwrap_or_default()
                    .iter()
                    .map(|&v| v as f32)
                    .collect(),
                score: sp.score as f32,
                regime: Self::payload_string(&sp.payload, "regime"),
                source: Self::payload_string(&sp.payload, "source"),
                symbol: Self::payload_string(&sp.payload, "symbol"),
                timestamp: Self::payload_f64(&sp.payload, "timestamp"),
                drift_score: Self::payload_f64(&sp.payload, "drift_score").map(|f| f as f32),
                confidence: Self::payload_f64(&sp.payload, "confidence").map(|f| f as f32),
                sentiment_score: Self::payload_f64(&sp.payload, "sentiment_score")
                    .map(|f| f as f32),
            })
            .collect();

        Ok(results)
    }

    /// Search for points by regime label.
    pub async fn search_by_regime(
        &self,
        query_point: &ProjectedPoint,
        regime: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedUmapPoint>, UmapError> {
        // For now, search broadly and filter; in production, use Qdrant filters
        let all_results = self.search_similar(query_point, limit * 5).await?;
        let filtered: Vec<RetrievedUmapPoint> = all_results
            .into_iter()
            .filter(|r| r.regime.as_deref() == Some(regime))
            .take(limit)
            .collect();
        Ok(filtered)
    }

    /// Get bridge statistics.
    pub fn stats(&self) -> UmapBridgeStats {
        let inner = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        UmapBridgeStats {
            points_stored: inner.points_stored,
            points_searched: inner.points_searched,
            errors: inner.errors,
            collection_name: self.config.collection_name.clone(),
        }
    }

    /// Delete all points in the collection.
    pub async fn clear(&self) -> Result<(), UmapError> {
        self.client
            .clear_collection(&self.config.collection_name)
            .await
            .map_err(|e| UmapError::QdrantError(e.to_string()))?;
        self.ensure_collection().await
    }
}

/// A point retrieved from Qdrant with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedUmapPoint {
    /// Qdrant point ID.
    pub id: String,
    /// UMAP coordinates.
    pub coordinates: Vec<f32>,
    /// Similarity score from the search.
    pub score: f32,
    /// Regime label.
    pub regime: Option<String>,
    /// Source identifier.
    pub source: Option<String>,
    /// Symbol.
    pub symbol: Option<String>,
    /// Timestamp.
    pub timestamp: Option<f64>,
    /// Drift score at storage time.
    pub drift_score: Option<f32>,
    /// Confidence from upstream model.
    pub confidence: Option<f32>,
    /// Sentiment score from upstream model.
    pub sentiment_score: Option<f32>,
}

// ---------------------------------------------------------------------------
// Regime Cluster Analysis
// ---------------------------------------------------------------------------

/// Cluster statistics for a group of UMAP projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Regime label for this cluster.
    pub regime: String,
    /// Number of points in the cluster.
    pub n_points: usize,
    /// Centroid of the cluster in UMAP space.
    pub centroid: Vec<f32>,
    /// Average intra-cluster distance.
    pub avg_intra_distance: f32,
    /// Maximum intra-cluster distance.
    pub max_intra_distance: f32,
    /// Standard deviation of distances from centroid.
    pub distance_std: f32,
}

/// Analyze regime clusters from a set of projected points.
///
/// Groups points by their regime label and computes cluster statistics.
pub fn analyze_regime_clusters(points: &[ProjectedPoint]) -> Vec<ClusterStats> {
    // Group by regime
    let mut regime_groups: HashMap<String, Vec<&ProjectedPoint>> = HashMap::new();
    for p in points {
        let regime = p.regime.clone().unwrap_or_else(|| "unknown".to_string());
        regime_groups.entry(regime).or_default().push(p);
    }

    let mut stats = Vec::new();
    for (regime, group) in &regime_groups {
        if group.is_empty() {
            continue;
        }

        let dim = group[0].dim();
        let n = group.len();

        // Compute centroid
        let mut centroid = vec![0.0f32; dim];
        for p in group {
            for (j, &c) in p.coordinates.iter().enumerate() {
                if j < dim {
                    centroid[j] += c;
                }
            }
        }
        for c in &mut centroid {
            *c /= n as f32;
        }

        // Compute distances from centroid
        let distances: Vec<f32> = group
            .iter()
            .map(|p| {
                p.coordinates
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let avg_dist = distances.iter().sum::<f32>() / n as f32;
        let max_dist = distances.iter().cloned().fold(0.0f32, f32::max);

        let variance = distances
            .iter()
            .map(|d| (d - avg_dist) * (d - avg_dist))
            .sum::<f32>()
            / n as f32;
        let std = variance.sqrt();

        stats.push(ClusterStats {
            regime: regime.clone(),
            n_points: n,
            centroid,
            avg_intra_distance: avg_dist,
            max_intra_distance: max_dist,
            distance_std: std,
        });
    }

    stats.sort_by(|a, b| b.n_points.cmp(&a.n_points));
    stats
}

/// Compute inter-cluster distances between regime centroids.
pub fn inter_cluster_distances(clusters: &[ClusterStats]) -> Vec<(String, String, f32)> {
    let mut distances = Vec::new();
    for i in 0..clusters.len() {
        for j in (i + 1)..clusters.len() {
            let dist: f32 = clusters[i]
                .centroid
                .iter()
                .zip(clusters[j].centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            distances.push((clusters[i].regime.clone(), clusters[j].regime.clone(), dist));
        }
    }
    distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    distances
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during Parametric UMAP operations.
#[derive(Debug, thiserror::Error)]
pub enum UmapError {
    #[error("Insufficient data: got {got} points, need at least {min}")]
    InsufficientData { got: usize, min: usize },

    #[error("Dimension mismatch at index {index}: expected {expected}, got {got}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        index: usize,
    },

    #[error("Model not fitted: call fit() before transform()")]
    NotFitted,

    #[error("No edges in fuzzy simplicial set — data may be degenerate")]
    NoEdges,

    #[error("Batch size mismatch: {points} points but {metas} metadata entries")]
    BatchSizeMismatch { points: usize, metas: usize },

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Loss computation error: {0}")]
    LossError(String),

    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    #[error("Qdrant error: {0}")]
    QdrantError(String),

    #[error("I/O error: {0}")]
    IoError(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate random test data of given dimensions.
    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut vec = Vec::with_capacity(dim);
            for j in 0..dim {
                let val = ((pseudo_random((seed as usize) * n * dim + i * dim + j) % 10000) as f32)
                    / 10000.0
                    - 0.5;
                vec.push(val);
            }
            data.push(vec);
        }
        data
    }

    /// Generate clustered test data (2 clusters in high-d space).
    fn clustered_data(n_per_cluster: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<String>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();

        // Cluster 0: centered at +1
        for i in 0..n_per_cluster {
            let mut vec = Vec::with_capacity(dim);
            for j in 0..dim {
                let noise = ((pseudo_random(i * dim + j) % 1000) as f32) / 5000.0;
                vec.push(1.0 + noise);
            }
            data.push(vec);
            labels.push("trending".to_string());
        }

        // Cluster 1: centered at -1
        for i in 0..n_per_cluster {
            let mut vec = Vec::with_capacity(dim);
            for j in 0..dim {
                let noise =
                    ((pseudo_random((n_per_cluster + i) * dim + j + 999) % 1000) as f32) / 5000.0;
                vec.push(-1.0 + noise);
            }
            data.push(vec);
            labels.push("volatile".to_string());
        }

        (data, labels)
    }

    #[test]
    fn test_config_defaults() {
        let config = UmapConfig::default();
        assert_eq!(config.n_neighbors, 15);
        assert_eq!(config.n_components, 2);
        assert_eq!(config.input_dim, 768);
        assert_eq!(config.min_dist, 0.1);
        assert_eq!(config.spread, 1.0);
    }

    #[test]
    fn test_config_presets() {
        let bert_base = UmapConfig::bert_base();
        assert_eq!(bert_base.input_dim, 768);
        assert_eq!(bert_base.n_components, 2);

        let bert_tiny = UmapConfig::bert_tiny();
        assert_eq!(bert_tiny.input_dim, 256);

        let regime_3d = UmapConfig::regime_3d();
        assert_eq!(regime_3d.n_components, 3);
    }

    #[test]
    fn test_config_builder() {
        let config = UmapConfig::default()
            .with_input_dim(128)
            .with_n_components(3)
            .with_n_neighbors(20)
            .with_n_epochs(100)
            .with_learning_rate(5e-4)
            .with_encoder_dims(vec![64, 32])
            .with_metric(DistanceMetric::Euclidean);

        assert_eq!(config.input_dim, 128);
        assert_eq!(config.n_components, 3);
        assert_eq!(config.n_neighbors, 20);
        assert_eq!(config.n_epochs, 100);
        assert_eq!(config.learning_rate, 5e-4);
        assert_eq!(config.encoder_hidden_dims, vec![64, 32]);
        assert_eq!(config.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_find_ab_params() {
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!(a > 0.0, "a should be positive, got {}", a);
        assert!(b > 0.0, "b should be positive, got {}", b);

        // Different min_dist should give different params
        let (a2, b2) = find_ab_params(1.0, 0.5);
        assert!((a - a2).abs() > 1e-6 || (b - b2).abs() > 1e-6);
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let data = vec![a.clone(), b.clone()];

        // Euclidean distance between (1,0,0) and (0,1,0) = sqrt(2)
        let (indices, dists) = FuzzySimplicialSet::compute_knn(&data, 1, DistanceMetric::Euclidean);
        assert_eq!(indices[0], vec![1]);
        assert!((dists[0][0] - std::f32::consts::SQRT_2).abs() < 0.01);

        // Cosine distance between orthogonal vectors = 1.0
        let (_, cos_dists) = FuzzySimplicialSet::compute_knn(&data, 1, DistanceMetric::Cosine);
        assert!((cos_dists[0][0] - 1.0).abs() < 0.01);

        // Manhattan distance = 2.0
        let (_, man_dists) = FuzzySimplicialSet::compute_knn(&data, 1, DistanceMetric::Manhattan);
        assert!((man_dists[0][0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_smooth_knn_dists() {
        let distances = vec![
            vec![0.1, 0.2, 0.5],
            vec![0.15, 0.3, 0.6],
            vec![0.2, 0.4, 0.8],
        ];

        let (sigmas, rhos) = smooth_knn_dists(&distances, 3, 1);

        assert_eq!(sigmas.len(), 3);
        assert_eq!(rhos.len(), 3);

        // Sigmas should be positive
        for sigma in &sigmas {
            assert!(*sigma > 0.0, "sigma should be positive, got {}", sigma);
        }
        // Rhos should be >= 0
        for rho in &rhos {
            assert!(*rho >= 0.0, "rho should be non-negative, got {}", rho);
        }
    }

    #[test]
    fn test_fuzzy_simplicial_set_build() {
        let data = random_data(20, 8, 42);
        let config = UmapConfig::default().with_input_dim(8).with_n_neighbors(5);

        let graph = FuzzySimplicialSet::build(&data, &config);

        assert_eq!(graph.n_points, 20);
        assert!(!graph.edges.is_empty(), "Should have edges");
        assert_eq!(graph.neighbor_indices.len(), 20);

        // Each point should have exactly 5 neighbors
        for neighbors in &graph.neighbor_indices {
            assert_eq!(neighbors.len(), 5);
        }

        // Edge weights should be in [0, 1]
        for edge in &graph.edges {
            assert!(edge.weight >= 0.0 && edge.weight <= 1.0001);
        }
    }

    #[test]
    fn test_epoch_edges() {
        let data = random_data(20, 8, 42);
        let config = UmapConfig::default().with_input_dim(8).with_n_neighbors(5);

        let graph = FuzzySimplicialSet::build(&data, &config);
        let (pairs, eps) = graph.compute_epoch_edges(100);

        assert!(!pairs.is_empty());
        assert_eq!(pairs.len(), eps.len());

        // All epochs_per_sample should be positive
        for e in &eps {
            assert!(*e > 0.0);
        }
    }

    #[test]
    fn test_umap_creation() {
        let config = UmapConfig::default().with_input_dim(16);
        let umap = ParametricUmap::new(config);
        assert!(umap.is_ok());

        let umap = umap.unwrap();
        assert_eq!(umap.state(), UmapState::Initialized);
        assert!(umap.training_stats().is_none());
        assert!(umap.graph().is_none());
    }

    #[test]
    fn test_umap_fit_insufficient_data() {
        let config = UmapConfig::default().with_input_dim(8);
        let mut umap = ParametricUmap::new(config).unwrap();

        let data = vec![vec![0.0f32; 8]]; // Only 1 point
        let result = umap.fit(&data);
        assert!(result.is_err());

        match result.unwrap_err() {
            UmapError::InsufficientData { got, min } => {
                assert_eq!(got, 1);
                assert_eq!(min, 2);
            }
            other => panic!("Expected InsufficientData, got: {:?}", other),
        }
    }

    #[test]
    fn test_umap_fit_dimension_mismatch() {
        let config = UmapConfig::default().with_input_dim(8);
        let mut umap = ParametricUmap::new(config).unwrap();

        let data = vec![vec![0.0f32; 8], vec![0.0f32; 10]]; // Wrong dim
        let result = umap.fit(&data);
        assert!(result.is_err());

        match result.unwrap_err() {
            UmapError::DimensionMismatch {
                expected,
                got,
                index,
            } => {
                assert_eq!(expected, 8);
                assert_eq!(got, 10);
                assert_eq!(index, 1);
            }
            other => panic!("Expected DimensionMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_umap_fit_and_transform() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(5) // Very few epochs for test speed
            .with_encoder_dims(vec![16, 8])
            .with_learning_rate(1e-3);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(20, 8, 123);

        // Fit
        let stats = umap.fit(&data).unwrap();
        assert_eq!(stats.epochs_completed, 5);
        assert_eq!(stats.n_points, 20);
        assert_eq!(stats.input_dim, 8);
        assert_eq!(stats.output_dim, 2);
        assert_eq!(stats.loss_history.len(), 5);
        assert_eq!(umap.state(), UmapState::Fitted);

        // Transform the same data
        let projected = umap.transform(&data).unwrap();
        assert_eq!(projected.len(), 20);
        for p in &projected {
            assert_eq!(p.dim(), 2);
            assert!(p.source_index.is_some());
        }
    }

    #[test]
    fn test_umap_transform_not_fitted() {
        let config = UmapConfig::default().with_input_dim(8);
        let umap = ParametricUmap::new(config).unwrap();

        let data = random_data(5, 8, 42);
        let result = umap.transform(&data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), UmapError::NotFitted));
    }

    #[test]
    fn test_umap_fit_transform() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16]);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(15, 8, 99);

        let (stats, points) = umap.fit_transform(&data).unwrap();
        assert_eq!(stats.epochs_completed, 3);
        assert_eq!(points.len(), 15);
    }

    #[test]
    fn test_umap_transform_one() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16]);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(15, 8, 42);
        umap.fit(&data).unwrap();

        let single_point = vec![0.1f32; 8];
        let projected = umap.transform_one(&single_point).unwrap();
        assert_eq!(projected.dim(), 2);
    }

    #[test]
    fn test_projected_point() {
        let p = ProjectedPoint::from_coords(vec![1.0, 2.0])
            .with_regime("trending")
            .with_timestamp(1234567890.0)
            .with_metadata("source", "finbert");

        assert_eq!(p.dim(), 2);
        assert_eq!(p.regime.as_deref(), Some("trending"));
        assert_eq!(p.timestamp, Some(1234567890.0));
        assert_eq!(p.metadata.get("source").unwrap(), "finbert");
    }

    #[test]
    fn test_projected_point_distance() {
        let p1 = ProjectedPoint::from_coords(vec![0.0, 0.0]);
        let p2 = ProjectedPoint::from_coords(vec![3.0, 4.0]);

        let dist = p1.distance_to(&p2);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_drift_detection() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16]);

        let mut umap = ParametricUmap::new(config).unwrap();

        // Train on data centered near 0
        let train_data = random_data(20, 8, 42);
        umap.fit(&train_data).unwrap();

        // Test with similar data (should have low drift)
        let similar_data = random_data(5, 8, 99);
        let report = umap.compute_drift(&similar_data).unwrap();
        assert!(report.drift_score >= 0.0);
        assert_eq!(report.n_points, 5);

        // Test with very different data (should have higher drift)
        let far_data: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                (0..8)
                    .map(|j| 100.0 + (pseudo_random(i * 8 + j) % 100) as f32)
                    .collect()
            })
            .collect();
        let far_report = umap.compute_drift(&far_data).unwrap();
        assert!(
            far_report.drift_score > report.drift_score,
            "Far data should have higher drift: {} vs {}",
            far_report.drift_score,
            report.drift_score
        );
    }

    #[test]
    fn test_drift_severity() {
        assert_eq!(DriftSeverity::Low, DriftSeverity::Low);
        assert_ne!(DriftSeverity::Low, DriftSeverity::High);
    }

    #[test]
    fn test_regime_cluster_analysis() {
        let (data, labels) = clustered_data(10, 8);

        // Create projected points with regime labels
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(5)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16]);

        let mut umap = ParametricUmap::new(config).unwrap();
        let (_, mut points) = umap.fit_transform(&data).unwrap();

        // Attach regime labels
        for (i, p) in points.iter_mut().enumerate() {
            p.regime = Some(labels[i].clone());
        }

        let clusters = analyze_regime_clusters(&points);
        assert_eq!(clusters.len(), 2); // "trending" and "volatile"

        for cluster in &clusters {
            assert_eq!(cluster.n_points, 10);
            assert!(!cluster.centroid.is_empty());
            assert!(cluster.avg_intra_distance >= 0.0);
        }
    }

    #[test]
    fn test_inter_cluster_distances() {
        let clusters = vec![
            ClusterStats {
                regime: "trending".to_string(),
                n_points: 10,
                centroid: vec![1.0, 0.0],
                avg_intra_distance: 0.5,
                max_intra_distance: 1.0,
                distance_std: 0.2,
            },
            ClusterStats {
                regime: "volatile".to_string(),
                n_points: 10,
                centroid: vec![-1.0, 0.0],
                avg_intra_distance: 0.6,
                max_intra_distance: 1.2,
                distance_std: 0.3,
            },
        ];

        let distances = inter_cluster_distances(&clusters);
        assert_eq!(distances.len(), 1);
        assert!((distances[0].2 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_umap_qdrant_config_default() {
        let config = UmapQdrantConfig::default();
        assert_eq!(config.collection_name, "umap_regime_embeddings");
        assert_eq!(config.n_components, 2);
        assert_eq!(config.max_batch_size, 100);
        assert!(!config.store_original_vector);
    }

    #[test]
    fn test_umap_projection_meta() {
        let meta = UmapProjectionMeta::now()
            .with_regime("trending_bullish")
            .with_source("finbert")
            .with_symbol("BTCUSDT")
            .with_confidence(0.92);

        assert_eq!(meta.regime.as_deref(), Some("trending_bullish"));
        assert_eq!(meta.source.as_deref(), Some("finbert"));
        assert_eq!(meta.symbol.as_deref(), Some("BTCUSDT"));
        assert_eq!(meta.confidence, Some(0.92));
        assert!(meta.timestamp > 0.0);
    }

    #[test]
    fn test_umap_projection_meta_with_drift() {
        let report = DriftReport {
            drift_score: 2.5,
            mean_embedding_distance: 0.8,
            max_embedding_distance: 1.5,
            mean_input_distance: 3.0,
            reference_scale: 1.2,
            n_points: 10,
            is_drifting: true,
            severity: DriftSeverity::High,
        };

        let meta = UmapProjectionMeta::now().with_drift(&report);
        assert_eq!(meta.drift_score, Some(2.5));
        assert_eq!(meta.drift_severity.as_deref(), Some("High"));
    }

    #[test]
    fn test_pseudo_random_deterministic() {
        let a = pseudo_random(42);
        let b = pseudo_random(42);
        assert_eq!(a, b);

        let c = pseudo_random(43);
        assert_ne!(a, c);
    }

    #[test]
    fn test_empty_cluster_analysis() {
        let points: Vec<ProjectedPoint> = vec![];
        let clusters = analyze_regime_clusters(&points);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_single_point_cluster() {
        let points = vec![ProjectedPoint::from_coords(vec![1.0, 2.0]).with_regime("test")];

        let clusters = analyze_regime_clusters(&points);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].n_points, 1);
        assert_eq!(clusters[0].avg_intra_distance, 0.0);
    }

    #[test]
    fn test_umap_bridge_stats_default() {
        // Test that the bridge stats struct works correctly
        let stats = UmapBridgeStats {
            points_stored: 0,
            points_searched: 0,
            errors: 0,
            collection_name: "test".to_string(),
        };
        assert_eq!(stats.points_stored, 0);
    }

    #[test]
    fn test_retrieved_umap_point() {
        let point = RetrievedUmapPoint {
            id: "test-id".to_string(),
            coordinates: vec![1.0, 2.0],
            score: 0.95,
            regime: Some("trending".to_string()),
            source: Some("finbert".to_string()),
            symbol: Some("BTCUSDT".to_string()),
            timestamp: Some(1234567890.0),
            drift_score: Some(0.5),
            confidence: Some(0.92),
            sentiment_score: Some(0.75),
        };

        assert_eq!(point.id, "test-id");
        assert_eq!(point.coordinates.len(), 2);
        assert_eq!(point.score, 0.95);
    }

    // 3-D output test
    #[test]
    fn test_umap_3d_output() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(3)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16, 8]);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(15, 8, 77);

        let (_, points) = umap.fit_transform(&data).unwrap();
        for p in &points {
            assert_eq!(p.dim(), 3);
        }
    }

    // Test with cosine metric
    #[test]
    fn test_umap_cosine_metric() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16])
            .with_metric(DistanceMetric::Cosine);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(15, 8, 55);

        let result = umap.fit(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_umap_manhattan_metric() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16])
            .with_metric(DistanceMetric::Manhattan);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(15, 8, 66);

        let result = umap.fit(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transform_empty_data() {
        let config = UmapConfig::default()
            .with_input_dim(8)
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_n_epochs(3)
            .with_encoder_dims(vec![16]);

        let mut umap = ParametricUmap::new(config).unwrap();
        let data = random_data(15, 8, 42);
        umap.fit(&data).unwrap();

        let empty: Vec<Vec<f32>> = vec![];
        let result = umap.transform(&empty).unwrap();
        assert!(result.is_empty());
    }
}
