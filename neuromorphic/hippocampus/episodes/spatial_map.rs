//! Market state space representation
//!
//! Part of the Hippocampus region
//! Component: episodes
//!
//! This module implements a "spatial map" of market states, inspired by the hippocampus's
//! role in spatial navigation and place cells. It maps market conditions to a multi-dimensional
//! state space, enabling:
//! - Pattern recognition across similar market states
//! - Experience retrieval based on state similarity
//! - Navigation through market "terrain" for decision making
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Market State Space                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                               │
//! │  Dimensions:                                                 │
//! │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
//! │  │ Volatility │ │   Trend    │ │  Volume    │              │
//! │  │  (ATR/σ)   │ │ (Momentum) │ │  (Ratio)   │              │
//! │  └────────────┘ └────────────┘ └────────────┘              │
//! │                                                               │
//! │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
//! │  │  Spread    │ │   Depth    │ │ Correlation│              │
//! │  │ (Liquidity)│ │(Order Book)│ │  (Cross)   │              │
//! │  └────────────┘ └────────────┘ └────────────┘              │
//! │                                                               │
//! │  ┌─────────────────────────────────────────┐                │
//! │  │         Place Cells (State Clusters)     │                │
//! │  │    • High-vol trending                   │                │
//! │  │    • Low-vol ranging                     │                │
//! │  │    • Crisis conditions                   │                │
//! │  │    • ...                                 │                │
//! │  └─────────────────────────────────────────┘                │
//! │                                                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Configuration for the spatial map
#[derive(Debug, Clone)]
pub struct SpatialMapConfig {
    /// Number of dimensions in the state space
    pub dimensions: usize,
    /// Maximum number of place cells (state clusters)
    pub max_place_cells: usize,
    /// Similarity threshold for matching states (0.0 - 1.0)
    pub similarity_threshold: f64,
    /// Decay rate for place cell activation
    pub activation_decay: f64,
    /// Minimum samples to form a place cell
    pub min_samples_for_cell: usize,
    /// History length for trajectory tracking
    pub trajectory_length: usize,
    /// Grid resolution for discretization (cells per dimension)
    pub grid_resolution: usize,
}

impl Default for SpatialMapConfig {
    fn default() -> Self {
        Self {
            dimensions: 8,
            max_place_cells: 100,
            similarity_threshold: 0.85,
            activation_decay: 0.95,
            min_samples_for_cell: 10,
            trajectory_length: 50,
            grid_resolution: 10,
        }
    }
}

/// A point in the market state space
#[derive(Debug, Clone)]
pub struct StatePoint {
    /// Unique identifier
    pub id: u64,
    /// Coordinates in state space (normalized 0.0 - 1.0)
    pub coordinates: Vec<f64>,
    /// Timestamp when this state was observed
    pub timestamp: i64,
    /// Symbol associated with this state
    pub symbol: String,
    /// Raw feature values (before normalization)
    pub raw_features: HashMap<String, f64>,
    /// Associated outcome (e.g., subsequent return)
    pub outcome: Option<f64>,
}

impl StatePoint {
    /// Create a new state point
    pub fn new(id: u64, coordinates: Vec<f64>, timestamp: i64, symbol: String) -> Self {
        Self {
            id,
            coordinates,
            timestamp,
            symbol,
            raw_features: HashMap::new(),
            outcome: None,
        }
    }

    /// Calculate Euclidean distance to another point
    pub fn distance(&self, other: &StatePoint) -> f64 {
        if self.coordinates.len() != other.coordinates.len() {
            return f64::MAX;
        }

        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate cosine similarity to another point
    pub fn cosine_similarity(&self, other: &StatePoint) -> f64 {
        if self.coordinates.len() != other.coordinates.len() {
            return 0.0;
        }

        let dot_product: f64 = self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mag_a: f64 = self
            .coordinates
            .iter()
            .map(|x| x.powi(2))
            .sum::<f64>()
            .sqrt();
        let mag_b: f64 = other
            .coordinates
            .iter()
            .map(|x| x.powi(2))
            .sum::<f64>()
            .sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot_product / (mag_a * mag_b)
    }
}

/// A "place cell" - a cluster of similar market states
#[derive(Debug, Clone)]
pub struct PlaceCell {
    /// Unique identifier
    pub id: u64,
    /// Centroid of the cell in state space
    pub centroid: Vec<f64>,
    /// Number of observations in this cell
    pub observation_count: usize,
    /// Running sum of coordinates (for centroid updates)
    coordinate_sum: Vec<f64>,
    /// Average outcome when in this state
    pub avg_outcome: f64,
    /// Outcome variance for this state
    pub outcome_variance: f64,
    /// Running sum of outcomes
    outcome_sum: f64,
    /// Running sum of squared outcomes
    outcome_sq_sum: f64,
    /// Current activation level (0.0 - 1.0)
    pub activation: f64,
    /// Last activation timestamp
    pub last_activated: i64,
    /// Label/description for this cell
    pub label: String,
    /// Associated episode IDs
    pub episode_ids: Vec<u64>,
}

impl PlaceCell {
    /// Create a new place cell from an initial point
    pub fn new(id: u64, initial_point: &StatePoint) -> Self {
        Self {
            id,
            centroid: initial_point.coordinates.clone(),
            observation_count: 1,
            coordinate_sum: initial_point.coordinates.clone(),
            avg_outcome: initial_point.outcome.unwrap_or(0.0),
            outcome_variance: 0.0,
            outcome_sum: initial_point.outcome.unwrap_or(0.0),
            outcome_sq_sum: initial_point.outcome.unwrap_or(0.0).powi(2),
            activation: 1.0,
            last_activated: initial_point.timestamp,
            label: String::new(),
            episode_ids: vec![initial_point.id],
        }
    }

    /// Update the cell with a new observation
    pub fn update(&mut self, point: &StatePoint) {
        // Update coordinate sum and centroid
        for (i, coord) in point.coordinates.iter().enumerate() {
            if i < self.coordinate_sum.len() {
                self.coordinate_sum[i] += coord;
            }
        }

        self.observation_count += 1;

        // Update centroid as running average
        for (i, sum) in self.coordinate_sum.iter().enumerate() {
            if i < self.centroid.len() {
                self.centroid[i] = sum / self.observation_count as f64;
            }
        }

        // Update outcome statistics
        if let Some(outcome) = point.outcome {
            self.outcome_sum += outcome;
            self.outcome_sq_sum += outcome.powi(2);
            self.avg_outcome = self.outcome_sum / self.observation_count as f64;

            if self.observation_count > 1 {
                let variance = (self.outcome_sq_sum / self.observation_count as f64)
                    - self.avg_outcome.powi(2);
                self.outcome_variance = variance.max(0.0);
            }
        }

        // Update activation
        self.activation = 1.0;
        self.last_activated = point.timestamp;

        // Track episode
        if !self.episode_ids.contains(&point.id) {
            self.episode_ids.push(point.id);
            // Keep bounded
            if self.episode_ids.len() > 1000 {
                self.episode_ids.remove(0);
            }
        }
    }

    /// Decay the activation over time
    pub fn decay_activation(&mut self, decay_rate: f64) {
        self.activation *= decay_rate;
    }

    /// Calculate distance from a point to this cell's centroid
    pub fn distance_to(&self, point: &StatePoint) -> f64 {
        if self.centroid.len() != point.coordinates.len() {
            return f64::MAX;
        }

        self.centroid
            .iter()
            .zip(point.coordinates.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Check if a point falls within this cell
    pub fn contains(&self, point: &StatePoint, threshold: f64) -> bool {
        self.distance_to(point) < threshold
    }
}

/// Dimension/feature definition for the state space
#[derive(Debug, Clone)]
pub struct StateDimension {
    /// Name of the dimension
    pub name: String,
    /// Minimum value (for normalization)
    pub min_value: f64,
    /// Maximum value (for normalization)
    pub max_value: f64,
    /// Current normalized value
    pub current_value: f64,
    /// Weight for similarity calculations
    pub weight: f64,
}

impl StateDimension {
    /// Create a new dimension
    pub fn new(name: &str, min_value: f64, max_value: f64) -> Self {
        Self {
            name: name.to_string(),
            min_value,
            max_value,
            current_value: 0.5,
            weight: 1.0,
        }
    }

    /// Normalize a raw value to 0.0 - 1.0
    pub fn normalize(&self, value: f64) -> f64 {
        if (self.max_value - self.min_value).abs() < f64::EPSILON {
            return 0.5;
        }
        ((value - self.min_value) / (self.max_value - self.min_value)).clamp(0.0, 1.0)
    }

    /// Denormalize a value from 0.0 - 1.0 to raw scale
    pub fn denormalize(&self, normalized: f64) -> f64 {
        self.min_value + normalized * (self.max_value - self.min_value)
    }
}

/// Trajectory through state space
#[derive(Debug, Clone)]
pub struct StateTrajectory {
    /// Sequence of state points
    pub points: VecDeque<StatePoint>,
    /// Maximum trajectory length
    max_length: usize,
}

impl StateTrajectory {
    /// Create a new trajectory
    pub fn new(max_length: usize) -> Self {
        Self {
            points: VecDeque::new(),
            max_length,
        }
    }

    /// Add a point to the trajectory
    pub fn add(&mut self, point: StatePoint) {
        self.points.push_back(point);
        while self.points.len() > self.max_length {
            self.points.pop_front();
        }
    }

    /// Calculate the total path length
    pub fn path_length(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 1..self.points.len() {
            total += self.points[i - 1].distance(&self.points[i]);
        }
        total
    }

    /// Calculate the net displacement (start to end)
    pub fn displacement(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }

        self.points
            .front()
            .unwrap()
            .distance(self.points.back().unwrap())
    }

    /// Calculate trajectory efficiency (displacement / path_length)
    pub fn efficiency(&self) -> f64 {
        let path = self.path_length();
        if path == 0.0 {
            return 0.0;
        }
        self.displacement() / path
    }

    /// Get the direction vector (last - first, normalized)
    pub fn direction(&self) -> Option<Vec<f64>> {
        if self.points.len() < 2 {
            return None;
        }

        let first = self.points.front().unwrap();
        let last = self.points.back().unwrap();

        let direction: Vec<f64> = first
            .coordinates
            .iter()
            .zip(last.coordinates.iter())
            .map(|(a, b)| b - a)
            .collect();

        let magnitude: f64 = direction.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if magnitude == 0.0 {
            return None;
        }

        Some(direction.iter().map(|x| x / magnitude).collect())
    }

    /// Get recent velocity (rate of change in state space)
    pub fn velocity(&self, lookback: usize) -> Option<Vec<f64>> {
        if self.points.len() < 2 {
            return None;
        }

        let lookback = lookback.min(self.points.len() - 1);
        let start_idx = self.points.len() - 1 - lookback;
        let start = &self.points[start_idx];
        let end = self.points.back().unwrap();

        let time_diff = (end.timestamp - start.timestamp) as f64;
        if time_diff == 0.0 {
            return None;
        }

        Some(
            start
                .coordinates
                .iter()
                .zip(end.coordinates.iter())
                .map(|(a, b)| (b - a) / time_diff * 1000.0) // per second
                .collect(),
        )
    }
}

/// Market state space representation (the "spatial map")
pub struct SpatialMap {
    /// Configuration
    config: SpatialMapConfig,
    /// Defined dimensions of state space
    dimensions: Vec<StateDimension>,
    /// Place cells (state clusters)
    place_cells: Vec<PlaceCell>,
    /// Current trajectory through state space
    trajectory: StateTrajectory,
    /// Symbol being tracked
    symbol: String,
    /// Next point ID
    next_point_id: u64,
    /// Next cell ID
    next_cell_id: u64,
    /// Grid cell counts for visualization
    grid_counts: HashMap<Vec<usize>, usize>,
}

impl Default for SpatialMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialMap {
    /// Create a new spatial map with default configuration
    pub fn new() -> Self {
        Self::with_config(SpatialMapConfig::default())
    }

    /// Create a new spatial map with custom configuration
    pub fn with_config(config: SpatialMapConfig) -> Self {
        // Define default market state dimensions
        let dimensions = vec![
            StateDimension::new("volatility", 0.0, 0.1), // ATR as % of price
            StateDimension::new("trend", -1.0, 1.0),     // Momentum indicator
            StateDimension::new("volume_ratio", 0.0, 5.0), // Relative to average
            StateDimension::new("spread", 0.0, 0.01),    // Bid-ask spread %
            StateDimension::new("depth_imbalance", -1.0, 1.0), // Order book imbalance
            StateDimension::new("correlation", -1.0, 1.0), // Cross-asset correlation
            StateDimension::new("regime", 0.0, 1.0),     // Regime indicator
            StateDimension::new("sentiment", -1.0, 1.0), // Market sentiment
        ];

        let trajectory_length = config.trajectory_length;

        Self {
            config,
            dimensions,
            place_cells: Vec::new(),
            trajectory: StateTrajectory::new(trajectory_length),
            symbol: String::new(),
            next_point_id: 1,
            next_cell_id: 1,
            grid_counts: HashMap::new(),
        }
    }

    /// Set the symbol being tracked
    pub fn set_symbol(&mut self, symbol: &str) {
        self.symbol = symbol.to_string();
    }

    /// Add a custom dimension
    pub fn add_dimension(&mut self, name: &str, min_value: f64, max_value: f64, weight: f64) {
        let mut dim = StateDimension::new(name, min_value, max_value);
        dim.weight = weight;
        self.dimensions.push(dim);
    }

    /// Record a new state observation
    pub fn record_state(&mut self, features: HashMap<String, f64>, timestamp: i64) -> StatePoint {
        // Normalize features into coordinates
        let coordinates: Vec<f64> = self
            .dimensions
            .iter()
            .map(|dim| {
                let value = features.get(&dim.name).copied().unwrap_or(0.0);
                dim.normalize(value)
            })
            .collect();

        let mut point = StatePoint::new(
            self.next_point_id,
            coordinates,
            timestamp,
            self.symbol.clone(),
        );
        point.raw_features = features;
        self.next_point_id += 1;

        // Add to trajectory
        self.trajectory.add(point.clone());

        // Update grid counts
        self.update_grid_counts(&point);

        // Update or create place cells
        self.update_place_cells(&point);

        // Decay all place cells
        for cell in &mut self.place_cells {
            cell.decay_activation(self.config.activation_decay);
        }

        point
    }

    /// Record state with outcome (for learning)
    pub fn record_state_with_outcome(
        &mut self,
        features: HashMap<String, f64>,
        timestamp: i64,
        outcome: f64,
    ) -> StatePoint {
        let mut point = self.record_state(features, timestamp);
        point.outcome = Some(outcome);

        // Update the place cell that contains this point with the outcome
        // Get the cell ID first to avoid borrow issues
        let cell_id = self.find_nearest_cell(&point).map(|c| c.id);
        if let Some(id) = cell_id {
            if let Some(c) = self.place_cells.iter_mut().find(|c| c.id == id) {
                c.update(&point);
            }
        }

        point
    }

    /// Update grid counts for visualization
    fn update_grid_counts(&mut self, point: &StatePoint) {
        let grid_cell: Vec<usize> = point
            .coordinates
            .iter()
            .map(|&c| {
                let idx = (c * self.config.grid_resolution as f64) as usize;
                idx.min(self.config.grid_resolution - 1)
            })
            .collect();

        *self.grid_counts.entry(grid_cell).or_insert(0) += 1;
    }

    /// Update place cells with new observation
    fn update_place_cells(&mut self, point: &StatePoint) {
        // Find the nearest place cell and get its ID and distance
        // to avoid borrow checker issues
        let nearest_info = self
            .find_nearest_cell(point)
            .map(|cell| (cell.id, cell.distance_to(point)));

        let threshold =
            (1.0 - self.config.similarity_threshold) * (self.dimensions.len() as f64).sqrt();

        if let Some((cell_id, distance)) = nearest_info {
            if distance < threshold {
                // Update existing cell
                if let Some(c) = self.place_cells.iter_mut().find(|c| c.id == cell_id) {
                    c.update(point);
                }
            } else if self.place_cells.len() < self.config.max_place_cells {
                // Create new cell
                self.create_place_cell(point);
            }
        } else {
            // No cells exist yet, create the first one
            self.create_place_cell(point);
        }
    }

    /// Create a new place cell
    fn create_place_cell(&mut self, point: &StatePoint) {
        let cell = PlaceCell::new(self.next_cell_id, point);
        self.next_cell_id += 1;
        self.place_cells.push(cell);
    }

    /// Find the nearest place cell to a point
    fn find_nearest_cell(&self, point: &StatePoint) -> Option<&PlaceCell> {
        if self.place_cells.is_empty() {
            return None;
        }

        self.place_cells.iter().min_by(|a, b| {
            a.distance_to(point)
                .partial_cmp(&b.distance_to(point))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find similar states from history
    pub fn find_similar_states(&self, point: &StatePoint, limit: usize) -> Vec<&PlaceCell> {
        let mut cells_with_distances: Vec<_> = self
            .place_cells
            .iter()
            .map(|cell| (cell, cell.distance_to(point)))
            .collect();

        cells_with_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        cells_with_distances
            .into_iter()
            .take(limit)
            .map(|(cell, _)| cell)
            .collect()
    }

    /// Get the current position in state space
    pub fn current_position(&self) -> Option<&StatePoint> {
        self.trajectory.points.back()
    }

    /// Get predicted outcome based on similar states
    pub fn predict_outcome(&self, point: &StatePoint) -> Option<f64> {
        let similar = self.find_similar_states(point, 5);
        if similar.is_empty() {
            return None;
        }

        // Weighted average by distance (inverse distance weighting)
        let mut total_weight = 0.0;
        let mut weighted_outcome = 0.0;

        for cell in similar {
            if cell.observation_count >= self.config.min_samples_for_cell {
                let distance = cell.distance_to(point).max(0.001); // Avoid division by zero
                let weight = 1.0 / distance;
                weighted_outcome += cell.avg_outcome * weight;
                total_weight += weight;
            }
        }

        if total_weight == 0.0 {
            None
        } else {
            Some(weighted_outcome / total_weight)
        }
    }

    /// Get the active place cells (above activation threshold)
    pub fn active_cells(&self, threshold: f64) -> Vec<&PlaceCell> {
        self.place_cells
            .iter()
            .filter(|cell| cell.activation >= threshold)
            .collect()
    }

    /// Get trajectory statistics
    pub fn trajectory_stats(&self) -> TrajectoryStats {
        TrajectoryStats {
            length: self.trajectory.points.len(),
            path_length: self.trajectory.path_length(),
            displacement: self.trajectory.displacement(),
            efficiency: self.trajectory.efficiency(),
            direction: self.trajectory.direction(),
        }
    }

    /// Get overall map statistics
    pub fn stats(&self) -> SpatialMapStats {
        let total_observations: usize = self.place_cells.iter().map(|c| c.observation_count).sum();

        let avg_activation = if self.place_cells.is_empty() {
            0.0
        } else {
            self.place_cells.iter().map(|c| c.activation).sum::<f64>()
                / self.place_cells.len() as f64
        };

        SpatialMapStats {
            dimensions: self.dimensions.len(),
            place_cells: self.place_cells.len(),
            total_observations,
            trajectory_length: self.trajectory.points.len(),
            avg_activation,
            grid_cells_occupied: self.grid_counts.len(),
        }
    }

    /// Label a place cell
    pub fn label_cell(&mut self, cell_id: u64, label: &str) {
        if let Some(cell) = self.place_cells.iter_mut().find(|c| c.id == cell_id) {
            cell.label = label.to_string();
        }
    }

    /// Get place cells by label
    pub fn get_cells_by_label(&self, label: &str) -> Vec<&PlaceCell> {
        self.place_cells
            .iter()
            .filter(|c| c.label == label)
            .collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.place_cells.clear();
        self.trajectory = StateTrajectory::new(self.config.trajectory_length);
        self.grid_counts.clear();
        self.next_point_id = 1;
        self.next_cell_id = 1;
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // No-op for basic processing
        Ok(())
    }

    /// Get the trajectory
    pub fn trajectory(&self) -> &StateTrajectory {
        &self.trajectory
    }

    /// Get all place cells
    pub fn place_cells(&self) -> &[PlaceCell] {
        &self.place_cells
    }

    /// Get the dimensions
    pub fn dimensions(&self) -> &[StateDimension] {
        &self.dimensions
    }
}

/// Trajectory statistics
#[derive(Debug, Clone)]
pub struct TrajectoryStats {
    /// Number of points in trajectory
    pub length: usize,
    /// Total path length
    pub path_length: f64,
    /// Net displacement
    pub displacement: f64,
    /// Efficiency (displacement / path_length)
    pub efficiency: f64,
    /// Direction vector
    pub direction: Option<Vec<f64>>,
}

/// Spatial map statistics
#[derive(Debug, Clone)]
pub struct SpatialMapStats {
    /// Number of dimensions
    pub dimensions: usize,
    /// Number of place cells
    pub place_cells: usize,
    /// Total observations recorded
    pub total_observations: usize,
    /// Current trajectory length
    pub trajectory_length: usize,
    /// Average activation across cells
    pub avg_activation: f64,
    /// Number of grid cells with data
    pub grid_cells_occupied: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = SpatialMap::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_state_point_distance() {
        let p1 = StatePoint::new(1, vec![0.0, 0.0, 0.0], 0, "TEST".to_string());
        let p2 = StatePoint::new(2, vec![1.0, 0.0, 0.0], 1, "TEST".to_string());
        let p3 = StatePoint::new(3, vec![1.0, 1.0, 1.0], 2, "TEST".to_string());

        assert!((p1.distance(&p2) - 1.0).abs() < 0.001);
        assert!((p1.distance(&p3) - 3.0_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_state_point_cosine_similarity() {
        let p1 = StatePoint::new(1, vec![1.0, 0.0], 0, "TEST".to_string());
        let p2 = StatePoint::new(2, vec![1.0, 0.0], 1, "TEST".to_string());
        let p3 = StatePoint::new(3, vec![0.0, 1.0], 2, "TEST".to_string());

        assert!((p1.cosine_similarity(&p2) - 1.0).abs() < 0.001);
        assert!(p1.cosine_similarity(&p3).abs() < 0.001);
    }

    #[test]
    fn test_dimension_normalization() {
        let dim = StateDimension::new("test", 0.0, 100.0);

        assert!((dim.normalize(0.0) - 0.0).abs() < 0.001);
        assert!((dim.normalize(50.0) - 0.5).abs() < 0.001);
        assert!((dim.normalize(100.0) - 1.0).abs() < 0.001);
        assert!((dim.normalize(150.0) - 1.0).abs() < 0.001); // Clamped
    }

    #[test]
    fn test_record_state() {
        let mut map = SpatialMap::new();
        map.set_symbol("BTCUSD");

        let mut features = HashMap::new();
        features.insert("volatility".to_string(), 0.02);
        features.insert("trend".to_string(), 0.5);
        features.insert("volume_ratio".to_string(), 1.5);

        let point = map.record_state(features, 1000);

        assert_eq!(point.symbol, "BTCUSD");
        assert!(!point.coordinates.is_empty());
        assert_eq!(map.trajectory().points.len(), 1);
    }

    #[test]
    fn test_place_cell_creation() {
        let mut map = SpatialMap::new();
        map.set_symbol("BTCUSD");

        // Record multiple similar states
        for i in 0..5 {
            let mut features = HashMap::new();
            features.insert("volatility".to_string(), 0.02 + (i as f64) * 0.001);
            features.insert("trend".to_string(), 0.5);
            map.record_state(features, i * 1000);
        }

        // Should have created at least one place cell
        assert!(!map.place_cells().is_empty());
    }

    #[test]
    fn test_find_similar_states() {
        let mut map = SpatialMap::new();
        map.set_symbol("BTCUSD");

        // Create distinct clusters
        for i in 0..10 {
            let mut features = HashMap::new();
            features.insert("volatility".to_string(), 0.02);
            features.insert("trend".to_string(), 0.5);
            map.record_state(features, i * 1000);
        }

        // Record a different state
        for i in 0..10 {
            let mut features = HashMap::new();
            features.insert("volatility".to_string(), 0.08);
            features.insert("trend".to_string(), -0.5);
            map.record_state(features, (i + 10) * 1000);
        }

        // Query for similar states
        let query = StatePoint::new(
            999,
            vec![0.2, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0,
            "TEST".to_string(),
        );
        let similar = map.find_similar_states(&query, 3);

        assert!(!similar.is_empty());
    }

    #[test]
    fn test_trajectory() {
        let mut trajectory = StateTrajectory::new(10);

        // Add points in a straight line
        for i in 0..5 {
            let point = StatePoint::new(
                i as u64,
                vec![i as f64 / 10.0, 0.0, 0.0],
                i * 1000,
                "TEST".to_string(),
            );
            trajectory.add(point);
        }

        assert_eq!(trajectory.points.len(), 5);
        assert!(trajectory.path_length() > 0.0);
        assert!(trajectory.efficiency() > 0.9); // Straight line is efficient

        // Direction should be mostly in first dimension
        if let Some(dir) = trajectory.direction() {
            assert!(dir[0] > 0.9);
        }
    }

    #[test]
    fn test_predict_outcome() {
        let mut map = SpatialMap::with_config(SpatialMapConfig {
            min_samples_for_cell: 2,
            ..Default::default()
        });
        map.set_symbol("BTCUSD");

        // Record states with known outcomes
        for i in 0..10 {
            let mut features = HashMap::new();
            features.insert("volatility".to_string(), 0.02);
            features.insert("trend".to_string(), 0.5);
            map.record_state_with_outcome(features, i * 1000, 0.01); // Positive outcome
        }

        // Query a similar state
        let query = StatePoint::new(
            999,
            vec![0.2, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0,
            "TEST".to_string(),
        );

        // Prediction should be close to the average outcome
        if let Some(prediction) = map.predict_outcome(&query) {
            assert!(prediction > 0.0);
        }
    }

    #[test]
    fn test_stats() {
        let mut map = SpatialMap::new();
        map.set_symbol("BTCUSD");

        let mut features = HashMap::new();
        features.insert("volatility".to_string(), 0.02);
        map.record_state(features.clone(), 1000);
        map.record_state(features.clone(), 2000);

        let stats = map.stats();
        assert_eq!(stats.dimensions, 8);
        assert!(stats.place_cells >= 1);
        assert_eq!(stats.trajectory_length, 2);
    }

    #[test]
    fn test_label_cells() {
        let mut map = SpatialMap::new();
        map.set_symbol("BTCUSD");

        let mut features = HashMap::new();
        features.insert("volatility".to_string(), 0.08);
        features.insert("trend".to_string(), -0.8);
        map.record_state(features, 1000);

        // Label the cell
        if let Some(cell) = map.place_cells().first() {
            let id = cell.id;
            map.label_cell(id, "high_vol_bearish");
        }

        let labeled = map.get_cells_by_label("high_vol_bearish");
        assert_eq!(labeled.len(), 1);
    }

    #[test]
    fn test_trajectory_max_length() {
        let config = SpatialMapConfig {
            trajectory_length: 5,
            ..Default::default()
        };
        let mut map = SpatialMap::with_config(config);

        for i in 0..10 {
            let mut features = HashMap::new();
            features.insert("volatility".to_string(), 0.02);
            map.record_state(features, i * 1000);
        }

        // Trajectory should be limited to max length
        assert_eq!(map.trajectory().points.len(), 5);
    }
}
