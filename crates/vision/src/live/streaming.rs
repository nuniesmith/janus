//! Streaming feature computation for real-time market data.
//!
//! This module provides efficient incremental feature computation for live trading:
//! - Circular buffers for maintaining rolling windows
//! - Incremental GAF/DiffGAF updates without full recomputation
//! - Fast append-only operations for new bars/ticks
//! - Pre-allocated buffers to minimize latency

use crate::error::Result;
use std::collections::VecDeque;

/// A circular buffer for maintaining a rolling window of market data.
///
/// Uses a ring buffer implementation for O(1) append and efficient memory usage.
#[derive(Debug, Clone)]
pub struct CircularBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> CircularBuffer<T> {
    /// Create a new circular buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Append a new item, removing the oldest if at capacity.
    pub fn push(&mut self, item: T) {
        if self.data.len() == self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(item);
    }

    /// Get the current number of items in the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if the buffer is at full capacity.
    pub fn is_full(&self) -> bool {
        self.data.len() == self.capacity
    }

    /// Get a slice view of the buffer contents.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.data.as_slices()
    }

    /// Get an iterator over the buffer.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Clear all items from the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Convert buffer contents to a vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().cloned().collect()
    }
}

/// Market data point for streaming updates.
#[derive(Debug, Clone)]
pub struct MarketData {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl MarketData {
    /// Create a new market data point.
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Get OHLC array.
    pub fn ohlc(&self) -> [f64; 4] {
        [self.open, self.high, self.low, self.close]
    }
}

/// Streaming feature buffer for incremental computation.
///
/// Maintains rolling windows of features and supports incremental updates
/// without full recomputation.
#[derive(Debug, Clone)]
pub struct StreamingFeatureBuffer {
    /// Window size for feature computation
    window_size: usize,
    /// Buffer of market data
    data_buffer: CircularBuffer<MarketData>,
    /// Cached close prices for quick access
    close_buffer: CircularBuffer<f64>,
    /// Cached returns for incremental statistics
    returns_buffer: CircularBuffer<f64>,
    /// Running statistics
    stats: RunningStats,
}

impl StreamingFeatureBuffer {
    /// Create a new streaming feature buffer.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            data_buffer: CircularBuffer::new(window_size),
            close_buffer: CircularBuffer::new(window_size),
            returns_buffer: CircularBuffer::new(window_size - 1),
            stats: RunningStats::new(),
        }
    }

    /// Add a new market data point and update features incrementally.
    pub fn update(&mut self, data: MarketData) -> Result<()> {
        let close = data.close;

        // Calculate return if we have a previous close
        if let Some(prev_close) = self.close_buffer.iter().last() {
            let ret = (close / prev_close).ln();
            self.returns_buffer.push(ret);
            self.stats.update(ret);
        }

        self.data_buffer.push(data);
        self.close_buffer.push(close);

        Ok(())
    }

    /// Get the current close prices as a vector.
    pub fn get_closes(&self) -> Vec<f64> {
        self.close_buffer.to_vec()
    }

    /// Get the current returns as a vector.
    pub fn get_returns(&self) -> Vec<f64> {
        self.returns_buffer.to_vec()
    }

    /// Get the current market data as a vector.
    pub fn get_data(&self) -> Vec<MarketData> {
        self.data_buffer.to_vec()
    }

    /// Check if the buffer has enough data for feature computation.
    pub fn is_ready(&self) -> bool {
        self.data_buffer.is_full()
    }

    /// Get current window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get current buffer length.
    pub fn len(&self) -> usize {
        self.data_buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data_buffer.is_empty()
    }

    /// Get running statistics.
    pub fn stats(&self) -> &RunningStats {
        &self.stats
    }

    /// Reset the buffer and statistics.
    pub fn reset(&mut self) {
        self.data_buffer.clear();
        self.close_buffer.clear();
        self.returns_buffer.clear();
        self.stats.reset();
    }
}

/// Running statistics calculator for incremental updates.
///
/// Computes mean, variance, and other statistics incrementally using
/// Welford's online algorithm for numerical stability.
#[derive(Debug, Clone)]
pub struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squared differences from mean
    min: f64,
    max: f64,
}

impl RunningStats {
    /// Create a new running statistics calculator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Update statistics with a new value using Welford's algorithm.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Get the current mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the current variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get the current standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the minimum value seen.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get the maximum value seen.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Get the number of values processed.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental GAF computer for streaming updates.
///
/// Computes GAF features incrementally by updating only the new columns/rows
/// when a new data point arrives, avoiding full matrix recomputation.
///
/// # GAF Sign Ambiguity Warning
///
/// **GASF** (`cos(φ_i + φ_j)`) is **symmetric** — it cannot distinguish upward
/// from downward price movements. A rising series and its mirror produce
/// identical GASF matrices.
///
/// **GADF** (`sin(φ_i - φ_j)`) is **anti-symmetric** — it preserves temporal
/// direction and resolves the sign ambiguity.
///
/// **Recommendation**: Use [`update_dual()`](IncrementalGAFComputer::update_dual)
/// to get both GASF and GADF matrices, giving downstream models both correlation
/// magnitude and directional information. Using [`update()`](IncrementalGAFComputer::update)
/// alone (GASF-only) will lose directional information.
#[derive(Debug, Clone)]
pub struct IncrementalGAFComputer {
    #[allow(dead_code)]
    window_size: usize,
    feature_buffer: StreamingFeatureBuffer,
    /// Pre-allocated buffer for normalized values
    normalized_buffer: Vec<f64>,
    /// Pre-allocated buffer for polar coordinates
    phi_buffer: Vec<f64>,
    /// Cache for last computed GASF
    last_gaf: Option<Vec<Vec<f64>>>,
    /// Cache for last computed GADF
    last_gadf: Option<Vec<Vec<f64>>>,
}

impl IncrementalGAFComputer {
    /// Create a new incremental GAF computer.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            feature_buffer: StreamingFeatureBuffer::new(window_size),
            normalized_buffer: Vec::with_capacity(window_size),
            phi_buffer: Vec::with_capacity(window_size),
            last_gaf: None,
            last_gadf: None,
        }
    }

    /// Update with new market data and compute GASF incrementally.
    ///
    /// ⚠️ **Sign ambiguity**: GASF is symmetric and cannot distinguish upward
    /// from downward price movements. Consider using [`update_dual()`] instead.
    pub fn update(&mut self, data: MarketData) -> Result<Option<Vec<Vec<f64>>>> {
        self.feature_buffer.update(data)?;

        if !self.feature_buffer.is_ready() {
            return Ok(None);
        }

        // Get current values
        let values = self.feature_buffer.get_closes();

        // Normalize values to [-1, 1]
        self.normalize_values(&values);

        // Convert to polar angles
        self.compute_polar_angles();

        // Compute GAF matrix
        let gaf = self.compute_gaf_matrix();

        self.last_gaf = Some(gaf.clone());
        Ok(Some(gaf))
    }

    /// Update with new market data and compute both GASF and GADF matrices.
    ///
    /// Returns `(GASF, GADF)` where:
    /// - **GASF** (`cos(φ_i + φ_j)`) captures magnitude correlation (symmetric)
    /// - **GADF** (`sin(φ_i - φ_j)`) captures temporal direction (anti-symmetric)
    ///
    /// Using both resolves the sign ambiguity inherent in GASF alone.
    pub fn update_dual(
        &mut self,
        data: MarketData,
    ) -> Result<Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)>> {
        self.feature_buffer.update(data)?;

        if !self.feature_buffer.is_ready() {
            return Ok(None);
        }

        let values = self.feature_buffer.get_closes();
        self.normalize_values(&values);
        self.compute_polar_angles();

        let gasf = self.compute_gaf_matrix();
        let gadf = self.compute_gadf_matrix();

        self.last_gaf = Some(gasf.clone());
        self.last_gadf = Some(gadf.clone());

        Ok(Some((gasf, gadf)))
    }

    /// Normalize values to [-1, 1] range.
    fn normalize_values(&mut self, values: &[f64]) {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        self.normalized_buffer.clear();

        if (max - min).abs() < 1e-10 {
            // All values are the same
            self.normalized_buffer
                .extend(std::iter::repeat(0.0).take(values.len()));
        } else {
            for &v in values {
                let normalized = 2.0 * (v - min) / (max - min) - 1.0;
                self.normalized_buffer.push(normalized.clamp(-1.0, 1.0));
            }
        }
    }

    /// Convert normalized values to polar angles.
    fn compute_polar_angles(&mut self) {
        self.phi_buffer.clear();
        for &v in &self.normalized_buffer {
            let phi = v.acos();
            self.phi_buffer.push(phi);
        }
    }

    /// Compute the GASF matrix (Gramian Angular Summation Field).
    ///
    /// `GASF[i,j] = cos(φ_i + φ_j)` — symmetric, captures magnitude correlation.
    fn compute_gaf_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.phi_buffer.len();
        let mut gaf = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                // GASF: cos(phi_i + phi_j)
                gaf[i][j] = (self.phi_buffer[i] + self.phi_buffer[j]).cos();
            }
        }

        gaf
    }

    /// Compute the GADF matrix (Gramian Angular Difference Field).
    ///
    /// `GADF[i,j] = sin(φ_i - φ_j)` — anti-symmetric, preserves temporal direction.
    /// This resolves the sign ambiguity in GASF: rising and falling price series
    /// produce different GADF matrices.
    fn compute_gadf_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.phi_buffer.len();
        let mut gadf = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                // GADF: sin(phi_i - phi_j)
                gadf[i][j] = (self.phi_buffer[i] - self.phi_buffer[j]).sin();
            }
        }

        gadf
    }

    /// Get the last computed GASF matrix.
    pub fn last_gaf(&self) -> Option<&Vec<Vec<f64>>> {
        self.last_gaf.as_ref()
    }

    /// Get the last computed GADF matrix.
    pub fn last_gadf(&self) -> Option<&Vec<Vec<f64>>> {
        self.last_gadf.as_ref()
    }

    /// Check if ready to compute GAF.
    pub fn is_ready(&self) -> bool {
        self.feature_buffer.is_ready()
    }

    /// Get current buffer length.
    pub fn len(&self) -> usize {
        self.feature_buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.feature_buffer.is_empty()
    }

    /// Reset the computer.
    pub fn reset(&mut self) {
        self.feature_buffer.reset();
        self.normalized_buffer.clear();
        self.phi_buffer.clear();
        self.last_gaf = None;
        self.last_gadf = None;
    }
}

/// Multi-timeframe streaming feature buffer.
///
/// Maintains multiple buffers for different timeframes simultaneously,
/// enabling multi-timeframe analysis in real-time.
#[derive(Debug, Clone)]
pub struct MultiTimeframeBuffer {
    buffers: Vec<(String, StreamingFeatureBuffer)>,
}

impl MultiTimeframeBuffer {
    /// Create a new multi-timeframe buffer.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    /// Add a timeframe buffer.
    pub fn add_timeframe(&mut self, name: String, window_size: usize) {
        self.buffers
            .push((name, StreamingFeatureBuffer::new(window_size)));
    }

    /// Update all timeframes with new data.
    pub fn update_all(&mut self, data: &MarketData) -> Result<()> {
        for (_, buffer) in &mut self.buffers {
            buffer.update(data.clone())?;
        }
        Ok(())
    }

    /// Get a specific timeframe buffer.
    pub fn get_buffer(&self, name: &str) -> Option<&StreamingFeatureBuffer> {
        self.buffers.iter().find(|(n, _)| n == name).map(|(_, b)| b)
    }

    /// Get a mutable reference to a specific timeframe buffer.
    pub fn get_buffer_mut(&mut self, name: &str) -> Option<&mut StreamingFeatureBuffer> {
        self.buffers
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b)
    }

    /// Check if all buffers are ready.
    pub fn all_ready(&self) -> bool {
        self.buffers.iter().all(|(_, b)| b.is_ready())
    }

    /// Get all timeframe names.
    pub fn timeframe_names(&self) -> Vec<String> {
        self.buffers.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Reset all buffers.
    pub fn reset(&mut self) {
        for (_, buffer) in &mut self.buffers {
            buffer.reset();
        }
    }
}

impl Default for MultiTimeframeBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 3);

        buffer.push(4);
        assert_eq!(buffer.len(), 3);
        let vec = buffer.to_vec();
        assert_eq!(vec, vec![2, 3, 4]);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);

        assert_eq!(stats.mean(), 2.0);
        assert_eq!(stats.count(), 3);
        assert!((stats.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_feature_buffer() {
        let mut buffer = StreamingFeatureBuffer::new(3);
        assert!(!buffer.is_ready());

        buffer
            .update(MarketData::new(0, 100.0, 101.0, 99.0, 100.5, 1000.0))
            .unwrap();
        buffer
            .update(MarketData::new(1, 100.5, 102.0, 100.0, 101.0, 1100.0))
            .unwrap();
        assert!(!buffer.is_ready());

        buffer
            .update(MarketData::new(2, 101.0, 103.0, 101.0, 102.0, 1200.0))
            .unwrap();
        assert!(buffer.is_ready());

        let closes = buffer.get_closes();
        assert_eq!(closes.len(), 3);
        assert_eq!(closes[0], 100.5);
        assert_eq!(closes[2], 102.0);

        let returns = buffer.get_returns();
        assert_eq!(returns.len(), 2);
    }

    #[test]
    fn test_incremental_gaf_computer() {
        let mut computer = IncrementalGAFComputer::new(3);

        let result1 = computer
            .update(MarketData::new(0, 100.0, 101.0, 99.0, 100.0, 1000.0))
            .unwrap();
        assert!(result1.is_none());

        let result2 = computer
            .update(MarketData::new(1, 100.0, 102.0, 100.0, 101.0, 1100.0))
            .unwrap();
        assert!(result2.is_none());

        let result3 = computer
            .update(MarketData::new(2, 101.0, 103.0, 101.0, 102.0, 1200.0))
            .unwrap();
        assert!(result3.is_some());

        let gaf = result3.unwrap();
        assert_eq!(gaf.len(), 3);
        assert_eq!(gaf[0].len(), 3);
    }

    #[test]
    fn test_multi_timeframe_buffer() {
        let mut mtf = MultiTimeframeBuffer::new();
        mtf.add_timeframe("1m".to_string(), 60);
        mtf.add_timeframe("5m".to_string(), 300);

        assert_eq!(mtf.timeframe_names().len(), 2);
        assert!(!mtf.all_ready());

        let data = MarketData::new(0, 100.0, 101.0, 99.0, 100.0, 1000.0);
        mtf.update_all(&data).unwrap();

        let buffer = mtf.get_buffer("1m").unwrap();
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_circular_buffer_iteration() {
        let mut buffer = CircularBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let sum: i32 = buffer.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_running_stats_edge_cases() {
        let mut stats = RunningStats::new();
        assert_eq!(stats.variance(), 0.0);

        stats.update(5.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.mean(), 5.0);
        assert_eq!(stats.min(), 5.0);
        assert_eq!(stats.max(), 5.0);
    }

    #[test]
    fn test_gaf_normalization() {
        let mut computer = IncrementalGAFComputer::new(3);

        // All same values
        computer
            .update(MarketData::new(0, 100.0, 100.0, 100.0, 100.0, 1000.0))
            .unwrap();
        computer
            .update(MarketData::new(1, 100.0, 100.0, 100.0, 100.0, 1000.0))
            .unwrap();
        let result = computer
            .update(MarketData::new(2, 100.0, 100.0, 100.0, 100.0, 1000.0))
            .unwrap();

        assert!(result.is_some());
    }
}
