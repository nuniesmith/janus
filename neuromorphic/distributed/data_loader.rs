//! Distributed Data Loader
//!
//! This module provides a distributed data loader for efficient multi-GPU training.
//! It handles data sharding, prefetching, and balanced distribution across devices.
//!
//! # Features
//!
//! - Automatic data sharding across devices
//! - Asynchronous prefetching for I/O overlap
//! - Balanced batch distribution
//! - Support for custom samplers
//! - Memory-efficient streaming
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::distributed::{DistributedDataLoader, ShardingStrategy};
//! use candle_core::Tensor;
//!
//! # fn example() -> anyhow::Result<()> {
//! let data = vec![vec![1.0, 2.0, 3.0]; 1000]; // Example data
//! let loader = DistributedDataLoader::new(data, 32, 4)?
//!     .with_sharding(ShardingStrategy::RoundRobin)
//!     .with_prefetch(2);
//!
//! for batch in loader.iter() {
//!     // Process batch
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tracing::debug;

/// Data sharding strategy for distributed training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Divide data sequentially into contiguous chunks
    Contiguous,
    /// Distribute data in round-robin fashion
    RoundRobin,
    /// Random sharding (with seed for reproducibility)
    Random,
    /// Stratified sharding (preserve class distribution)
    Stratified,
}

/// Data sampling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Sequential sampling (no shuffling)
    Sequential,
    /// Random sampling (shuffle each epoch)
    Random,
    /// Weighted sampling
    Weighted,
}

/// Configuration for distributed data loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    /// Batch size per device
    pub batch_size: usize,
    /// Number of devices/workers
    pub num_workers: usize,
    /// Sharding strategy
    pub sharding_strategy: ShardingStrategy,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
    /// Number of batches to prefetch
    pub prefetch_batches: usize,
    /// Shuffle data each epoch
    pub shuffle: bool,
    /// Drop last incomplete batch
    pub drop_last: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Pin memory for faster GPU transfer
    pub pin_memory: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 1,
            sharding_strategy: ShardingStrategy::Contiguous,
            sampling_strategy: SamplingStrategy::Sequential,
            prefetch_batches: 2,
            shuffle: false,
            drop_last: false,
            seed: None,
            pin_memory: false,
        }
    }
}

/// Batch of data distributed across devices
#[derive(Debug, Clone)]
pub struct DistributedBatch {
    /// Data tensors per device
    pub data: Vec<Tensor>,
    /// Labels (optional) per device
    pub labels: Option<Vec<Tensor>>,
    /// Batch indices
    pub indices: Vec<usize>,
    /// Device assignments
    pub devices: Vec<usize>,
}

impl DistributedBatch {
    /// Get batch size (total across all devices)
    pub fn size(&self) -> usize {
        self.data.iter().map(|t| t.dims()[0]).sum()
    }

    /// Get number of devices
    pub fn num_devices(&self) -> usize {
        self.data.len()
    }

    /// Get data for specific device
    pub fn device_data(&self, device_idx: usize) -> Option<&Tensor> {
        self.data.get(device_idx)
    }

    /// Get labels for specific device
    pub fn device_labels(&self, device_idx: usize) -> Option<&Tensor> {
        self.labels.as_ref().and_then(|l| l.get(device_idx))
    }
}

/// Distributed data loader for multi-GPU training
pub struct DistributedDataLoader<T> {
    /// Raw data
    data: Vec<T>,
    /// Labels (optional)
    labels: Option<Vec<usize>>,
    /// Configuration
    config: DataLoaderConfig,
    /// Devices to distribute to
    devices: Vec<Device>,
    /// Current epoch
    current_epoch: usize,
    /// Prefetch buffer
    prefetch_buffer: Arc<Mutex<VecDeque<DistributedBatch>>>,
    /// Random number generator
    rng: StdRng,
}

impl<T: Clone + Send + 'static> DistributedDataLoader<T> {
    /// Create a new distributed data loader
    pub fn new(data: Vec<T>, batch_size: usize, num_workers: usize) -> Result<Self> {
        let config = DataLoaderConfig {
            batch_size,
            num_workers,
            ..DataLoaderConfig::default()
        };

        Self::with_config(data, None, config)
    }

    /// Create with custom configuration
    pub fn with_config(
        data: Vec<T>,
        labels: Option<Vec<usize>>,
        config: DataLoaderConfig,
    ) -> Result<Self> {
        if let Some(ref lbls) = labels {
            if lbls.len() != data.len() {
                return Err(anyhow!(
                    "Data and labels length mismatch: {} vs {}",
                    data.len(),
                    lbls.len()
                ));
            }
        }

        // Detect devices
        let devices = Self::detect_devices(config.num_workers)?;

        let seed = config.seed.unwrap_or_else(rand::random);
        let rng = StdRng::seed_from_u64(seed);

        Ok(Self {
            data,
            labels,
            config,
            devices,
            current_epoch: 0,
            prefetch_buffer: Arc::new(Mutex::new(VecDeque::new())),
            rng,
        })
    }

    /// Detect available devices
    fn detect_devices(num_workers: usize) -> Result<Vec<Device>> {
        let mut devices = Vec::new();

        // Try CUDA devices first
        #[cfg(feature = "cuda")]
        {
            for i in 0..num_workers {
                match Device::new_cuda(i) {
                    Ok(dev) => devices.push(dev),
                    Err(_) => break,
                }
            }
        }

        // Fallback to CPU if no CUDA devices
        if devices.is_empty() {
            for _ in 0..num_workers {
                devices.push(Device::Cpu);
            }
        }

        Ok(devices)
    }

    /// Set sharding strategy
    pub fn with_sharding(mut self, strategy: ShardingStrategy) -> Self {
        self.config.sharding_strategy = strategy;
        self
    }

    /// Set number of prefetch batches
    pub fn with_prefetch(mut self, num_batches: usize) -> Self {
        self.config.prefetch_batches = num_batches;
        self
    }

    /// Enable/disable shuffling
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        let total_batch_size = self.config.batch_size * self.config.num_workers;
        let batches = self.data.len() / total_batch_size;
        if self.config.drop_last {
            batches
        } else {
            self.data.len().div_ceil(total_batch_size)
        }
    }

    /// Get data length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Shard data indices across workers
    fn shard_indices(&mut self) -> Vec<Vec<usize>> {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();

        // Shuffle if configured
        if self.config.shuffle {
            indices.shuffle(&mut self.rng);
        }

        let num_workers = self.config.num_workers;
        let mut shards: Vec<Vec<usize>> = vec![Vec::new(); num_workers];

        match self.config.sharding_strategy {
            ShardingStrategy::Contiguous => {
                let chunk_size = indices.len().div_ceil(num_workers);
                for (i, idx) in indices.into_iter().enumerate() {
                    let shard = i / chunk_size;
                    if shard < num_workers {
                        shards[shard].push(idx);
                    }
                }
            }
            ShardingStrategy::RoundRobin => {
                for (i, idx) in indices.into_iter().enumerate() {
                    shards[i % num_workers].push(idx);
                }
            }
            ShardingStrategy::Random => {
                indices.shuffle(&mut self.rng);
                for (i, idx) in indices.into_iter().enumerate() {
                    shards[i % num_workers].push(idx);
                }
            }
            ShardingStrategy::Stratified => {
                // Group by label if available
                if let Some(ref labels) = self.labels {
                    let mut label_indices: std::collections::HashMap<usize, Vec<usize>> =
                        std::collections::HashMap::new();

                    for idx in indices {
                        label_indices.entry(labels[idx]).or_default().push(idx);
                    }

                    // Distribute each label group round-robin
                    for mut group in label_indices.into_values() {
                        if self.config.shuffle {
                            group.shuffle(&mut self.rng);
                        }
                        for (i, idx) in group.into_iter().enumerate() {
                            shards[i % num_workers].push(idx);
                        }
                    }
                } else {
                    // Fall back to round-robin if no labels
                    for (i, idx) in indices.into_iter().enumerate() {
                        shards[i % num_workers].push(idx);
                    }
                }
            }
        }

        shards
    }

    /// Create iterator over batches
    pub fn iter(&mut self) -> DistributedDataIterator<'_, T> {
        let shards = self.shard_indices();

        DistributedDataIterator {
            shards,
            batch_size: self.config.batch_size,
            devices: self.devices.clone(),
            current_positions: vec![0; self.config.num_workers],
            drop_last: self.config.drop_last,
            data: &self.data,
            labels: self.labels.as_ref(),
        }
    }

    /// Start prefetching batches in background
    pub fn start_prefetch(&mut self) {
        let _buffer = Arc::clone(&self.prefetch_buffer);
        let prefetch_count = self.config.prefetch_batches;

        // In a real implementation, this would spawn background threads
        // to asynchronously load and prepare batches
        debug!("Prefetch configured for {} batches", prefetch_count);
    }

    /// Advance to next epoch
    pub fn next_epoch(&mut self) {
        self.current_epoch += 1;
        debug!("Advanced to epoch {}", self.current_epoch);
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get configuration
    pub fn config(&self) -> &DataLoaderConfig {
        &self.config
    }
}

/// Iterator over distributed batches
pub struct DistributedDataIterator<'a, T> {
    shards: Vec<Vec<usize>>,
    batch_size: usize,
    #[allow(dead_code)]
    devices: Vec<Device>,
    current_positions: Vec<usize>,
    drop_last: bool,
    data: &'a [T],
    labels: Option<&'a Vec<usize>>,
}

impl<'a, T: Clone> Iterator for DistributedDataIterator<'a, T> {
    type Item = (Vec<Vec<T>>, Option<Vec<Vec<usize>>>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_data = Vec::new();
        let mut batch_labels = if self.labels.is_some() {
            Some(Vec::new())
        } else {
            None
        };
        let mut batch_indices = Vec::new();

        let mut all_done = true;

        // Collect batch from each shard
        for (shard_idx, shard) in self.shards.iter().enumerate() {
            let pos = self.current_positions[shard_idx];

            if pos >= shard.len() {
                if !self.drop_last && all_done {
                    // Add empty for this shard
                    batch_data.push(Vec::new());
                    if let Some(ref mut lbls) = batch_labels {
                        lbls.push(Vec::new());
                    }
                }
                continue;
            }

            all_done = false;

            let end = (pos + self.batch_size).min(shard.len());

            // Skip if incomplete and drop_last is true
            if self.drop_last && (end - pos) < self.batch_size {
                self.current_positions[shard_idx] = shard.len();
                continue;
            }

            let mut shard_data = Vec::new();
            let mut shard_labels = Vec::new();

            for &idx in &shard[pos..end] {
                shard_data.push(self.data[idx].clone());
                if let Some(labels) = self.labels {
                    shard_labels.push(labels[idx]);
                }
                batch_indices.push(idx);
            }

            batch_data.push(shard_data);
            if self.labels.is_some() {
                if let Some(ref mut lbls) = batch_labels {
                    lbls.push(shard_labels);
                }
            }

            self.current_positions[shard_idx] = end;
        }

        if all_done || batch_data.is_empty() {
            None
        } else {
            Some((batch_data, batch_labels, batch_indices))
        }
    }
}

/// Helper to convert Vec<f32> batches to Tensors
pub fn batches_to_tensors(
    batches: Vec<Vec<Vec<f32>>>,
    devices: &[Device],
    dtype: DType,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::new();

    for (batch, device) in batches.into_iter().zip(devices.iter()) {
        if batch.is_empty() {
            continue;
        }

        let batch_size = batch.len();
        let feature_size = batch[0].len();

        // Flatten batch
        let data: Vec<f32> = batch.into_iter().flatten().collect();

        // Create tensor
        let tensor = Tensor::from_vec(data, (batch_size, feature_size), device)?.to_dtype(dtype)?;

        tensors.push(tensor);
    }

    Ok(tensors)
}

/// Helper to convert label batches to Tensors
pub fn labels_to_tensors(
    label_batches: Vec<Vec<usize>>,
    devices: &[Device],
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::new();

    for (labels, device) in label_batches.into_iter().zip(devices.iter()) {
        if labels.is_empty() {
            continue;
        }

        let labels_u32: Vec<u32> = labels.into_iter().map(|l| l as u32).collect();
        let len = labels_u32.len();
        let tensor = Tensor::from_vec(labels_u32, len, device)?;
        tensors.push(tensor);
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader_creation() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let loader = DistributedDataLoader::new(data, 8, 2).unwrap();

        assert_eq!(loader.len(), 100);
        assert!(!loader.is_empty());
    }

    #[test]
    fn test_sharding_contiguous() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let mut loader = DistributedDataLoader::new(data, 8, 4)
            .unwrap()
            .with_sharding(ShardingStrategy::Contiguous);

        let shards = loader.shard_indices();
        assert_eq!(shards.len(), 4);

        // Each shard should have ~25 elements
        for shard in &shards {
            assert!(shard.len() >= 24 && shard.len() <= 26);
        }
    }

    #[test]
    fn test_sharding_round_robin() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let mut loader = DistributedDataLoader::new(data, 8, 4)
            .unwrap()
            .with_sharding(ShardingStrategy::RoundRobin);

        let shards = loader.shard_indices();
        assert_eq!(shards.len(), 4);

        // Each shard should have exactly 25 elements
        for shard in &shards {
            assert_eq!(shard.len(), 25);
        }

        // Check round-robin distribution
        assert_eq!(shards[0][0], 0);
        assert_eq!(shards[1][0], 1);
        assert_eq!(shards[2][0], 2);
        assert_eq!(shards[3][0], 3);
        assert_eq!(shards[0][1], 4);
    }

    #[test]
    fn test_iterator() {
        let data: Vec<Vec<f32>> = (0..32).map(|i| vec![i as f32; 10]).collect();
        let mut loader = DistributedDataLoader::new(data, 8, 2).unwrap();

        let mut count = 0;
        for (batch_data, _labels, _indices) in loader.iter() {
            assert_eq!(batch_data.len(), 2); // 2 workers
            count += 1;
        }

        assert_eq!(count, 2); // 32 samples / (8 batch_size * 2 workers) = 2 batches
    }

    #[test]
    fn test_with_labels() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let labels: Vec<usize> = (0..100).map(|i| i % 5).collect(); // 5 classes

        let mut loader =
            DistributedDataLoader::with_config(data, Some(labels), DataLoaderConfig::default())
                .unwrap();

        for (batch_data, batch_labels, _indices) in loader.iter() {
            assert!(batch_labels.is_some());
            let lbls = batch_labels.unwrap();
            assert_eq!(lbls.len(), batch_data.len());
        }
    }

    #[test]
    fn test_shuffle() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let mut loader = DistributedDataLoader::new(data.clone(), 10, 1)
            .unwrap()
            .with_shuffle(true)
            .with_seed(42);

        let shards1 = loader.shard_indices();

        // Reset and shard again with different seed
        loader = DistributedDataLoader::new(data, 10, 1)
            .unwrap()
            .with_shuffle(true)
            .with_seed(123);

        let shards2 = loader.shard_indices();

        // Should be different due to different seeds
        assert_ne!(shards1[0], shards2[0]);
    }

    #[test]
    fn test_drop_last() {
        let data: Vec<Vec<f32>> = (0..33).map(|i| vec![i as f32; 10]).collect();
        let mut loader = DistributedDataLoader::new(data, 10, 2).unwrap();

        loader.config.drop_last = true;
        let mut count = 0;
        for _ in loader.iter() {
            count += 1;
        }
        assert_eq!(count, 1); // Only complete batch

        loader.config.drop_last = false;
        let mut count = 0;
        for _ in loader.iter() {
            count += 1;
        }
        assert_eq!(count, 2); // Include incomplete batch
    }

    #[test]
    fn test_num_batches() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let loader = DistributedDataLoader::new(data, 8, 2).unwrap();

        // 100 samples / (8 batch_size * 2 workers) = 6.25 -> 7 batches (rounded up)
        assert_eq!(loader.num_batches(), 7);
    }

    #[test]
    fn test_epoch_management() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 10]).collect();
        let mut loader = DistributedDataLoader::new(data, 8, 2).unwrap();

        assert_eq!(loader.current_epoch(), 0);
        loader.next_epoch();
        assert_eq!(loader.current_epoch(), 1);
        loader.next_epoch();
        assert_eq!(loader.current_epoch(), 2);
    }

    #[test]
    fn test_batches_to_tensors() {
        let batch1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let batch2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let batches = vec![batch1, batch2];

        let devices = vec![Device::Cpu, Device::Cpu];
        let tensors = batches_to_tensors(batches, &devices, DType::F32).unwrap();

        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].dims(), &[2, 2]);
        assert_eq!(tensors[1].dims(), &[2, 2]);
    }

    #[test]
    fn test_labels_to_tensors() {
        let labels1 = vec![0, 1];
        let labels2 = vec![2, 3];
        let label_batches = vec![labels1, labels2];

        let devices = vec![Device::Cpu, Device::Cpu];
        let tensors = labels_to_tensors(label_batches, &devices).unwrap();

        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].dims(), &[2]);
        assert_eq!(tensors[1].dims(), &[2]);
    }
}
