//! Distributed Training Coordinator
//!
//! This module provides distributed training coordination for multi-GPU and multi-node setups.
//! It handles device management, gradient synchronization, and distributed checkpoint coordination.
//!
//! # Features
//!
//! - Multi-GPU data parallelism
//! - Gradient synchronization across devices
//! - Distributed checkpoint coordination
//! - Device topology detection
//! - Performance monitoring and profiling
//!
//! # Example
//!
//! ```no_run
//! use janus_neuromorphic::distributed::TrainingCoordinator;
//! use candle_core::Device;
//!
//! # fn example() -> anyhow::Result<()> {
//! let coordinator = TrainingCoordinator::new()?;
//! let devices = coordinator.available_devices();
//! println!("Training on {} devices", devices.len());
//!
//! // Distribute model across devices
//! coordinator.sync_gradients()?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tracing::{debug, info};

/// Device information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device index
    pub index: usize,
    /// Device type (CPU, CUDA, Metal)
    pub device_type: String,
    /// Memory capacity in bytes (if available)
    pub memory_total: Option<usize>,
    /// Current memory usage in bytes
    pub memory_used: Option<usize>,
    /// Device name/identifier
    pub name: String,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(usize, usize)>,
}

/// Distributed training strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingStrategy {
    /// Data parallelism - replicate model, split data
    DataParallel,
    /// Model parallelism - split model across devices
    ModelParallel,
    /// Pipeline parallelism - split model into stages
    PipelineParallel,
    /// Hybrid - combination of strategies
    Hybrid,
}

/// Gradient synchronization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncMethod {
    /// All-reduce (average gradients across all devices)
    AllReduce,
    /// Parameter server (centralized gradient aggregation)
    ParameterServer,
    /// Ring all-reduce (bandwidth-optimal)
    RingAllReduce,
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Training strategy
    pub strategy: TrainingStrategy,
    /// Gradient synchronization method
    pub sync_method: SyncMethod,
    /// Number of gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Use mixed precision training
    pub mixed_precision: bool,
    /// Gradient clipping threshold
    pub gradient_clip_norm: Option<f32>,
    /// Sync every N steps (0 = sync every step)
    pub sync_frequency: usize,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Compression ratio (if enabled)
    pub compression_ratio: f32,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            strategy: TrainingStrategy::DataParallel,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation_steps: 1,
            mixed_precision: false,
            gradient_clip_norm: Some(1.0),
            sync_frequency: 1,
            gradient_compression: false,
            compression_ratio: 0.1,
        }
    }
}

/// Performance metrics for distributed training
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedMetrics {
    /// Total training steps
    pub total_steps: usize,
    /// Total samples processed
    pub total_samples: usize,
    /// Average gradient sync time (ms)
    pub avg_sync_time_ms: f64,
    /// Peak memory usage per device (bytes)
    pub peak_memory_per_device: HashMap<usize, usize>,
    /// Throughput (samples/second)
    pub throughput: f64,
    /// Communication overhead (%)
    pub communication_overhead_pct: f64,
    /// Last update timestamp (not serialized)
    #[serde(skip)]
    pub last_update: Option<Instant>,
}

/// Distributed training coordinator
pub struct TrainingCoordinator {
    /// Available devices
    devices: Vec<Device>,
    /// Device information
    device_info: Vec<DeviceInfo>,
    /// Configuration
    config: DistributedConfig,
    /// Performance metrics
    metrics: Arc<RwLock<DistributedMetrics>>,
    /// Gradient accumulation buffer
    gradient_buffer: Arc<Mutex<HashMap<String, Vec<Tensor>>>>,
    /// Current step counter
    step_counter: Arc<Mutex<usize>>,
    /// Rank (for multi-node setups)
    rank: usize,
    /// World size (total number of processes)
    world_size: usize,
}

impl TrainingCoordinator {
    /// Create a new training coordinator with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(DistributedConfig::default())
    }

    /// Create a new training coordinator with custom configuration
    pub fn with_config(config: DistributedConfig) -> Result<Self> {
        let devices = Self::detect_devices()?;
        let device_info = Self::gather_device_info(&devices)?;

        info!(
            "Initialized distributed training coordinator with {} devices",
            devices.len()
        );
        for info in &device_info {
            info!(
                "  Device {}: {} ({})",
                info.index, info.name, info.device_type
            );
        }

        Ok(Self {
            devices,
            device_info,
            config,
            metrics: Arc::new(RwLock::new(DistributedMetrics::default())),
            gradient_buffer: Arc::new(Mutex::new(HashMap::new())),
            step_counter: Arc::new(Mutex::new(0)),
            rank: 0,
            world_size: 1,
        })
    }

    /// Detect available devices
    fn detect_devices() -> Result<Vec<Device>> {
        let mut devices = Vec::new();

        // Try to detect CUDA devices
        #[cfg(feature = "cuda")]
        {
            match Device::cuda_if_available(0) {
                Ok(device) => {
                    if let Device::Cuda(_) = device {
                        // CUDA available, detect all GPUs
                        let mut gpu_idx = 0;
                        loop {
                            match Device::new_cuda(gpu_idx) {
                                Ok(dev) => {
                                    devices.push(dev);
                                    gpu_idx += 1;
                                }
                                Err(_) => break,
                            }
                        }
                        info!("Detected {} CUDA devices", devices.len());
                    }
                }
                Err(_) => {
                    debug!("CUDA not available");
                }
            }
        }

        // Try Metal (for Apple Silicon)
        #[cfg(feature = "metal")]
        {
            if devices.is_empty() {
                match Device::new_metal(0) {
                    Ok(device) => {
                        devices.push(device);
                        info!("Using Metal device");
                    }
                    Err(_) => {
                        debug!("Metal not available");
                    }
                }
            }
        }

        // Fallback to CPU
        if devices.is_empty() {
            devices.push(Device::Cpu);
            info!("Using CPU device");
        }

        Ok(devices)
    }

    /// Gather information about devices
    fn gather_device_info(devices: &[Device]) -> Result<Vec<DeviceInfo>> {
        devices
            .iter()
            .enumerate()
            .map(|(idx, device)| {
                let (device_type, name) = match device {
                    Device::Cpu => ("CPU".to_string(), "CPU".to_string()),
                    Device::Cuda(_) => ("CUDA".to_string(), format!("CUDA Device {}", idx)),
                    Device::Metal(_) => ("Metal".to_string(), format!("Metal Device {}", idx)),
                };

                Ok(DeviceInfo {
                    index: idx,
                    device_type,
                    memory_total: None, // Would need platform-specific queries
                    memory_used: None,
                    name,
                    compute_capability: None,
                })
            })
            .collect()
    }

    /// Get available devices
    pub fn available_devices(&self) -> &[Device] {
        &self.devices
    }

    /// Get device information
    pub fn device_info(&self) -> &[DeviceInfo] {
        &self.device_info
    }

    /// Get primary device (rank 0)
    pub fn primary_device(&self) -> &Device {
        &self.devices[0]
    }

    /// Get device by index
    pub fn device(&self, index: usize) -> Option<&Device> {
        self.devices.get(index)
    }

    /// Get current rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if this is the primary rank
    pub fn is_primary(&self) -> bool {
        self.rank == 0
    }

    /// Accumulate gradient for a parameter
    pub fn accumulate_gradient(&self, param_name: &str, gradient: Tensor) -> Result<()> {
        let mut buffer = self.gradient_buffer.lock().unwrap();
        buffer
            .entry(param_name.to_string())
            .or_default()
            .push(gradient);
        Ok(())
    }

    /// Synchronize gradients across devices
    pub fn sync_gradients(&self) -> Result<HashMap<String, Tensor>> {
        let start = Instant::now();
        let mut buffer = self.gradient_buffer.lock().unwrap();

        if buffer.is_empty() {
            return Ok(HashMap::new());
        }

        let mut synced_gradients = HashMap::new();

        for (param_name, gradients) in buffer.iter() {
            if gradients.is_empty() {
                continue;
            }

            // Synchronize based on method
            let synced = match self.config.sync_method {
                SyncMethod::AllReduce => self.all_reduce(gradients)?,
                SyncMethod::ParameterServer => self.parameter_server_sync(gradients)?,
                SyncMethod::RingAllReduce => self.ring_all_reduce(gradients)?,
            };

            // Apply gradient clipping if configured
            let final_grad = if let Some(max_norm) = self.config.gradient_clip_norm {
                self.clip_gradient(&synced, max_norm)?
            } else {
                synced
            };

            synced_gradients.insert(param_name.clone(), final_grad);
        }

        // Clear buffer after sync
        buffer.clear();

        // Update metrics
        let sync_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.write().unwrap();
        metrics.avg_sync_time_ms = if metrics.total_steps == 0 {
            sync_time_ms
        } else {
            (metrics.avg_sync_time_ms * metrics.total_steps as f64 + sync_time_ms)
                / (metrics.total_steps + 1) as f64
        };
        metrics.total_steps += 1;
        metrics.last_update = Some(Instant::now());

        debug!(
            "Synchronized {} gradients in {:.2}ms",
            synced_gradients.len(),
            sync_time_ms
        );

        Ok(synced_gradients)
    }

    /// All-reduce gradient synchronization (average across devices)
    fn all_reduce(&self, gradients: &[Tensor]) -> Result<Tensor> {
        if gradients.is_empty() {
            return Err(anyhow!("No gradients to reduce"));
        }

        // For single device, just return the gradient
        if self.devices.len() == 1 {
            return Ok(gradients[0].clone());
        }

        // Stack gradients and compute mean
        // In a real multi-GPU setup, this would use NCCL or similar
        let stacked = Tensor::stack(gradients, 0)?;
        let mean = stacked.mean(0)?;

        Ok(mean)
    }

    /// Parameter server gradient synchronization
    fn parameter_server_sync(&self, gradients: &[Tensor]) -> Result<Tensor> {
        // Primary device (rank 0) acts as parameter server
        if self.is_primary() {
            // Collect gradients from all devices and average
            self.all_reduce(gradients)
        } else {
            // Send gradient to primary and receive averaged gradient
            // In real implementation, would use network communication
            self.all_reduce(gradients)
        }
    }

    /// Ring all-reduce gradient synchronization (bandwidth-optimal)
    fn ring_all_reduce(&self, gradients: &[Tensor]) -> Result<Tensor> {
        let n_devices = gradients.len();
        if n_devices <= 1 {
            return self.all_reduce(gradients);
        }

        let first = &gradients[0];
        let shape = first.shape();
        let total_elems = shape.elem_count();

        if total_elems == 0 {
            return Ok(first.clone());
        }

        let primary_device = self.primary_device();

        // 1. Partition: Split tensor into N chunks (where N = devices)
        // This simulates the ring segmentation to reduce peak memory
        let chunk_size = total_elems.div_ceil(n_devices);
        let mut reduced_chunks = Vec::with_capacity(n_devices);

        // 2. Scatter-Reduce: Process each chunk sequentially
        // In a physical ring, these would be pipelined across links.
        // Here, we save memory by accumulating one chunk at a time.
        for i in 0..n_devices {
            let start = i * chunk_size;
            if start >= total_elems {
                break;
            }
            let len = usize::min(chunk_size, total_elems - start);

            let mut chunk_sum: Option<Tensor> = None;

            for grad in gradients {
                // Efficiently view the chunk without copying the full tensor
                let flat = grad.flatten_all()?;
                let chunk = flat.narrow(0, start, len)?;

                // Move chunk to aggregator device (simulating receiving from peer)
                let chunk_local = chunk.to_device(primary_device)?;

                chunk_sum = match chunk_sum {
                    Some(acc) => Some(acc.add(&chunk_local)?),
                    None => Some(chunk_local),
                };
            }

            if let Some(sum) = chunk_sum {
                // Average the accumulated chunk
                let mean = (sum / (n_devices as f64))?;
                reduced_chunks.push(mean);
            }
        }

        // 3. All-Gather: Stitch the reduced chunks back together
        if reduced_chunks.is_empty() {
            return self.all_reduce(gradients);
        }

        let flat_result = Tensor::cat(&reduced_chunks, 0)?;
        let result = flat_result.reshape(shape)?;

        Ok(result)
    }

    /// Clip gradient by norm
    fn clip_gradient(&self, gradient: &Tensor, max_norm: f32) -> Result<Tensor> {
        let grad_norm = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        if grad_norm > max_norm {
            let scale = max_norm / grad_norm;
            Ok(gradient.affine(scale as f64, 0.0)?)
        } else {
            Ok(gradient.clone())
        }
    }

    /// Broadcast tensor from primary to all devices
    pub fn broadcast(&self, tensor: &Tensor, device_idx: usize) -> Result<Vec<Tensor>> {
        let mut result = Vec::new();

        for (idx, device) in self.devices.iter().enumerate() {
            if idx == device_idx {
                result.push(tensor.clone());
            } else {
                // Copy tensor to target device
                let copied = tensor.to_device(device)?;
                result.push(copied);
            }
        }

        Ok(result)
    }

    /// Gather tensors from all devices to primary
    pub fn gather(&self, tensors: &[Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow!("No tensors to gather"));
        }

        // Move all tensors to primary device and stack
        let primary_device = self.primary_device();
        let mut moved = Vec::new();

        for tensor in tensors {
            moved.push(tensor.to_device(primary_device)?);
        }

        Ok(Tensor::stack(&moved, 0)?)
    }

    /// Scatter tensor from primary to all devices
    pub fn scatter(&self, tensor: &Tensor, dim: usize) -> Result<Vec<Tensor>> {
        let size = tensor.dim(dim)?;
        let chunk_size = size.div_ceil(self.devices.len());

        let mut result = Vec::new();
        for (idx, device) in self.devices.iter().enumerate() {
            let start = idx * chunk_size;
            let end = ((idx + 1) * chunk_size).min(size);

            if start >= size {
                break;
            }

            let chunk = tensor.narrow(dim, start, end - start)?;
            let moved = chunk.to_device(device)?;
            result.push(moved);
        }

        Ok(result)
    }

    /// Get current metrics
    pub fn metrics(&self) -> DistributedMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Update sample count for throughput calculation
    pub fn update_samples(&self, num_samples: usize) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_samples += num_samples;

        if let Some(last_update) = metrics.last_update {
            let elapsed = last_update.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                metrics.throughput = num_samples as f64 / elapsed;
            }
        }
    }

    /// Record peak memory usage for a device
    pub fn record_memory_usage(&self, device_idx: usize, bytes: usize) {
        let mut metrics = self.metrics.write().unwrap();
        metrics
            .peak_memory_per_device
            .entry(device_idx)
            .and_modify(|peak| {
                if bytes > *peak {
                    *peak = bytes;
                }
            })
            .or_insert(bytes);
    }

    /// Reset metrics
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().unwrap();
        *metrics = DistributedMetrics::default();
    }

    /// Get configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Set gradient accumulation steps
    pub fn set_gradient_accumulation(&mut self, steps: usize) {
        self.config.gradient_accumulation_steps = steps;
    }

    /// Enable/disable mixed precision
    pub fn set_mixed_precision(&mut self, enabled: bool) {
        self.config.mixed_precision = enabled;
    }

    /// Increment step counter
    pub fn increment_step(&self) -> usize {
        let mut counter = self.step_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        *self.step_counter.lock().unwrap()
    }

    /// Check if should sync at current step
    pub fn should_sync(&self) -> bool {
        if self.config.sync_frequency == 0 {
            return true;
        }

        let step = self.current_step();
        step.is_multiple_of(self.config.sync_frequency)
    }
}

impl Default for TrainingCoordinator {
    fn default() -> Self {
        Self::new().expect("Failed to create training coordinator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = TrainingCoordinator::new().unwrap();
        assert!(!coordinator.available_devices().is_empty());
        assert_eq!(coordinator.rank(), 0);
        assert_eq!(coordinator.world_size(), 1);
    }

    #[test]
    fn test_device_detection() {
        let devices = TrainingCoordinator::detect_devices().unwrap();
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_is_primary() {
        let coordinator = TrainingCoordinator::new().unwrap();
        assert!(coordinator.is_primary());
    }

    #[test]
    fn test_step_counter() {
        let coordinator = TrainingCoordinator::new().unwrap();
        assert_eq!(coordinator.current_step(), 0);
        coordinator.increment_step();
        assert_eq!(coordinator.current_step(), 1);
        coordinator.increment_step();
        assert_eq!(coordinator.current_step(), 2);
    }

    #[test]
    fn test_sync_frequency() {
        let mut config = DistributedConfig::default();
        config.sync_frequency = 5;
        let coordinator = TrainingCoordinator::with_config(config).unwrap();

        for i in 0..10 {
            coordinator.increment_step();
            let should_sync = coordinator.should_sync();
            assert_eq!(should_sync, i % 5 == 4);
        }
    }

    #[test]
    fn test_metrics_update() {
        let coordinator = TrainingCoordinator::new().unwrap();
        coordinator.update_samples(100);
        let metrics = coordinator.metrics();
        assert_eq!(metrics.total_samples, 100);
    }

    #[test]
    fn test_memory_recording() {
        let coordinator = TrainingCoordinator::new().unwrap();
        coordinator.record_memory_usage(0, 1024);
        coordinator.record_memory_usage(0, 2048);
        coordinator.record_memory_usage(0, 512);

        let metrics = coordinator.metrics();
        assert_eq!(metrics.peak_memory_per_device.get(&0), Some(&2048));
    }

    #[test]
    fn test_gradient_accumulation() {
        let coordinator = TrainingCoordinator::new().unwrap();
        let device = coordinator.primary_device();

        let grad1 = Tensor::ones((2, 2), candle_core::DType::F32, device).unwrap();
        coordinator.accumulate_gradient("param1", grad1).unwrap();

        let synced = coordinator.sync_gradients().unwrap();
        assert_eq!(synced.len(), 1);
        assert!(synced.contains_key("param1"));
    }

    #[test]
    fn test_multi_gpu_sync() {
        let coordinator = TrainingCoordinator::new().unwrap();
        if coordinator.available_devices().len() < 2 {
            return;
        }

        // Test gradient sync across multiple devices
        let device0 = &coordinator.devices[0];
        let device1 = &coordinator.devices[1];

        let grad0 = Tensor::ones((2, 2), candle_core::DType::F32, device0).unwrap();
        let grad1 = Tensor::ones((2, 2), candle_core::DType::F32, device1)
            .unwrap()
            .affine(2.0, 0.0)
            .unwrap();

        coordinator.accumulate_gradient("param1", grad0).unwrap();
        coordinator.accumulate_gradient("param1", grad1).unwrap();

        let synced = coordinator.sync_gradients().unwrap();
        assert!(synced.contains_key("param1"));

        // Average should be 1.5
        let result = &synced["param1"];
        let values = result.to_vec2::<f32>().unwrap();
        assert!((values[0][0] - 1.5).abs() < 1e-5);
    }
}
