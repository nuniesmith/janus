//! Candle Tensor ↔ GPU Buffer Bridge
//!
//! Provides bidirectional data transfer between Candle tensors (CPU) and
//! GPU buffers managed by the `GpuRuntime`. This bridge enables seamless
//! acceleration of tensor operations: data is uploaded to the GPU for
//! compute kernels, and results are downloaded back into Candle tensors.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
//! │   Candle     │  upload  │   GPU        │  dispatch│   GPU        │
//! │   Tensor     │────────▶│   Buffer     │────────▶│   Kernel     │
//! │   (CPU)      │         │   (VRAM)     │         │   Output     │
//! └──────────────┘         └──────────────┘         └──────┬───────┘
//!        ▲                                                  │
//!        │                  download                        │
//!        └──────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Conversions
//!
//! - `f32` tensors ↔ GPU storage buffers
//! - `f64` tensors → `f32` GPU buffers (with precision warning)
//! - `u32` tensors ↔ GPU uniform/storage buffers
//! - Batch uploads of multiple tensors
//! - Shape-preserving round-trips

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::runtime::{GpuBufferUsage, GpuError, GpuRuntime};

// ---------------------------------------------------------------------------
// Transfer statistics
// ---------------------------------------------------------------------------

/// Accumulated statistics for tensor ↔ GPU transfers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorTransferStats {
    /// Number of tensors uploaded to GPU.
    pub uploads: u64,
    /// Number of tensors downloaded from GPU.
    pub downloads: u64,
    /// Total bytes uploaded.
    pub bytes_uploaded: u64,
    /// Total bytes downloaded.
    pub bytes_downloaded: u64,
    /// Total time spent on uploads.
    pub upload_duration: Duration,
    /// Total time spent on downloads.
    pub download_duration: Duration,
    /// Number of f64→f32 precision truncations.
    pub precision_truncations: u64,
    /// Number of failed transfers.
    pub errors: u64,
    /// Number of batch operations performed.
    pub batch_operations: u64,
}

impl TensorTransferStats {
    /// Record an upload.
    pub fn record_upload(&mut self, bytes: u64, duration: Duration) {
        self.uploads += 1;
        self.bytes_uploaded += bytes;
        self.upload_duration += duration;
    }

    /// Record a download.
    pub fn record_download(&mut self, bytes: u64, duration: Duration) {
        self.downloads += 1;
        self.bytes_downloaded += bytes;
        self.download_duration += duration;
    }

    /// Record a precision truncation event.
    pub fn record_truncation(&mut self) {
        self.precision_truncations += 1;
    }

    /// Record an error.
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Record a batch operation.
    pub fn record_batch(&mut self) {
        self.batch_operations += 1;
    }

    /// Average upload throughput in MB/s.
    pub fn upload_throughput_mbps(&self) -> f64 {
        let secs = self.upload_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.bytes_uploaded as f64 / (1024.0 * 1024.0)) / secs
    }

    /// Average download throughput in MB/s.
    pub fn download_throughput_mbps(&self) -> f64 {
        let secs = self.download_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.bytes_downloaded as f64 / (1024.0 * 1024.0)) / secs
    }

    /// Total bytes transferred (upload + download).
    pub fn total_bytes(&self) -> u64 {
        self.bytes_uploaded + self.bytes_downloaded
    }

    /// Total transfers (upload + download).
    pub fn total_transfers(&self) -> u64 {
        self.uploads + self.downloads
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for TensorTransferStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "uploads={} ({:.2} MB, {:.1} MB/s), downloads={} ({:.2} MB, {:.1} MB/s), \
             truncations={}, errors={}, batches={}",
            self.uploads,
            self.bytes_uploaded as f64 / (1024.0 * 1024.0),
            self.upload_throughput_mbps(),
            self.downloads,
            self.bytes_downloaded as f64 / (1024.0 * 1024.0),
            self.download_throughput_mbps(),
            self.precision_truncations,
            self.errors,
            self.batch_operations,
        )
    }
}

// ---------------------------------------------------------------------------
// Tensor shape descriptor
// ---------------------------------------------------------------------------

/// Describes the shape and element type of a tensor for GPU transfer.
///
/// This is a lightweight descriptor that doesn't hold actual data;
/// it records enough metadata to reconstruct a Candle tensor from
/// raw GPU buffer bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Shape dimensions (e.g., [batch, seq_len, hidden_dim]).
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: TensorDType,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor.
    pub fn new(shape: Vec<usize>, dtype: TensorDType) -> Self {
        Self {
            shape,
            dtype,
            label: None,
        }
    }

    /// Builder: set a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Whether the shape is scalar (0-dim or single element).
    pub fn is_scalar(&self) -> bool {
        self.num_elements() <= 1
    }

    /// Create a descriptor for a 1-D f32 vector.
    pub fn f32_vec(len: usize) -> Self {
        Self::new(vec![len], TensorDType::F32)
    }

    /// Create a descriptor for a 2-D f32 matrix.
    pub fn f32_matrix(rows: usize, cols: usize) -> Self {
        Self::new(vec![rows, cols], TensorDType::F32)
    }

    /// Create a descriptor for a 3-D f32 tensor (batch, rows, cols).
    pub fn f32_tensor3(batch: usize, rows: usize, cols: usize) -> Self {
        Self::new(vec![batch, rows, cols], TensorDType::F32)
    }
}

impl fmt::Display for TensorDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_str: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        write!(
            f,
            "[{}] {:?} ({} bytes)",
            shape_str.join("×"),
            self.dtype,
            self.size_bytes()
        )?;
        if let Some(label) = &self.label {
            write!(f, " '{label}'")?;
        }
        Ok(())
    }
}

/// Supported tensor element data types for GPU transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorDType {
    /// 32-bit float (native GPU type).
    F32,
    /// 64-bit float (will be truncated to f32 for GPU).
    F64,
    /// 32-bit unsigned integer.
    U32,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit float (half precision).
    F16,
    /// 16-bit bfloat.
    BF16,
}

impl TensorDType {
    /// Size of one element in bytes.
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::U32 => 4,
            Self::U8 => 1,
            Self::F16 => 2,
            Self::BF16 => 2,
        }
    }

    /// Whether this type needs conversion before GPU upload.
    pub const fn needs_conversion(&self) -> bool {
        matches!(self, Self::F64 | Self::F16 | Self::BF16)
    }

    /// The GPU-native type this will be converted to.
    pub const fn gpu_native_type(&self) -> TensorDType {
        match self {
            Self::F64 => Self::F32,
            Self::F16 => Self::F32,
            Self::BF16 => Self::F32,
            other => *other,
        }
    }
}

// ---------------------------------------------------------------------------
// Tracked GPU tensor
// ---------------------------------------------------------------------------

/// A tensor that has been uploaded to the GPU.
///
/// Tracks the buffer ID, shape, and metadata needed to download
/// and reconstruct a Candle tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTensor {
    /// GPU buffer ID (from `GpuRuntime`).
    pub buffer_id: u64,
    /// Tensor descriptor (shape, dtype).
    pub descriptor: TensorDescriptor,
    /// Whether this tensor was uploaded with precision truncation.
    pub truncated: bool,
    /// Timestamp when the tensor was uploaded.
    pub uploaded_at: std::time::SystemTime,
}

impl GpuTensor {
    /// Create a new tracked GPU tensor.
    pub fn new(buffer_id: u64, descriptor: TensorDescriptor, truncated: bool) -> Self {
        Self {
            buffer_id,
            descriptor,
            truncated,
            uploaded_at: std::time::SystemTime::now(),
        }
    }

    /// Buffer ID for use with the runtime.
    pub fn id(&self) -> u64 {
        self.buffer_id
    }

    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.descriptor.shape
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.descriptor.num_elements()
    }

    /// Size in bytes on GPU.
    pub fn gpu_size_bytes(&self) -> usize {
        self.num_elements() * self.descriptor.dtype.gpu_native_type().size_bytes()
    }
}

impl fmt::Display for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuTensor(id={}, {})", self.buffer_id, self.descriptor)?;
        if self.truncated {
            write!(f, " [truncated]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tensor Bridge
// ---------------------------------------------------------------------------

/// Bridge between Candle tensors and GPU buffers.
///
/// Manages the lifecycle of tensors on the GPU, including:
/// - Upload (CPU → GPU)
/// - Download (GPU → CPU)
/// - Shape tracking and validation
/// - Batch operations
/// - Transfer statistics
pub struct GpuTensorBridge {
    /// All tracked GPU tensors keyed by buffer ID.
    tensors: HashMap<u64, GpuTensor>,
    /// Named tensor registry for convenience lookups.
    named: HashMap<String, u64>,
    /// Transfer statistics.
    stats: TensorTransferStats,
    /// Maximum number of tensors to track.
    max_tracked: usize,
}

impl fmt::Debug for GpuTensorBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTensorBridge")
            .field("tracked_tensors", &self.tensors.len())
            .field("named_tensors", &self.named.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl GpuTensorBridge {
    // ── Construction ───────────────────────────────────────────────────

    /// Create a new tensor bridge.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            named: HashMap::new(),
            stats: TensorTransferStats::default(),
            max_tracked: 1024,
        }
    }

    /// Create a tensor bridge with a custom maximum tracked tensor count.
    pub fn with_capacity(max_tracked: usize) -> Self {
        Self {
            tensors: HashMap::with_capacity(max_tracked.min(256)),
            named: HashMap::new(),
            stats: TensorTransferStats::default(),
            max_tracked,
        }
    }

    // ── Upload (CPU → GPU) ─────────────────────────────────────────────

    /// Upload an f32 slice to the GPU as a storage buffer.
    ///
    /// Returns a `GpuTensor` handle that tracks the buffer ID and shape.
    pub fn upload_f32(
        &mut self,
        runtime: &mut GpuRuntime,
        data: &[f32],
        shape: Vec<usize>,
        label: Option<&str>,
    ) -> Result<GpuTensor, GpuError> {
        // Validate shape
        let expected_elements: usize = shape.iter().product();
        if expected_elements != data.len() {
            return Err(GpuError::InvalidDimension(format!(
                "Shape {:?} expects {} elements but data has {}",
                shape,
                expected_elements,
                data.len()
            )));
        }

        self.check_capacity()?;

        let start = Instant::now();
        let buffer_id = runtime.upload_f32(data, label)?;
        let duration = start.elapsed();

        let descriptor = TensorDescriptor::new(shape, TensorDType::F32)
            .with_label(label.unwrap_or("unnamed").to_string());

        let gpu_tensor = GpuTensor::new(buffer_id, descriptor, false);

        self.stats.record_upload(data.len() as u64 * 4, duration);

        if let Some(name) = label {
            self.named.insert(name.to_string(), buffer_id);
        }

        self.tensors.insert(buffer_id, gpu_tensor.clone());

        debug!(
            buffer_id,
            elements = data.len(),
            label = label.unwrap_or("none"),
            "Tensor uploaded to GPU"
        );

        Ok(gpu_tensor)
    }

    /// Upload an f64 slice to the GPU, truncating to f32.
    ///
    /// A precision truncation warning is logged and recorded in stats.
    pub fn upload_f64_as_f32(
        &mut self,
        runtime: &mut GpuRuntime,
        data: &[f64],
        shape: Vec<usize>,
        label: Option<&str>,
    ) -> Result<GpuTensor, GpuError> {
        let expected_elements: usize = shape.iter().product();
        if expected_elements != data.len() {
            return Err(GpuError::InvalidDimension(format!(
                "Shape {:?} expects {} elements but data has {}",
                shape,
                expected_elements,
                data.len()
            )));
        }

        warn!(
            elements = data.len(),
            label = label.unwrap_or("none"),
            "Truncating f64 tensor to f32 for GPU upload"
        );
        self.stats.record_truncation();

        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        self.upload_f32(runtime, &f32_data, shape, label)
    }

    /// Upload a u32 slice as a GPU buffer.
    pub fn upload_u32(
        &mut self,
        runtime: &mut GpuRuntime,
        data: &[u32],
        shape: Vec<usize>,
        usage: GpuBufferUsage,
        label: Option<&str>,
    ) -> Result<GpuTensor, GpuError> {
        let expected_elements: usize = shape.iter().product();
        if expected_elements != data.len() {
            return Err(GpuError::InvalidDimension(format!(
                "Shape {:?} expects {} elements but data has {}",
                shape,
                expected_elements,
                data.len()
            )));
        }

        self.check_capacity()?;

        let start = Instant::now();
        let buffer_id = runtime.upload_u32(data, usage, label)?;
        let duration = start.elapsed();

        let descriptor = TensorDescriptor::new(shape, TensorDType::U32)
            .with_label(label.unwrap_or("unnamed").to_string());

        let gpu_tensor = GpuTensor::new(buffer_id, descriptor, false);

        self.stats.record_upload(data.len() as u64 * 4, duration);

        if let Some(name) = label {
            self.named.insert(name.to_string(), buffer_id);
        }

        self.tensors.insert(buffer_id, gpu_tensor.clone());
        Ok(gpu_tensor)
    }

    /// Upload raw bytes as a GPU buffer with a given descriptor.
    pub fn upload_raw(
        &mut self,
        runtime: &mut GpuRuntime,
        data: &[u8],
        descriptor: TensorDescriptor,
        usage: GpuBufferUsage,
    ) -> Result<GpuTensor, GpuError> {
        let expected_bytes = descriptor.size_bytes();
        if data.len() != expected_bytes {
            return Err(GpuError::InvalidDimension(format!(
                "Descriptor expects {} bytes but data has {}",
                expected_bytes,
                data.len()
            )));
        }

        self.check_capacity()?;

        let start = Instant::now();
        let label_str = descriptor.label.as_deref();
        let buffer_id = runtime.upload_buffer(data, usage, label_str)?;
        let duration = start.elapsed();

        let gpu_tensor = GpuTensor::new(buffer_id, descriptor.clone(), false);

        self.stats.record_upload(data.len() as u64, duration);

        if let Some(name) = &descriptor.label {
            self.named.insert(name.clone(), buffer_id);
        }

        self.tensors.insert(buffer_id, gpu_tensor.clone());
        Ok(gpu_tensor)
    }

    // ── Download (GPU → CPU) ───────────────────────────────────────────

    /// Download a GPU tensor as an f32 vector.
    ///
    /// The returned vector is flat (row-major). Use the tensor's
    /// shape information to reshape if needed.
    pub fn download_f32(
        &mut self,
        runtime: &mut GpuRuntime,
        tensor: &GpuTensor,
    ) -> Result<Vec<f32>, GpuError> {
        let start = Instant::now();
        let data = runtime.download_f32(tensor.buffer_id)?;
        let duration = start.elapsed();

        self.stats.record_download(data.len() as u64 * 4, duration);

        debug!(
            buffer_id = tensor.buffer_id,
            elements = data.len(),
            "Tensor downloaded from GPU"
        );

        Ok(data)
    }

    /// Download a GPU tensor as an f32 vector and reshape to 2-D.
    ///
    /// Returns a Vec of rows, each row being a Vec<f32>.
    pub fn download_f32_2d(
        &mut self,
        runtime: &mut GpuRuntime,
        tensor: &GpuTensor,
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        if tensor.descriptor.ndim() < 2 {
            return Err(GpuError::InvalidDimension(
                "Tensor must have at least 2 dimensions for 2D download".into(),
            ));
        }

        let flat = self.download_f32(runtime, tensor)?;
        let cols = *tensor.shape().last().unwrap_or(&1);
        let rows = flat.len() / cols;

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(flat[start..end].to_vec());
        }

        Ok(result)
    }

    /// Download a GPU tensor as raw bytes.
    pub fn download_raw(
        &mut self,
        runtime: &mut GpuRuntime,
        tensor: &GpuTensor,
    ) -> Result<Vec<u8>, GpuError> {
        let start = Instant::now();
        let data = runtime.download_buffer(tensor.buffer_id)?;
        let duration = start.elapsed();

        self.stats.record_download(data.len() as u64, duration);

        Ok(data)
    }

    /// Download and convert a GPU tensor to f64 (upcast from f32).
    pub fn download_as_f64(
        &mut self,
        runtime: &mut GpuRuntime,
        tensor: &GpuTensor,
    ) -> Result<Vec<f64>, GpuError> {
        let f32_data = self.download_f32(runtime, tensor)?;
        Ok(f32_data.iter().map(|&v| v as f64).collect())
    }

    // ── Batch operations ───────────────────────────────────────────────

    /// Upload multiple f32 tensors in batch.
    ///
    /// Returns a vector of `GpuTensor` handles in the same order as
    /// the input data.
    pub fn batch_upload_f32(
        &mut self,
        runtime: &mut GpuRuntime,
        items: &[(&[f32], Vec<usize>, Option<&str>)],
    ) -> Result<Vec<GpuTensor>, GpuError> {
        self.stats.record_batch();

        let mut results = Vec::with_capacity(items.len());
        for (data, shape, label) in items {
            let tensor = self.upload_f32(runtime, data, shape.clone(), *label)?;
            results.push(tensor);
        }

        debug!(count = results.len(), "Batch upload completed");
        Ok(results)
    }

    /// Download multiple GPU tensors as f32 vectors.
    pub fn batch_download_f32(
        &mut self,
        runtime: &mut GpuRuntime,
        tensors: &[&GpuTensor],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        self.stats.record_batch();

        let mut results = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            let data = self.download_f32(runtime, tensor)?;
            results.push(data);
        }

        debug!(count = results.len(), "Batch download completed");
        Ok(results)
    }

    // ── Named tensor management ────────────────────────────────────────

    /// Look up a GPU tensor by its label/name.
    pub fn get_by_name(&self, name: &str) -> Option<&GpuTensor> {
        self.named.get(name).and_then(|id| self.tensors.get(id))
    }

    /// Look up a GPU tensor by buffer ID.
    pub fn get_by_id(&self, id: u64) -> Option<&GpuTensor> {
        self.tensors.get(&id)
    }

    /// Check if a named tensor exists.
    pub fn has_named(&self, name: &str) -> bool {
        self.named.contains_key(name)
    }

    /// List all named tensors.
    pub fn named_tensors(&self) -> Vec<(&str, &GpuTensor)> {
        self.named
            .iter()
            .filter_map(|(name, id)| self.tensors.get(id).map(|t| (name.as_str(), t)))
            .collect()
    }

    /// Assign a name to an existing buffer ID.
    pub fn name_tensor(&mut self, name: impl Into<String>, buffer_id: u64) -> Result<(), GpuError> {
        if !self.tensors.contains_key(&buffer_id) {
            return Err(GpuError::BufferError(format!(
                "Buffer {buffer_id} not tracked by bridge"
            )));
        }
        self.named.insert(name.into(), buffer_id);
        Ok(())
    }

    // ── Release ────────────────────────────────────────────────────────

    /// Release a GPU tensor, returning its buffer to the runtime pool.
    pub fn release(
        &mut self,
        runtime: &mut GpuRuntime,
        tensor: &GpuTensor,
    ) -> Result<(), GpuError> {
        let id = tensor.buffer_id;
        runtime.release_buffer(id)?;
        self.tensors.remove(&id);

        // Remove from named registry if present.
        self.named.retain(|_, v| *v != id);

        debug!(buffer_id = id, "GPU tensor released");
        Ok(())
    }

    /// Release a tensor by name.
    pub fn release_named(&mut self, runtime: &mut GpuRuntime, name: &str) -> Result<(), GpuError> {
        let id = self
            .named
            .get(name)
            .copied()
            .ok_or_else(|| GpuError::BufferError(format!("Named tensor '{name}' not found")))?;

        runtime.release_buffer(id)?;
        self.tensors.remove(&id);
        self.named.remove(name);

        debug!(buffer_id = id, name, "Named GPU tensor released");
        Ok(())
    }

    /// Release all tracked tensors.
    pub fn release_all(&mut self, runtime: &mut GpuRuntime) -> Result<(), GpuError> {
        let ids: Vec<u64> = self.tensors.keys().copied().collect();
        for id in ids {
            runtime.release_buffer(id)?;
        }
        self.tensors.clear();
        self.named.clear();
        info!("All GPU tensors released");
        Ok(())
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// Number of tracked GPU tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are tracked.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total GPU memory used by tracked tensors (in bytes).
    pub fn total_gpu_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.gpu_size_bytes()).sum()
    }

    /// Transfer statistics.
    pub fn stats(&self) -> &TensorTransferStats {
        &self.stats
    }

    /// Mutable access to transfer statistics.
    pub fn stats_mut(&mut self) -> &mut TensorTransferStats {
        &mut self.stats
    }

    /// Reset transfer statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Generate a summary of all tracked tensors.
    pub fn summary(&self) -> Vec<TensorSummary> {
        let mut summaries = Vec::with_capacity(self.tensors.len());
        for (id, tensor) in &self.tensors {
            let name = self
                .named
                .iter()
                .find(|(_, v)| **v == *id)
                .map(|(k, _)| k.clone());

            summaries.push(TensorSummary {
                buffer_id: *id,
                name,
                shape: tensor.descriptor.shape.clone(),
                dtype: tensor.descriptor.dtype,
                gpu_bytes: tensor.gpu_size_bytes(),
                truncated: tensor.truncated,
            });
        }
        summaries.sort_by_key(|s| s.buffer_id);
        summaries
    }

    // ── Internal ───────────────────────────────────────────────────────

    fn check_capacity(&mut self) -> Result<(), GpuError> {
        if self.tensors.len() >= self.max_tracked {
            return Err(GpuError::Internal(format!(
                "Maximum tracked tensors reached ({})",
                self.max_tracked
            )));
        }
        Ok(())
    }
}

impl Default for GpuTensorBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tensor summary
// ---------------------------------------------------------------------------

/// Summary information about a tracked GPU tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSummary {
    /// GPU buffer ID.
    pub buffer_id: u64,
    /// Named alias (if any).
    pub name: Option<String>,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: TensorDType,
    /// GPU memory usage in bytes.
    pub gpu_bytes: usize,
    /// Whether precision was truncated during upload.
    pub truncated: bool,
}

impl fmt::Display for TensorSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_str: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        write!(f, "buf={}", self.buffer_id)?;
        if let Some(name) = &self.name {
            write!(f, " '{name}'")?;
        }
        write!(
            f,
            " [{}] {:?} {} bytes",
            shape_str.join("×"),
            self.dtype,
            self.gpu_bytes
        )?;
        if self.truncated {
            write!(f, " [truncated]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility functions for type conversion
// ---------------------------------------------------------------------------

/// Convert a flat f32 vector into a 2-D vector of rows.
pub fn reshape_2d(data: &[f32], rows: usize, cols: usize) -> Result<Vec<Vec<f32>>, GpuError> {
    if data.len() != rows * cols {
        return Err(GpuError::InvalidDimension(format!(
            "Cannot reshape {} elements into {}×{}",
            data.len(),
            rows,
            cols
        )));
    }

    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        result.push(data[i * cols..(i + 1) * cols].to_vec());
    }
    Ok(result)
}

/// Flatten a 2-D vector into a flat f32 vector (row-major).
pub fn flatten_2d(data: &[Vec<f32>]) -> Vec<f32> {
    let total: usize = data.iter().map(|r| r.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for row in data {
        flat.extend_from_slice(row);
    }
    flat
}

/// Convert a Vec<f64> to Vec<f32>.
pub fn f64_to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&v| v as f32).collect()
}

/// Convert a Vec<f32> to Vec<f64>.
pub fn f32_to_f64(data: &[f32]) -> Vec<f64> {
    data.iter().map(|&v| v as f64).collect()
}

/// Compute strides for a given shape (row-major / C-contiguous).
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a flat index to multi-dimensional indices given a shape.
pub fn flat_to_multi_index(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = flat_idx;
    for &stride in &strides {
        indices.push(remaining / stride);
        remaining %= stride;
    }
    indices
}

/// Convert multi-dimensional indices to a flat index given a shape.
pub fn multi_to_flat_index(indices: &[usize], shape: &[usize]) -> Result<usize, GpuError> {
    if indices.len() != shape.len() {
        return Err(GpuError::InvalidDimension(format!(
            "Index dimensions {} != shape dimensions {}",
            indices.len(),
            shape.len()
        )));
    }

    let strides = compute_strides(shape);
    let mut flat = 0;
    for (i, (&idx, &stride)) in indices.iter().zip(strides.iter()).enumerate() {
        if idx >= shape[i] {
            return Err(GpuError::InvalidDimension(format!(
                "Index {} out of bounds for dimension {} (size {})",
                idx, i, shape[i]
            )));
        }
        flat += idx * stride;
    }
    Ok(flat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::runtime::{GpuConfig, GpuRuntime};

    fn sim_runtime() -> GpuRuntime {
        GpuRuntime::new_simulated(GpuConfig::default())
    }

    // ── TensorTransferStats ────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = TensorTransferStats::default();
        assert_eq!(stats.uploads, 0);
        assert_eq!(stats.downloads, 0);
        assert_eq!(stats.total_bytes(), 0);
        assert_eq!(stats.total_transfers(), 0);
        assert_eq!(stats.upload_throughput_mbps(), 0.0);
        assert_eq!(stats.download_throughput_mbps(), 0.0);
    }

    #[test]
    fn test_stats_record_upload() {
        let mut stats = TensorTransferStats::default();
        stats.record_upload(1024, Duration::from_millis(10));
        assert_eq!(stats.uploads, 1);
        assert_eq!(stats.bytes_uploaded, 1024);
        assert_eq!(stats.total_transfers(), 1);
    }

    #[test]
    fn test_stats_record_download() {
        let mut stats = TensorTransferStats::default();
        stats.record_download(2048, Duration::from_millis(5));
        assert_eq!(stats.downloads, 1);
        assert_eq!(stats.bytes_downloaded, 2048);
    }

    #[test]
    fn test_stats_truncation() {
        let mut stats = TensorTransferStats::default();
        stats.record_truncation();
        stats.record_truncation();
        assert_eq!(stats.precision_truncations, 2);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = TensorTransferStats::default();
        stats.record_upload(100, Duration::from_millis(1));
        stats.record_download(200, Duration::from_millis(2));
        stats.reset();
        assert_eq!(stats.uploads, 0);
        assert_eq!(stats.downloads, 0);
        assert_eq!(stats.total_bytes(), 0);
    }

    #[test]
    fn test_stats_display() {
        let mut stats = TensorTransferStats::default();
        stats.record_upload(1024 * 1024, Duration::from_millis(10));
        let s = format!("{stats}");
        assert!(s.contains("uploads=1"));
        assert!(s.contains("downloads=0"));
    }

    // ── TensorDescriptor ───────────────────────────────────────────────

    #[test]
    fn test_descriptor_new() {
        let desc = TensorDescriptor::new(vec![3, 4], TensorDType::F32);
        assert_eq!(desc.num_elements(), 12);
        assert_eq!(desc.size_bytes(), 48);
        assert_eq!(desc.ndim(), 2);
        assert!(!desc.is_scalar());
    }

    #[test]
    fn test_descriptor_f32_vec() {
        let desc = TensorDescriptor::f32_vec(100);
        assert_eq!(desc.shape, vec![100]);
        assert_eq!(desc.num_elements(), 100);
        assert_eq!(desc.size_bytes(), 400);
    }

    #[test]
    fn test_descriptor_f32_matrix() {
        let desc = TensorDescriptor::f32_matrix(3, 4);
        assert_eq!(desc.shape, vec![3, 4]);
        assert_eq!(desc.num_elements(), 12);
    }

    #[test]
    fn test_descriptor_f32_tensor3() {
        let desc = TensorDescriptor::f32_tensor3(2, 3, 4);
        assert_eq!(desc.shape, vec![2, 3, 4]);
        assert_eq!(desc.num_elements(), 24);
    }

    #[test]
    fn test_descriptor_with_label() {
        let desc = TensorDescriptor::f32_vec(10).with_label("weights");
        assert_eq!(desc.label.as_deref(), Some("weights"));
    }

    #[test]
    fn test_descriptor_scalar() {
        let desc = TensorDescriptor::new(vec![1], TensorDType::F32);
        assert!(desc.is_scalar());

        let desc = TensorDescriptor::new(vec![2], TensorDType::F32);
        assert!(!desc.is_scalar());
    }

    #[test]
    fn test_descriptor_display() {
        let desc = TensorDescriptor::f32_matrix(3, 4).with_label("W");
        let s = format!("{desc}");
        assert!(s.contains("3×4"));
        assert!(s.contains("48 bytes"));
        assert!(s.contains("'W'"));
    }

    // ── TensorDType ────────────────────────────────────────────────────

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(TensorDType::F32.size_bytes(), 4);
        assert_eq!(TensorDType::F64.size_bytes(), 8);
        assert_eq!(TensorDType::U32.size_bytes(), 4);
        assert_eq!(TensorDType::U8.size_bytes(), 1);
        assert_eq!(TensorDType::F16.size_bytes(), 2);
        assert_eq!(TensorDType::BF16.size_bytes(), 2);
    }

    #[test]
    fn test_dtype_needs_conversion() {
        assert!(!TensorDType::F32.needs_conversion());
        assert!(TensorDType::F64.needs_conversion());
        assert!(!TensorDType::U32.needs_conversion());
        assert!(!TensorDType::U8.needs_conversion());
        assert!(TensorDType::F16.needs_conversion());
        assert!(TensorDType::BF16.needs_conversion());
    }

    #[test]
    fn test_dtype_gpu_native() {
        assert_eq!(TensorDType::F32.gpu_native_type(), TensorDType::F32);
        assert_eq!(TensorDType::F64.gpu_native_type(), TensorDType::F32);
        assert_eq!(TensorDType::U32.gpu_native_type(), TensorDType::U32);
        assert_eq!(TensorDType::F16.gpu_native_type(), TensorDType::F32);
        assert_eq!(TensorDType::BF16.gpu_native_type(), TensorDType::F32);
    }

    // ── GpuTensor ──────────────────────────────────────────────────────

    #[test]
    fn test_gpu_tensor_new() {
        let desc = TensorDescriptor::f32_matrix(3, 4);
        let tensor = GpuTensor::new(42, desc, false);
        assert_eq!(tensor.id(), 42);
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.num_elements(), 12);
        assert_eq!(tensor.gpu_size_bytes(), 48);
        assert!(!tensor.truncated);
    }

    #[test]
    fn test_gpu_tensor_truncated() {
        let desc = TensorDescriptor::new(vec![10], TensorDType::F64);
        let tensor = GpuTensor::new(1, desc, true);
        assert!(tensor.truncated);
        // GPU uses f32 even though original was f64
        assert_eq!(tensor.gpu_size_bytes(), 10 * 4);
    }

    #[test]
    fn test_gpu_tensor_display() {
        let desc = TensorDescriptor::f32_vec(5).with_label("bias");
        let tensor = GpuTensor::new(7, desc, false);
        let s = format!("{tensor}");
        assert!(s.contains("id=7"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_gpu_tensor_display_truncated() {
        let desc = TensorDescriptor::f32_vec(5);
        let tensor = GpuTensor::new(7, desc, true);
        let s = format!("{tensor}");
        assert!(s.contains("[truncated]"));
    }

    // ── GpuTensorBridge ────────────────────────────────────────────────

    #[test]
    fn test_bridge_new() {
        let bridge = GpuTensorBridge::new();
        assert!(bridge.is_empty());
        assert_eq!(bridge.len(), 0);
        assert_eq!(bridge.total_gpu_bytes(), 0);
    }

    #[test]
    fn test_bridge_upload_f32() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = bridge
            .upload_f32(&mut rt, &data, vec![2, 3], Some("W"))
            .unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.num_elements(), 6);
        assert!(!tensor.truncated);
        assert_eq!(bridge.len(), 1);
        assert_eq!(bridge.total_gpu_bytes(), 24);
    }

    #[test]
    fn test_bridge_upload_f32_shape_mismatch() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0, 3.0];
        let result = bridge.upload_f32(&mut rt, &data, vec![2, 3], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_bridge_upload_download_roundtrip() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = bridge
            .upload_f32(&mut rt, &data, vec![4], Some("test"))
            .unwrap();

        let downloaded = bridge.download_f32(&mut rt, &tensor).unwrap();
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_bridge_upload_f64_as_f32() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f64, 2.5, std::f64::consts::PI, 4.0];
        let tensor = bridge
            .upload_f64_as_f32(&mut rt, &data, vec![4], Some("f64test"))
            .unwrap();

        assert!(!tensor.truncated); // truncated flag is on the inner upload

        let downloaded = bridge.download_f32(&mut rt, &tensor).unwrap();
        assert_eq!(downloaded.len(), 4);
        assert!((downloaded[0] - 1.0).abs() < 1e-6);
        assert!((downloaded[2] - std::f64::consts::PI as f32).abs() < 1e-4);

        assert!(bridge.stats().precision_truncations >= 1);
    }

    #[test]
    fn test_bridge_upload_u32() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![10u32, 20, 30];
        let tensor = bridge
            .upload_u32(
                &mut rt,
                &data,
                vec![3],
                GpuBufferUsage::Storage,
                Some("indices"),
            )
            .unwrap();

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(bridge.len(), 1);
    }

    #[test]
    fn test_bridge_download_f32_2d() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = bridge.upload_f32(&mut rt, &data, vec![2, 3], None).unwrap();

        let result = bridge.download_f32_2d(&mut rt, &tensor).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_bridge_download_as_f64() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.5f32, 2.5, 3.5];
        let tensor = bridge.upload_f32(&mut rt, &data, vec![3], None).unwrap();

        let f64_data = bridge.download_as_f64(&mut rt, &tensor).unwrap();
        assert_eq!(f64_data.len(), 3);
        assert!((f64_data[0] - 1.5).abs() < 1e-6);
        assert!((f64_data[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_bridge_named_tensors() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0];
        bridge
            .upload_f32(&mut rt, &data, vec![2], Some("weights"))
            .unwrap();
        bridge
            .upload_f32(&mut rt, &data, vec![2], Some("bias"))
            .unwrap();

        assert!(bridge.has_named("weights"));
        assert!(bridge.has_named("bias"));
        assert!(!bridge.has_named("nonexistent"));

        let w = bridge.get_by_name("weights").unwrap();
        assert_eq!(w.shape(), &[2]);

        let named = bridge.named_tensors();
        assert_eq!(named.len(), 2);
    }

    #[test]
    fn test_bridge_get_by_id() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32];
        let tensor = bridge.upload_f32(&mut rt, &data, vec![1], None).unwrap();

        let found = bridge.get_by_id(tensor.buffer_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().shape(), &[1]);
    }

    #[test]
    fn test_bridge_name_tensor() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32];
        let tensor = bridge.upload_f32(&mut rt, &data, vec![1], None).unwrap();

        bridge.name_tensor("my_tensor", tensor.buffer_id).unwrap();
        assert!(bridge.has_named("my_tensor"));
    }

    #[test]
    fn test_bridge_name_tensor_invalid_id() {
        let mut bridge = GpuTensorBridge::new();
        let result = bridge.name_tensor("bad", 9999);
        assert!(result.is_err());
    }

    #[test]
    fn test_bridge_release() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0];
        let tensor = bridge
            .upload_f32(&mut rt, &data, vec![2], Some("to_release"))
            .unwrap();

        assert_eq!(bridge.len(), 1);
        bridge.release(&mut rt, &tensor).unwrap();
        assert_eq!(bridge.len(), 0);
        assert!(!bridge.has_named("to_release"));
    }

    #[test]
    fn test_bridge_release_named() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32];
        bridge
            .upload_f32(&mut rt, &data, vec![1], Some("named"))
            .unwrap();

        assert!(bridge.has_named("named"));
        bridge.release_named(&mut rt, "named").unwrap();
        assert!(!bridge.has_named("named"));
        assert_eq!(bridge.len(), 0);
    }

    #[test]
    fn test_bridge_release_named_missing() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();
        let result = bridge.release_named(&mut rt, "missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_bridge_release_all() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        bridge
            .upload_f32(&mut rt, &[1.0], vec![1], Some("a"))
            .unwrap();
        bridge
            .upload_f32(&mut rt, &[2.0], vec![1], Some("b"))
            .unwrap();
        bridge.upload_f32(&mut rt, &[3.0], vec![1], None).unwrap();

        assert_eq!(bridge.len(), 3);
        bridge.release_all(&mut rt).unwrap();
        assert_eq!(bridge.len(), 0);
        assert!(bridge.is_empty());
    }

    #[test]
    fn test_bridge_batch_upload() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let d1 = vec![1.0f32, 2.0];
        let d2 = vec![3.0f32, 4.0, 5.0];
        let d3 = vec![6.0f32];

        let items: Vec<(&[f32], Vec<usize>, Option<&str>)> = vec![
            (&d1, vec![2], Some("a")),
            (&d2, vec![3], Some("b")),
            (&d3, vec![1], Some("c")),
        ];

        let tensors = bridge.batch_upload_f32(&mut rt, &items).unwrap();
        assert_eq!(tensors.len(), 3);
        assert_eq!(bridge.len(), 3);
        assert!(bridge.stats().batch_operations >= 1);
    }

    #[test]
    fn test_bridge_batch_download() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let d1 = vec![1.0f32, 2.0];
        let d2 = vec![3.0f32, 4.0, 5.0];

        let t1 = bridge.upload_f32(&mut rt, &d1, vec![2], None).unwrap();
        let t2 = bridge.upload_f32(&mut rt, &d2, vec![3], None).unwrap();

        let results = bridge.batch_download_f32(&mut rt, &[&t1, &t2]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], d1);
        assert_eq!(results[1], d2);
    }

    #[test]
    fn test_bridge_summary() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        bridge
            .upload_f32(&mut rt, &[1.0, 2.0], vec![2], Some("w"))
            .unwrap();
        bridge.upload_f32(&mut rt, &[3.0], vec![1], None).unwrap();

        let summary = bridge.summary();
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_bridge_stats() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32; 100];
        let tensor = bridge.upload_f32(&mut rt, &data, vec![100], None).unwrap();

        bridge.download_f32(&mut rt, &tensor).unwrap();

        let stats = bridge.stats();
        assert_eq!(stats.uploads, 1);
        assert_eq!(stats.downloads, 1);
        assert!(stats.bytes_uploaded > 0);
        assert!(stats.bytes_downloaded > 0);
    }

    #[test]
    fn test_bridge_reset_stats() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        bridge.upload_f32(&mut rt, &[1.0], vec![1], None).unwrap();
        assert!(bridge.stats().uploads > 0);

        bridge.reset_stats();
        assert_eq!(bridge.stats().uploads, 0);
    }

    #[test]
    fn test_bridge_capacity_limit() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::with_capacity(2);

        bridge.upload_f32(&mut rt, &[1.0], vec![1], None).unwrap();
        bridge.upload_f32(&mut rt, &[2.0], vec![1], None).unwrap();

        // Third should fail.
        let result = bridge.upload_f32(&mut rt, &[3.0], vec![1], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_bridge_total_gpu_bytes() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        bridge
            .upload_f32(&mut rt, &[1.0, 2.0], vec![2], None)
            .unwrap();
        bridge
            .upload_f32(&mut rt, &[3.0, 4.0, 5.0], vec![3], None)
            .unwrap();

        // 2*4 + 3*4 = 20 bytes
        assert_eq!(bridge.total_gpu_bytes(), 20);
    }

    // ── Utility function tests ─────────────────────────────────────────

    #[test]
    fn test_reshape_2d() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = reshape_2d(&data, 2, 3).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_2d_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = reshape_2d(&data, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_flatten_2d() {
        let data = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let flat = flatten_2d(&data);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flatten_2d_empty() {
        let data: Vec<Vec<f32>> = vec![];
        let flat = flatten_2d(&data);
        assert!(flat.is_empty());
    }

    #[test]
    fn test_f64_to_f32_conversion() {
        let data = vec![1.0f64, 2.5, -std::f64::consts::PI];
        let result = f64_to_f32(&data);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.5).abs() < 1e-6);
        assert!((result[2] - (-std::f64::consts::PI as f32)).abs() < 1e-4);
    }

    #[test]
    fn test_f32_to_f64_conversion() {
        let data = vec![1.5f32, 2.5, 3.5];
        let result = f32_to_f64(&data);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_strides() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);

        let strides = compute_strides(&[5]);
        assert_eq!(strides, vec![1]);

        let strides = compute_strides(&[]);
        assert!(strides.is_empty());
    }

    #[test]
    fn test_flat_to_multi_index() {
        let shape = [2, 3, 4];

        let idx = flat_to_multi_index(0, &shape);
        assert_eq!(idx, vec![0, 0, 0]);

        let idx = flat_to_multi_index(5, &shape);
        assert_eq!(idx, vec![0, 1, 1]);

        let idx = flat_to_multi_index(23, &shape);
        assert_eq!(idx, vec![1, 2, 3]);
    }

    #[test]
    fn test_multi_to_flat_index() {
        let shape = [2, 3, 4];

        let flat = multi_to_flat_index(&[0, 0, 0], &shape).unwrap();
        assert_eq!(flat, 0);

        let flat = multi_to_flat_index(&[0, 1, 1], &shape).unwrap();
        assert_eq!(flat, 5);

        let flat = multi_to_flat_index(&[1, 2, 3], &shape).unwrap();
        assert_eq!(flat, 23);
    }

    #[test]
    fn test_multi_to_flat_index_out_of_bounds() {
        let shape = [2, 3];
        let result = multi_to_flat_index(&[2, 0], &shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_to_flat_index_wrong_dims() {
        let shape = [2, 3];
        let result = multi_to_flat_index(&[0, 0, 0], &shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_multi_roundtrip() {
        let shape = [3, 4, 5];
        for flat in 0..60 {
            let multi = flat_to_multi_index(flat, &shape);
            let recovered = multi_to_flat_index(&multi, &shape).unwrap();
            assert_eq!(flat, recovered, "Roundtrip failed for flat={flat}");
        }
    }

    #[test]
    fn test_tensor_summary_display() {
        let summary = TensorSummary {
            buffer_id: 42,
            name: Some("weights".into()),
            shape: vec![3, 4],
            dtype: TensorDType::F32,
            gpu_bytes: 48,
            truncated: false,
        };
        let s = format!("{summary}");
        assert!(s.contains("buf=42"));
        assert!(s.contains("'weights'"));
        assert!(s.contains("3×4"));
        assert!(s.contains("48 bytes"));
    }

    #[test]
    fn test_tensor_summary_display_truncated() {
        let summary = TensorSummary {
            buffer_id: 1,
            name: None,
            shape: vec![10],
            dtype: TensorDType::F32,
            gpu_bytes: 40,
            truncated: true,
        };
        let s = format!("{summary}");
        assert!(s.contains("[truncated]"));
        assert!(!s.contains("'"));
    }

    #[test]
    fn test_bridge_upload_raw() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data: Vec<u8> = vec![0, 0, 128, 63, 0, 0, 0, 64]; // 1.0f32, 2.0f32 in LE
        let desc = TensorDescriptor::new(vec![2], TensorDType::F32).with_label("raw_test");

        let tensor = bridge
            .upload_raw(&mut rt, &data, desc, GpuBufferUsage::Storage)
            .unwrap();

        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(bridge.len(), 1);
    }

    #[test]
    fn test_bridge_upload_raw_size_mismatch() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data: Vec<u8> = vec![0, 1, 2]; // 3 bytes, but shape says 2 f32 = 8 bytes
        let desc = TensorDescriptor::new(vec![2], TensorDType::F32);

        let result = bridge.upload_raw(&mut rt, &data, desc, GpuBufferUsage::Storage);
        assert!(result.is_err());
    }

    #[test]
    fn test_bridge_download_raw() {
        let mut rt = sim_runtime();
        let mut bridge = GpuTensorBridge::new();

        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = bridge.upload_f32(&mut rt, &data, vec![3], None).unwrap();

        let raw = bridge.download_raw(&mut rt, &tensor).unwrap();
        assert_eq!(raw.len(), 12); // 3 * 4 bytes
    }
}
