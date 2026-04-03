//! wgpu GPU Runtime — Adapter Discovery, Device/Queue, Buffer Management
//!
//! This module provides the core GPU runtime infrastructure built on wgpu.
//! It handles device enumeration, adapter selection, buffer allocation and
//! transfer, and exposes a high-level API for dispatching compute work.
//!
//! When the `gpu` feature is not enabled (or no GPU adapter is found at
//! runtime), all operations gracefully degrade to `GpuStatus::Unavailable`
//! and callers can fall back to CPU paths.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the GPU runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuError {
    /// No suitable GPU adapter was found.
    NoAdapter,
    /// The adapter could not provide a device + queue.
    DeviceCreationFailed(String),
    /// A shader failed to compile.
    ShaderCompilationFailed(String),
    /// A compute pipeline could not be created.
    PipelineCreationFailed(String),
    /// A GPU buffer operation failed.
    BufferError(String),
    /// Data transfer between CPU and GPU failed.
    TransferError(String),
    /// The requested operation requires a dimension that is invalid.
    InvalidDimension(String),
    /// A dispatch / submission error.
    DispatchError(String),
    /// The GPU runtime has not been initialised.
    NotInitialised,
    /// The GPU feature is not compiled in.
    FeatureDisabled,
    /// Generic internal error.
    Internal(String),
    /// Timeout waiting for GPU work to complete.
    Timeout(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "GPU: no suitable adapter found"),
            Self::DeviceCreationFailed(e) => write!(f, "GPU: device creation failed: {e}"),
            Self::ShaderCompilationFailed(e) => write!(f, "GPU: shader compilation failed: {e}"),
            Self::PipelineCreationFailed(e) => write!(f, "GPU: pipeline creation failed: {e}"),
            Self::BufferError(e) => write!(f, "GPU: buffer error: {e}"),
            Self::TransferError(e) => write!(f, "GPU: transfer error: {e}"),
            Self::InvalidDimension(e) => write!(f, "GPU: invalid dimension: {e}"),
            Self::DispatchError(e) => write!(f, "GPU: dispatch error: {e}"),
            Self::NotInitialised => write!(f, "GPU: runtime not initialised"),
            Self::FeatureDisabled => write!(f, "GPU: feature not compiled"),
            Self::Internal(e) => write!(f, "GPU: internal error: {e}"),
            Self::Timeout(e) => write!(f, "GPU: timeout: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GPU runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Preferred GPU backend (Vulkan, Metal, DX12, …). `None` = auto-detect.
    pub preferred_backend: Option<GpuBackend>,

    /// Maximum GPU memory budget in bytes (soft limit for buffer pool).
    /// Default: 512 MiB.
    pub max_memory_bytes: u64,

    /// Whether to enable shader validation / debug labels.
    pub debug_mode: bool,

    /// Power preference when selecting an adapter.
    pub power_preference: GpuPowerPreference,

    /// Maximum number of buffers to keep in the pool for reuse.
    pub buffer_pool_capacity: usize,

    /// Default workgroup size for 1-D dispatches.
    pub default_workgroup_size: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: None,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MiB
            debug_mode: false,
            power_preference: GpuPowerPreference::HighPerformance,
            buffer_pool_capacity: 64,
            default_workgroup_size: 256,
        }
    }
}

impl GpuConfig {
    /// Builder-style: set preferred backend.
    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Builder-style: set memory budget.
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Builder-style: enable debug mode.
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug_mode = debug;
        self
    }

    /// Builder-style: set power preference.
    pub fn with_power_preference(mut self, pref: GpuPowerPreference) -> Self {
        self.power_preference = pref;
        self
    }

    /// Builder-style: set default workgroup size.
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.default_workgroup_size = size;
        self
    }
}

/// GPU backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Dx11,
    Gl,
    BrowserWebGpu,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::Dx12 => write!(f, "DX12"),
            Self::Dx11 => write!(f, "DX11"),
            Self::Gl => write!(f, "OpenGL"),
            Self::BrowserWebGpu => write!(f, "WebGPU"),
        }
    }
}

/// Power preference for adapter selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuPowerPreference {
    /// Prefer low-power integrated GPU.
    LowPower,
    /// Prefer high-performance discrete GPU.
    HighPerformance,
}

// ---------------------------------------------------------------------------
// Device Info
// ---------------------------------------------------------------------------

/// Information about a discovered GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Vendor name or PCI ID string.
    pub vendor: String,
    /// Backend in use.
    pub backend: String,
    /// Device type (discrete, integrated, virtual, …).
    pub device_type: String,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,
    /// Maximum compute workgroup size (x).
    pub max_workgroup_size_x: u32,
    /// Maximum compute workgroup size (y).
    pub max_workgroup_size_y: u32,
    /// Maximum compute workgroup size (z).
    pub max_workgroup_size_z: u32,
    /// Maximum total invocations per workgroup.
    pub max_workgroup_invocations: u32,
    /// Maximum dispatch count per dimension.
    pub max_dispatch_x: u32,
    /// Maximum storage buffers per shader stage.
    pub max_storage_buffers: u32,
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {}, max_buf={}MiB, max_wg={}x{}x{}/{})",
            self.name,
            self.backend,
            self.device_type,
            self.max_buffer_size / (1024 * 1024),
            self.max_workgroup_size_x,
            self.max_workgroup_size_y,
            self.max_workgroup_size_z,
            self.max_workgroup_invocations,
        )
    }
}

// ---------------------------------------------------------------------------
// GPU Status
// ---------------------------------------------------------------------------

/// Current status of the GPU runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuStatus {
    /// GPU is available and ready.
    Ready,
    /// GPU feature is compiled but no adapter was found.
    Unavailable,
    /// Runtime has not been initialised yet.
    Uninitialised,
    /// An error occurred during initialisation.
    Error(String),
}

impl fmt::Display for GpuStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "GPU Ready"),
            Self::Unavailable => write!(f, "GPU Unavailable"),
            Self::Uninitialised => write!(f, "GPU Uninitialised"),
            Self::Error(e) => write!(f, "GPU Error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer abstraction
// ---------------------------------------------------------------------------

/// Usage flags for GPU buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBufferUsage {
    /// Storage buffer (read/write from shaders).
    Storage,
    /// Uniform buffer (read-only constants).
    Uniform,
    /// Buffer that can be mapped for CPU read-back.
    MapRead,
    /// Buffer that can be mapped for CPU write (staging).
    MapWrite,
    /// Combined storage + copy source (for readback).
    StorageReadback,
    /// Combined staging + copy destination.
    StagingUpload,
}

/// A tracked GPU buffer.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Unique identifier for this buffer.
    pub id: u64,
    /// Size in bytes.
    pub size: u64,
    /// Usage flags.
    pub usage: GpuBufferUsage,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// The raw bytes currently held (CPU-side shadow copy for staging).
    /// For storage-only buffers this may be empty after upload.
    shadow: Vec<u8>,
}

impl GpuBuffer {
    /// Create a new buffer descriptor.
    fn new(id: u64, size: u64, usage: GpuBufferUsage, label: Option<String>) -> Self {
        Self {
            id,
            size,
            usage,
            label,
            shadow: Vec::new(),
        }
    }

    /// Create a buffer from data (keeps a shadow copy).
    fn from_data(id: u64, data: &[u8], usage: GpuBufferUsage, label: Option<String>) -> Self {
        Self {
            id,
            size: data.len() as u64,
            usage,
            label,
            shadow: data.to_vec(),
        }
    }

    /// Return the shadow copy (CPU-side data).
    pub fn shadow_data(&self) -> &[u8] {
        &self.shadow
    }

    /// Interpret the shadow data as a slice of f32.
    pub fn as_f32_slice(&self) -> &[f32] {
        if self.shadow.is_empty() {
            return &[];
        }
        let ptr = self.shadow.as_ptr() as *const f32;
        let len = self.shadow.len() / std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

// ---------------------------------------------------------------------------
// Buffer Pool — simple recycling allocator
// ---------------------------------------------------------------------------

/// A pool of reusable GPU buffers to avoid repeated allocation overhead.
#[derive(Debug)]
struct BufferPool {
    /// Available (unused) buffers keyed by size bucket.
    available: HashMap<u64, Vec<GpuBuffer>>,
    /// Maximum number of buffers to keep idle.
    capacity: usize,
    /// Total number of buffers in the pool.
    total_count: usize,
}

impl BufferPool {
    fn new(capacity: usize) -> Self {
        Self {
            available: HashMap::new(),
            capacity,
            total_count: 0,
        }
    }

    /// Try to acquire a buffer of at least `size` bytes.
    fn acquire(&mut self, size: u64) -> Option<GpuBuffer> {
        // Round up to nearest 256-byte alignment (common GPU requirement).
        let bucket = Self::bucket(size);
        if let Some(pool) = self.available.get_mut(&bucket) {
            if let Some(buf) = pool.pop() {
                self.total_count -= 1;
                return Some(buf);
            }
        }
        None
    }

    /// Return a buffer to the pool for reuse.
    fn release(&mut self, mut buf: GpuBuffer) {
        if self.total_count >= self.capacity {
            // Pool is full — just drop it.
            return;
        }
        let bucket = Self::bucket(buf.size);
        buf.shadow.clear();
        self.available.entry(bucket).or_default().push(buf);
        self.total_count += 1;
    }

    /// Clear the pool.
    fn clear(&mut self) {
        self.available.clear();
        self.total_count = 0;
    }

    /// Compute the size bucket (round up to 256-byte alignment).
    fn bucket(size: u64) -> u64 {
        (size + 255) & !255
    }
}

// ---------------------------------------------------------------------------
// Compute dispatch descriptor
// ---------------------------------------------------------------------------

/// Describes a single compute dispatch.
#[derive(Debug, Clone)]
pub struct ComputeDispatch {
    /// Number of workgroups in X.
    pub workgroups_x: u32,
    /// Number of workgroups in Y.
    pub workgroups_y: u32,
    /// Number of workgroups in Z.
    pub workgroups_z: u32,
}

impl ComputeDispatch {
    /// 1-D dispatch with given number of workgroups.
    pub fn new_1d(x: u32) -> Self {
        Self {
            workgroups_x: x,
            workgroups_y: 1,
            workgroups_z: 1,
        }
    }

    /// 2-D dispatch.
    pub fn new_2d(x: u32, y: u32) -> Self {
        Self {
            workgroups_x: x,
            workgroups_y: y,
            workgroups_z: 1,
        }
    }

    /// 3-D dispatch.
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self {
            workgroups_x: x,
            workgroups_y: y,
            workgroups_z: z,
        }
    }

    /// Compute the number of workgroups required to cover `n` elements
    /// with the given workgroup size.
    pub fn for_elements(n: u32, workgroup_size: u32) -> Self {
        let groups = n.div_ceil(workgroup_size);
        Self::new_1d(groups)
    }
}

// ---------------------------------------------------------------------------
// Runtime stats
// ---------------------------------------------------------------------------

/// Accumulated statistics for the GPU runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRuntimeStats {
    /// Total number of dispatches submitted.
    pub total_dispatches: u64,
    /// Total number of buffers allocated.
    pub total_buffers_allocated: u64,
    /// Total bytes uploaded to GPU.
    pub total_bytes_uploaded: u64,
    /// Total bytes downloaded from GPU.
    pub total_bytes_downloaded: u64,
    /// Total bytes currently allocated.
    pub current_allocated_bytes: u64,
    /// Number of buffers currently alive.
    pub current_buffer_count: u64,
    /// Number of buffer pool hits (reused).
    pub pool_hits: u64,
    /// Number of buffer pool misses (new allocation).
    pub pool_misses: u64,
}

impl Default for GpuRuntimeStats {
    fn default() -> Self {
        Self {
            total_dispatches: 0,
            total_buffers_allocated: 0,
            total_bytes_uploaded: 0,
            total_bytes_downloaded: 0,
            current_allocated_bytes: 0,
            current_buffer_count: 0,
            pool_hits: 0,
            pool_misses: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU Runtime
// ---------------------------------------------------------------------------

/// The main GPU runtime.
///
/// This is a CPU-side simulation of a wgpu-backed compute runtime.
/// When actual wgpu is wired in, the `device`, `queue`, and `adapter`
/// fields become real wgpu handles; the public API stays the same.
///
/// The simulation mode allows all upstream code (kernels, tensor bridge,
/// tests) to be developed and verified without requiring a physical GPU.
pub struct GpuRuntime {
    /// Configuration used to create this runtime.
    config: GpuConfig,
    /// Device info (populated on init).
    device_info: Option<GpuDeviceInfo>,
    /// Current status.
    status: GpuStatus,
    /// Monotonically increasing buffer ID.
    next_buffer_id: AtomicU64,
    /// Buffer pool for recycling.
    buffer_pool: BufferPool,
    /// All live buffers (id → buffer).
    live_buffers: HashMap<u64, GpuBuffer>,
    /// Runtime statistics.
    stats: GpuRuntimeStats,
}

impl fmt::Debug for GpuRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuRuntime")
            .field("status", &self.status)
            .field("device_info", &self.device_info)
            .field("live_buffers", &self.live_buffers.len())
            .finish()
    }
}

impl GpuRuntime {
    // ── Construction ───────────────────────────────────────────────────

    /// Create a new GPU runtime with the given configuration.
    ///
    /// This performs adapter discovery and device creation. If no GPU
    /// is available the runtime enters `GpuStatus::Unavailable` and all
    /// compute operations will return `GpuError::NoAdapter`.
    pub async fn new(config: GpuConfig) -> Result<Self, GpuError> {
        info!("Initialising GPU runtime…");

        let mut runtime = Self {
            config: config.clone(),
            device_info: None,
            status: GpuStatus::Uninitialised,
            next_buffer_id: AtomicU64::new(1),
            buffer_pool: BufferPool::new(config.buffer_pool_capacity),
            live_buffers: HashMap::new(),
            stats: GpuRuntimeStats::default(),
        };

        // Attempt adapter discovery.
        match runtime.discover_adapter().await {
            Ok(info) => {
                info!("GPU adapter found: {}", info);
                runtime.device_info = Some(info);
                runtime.status = GpuStatus::Ready;
            }
            Err(e) => {
                warn!("No GPU adapter found, falling back to CPU: {e}");
                runtime.status = GpuStatus::Unavailable;
            }
        }

        Ok(runtime)
    }

    /// Create a runtime in CPU-simulation mode (no real GPU).
    /// Useful for testing, CI, and environments without a GPU.
    pub fn new_simulated(config: GpuConfig) -> Self {
        let info = GpuDeviceInfo {
            name: "Simulated GPU (CPU fallback)".into(),
            vendor: "JANUS".into(),
            backend: "CPU-Sim".into(),
            device_type: "Virtual".into(),
            max_buffer_size: config.max_memory_bytes,
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_workgroup_size_z: 64,
            max_workgroup_invocations: 256,
            max_dispatch_x: 65535,
            max_storage_buffers: 8,
        };

        info!("GPU runtime initialised in SIMULATED mode: {}", info);

        Self {
            config,
            device_info: Some(info),
            status: GpuStatus::Ready,
            next_buffer_id: AtomicU64::new(1),
            buffer_pool: BufferPool::new(64),
            live_buffers: HashMap::new(),
            stats: GpuRuntimeStats::default(),
        }
    }

    // ── Adapter discovery ──────────────────────────────────────────────

    /// Attempt to discover a GPU adapter.
    ///
    /// This is a scaffolded implementation. When `wgpu` is added as a
    /// real dependency the body of this function becomes:
    ///
    /// ```rust,ignore
    /// let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    ///     backends: wgpu::Backends::all(),
    ///     ..Default::default()
    /// });
    /// let adapter = instance
    ///     .request_adapter(&wgpu::RequestAdapterOptions {
    ///         power_preference: match self.config.power_preference {
    ///             GpuPowerPreference::LowPower => wgpu::PowerPreference::LowPower,
    ///             GpuPowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
    ///         },
    ///         compatible_surface: None,
    ///         force_fallback_adapter: false,
    ///     })
    ///     .await
    ///     .ok_or(GpuError::NoAdapter)?;
    ///
    /// let info = adapter.get_info();
    /// let limits = adapter.limits();
    /// ...
    /// ```
    async fn discover_adapter(&self) -> Result<GpuDeviceInfo, GpuError> {
        debug!(
            "Probing GPU adapters (power_pref={:?})…",
            self.config.power_preference
        );

        // ── Attempt real wgpu discovery ────────────────────────────────
        //
        // For now we do a best-effort probe. On headless CI or machines
        // without a GPU this will fail gracefully and the runtime will
        // use simulated mode.
        //
        // The actual wgpu integration is prepared but not yet linked as
        // a hard dependency. When wgpu is added to Cargo.toml, replace
        // this block with real adapter enumeration.

        // Simulate discovery failure when running without wgpu dependency.
        // This causes the runtime to enter Unavailable mode, which is the
        // correct behaviour until the wgpu crate is linked.
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::NoAdapter)
        }

        #[cfg(feature = "cuda")]
        {
            // When the `gpu` feature is enabled, we attempt real discovery.
            // This path will be filled in when wgpu is added as a dependency.
            Err(GpuError::NoAdapter)
        }
    }

    // ── Status & info ──────────────────────────────────────────────────

    /// Current runtime status.
    pub fn status(&self) -> &GpuStatus {
        &self.status
    }

    /// Whether the runtime is ready for compute work.
    pub fn is_ready(&self) -> bool {
        self.status == GpuStatus::Ready
    }

    /// Device information (if available).
    pub fn device_info(&self) -> Option<&GpuDeviceInfo> {
        self.device_info.as_ref()
    }

    /// Human-readable device name.
    pub fn device_name(&self) -> &str {
        self.device_info
            .as_ref()
            .map(|i| i.name.as_str())
            .unwrap_or("N/A")
    }

    /// The config this runtime was created with.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Accumulated runtime statistics.
    pub fn stats(&self) -> &GpuRuntimeStats {
        &self.stats
    }

    /// Default workgroup size from config.
    pub fn default_workgroup_size(&self) -> u32 {
        self.config.default_workgroup_size
    }

    // ── Buffer management ──────────────────────────────────────────────

    /// Allocate a new GPU buffer of the given size and usage.
    pub fn allocate_buffer(
        &mut self,
        size: u64,
        usage: GpuBufferUsage,
        label: Option<&str>,
    ) -> Result<u64, GpuError> {
        self.check_ready()?;
        self.check_memory(size)?;

        // Try to reuse a pooled buffer.
        if let Some(mut buf) = self.buffer_pool.acquire(size) {
            let id = self.next_id();
            buf.id = id;
            buf.usage = usage;
            buf.label = label.map(|s| s.to_string());
            buf.size = size;
            self.stats.pool_hits += 1;
            debug!(id, size, "Buffer reused from pool");
            self.live_buffers.insert(id, buf);
            self.stats.current_buffer_count += 1;
            self.stats.current_allocated_bytes += size;
            return Ok(id);
        }

        self.stats.pool_misses += 1;
        let id = self.next_id();
        let buf = GpuBuffer::new(id, size, usage, label.map(|s| s.to_string()));

        self.stats.total_buffers_allocated += 1;
        self.stats.current_buffer_count += 1;
        self.stats.current_allocated_bytes += size;
        self.live_buffers.insert(id, buf);

        debug!(id, size, ?usage, "Buffer allocated");
        Ok(id)
    }

    /// Upload data into a new storage buffer and return the buffer id.
    pub fn upload_buffer(
        &mut self,
        data: &[u8],
        usage: GpuBufferUsage,
        label: Option<&str>,
    ) -> Result<u64, GpuError> {
        self.check_ready()?;
        let size = data.len() as u64;
        self.check_memory(size)?;

        let id = self.next_id();
        let buf = GpuBuffer::from_data(id, data, usage, label.map(|s| s.to_string()));

        self.stats.total_buffers_allocated += 1;
        self.stats.current_buffer_count += 1;
        self.stats.current_allocated_bytes += size;
        self.stats.total_bytes_uploaded += size;
        self.live_buffers.insert(id, buf);

        debug!(id, size, "Buffer uploaded");
        Ok(id)
    }

    /// Upload an f32 slice as a storage buffer.
    pub fn upload_f32(&mut self, data: &[f32], label: Option<&str>) -> Result<u64, GpuError> {
        let bytes = bytemuck_cast_slice_f32(data);
        self.upload_buffer(&bytes, GpuBufferUsage::Storage, label)
    }

    /// Upload a u32 slice as a uniform/storage buffer.
    pub fn upload_u32(
        &mut self,
        data: &[u32],
        usage: GpuBufferUsage,
        label: Option<&str>,
    ) -> Result<u64, GpuError> {
        let bytes = bytemuck_cast_slice_u32(data);
        self.upload_buffer(&bytes, usage, label)
    }

    /// Download (read back) a buffer's contents as bytes.
    pub fn download_buffer(&mut self, id: u64) -> Result<Vec<u8>, GpuError> {
        self.check_ready()?;
        let buf = self
            .live_buffers
            .get(&id)
            .ok_or_else(|| GpuError::BufferError(format!("Buffer {id} not found")))?;

        let data = buf.shadow.clone();
        self.stats.total_bytes_downloaded += data.len() as u64;
        Ok(data)
    }

    /// Download a buffer and interpret as f32 slice.
    pub fn download_f32(&mut self, id: u64) -> Result<Vec<f32>, GpuError> {
        let bytes = self.download_buffer(id)?;
        Ok(bytes_to_f32_vec(&bytes))
    }

    /// Write new data into an existing buffer (must be same size or smaller).
    pub fn write_buffer(&mut self, id: u64, data: &[u8]) -> Result<(), GpuError> {
        self.check_ready()?;
        let buf = self
            .live_buffers
            .get_mut(&id)
            .ok_or_else(|| GpuError::BufferError(format!("Buffer {id} not found")))?;

        if data.len() as u64 > buf.size {
            return Err(GpuError::BufferError(format!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                buf.size
            )));
        }

        buf.shadow = data.to_vec();
        self.stats.total_bytes_uploaded += data.len() as u64;
        Ok(())
    }

    /// Release a buffer (returns it to the pool if space allows).
    pub fn release_buffer(&mut self, id: u64) -> Result<(), GpuError> {
        if let Some(buf) = self.live_buffers.remove(&id) {
            self.stats.current_allocated_bytes =
                self.stats.current_allocated_bytes.saturating_sub(buf.size);
            self.stats.current_buffer_count = self.stats.current_buffer_count.saturating_sub(1);
            self.buffer_pool.release(buf);
            debug!(id, "Buffer released");
        }
        Ok(())
    }

    /// Get a reference to a live buffer.
    pub fn get_buffer(&self, id: u64) -> Result<&GpuBuffer, GpuError> {
        self.live_buffers
            .get(&id)
            .ok_or_else(|| GpuError::BufferError(format!("Buffer {id} not found")))
    }

    /// Get a mutable reference to a live buffer.
    pub fn get_buffer_mut(&mut self, id: u64) -> Result<&mut GpuBuffer, GpuError> {
        self.live_buffers
            .get_mut(&id)
            .ok_or_else(|| GpuError::BufferError(format!("Buffer {id} not found")))
    }

    // ── Simulated compute dispatch ─────────────────────────────────────

    /// Submit a simulated compute dispatch.
    ///
    /// In simulation mode this calls the provided `cpu_fallback` closure
    /// which performs the equivalent work on the CPU. When real wgpu is
    /// wired in, this will encode a compute pass and submit to the queue.
    ///
    /// # Arguments
    ///
    /// * `label` – human-readable name for the dispatch (profiling).
    /// * `dispatch` – workgroup counts.
    /// * `input_ids` – buffer ids bound as inputs.
    /// * `output_id` – buffer id bound as output.
    /// * `cpu_fallback` – closure that performs the work on CPU and
    ///   writes results into the output `Vec<u8>`.
    pub fn dispatch_compute<F>(
        &mut self,
        label: &str,
        _dispatch: &ComputeDispatch,
        input_ids: &[u64],
        output_id: u64,
        cpu_fallback: F,
    ) -> Result<(), GpuError>
    where
        F: FnOnce(&[&[u8]]) -> Vec<u8>,
    {
        self.check_ready()?;

        // Gather input data.
        let inputs: Vec<&[u8]> = input_ids
            .iter()
            .map(|id| {
                self.live_buffers
                    .get(id)
                    .map(|b| b.shadow.as_slice())
                    .ok_or_else(|| GpuError::BufferError(format!("Input buffer {id} not found")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Run the CPU fallback.
        let result = cpu_fallback(&inputs);

        // Write result to output buffer.
        let out_buf = self
            .live_buffers
            .get_mut(&output_id)
            .ok_or_else(|| GpuError::BufferError(format!("Output buffer {output_id} not found")))?;

        out_buf.shadow = result;
        out_buf.size = out_buf.shadow.len() as u64;

        self.stats.total_dispatches += 1;
        debug!(label, "Compute dispatch completed (simulated)");
        Ok(())
    }

    // ── High-level numeric operations (simulated on CPU) ───────────────

    /// Matrix multiply: C = A × B.
    ///
    /// * `a` – row-major matrix of shape (M, K)
    /// * `b` – row-major matrix of shape (K, N)
    ///
    /// Returns row-major matrix of shape (M, N).
    pub fn matmul(
        &mut self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        _k2: usize,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        if a.len() != m * k {
            return Err(GpuError::InvalidDimension(format!(
                "A has {} elements but expected {}",
                a.len(),
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(GpuError::InvalidDimension(format!(
                "B has {} elements but expected {}",
                b.len(),
                k * n
            )));
        }

        // Upload A and B.
        let a_id = self.upload_f32(a, Some("matmul_A"))?;
        let b_id = self.upload_f32(b, Some("matmul_B"))?;

        // Allocate output.
        let out_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let out_id =
            self.allocate_buffer(out_size, GpuBufferUsage::StorageReadback, Some("matmul_C"))?;

        // Dispatch.
        let dispatch = ComputeDispatch::new_2d((n as u32).div_ceil(16), (m as u32).div_ceil(16));

        let m_cap = m;
        let k_cap = k;
        let n_cap = n;

        self.dispatch_compute("matmul", &dispatch, &[a_id, b_id], out_id, move |inputs| {
            let a_data = bytes_to_f32_vec(inputs[0]);
            let b_data = bytes_to_f32_vec(inputs[1]);
            let mut c = vec![0.0f32; m_cap * n_cap];
            for i in 0..m_cap {
                for j in 0..n_cap {
                    let mut sum = 0.0f32;
                    for p in 0..k_cap {
                        sum += a_data[i * k_cap + p] * b_data[p * n_cap + j];
                    }
                    c[i * n_cap + j] = sum;
                }
            }
            bytemuck_cast_slice_f32(&c)
        })?;

        let result = self.download_f32(out_id)?;

        // Cleanup.
        self.release_buffer(a_id)?;
        self.release_buffer(b_id)?;
        self.release_buffer(out_id)?;

        Ok(result)
    }

    /// Softmax over a 1-D vector of logits.
    pub fn softmax(&mut self, logits: &[f32], _len: usize) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        let id = self.upload_f32(logits, Some("softmax_in"))?;
        let out_size = std::mem::size_of_val(logits) as u64;
        let out_id = self.allocate_buffer(
            out_size,
            GpuBufferUsage::StorageReadback,
            Some("softmax_out"),
        )?;

        let dispatch =
            ComputeDispatch::for_elements(logits.len() as u32, self.config.default_workgroup_size);

        self.dispatch_compute("softmax", &dispatch, &[id], out_id, |inputs| {
            let data = bytes_to_f32_vec(inputs[0]);
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let result: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
            bytemuck_cast_slice_f32(&result)
        })?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(id)?;
        self.release_buffer(out_id)?;
        Ok(result)
    }

    /// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    pub fn gelu(&mut self, input: &[f32]) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        let id = self.upload_f32(input, Some("gelu_in"))?;
        let out_size = std::mem::size_of_val(input) as u64;
        let out_id =
            self.allocate_buffer(out_size, GpuBufferUsage::StorageReadback, Some("gelu_out"))?;

        let dispatch =
            ComputeDispatch::for_elements(input.len() as u32, self.config.default_workgroup_size);

        self.dispatch_compute("gelu", &dispatch, &[id], out_id, |inputs| {
            let data = bytes_to_f32_vec(inputs[0]);
            let sqrt_2_over_pi: f32 = (2.0f32 / std::f32::consts::PI).sqrt();
            let result: Vec<f32> = data
                .iter()
                .map(|&x| {
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect();
            bytemuck_cast_slice_f32(&result)
        })?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(id)?;
        self.release_buffer(out_id)?;
        Ok(result)
    }

    /// Layer normalization over the last `dim` elements of a flat buffer.
    /// `input` is treated as (batch, dim). Epsilon = 1e-5.
    pub fn layer_norm(
        &mut self,
        input: &[f32],
        dim: usize,
        gamma: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        if gamma.len() != dim || beta.len() != dim {
            return Err(GpuError::InvalidDimension(
                "gamma/beta length must match dim".into(),
            ));
        }
        if input.len() % dim != 0 {
            return Err(GpuError::InvalidDimension(
                "input length must be divisible by dim".into(),
            ));
        }

        let batch = input.len() / dim;
        let in_id = self.upload_f32(input, Some("ln_in"))?;
        let g_id = self.upload_f32(gamma, Some("ln_gamma"))?;
        let b_id = self.upload_f32(beta, Some("ln_beta"))?;
        let out_size = std::mem::size_of_val(input) as u64;
        let out_id =
            self.allocate_buffer(out_size, GpuBufferUsage::StorageReadback, Some("ln_out"))?;

        let dispatch =
            ComputeDispatch::for_elements(batch as u32, self.config.default_workgroup_size);
        let dim_cap = dim;

        self.dispatch_compute(
            "layer_norm",
            &dispatch,
            &[in_id, g_id, b_id],
            out_id,
            move |inputs| {
                let data = bytes_to_f32_vec(inputs[0]);
                let g = bytes_to_f32_vec(inputs[1]);
                let b = bytes_to_f32_vec(inputs[2]);
                let batch_size = data.len() / dim_cap;
                let eps = 1e-5f32;
                let mut out = vec![0.0f32; data.len()];

                for row in 0..batch_size {
                    let start = row * dim_cap;
                    let end = start + dim_cap;
                    let slice = &data[start..end];

                    let mean: f32 = slice.iter().sum::<f32>() / dim_cap as f32;
                    let var: f32 =
                        slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / dim_cap as f32;
                    let inv_std = 1.0 / (var + eps).sqrt();

                    for j in 0..dim_cap {
                        out[start + j] = g[j] * (slice[j] - mean) * inv_std + b[j];
                    }
                }

                bytemuck_cast_slice_f32(&out)
            },
        )?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(in_id)?;
        self.release_buffer(g_id)?;
        self.release_buffer(b_id)?;
        self.release_buffer(out_id)?;
        Ok(result)
    }

    /// Pairwise Euclidean distance matrix.
    /// Input: (n, d) row-major. Output: (n, n) distance matrix.
    pub fn pairwise_distance(
        &mut self,
        points: &[f32],
        n: usize,
        d: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        if points.len() != n * d {
            return Err(GpuError::InvalidDimension(format!(
                "Expected {} elements, got {}",
                n * d,
                points.len()
            )));
        }

        let in_id = self.upload_f32(points, Some("dist_in"))?;
        let out_size = (n * n * std::mem::size_of::<f32>()) as u64;
        let out_id =
            self.allocate_buffer(out_size, GpuBufferUsage::StorageReadback, Some("dist_out"))?;

        let dispatch = ComputeDispatch::new_2d((n as u32).div_ceil(16), (n as u32).div_ceil(16));

        let n_cap = n;
        let d_cap = d;

        self.dispatch_compute(
            "pairwise_distance",
            &dispatch,
            &[in_id],
            out_id,
            move |inputs| {
                let data = bytes_to_f32_vec(inputs[0]);
                let mut dist = vec![0.0f32; n_cap * n_cap];
                for i in 0..n_cap {
                    for j in i..n_cap {
                        let mut sum = 0.0f32;
                        for k in 0..d_cap {
                            let diff = data[i * d_cap + k] - data[j * d_cap + k];
                            sum += diff * diff;
                        }
                        let d_val = sum.sqrt();
                        dist[i * n_cap + j] = d_val;
                        dist[j * n_cap + i] = d_val;
                    }
                }
                bytemuck_cast_slice_f32(&dist)
            },
        )?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(in_id)?;
        self.release_buffer(out_id)?;
        Ok(result)
    }

    /// Reduce: sum of an f32 buffer.
    pub fn reduce_sum(&mut self, input: &[f32]) -> Result<f32, GpuError> {
        self.check_ready()?;

        let in_id = self.upload_f32(input, Some("reduce_in"))?;
        let out_size = std::mem::size_of::<f32>() as u64;
        let out_id = self.allocate_buffer(
            out_size,
            GpuBufferUsage::StorageReadback,
            Some("reduce_out"),
        )?;

        let dispatch = ComputeDispatch::new_1d(1);

        self.dispatch_compute("reduce_sum", &dispatch, &[in_id], out_id, |inputs| {
            let data = bytes_to_f32_vec(inputs[0]);
            let sum: f32 = data.iter().sum();
            bytemuck_cast_slice_f32(&[sum])
        })?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(in_id)?;
        self.release_buffer(out_id)?;
        Ok(result.first().copied().unwrap_or(0.0))
    }

    /// Reduce: max of an f32 buffer.
    pub fn reduce_max(&mut self, input: &[f32]) -> Result<f32, GpuError> {
        self.check_ready()?;

        let in_id = self.upload_f32(input, Some("reduce_in"))?;
        let out_size = std::mem::size_of::<f32>() as u64;
        let out_id = self.allocate_buffer(
            out_size,
            GpuBufferUsage::StorageReadback,
            Some("reduce_out"),
        )?;

        let dispatch = ComputeDispatch::new_1d(1);

        self.dispatch_compute("reduce_max", &dispatch, &[in_id], out_id, |inputs| {
            let data = bytes_to_f32_vec(inputs[0]);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            bytemuck_cast_slice_f32(&[max])
        })?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(in_id)?;
        self.release_buffer(out_id)?;
        Ok(result.first().copied().unwrap_or(f32::NEG_INFINITY))
    }

    /// Element-wise multiply: c[i] = a[i] * b[i].
    pub fn elementwise_mul(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        if a.len() != b.len() {
            return Err(GpuError::InvalidDimension(
                "Vectors must have same length".into(),
            ));
        }

        let a_id = self.upload_f32(a, Some("mul_a"))?;
        let b_id = self.upload_f32(b, Some("mul_b"))?;
        let out_size = std::mem::size_of_val(a) as u64;
        let out_id =
            self.allocate_buffer(out_size, GpuBufferUsage::StorageReadback, Some("mul_out"))?;

        let dispatch =
            ComputeDispatch::for_elements(a.len() as u32, self.config.default_workgroup_size);

        self.dispatch_compute(
            "elementwise_mul",
            &dispatch,
            &[a_id, b_id],
            out_id,
            |inputs| {
                let a_data = bytes_to_f32_vec(inputs[0]);
                let b_data = bytes_to_f32_vec(inputs[1]);
                let result: Vec<f32> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(x, y)| x * y)
                    .collect();
                bytemuck_cast_slice_f32(&result)
            },
        )?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(a_id)?;
        self.release_buffer(b_id)?;
        self.release_buffer(out_id)?;
        Ok(result)
    }

    /// Element-wise add: c[i] = a[i] + b[i].
    pub fn elementwise_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        if a.len() != b.len() {
            return Err(GpuError::InvalidDimension(
                "Vectors must have same length".into(),
            ));
        }

        let a_id = self.upload_f32(a, Some("add_a"))?;
        let b_id = self.upload_f32(b, Some("add_b"))?;
        let out_size = std::mem::size_of_val(a) as u64;
        let out_id =
            self.allocate_buffer(out_size, GpuBufferUsage::StorageReadback, Some("add_out"))?;

        let dispatch =
            ComputeDispatch::for_elements(a.len() as u32, self.config.default_workgroup_size);

        self.dispatch_compute(
            "elementwise_add",
            &dispatch,
            &[a_id, b_id],
            out_id,
            |inputs| {
                let a_data = bytes_to_f32_vec(inputs[0]);
                let b_data = bytes_to_f32_vec(inputs[1]);
                let result: Vec<f32> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(x, y)| x + y)
                    .collect();
                bytemuck_cast_slice_f32(&result)
            },
        )?;

        let result = self.download_f32(out_id)?;
        self.release_buffer(a_id)?;
        self.release_buffer(b_id)?;
        self.release_buffer(out_id)?;
        Ok(result)
    }

    /// Scaled dot-product attention (single-head).
    ///
    /// * `q` — queries, shape (seq_q, d_k)
    /// * `k` — keys, shape (seq_k, d_k)
    /// * `v` — values, shape (seq_k, d_v)
    ///
    /// Returns output of shape (seq_q, d_v).
    pub fn attention(
        &mut self,
        q: &[f32],
        seq_q: usize,
        k: &[f32],
        seq_k: usize,
        v: &[f32],
        d_k: usize,
        d_v: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.check_ready()?;

        // Validate dimensions.
        if q.len() != seq_q * d_k {
            return Err(GpuError::InvalidDimension("Q dimension mismatch".into()));
        }
        if k.len() != seq_k * d_k {
            return Err(GpuError::InvalidDimension("K dimension mismatch".into()));
        }
        if v.len() != seq_k * d_v {
            return Err(GpuError::InvalidDimension("V dimension mismatch".into()));
        }

        let scale = 1.0 / (d_k as f32).sqrt();

        // QK^T → (seq_q, seq_k)
        // Transpose K: (seq_k, d_k) → (d_k, seq_k)
        let mut kt = vec![0.0f32; d_k * seq_k];
        for i in 0..seq_k {
            for j in 0..d_k {
                kt[j * seq_k + i] = k[i * d_k + j];
            }
        }

        // scores = Q × K^T
        let mut scores = self.matmul(q, seq_q, d_k, &kt, d_k, seq_k)?;

        // Scale
        for s in scores.iter_mut() {
            *s *= scale;
        }

        // Row-wise softmax
        let mut attn_weights = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            let row = &scores[i * seq_k..(i + 1) * seq_k];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..seq_k {
                attn_weights[i * seq_k + j] = exps[j] / sum;
            }
        }

        // output = attn_weights × V → (seq_q, d_v)
        let output = self.matmul(&attn_weights, seq_q, seq_k, v, seq_k, d_v)?;

        Ok(output)
    }

    // ── Cleanup ────────────────────────────────────────────────────────

    /// Release all live buffers and clear the pool.
    pub fn clear_all_buffers(&mut self) {
        self.live_buffers.clear();
        self.buffer_pool.clear();
        self.stats.current_buffer_count = 0;
        self.stats.current_allocated_bytes = 0;
        info!("All GPU buffers cleared");
    }

    /// Shut down the runtime.
    pub fn shutdown(&mut self) {
        self.clear_all_buffers();
        self.status = GpuStatus::Uninitialised;
        info!("GPU runtime shut down");
    }

    // ── Internal helpers ───────────────────────────────────────────────

    fn next_id(&self) -> u64 {
        self.next_buffer_id.fetch_add(1, Ordering::Relaxed)
    }

    fn check_ready(&self) -> Result<(), GpuError> {
        match &self.status {
            GpuStatus::Ready => Ok(()),
            GpuStatus::Unavailable => Err(GpuError::NoAdapter),
            GpuStatus::Uninitialised => Err(GpuError::NotInitialised),
            GpuStatus::Error(e) => Err(GpuError::Internal(e.clone())),
        }
    }

    fn check_memory(&self, additional: u64) -> Result<(), GpuError> {
        if self.stats.current_allocated_bytes + additional > self.config.max_memory_bytes {
            return Err(GpuError::BufferError(format!(
                "Would exceed memory budget: current={}B + requested={}B > max={}B",
                self.stats.current_allocated_bytes, additional, self.config.max_memory_bytes
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Byte ↔ f32/u32 conversion helpers (no bytemuck dependency — manual)
// ---------------------------------------------------------------------------

/// Cast an f32 slice to a Vec<u8> (little-endian).
fn bytemuck_cast_slice_f32(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Cast a u32 slice to a Vec<u8> (little-endian).
fn bytemuck_cast_slice_u32(data: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Interpret a byte slice as a Vec<f32> (little-endian).
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let count = bytes.len() / 4;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let arr = [
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ];
        result.push(f32::from_le_bytes(arr));
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sim_runtime() -> GpuRuntime {
        GpuRuntime::new_simulated(GpuConfig::default())
    }

    #[test]
    fn test_simulated_runtime_is_ready() {
        let rt = sim_runtime();
        assert!(rt.is_ready());
        assert_eq!(*rt.status(), GpuStatus::Ready);
        assert!(!rt.device_name().is_empty());
    }

    #[test]
    fn test_buffer_alloc_release() {
        let mut rt = sim_runtime();
        let id = rt
            .allocate_buffer(1024, GpuBufferUsage::Storage, Some("test"))
            .unwrap();
        assert!(id > 0);
        assert_eq!(rt.stats().current_buffer_count, 1);
        assert_eq!(rt.stats().current_allocated_bytes, 1024);

        rt.release_buffer(id).unwrap();
        assert_eq!(rt.stats().current_buffer_count, 0);
    }

    #[test]
    fn test_upload_download_f32() {
        let mut rt = sim_runtime();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let id = rt.upload_f32(&data, Some("test")).unwrap();
        let result = rt.download_f32(id).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_upload_download_bytes() {
        let mut rt = sim_runtime();
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let id = rt
            .upload_buffer(&data, GpuBufferUsage::Storage, None)
            .unwrap();
        let result = rt.download_buffer(id).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_write_buffer() {
        let mut rt = sim_runtime();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let id = rt.upload_f32(&data, None).unwrap();

        let new_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let bytes = bytemuck_cast_slice_f32(&new_data);
        rt.write_buffer(id, &bytes).unwrap();

        let result = rt.download_f32(id).unwrap();
        assert_eq!(result, new_data);
    }

    #[test]
    fn test_matmul_identity() {
        let mut rt = sim_runtime();
        // 2x2 identity × [1,2; 3,4] = [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0f32];
        let b = vec![1.0, 2.0, 3.0, 4.0f32];
        let c = rt.matmul(&a, 2, 2, &b, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
        assert!((c[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_simple() {
        let mut rt = sim_runtime();
        // [1,2,3; 4,5,6] × [7,8; 9,10; 11,12] = [58,64; 139,154]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32];
        let c = rt.matmul(&a, 2, 3, &b, 3, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 1e-4);
        assert!((c[1] - 64.0).abs() < 1e-4);
        assert!((c[2] - 139.0).abs() < 1e-4);
        assert!((c[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let mut rt = sim_runtime();
        let logits = vec![1.0, 2.0, 3.0f32];
        let probs = rt.softmax(&logits, 3).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1, got {sum}"
        );
        // Probabilities should be increasing.
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_gelu() {
        let mut rt = sim_runtime();
        let input = vec![0.0, 1.0, -1.0, 2.0f32];
        let result = rt.gelu(&input).unwrap();
        assert_eq!(result.len(), 4);
        // GELU(0) ≈ 0
        assert!(result[0].abs() < 1e-5);
        // GELU(1) ≈ 0.841
        assert!((result[1] - 0.841).abs() < 0.01, "GELU(1) = {}", result[1]);
        // GELU(-1) ≈ -0.159
        assert!(
            (result[2] - (-0.159)).abs() < 0.01,
            "GELU(-1) = {}",
            result[2]
        );
    }

    #[test]
    fn test_layer_norm() {
        let mut rt = sim_runtime();
        // Single row, dim=4
        let input = vec![1.0, 2.0, 3.0, 4.0f32];
        let gamma = vec![1.0, 1.0, 1.0, 1.0f32];
        let beta = vec![0.0, 0.0, 0.0, 0.0f32];
        let result = rt.layer_norm(&input, 4, &gamma, &beta).unwrap();
        assert_eq!(result.len(), 4);
        // Mean should be ~0 after normalization.
        let mean: f32 = result.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "Mean after LN should be ~0, got {mean}");
    }

    #[test]
    fn test_pairwise_distance() {
        let mut rt = sim_runtime();
        // 3 points in 2D
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0f32];
        let dist = rt.pairwise_distance(&points, 3, 2).unwrap();
        assert_eq!(dist.len(), 9);
        // d(0,0) = 0
        assert!(dist[0].abs() < 1e-6);
        // d(0,1) = 1.0
        assert!((dist[1] - 1.0).abs() < 1e-5);
        // d(0,2) = 1.0
        assert!((dist[2] - 1.0).abs() < 1e-5);
        // d(1,2) = sqrt(2) ≈ 1.414
        assert!((dist[5] - std::f32::consts::SQRT_2).abs() < 1e-4);
        // Symmetric
        assert!((dist[1] - dist[3]).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum() {
        let mut rt = sim_runtime();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0f32];
        let sum = rt.reduce_sum(&data).unwrap();
        assert!((sum - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_max() {
        let mut rt = sim_runtime();
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0f32];
        let max = rt.reduce_max(&data).unwrap();
        assert!((max - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_elementwise_mul() {
        let mut rt = sim_runtime();
        let a = vec![1.0, 2.0, 3.0f32];
        let b = vec![4.0, 5.0, 6.0f32];
        let c = rt.elementwise_mul(&a, &b).unwrap();
        assert_eq!(c, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_elementwise_add() {
        let mut rt = sim_runtime();
        let a = vec![1.0, 2.0, 3.0f32];
        let b = vec![4.0, 5.0, 6.0f32];
        let c = rt.elementwise_add(&a, &b).unwrap();
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_attention_simple() {
        let mut rt = sim_runtime();
        // seq_q=2, seq_k=2, d_k=2, d_v=2
        let q = vec![1.0, 0.0, 0.0, 1.0f32];
        let k = vec![1.0, 0.0, 0.0, 1.0f32];
        let v = vec![1.0, 2.0, 3.0, 4.0f32];
        let out = rt.attention(&q, 2, &k, 2, &v, 2, 2).unwrap();
        assert_eq!(out.len(), 4); // (2, 2)
        // With identity-like Q and K, attention should mix V rows.
        // Just verify non-NaN and reasonable range.
        for &val in &out {
            assert!(!val.is_nan(), "Attention output contains NaN");
            assert!(val.abs() < 100.0, "Attention output too large: {val}");
        }
    }

    #[test]
    fn test_memory_budget_exceeded() {
        let config = GpuConfig {
            max_memory_bytes: 1024,
            ..GpuConfig::default()
        };
        let mut rt = GpuRuntime::new_simulated(config);
        let result = rt.allocate_buffer(2048, GpuBufferUsage::Storage, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_pool_reuse() {
        let mut rt = sim_runtime();
        let id1 = rt
            .allocate_buffer(256, GpuBufferUsage::Storage, None)
            .unwrap();
        rt.release_buffer(id1).unwrap();
        assert_eq!(rt.stats().pool_misses, 1);

        // Next allocation of same bucket should be a pool hit.
        let _id2 = rt
            .allocate_buffer(256, GpuBufferUsage::Storage, None)
            .unwrap();
        assert_eq!(rt.stats().pool_hits, 1);
    }

    #[test]
    fn test_clear_all_buffers() {
        let mut rt = sim_runtime();
        rt.upload_f32(&[1.0, 2.0, 3.0], None).unwrap();
        rt.upload_f32(&[4.0, 5.0, 6.0], None).unwrap();
        assert_eq!(rt.stats().current_buffer_count, 2);

        rt.clear_all_buffers();
        assert_eq!(rt.stats().current_buffer_count, 0);
        assert_eq!(rt.stats().current_allocated_bytes, 0);
    }

    #[test]
    fn test_shutdown() {
        let mut rt = sim_runtime();
        rt.upload_f32(&[1.0], None).unwrap();
        rt.shutdown();
        assert_eq!(*rt.status(), GpuStatus::Uninitialised);
        assert!(
            rt.allocate_buffer(64, GpuBufferUsage::Storage, None)
                .is_err()
        );
    }

    #[test]
    fn test_dispatch_stats() {
        let mut rt = sim_runtime();
        let _ = rt.softmax(&[1.0, 2.0, 3.0], 3).unwrap();
        assert!(rt.stats().total_dispatches >= 1);
        assert!(rt.stats().total_bytes_uploaded > 0);
    }

    #[test]
    fn test_compute_dispatch_for_elements() {
        let d = ComputeDispatch::for_elements(1000, 256);
        assert_eq!(d.workgroups_x, 4); // ceil(1000/256) = 4
        assert_eq!(d.workgroups_y, 1);
        assert_eq!(d.workgroups_z, 1);
    }

    #[test]
    fn test_byte_conversion_roundtrip() {
        let original = vec![1.5f32, -2.5, std::f32::consts::PI, 0.0, f32::MAX, f32::MIN];
        let bytes = bytemuck_cast_slice_f32(&original);
        let recovered = bytes_to_f32_vec(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_gpu_config_builder() {
        let config = GpuConfig::default()
            .with_backend(GpuBackend::Vulkan)
            .with_memory_budget(1024 * 1024 * 1024)
            .with_debug(true)
            .with_power_preference(GpuPowerPreference::LowPower)
            .with_workgroup_size(128);

        assert_eq!(config.preferred_backend, Some(GpuBackend::Vulkan));
        assert_eq!(config.max_memory_bytes, 1024 * 1024 * 1024);
        assert!(config.debug_mode);
        assert_eq!(config.power_preference, GpuPowerPreference::LowPower);
        assert_eq!(config.default_workgroup_size, 128);
    }

    #[test]
    fn test_device_info_display() {
        let info = GpuDeviceInfo {
            name: "Test GPU".into(),
            vendor: "Test".into(),
            backend: "Vulkan".into(),
            device_type: "Discrete".into(),
            max_buffer_size: 1024 * 1024 * 1024,
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_workgroup_size_z: 64,
            max_workgroup_invocations: 256,
            max_dispatch_x: 65535,
            max_storage_buffers: 8,
        };
        let s = format!("{info}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("Vulkan"));
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let mut rt = sim_runtime();
        let a = vec![1.0f32; 6]; // 2×3
        let b = vec![1.0f32; 4]; // should be 3×N but is too small
        let result = rt.matmul(&a, 2, 3, &b, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_not_found() {
        let mut rt = sim_runtime();
        let result = rt.download_buffer(9999);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_error_display() {
        let e = GpuError::NoAdapter;
        assert!(format!("{e}").contains("no suitable adapter"));

        let e = GpuError::BufferError("test".into());
        assert!(format!("{e}").contains("buffer error"));
    }
}
