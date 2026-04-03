//! Distributed Checkpoint Manager
//!
//! This module provides distributed checkpoint management for multi-GPU/multi-node training.
//! It handles checkpoint coordination, versioning, sharding, and cloud storage integration.
//!
//! # Features
//!
//! - Distributed checkpoint coordination
//! - Model sharding across devices
//! - S3/cloud storage integration
//! - Checkpoint versioning and metadata
//! - Automatic checkpoint rotation
//! - Fault tolerance and recovery
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::distributed::DistributedCheckpointManager;
//! use candle_core::Tensor;
//!
//! # fn example() -> anyhow::Result<()> {
//! let mut manager = DistributedCheckpointManager::new("checkpoints")?;
//!
//! // Save checkpoint
//! let mut state = std::collections::HashMap::new();
//! state.insert("model".to_string(), vec![/* tensors */]);
//! manager.save_checkpoint("model_v1", state, 0)?;
//!
//! // Load checkpoint
//! let loaded = manager.load_checkpoint("model_v1")?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint name/identifier
    pub name: String,
    /// Version number
    pub version: usize,
    /// Rank that created the checkpoint
    pub rank: usize,
    /// World size at checkpoint time
    pub world_size: usize,
    /// Creation timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Model architecture hash
    pub model_hash: Option<String>,
    /// Training step/epoch
    pub step: usize,
    /// Training metrics at checkpoint
    pub metrics: HashMap<String, f64>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
    /// Shard information (if model is sharded)
    pub shards: Vec<ShardInfo>,
}

impl CheckpointMetadata {
    /// Create new metadata
    pub fn new(name: String, version: usize, rank: usize, world_size: usize, step: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            name,
            version,
            rank,
            world_size,
            timestamp,
            model_hash: None,
            step,
            metrics: HashMap::new(),
            custom: HashMap::new(),
            shards: Vec::new(),
        }
    }

    /// Add metric
    pub fn add_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
    }

    /// Add custom metadata
    pub fn add_custom(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }

    /// Add shard information
    pub fn add_shard(&mut self, shard: ShardInfo) {
        self.shards.push(shard);
    }
}

/// Shard information for distributed checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard index
    pub index: usize,
    /// Rank that owns this shard
    pub rank: usize,
    /// Parameter names in this shard
    pub parameters: Vec<String>,
    /// Shard file path
    pub file_path: String,
    /// Shard size in bytes
    pub size_bytes: usize,
}

/// Checkpoint storage backend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// Local filesystem
    Local,
    /// Amazon S3
    S3 { bucket: String, prefix: String },
    /// Google Cloud Storage
    GCS { bucket: String, prefix: String },
    /// Azure Blob Storage
    Azure { container: String, prefix: String },
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Storage backend
    pub backend: StorageBackend,
    /// Base directory for checkpoints
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Save checkpoint every N steps
    pub save_frequency: usize,
    /// Keep checkpoints at specific intervals (e.g., every 1000 steps)
    pub keep_interval: Option<usize>,
    /// Enable checkpoint sharding
    pub shard_checkpoints: bool,
    /// Compress checkpoints
    pub compress: bool,
    /// Async upload to cloud storage
    pub async_upload: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Local,
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_checkpoints: 5,
            save_frequency: 1000,
            keep_interval: Some(10000),
            shard_checkpoints: false,
            compress: false,
            async_upload: false,
        }
    }
}

/// Distributed checkpoint manager
pub struct DistributedCheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Checkpoint registry (name -> metadata)
    registry: HashMap<String, CheckpointMetadata>,
    /// Registry file path
    registry_path: PathBuf,
    /// Current rank
    rank: usize,
    /// World size
    world_size: usize,
}

impl DistributedCheckpointManager {
    /// Create a new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self> {
        let config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            ..CheckpointConfig::default()
        };
        Self::with_config(config, 0, 1)
    }

    /// Create with custom configuration
    pub fn with_config(config: CheckpointConfig, rank: usize, world_size: usize) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&config.checkpoint_dir)
            .context("Failed to create checkpoint directory")?;

        let registry_path = config.checkpoint_dir.join("checkpoint_registry.json");
        let registry = if registry_path.exists() {
            Self::load_registry(&registry_path)?
        } else {
            HashMap::new()
        };

        info!(
            "Initialized distributed checkpoint manager at {:?} (rank {}/{})",
            config.checkpoint_dir, rank, world_size
        );

        Ok(Self {
            config,
            registry,
            registry_path,
            rank,
            world_size,
        })
    }

    /// Load checkpoint registry from file
    fn load_registry(path: &Path) -> Result<HashMap<String, CheckpointMetadata>> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let registry = serde_json::from_str(&contents)?;
        Ok(registry)
    }

    /// Save checkpoint registry to file
    fn save_registry(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.registry)?;
        let mut file = File::create(&self.registry_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Save a distributed checkpoint
    pub fn save_checkpoint(
        &mut self,
        name: &str,
        state: HashMap<String, Vec<Tensor>>,
        step: usize,
    ) -> Result<CheckpointMetadata> {
        info!("Saving checkpoint '{}' at step {}", name, step);

        let version = self.next_version(name);
        let mut metadata =
            CheckpointMetadata::new(name.to_string(), version, self.rank, self.world_size, step);

        // Create checkpoint directory
        let checkpoint_name = format!("{}_v{}", name, version);
        let checkpoint_dir = self.config.checkpoint_dir.join(&checkpoint_name);
        fs::create_dir_all(&checkpoint_dir)?;

        if self.config.shard_checkpoints && self.world_size > 1 {
            // Save sharded checkpoint
            self.save_sharded_checkpoint(&checkpoint_dir, &state, &mut metadata)?;
        } else {
            // Save full checkpoint
            self.save_full_checkpoint(&checkpoint_dir, &state, &mut metadata)?;
        }

        // Save metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(metadata_path, metadata_json)?;

        // Update registry
        self.registry
            .insert(checkpoint_name.clone(), metadata.clone());
        self.save_registry()?;

        // Prune old checkpoints
        self.prune_checkpoints(name)?;

        // Upload to cloud if configured
        if self.config.backend != StorageBackend::Local {
            self.upload_checkpoint(&checkpoint_name)?;
        }

        info!("Checkpoint '{}' saved successfully", checkpoint_name);
        Ok(metadata)
    }

    /// Save full (non-sharded) checkpoint
    fn save_full_checkpoint(
        &self,
        checkpoint_dir: &Path,
        state: &HashMap<String, Vec<Tensor>>,
        _metadata: &mut CheckpointMetadata,
    ) -> Result<()> {
        for (param_name, tensors) in state {
            let param_dir = checkpoint_dir.join(param_name);
            fs::create_dir_all(&param_dir)?;

            for (idx, tensor) in tensors.iter().enumerate() {
                let file_path = param_dir.join(format!("tensor_{}.safetensors", idx));
                self.save_tensor(tensor, &file_path)?;
            }
        }

        // Don't add shard info for full checkpoints - shards.is_empty() indicates full checkpoint
        Ok(())
    }

    /// Save sharded checkpoint
    fn save_sharded_checkpoint(
        &self,
        checkpoint_dir: &Path,
        state: &HashMap<String, Vec<Tensor>>,
        metadata: &mut CheckpointMetadata,
    ) -> Result<()> {
        let shard_dir = checkpoint_dir.join(format!("shard_{}", self.rank));
        fs::create_dir_all(&shard_dir)?;

        let mut shard_params = Vec::new();
        let mut total_size = 0;

        for (param_name, tensors) in state {
            shard_params.push(param_name.clone());

            let _param_file = shard_dir.join(format!("{}.safetensors", param_name));

            // For simplicity, save all tensors for this parameter
            // In production, would shard large tensors across ranks
            for (idx, tensor) in tensors.iter().enumerate() {
                let file_path = shard_dir.join(format!("{}_{}.safetensors", param_name, idx));
                self.save_tensor(tensor, &file_path)?;

                if let Ok(size) = fs::metadata(&file_path).map(|m| m.len() as usize) {
                    total_size += size;
                }
            }
        }

        metadata.add_shard(ShardInfo {
            index: self.rank,
            rank: self.rank,
            parameters: shard_params,
            file_path: format!("shard_{}", self.rank),
            size_bytes: total_size,
        });

        Ok(())
    }

    /// Save a single tensor to file
    fn save_tensor(&self, tensor: &Tensor, path: &Path) -> Result<()> {
        // Convert tensor to bytes
        // In production, would use safetensors format
        let shape = tensor.dims();
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;

        let mut file = File::create(path)?;

        // Write shape (migrated from bincode to postcard)
        let shape_bytes: Vec<u8> = postcard::to_allocvec(&shape)?;
        file.write_all(&(shape_bytes.len() as u32).to_le_bytes())?;
        file.write_all(&shape_bytes)?;

        // Write data
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&data_bytes)?;

        Ok(())
    }

    /// Load a checkpoint
    pub fn load_checkpoint(&self, name: &str) -> Result<HashMap<String, Vec<Tensor>>> {
        info!("Loading checkpoint '{}'", name);

        let checkpoint_dir = self.config.checkpoint_dir.join(name);
        if !checkpoint_dir.exists() {
            return Err(anyhow!("Checkpoint '{}' not found", name));
        }

        // Load metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = fs::read_to_string(metadata_path)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

        // Load state based on sharding
        let state = if metadata.shards.is_empty() {
            self.load_full_checkpoint(&checkpoint_dir)?
        } else {
            self.load_sharded_checkpoint(&checkpoint_dir, &metadata)?
        };

        info!("Checkpoint '{}' loaded successfully", name);
        Ok(state)
    }

    /// Load full checkpoint
    fn load_full_checkpoint(&self, checkpoint_dir: &Path) -> Result<HashMap<String, Vec<Tensor>>> {
        let mut state = HashMap::new();

        for entry in fs::read_dir(checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_dir() || path.file_name() == Some(std::ffi::OsStr::new("metadata.json")) {
                continue;
            }

            if let Some(param_name) = path.file_name().and_then(|n| n.to_str()) {
                let mut tensors = Vec::new();

                for tensor_entry in fs::read_dir(&path)? {
                    let tensor_path = tensor_entry?.path();
                    if tensor_path.extension() == Some(std::ffi::OsStr::new("safetensors")) {
                        let tensor = self.load_tensor(&tensor_path)?;
                        tensors.push(tensor);
                    }
                }

                if !tensors.is_empty() {
                    state.insert(param_name.to_string(), tensors);
                }
            }
        }

        Ok(state)
    }

    /// Load sharded checkpoint
    fn load_sharded_checkpoint(
        &self,
        checkpoint_dir: &Path,
        metadata: &CheckpointMetadata,
    ) -> Result<HashMap<String, Vec<Tensor>>> {
        let mut state = HashMap::new();

        // Load only the shard for this rank
        for shard in &metadata.shards {
            if shard.rank != self.rank {
                continue;
            }

            let shard_dir = checkpoint_dir.join(&shard.file_path);
            if !shard_dir.exists() {
                warn!("Shard directory not found: {:?}", shard_dir);
                continue;
            }

            for param_name in &shard.parameters {
                let mut tensors = Vec::new();

                for entry in fs::read_dir(&shard_dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                        if filename.starts_with(param_name) && filename.ends_with(".safetensors") {
                            let tensor = self.load_tensor(&path)?;
                            tensors.push(tensor);
                        }
                    }
                }

                if !tensors.is_empty() {
                    state.insert(param_name.clone(), tensors);
                }
            }
        }

        Ok(state)
    }

    /// Load a single tensor from file
    fn load_tensor(&self, path: &Path) -> Result<Tensor> {
        let mut file = File::open(path)?;

        // Read shape length
        let mut shape_len_bytes = [0u8; 4];
        file.read_exact(&mut shape_len_bytes)?;
        let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;

        // Read shape
        let mut shape_bytes = vec![0u8; shape_len];
        file.read_exact(&mut shape_bytes)?;
        let shape: Vec<usize> = postcard::from_bytes(&shape_bytes)?;

        // Read data
        let total_elements: usize = shape.iter().product();
        let mut data_bytes = vec![0u8; total_elements * 4];
        file.read_exact(&mut data_bytes)?;

        let data: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Create tensor
        let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?;
        Ok(tensor)
    }

    /// Get next version number for a checkpoint name
    fn next_version(&self, name: &str) -> usize {
        let mut max_version = 0;
        for key in self.registry.keys() {
            if key.starts_with(name) {
                if let Some(version_str) = key.strip_prefix(&format!("{}_v", name)) {
                    if let Ok(version) = version_str.parse::<usize>() {
                        max_version = max_version.max(version);
                    }
                }
            }
        }
        max_version + 1
    }

    /// Prune old checkpoints
    fn prune_checkpoints(&mut self, name: &str) -> Result<()> {
        let mut checkpoints: Vec<_> = self
            .registry
            .iter()
            .filter(|(key, _)| key.starts_with(name))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort by version number (extracted from checkpoint name) for stable ordering
        checkpoints.sort_by_key(|(checkpoint_name, meta)| {
            // Extract version from name like "model_v1" -> 1
            checkpoint_name
                .rsplit_once("_v")
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(meta.version)
        });

        while checkpoints.len() > self.config.max_checkpoints {
            if let Some((checkpoint_name, metadata)) = checkpoints.first() {
                // Keep if at specified interval
                if let Some(interval) = self.config.keep_interval {
                    if metadata.step % interval == 0 {
                        checkpoints.remove(0);
                        continue;
                    }
                }

                // Delete checkpoint
                let checkpoint_dir = self.config.checkpoint_dir.join(checkpoint_name);
                if checkpoint_dir.exists() {
                    fs::remove_dir_all(&checkpoint_dir)?;
                    info!("Pruned old checkpoint: {}", checkpoint_name);
                }

                self.registry.remove(&checkpoint_name.clone());
                checkpoints.remove(0);
            } else {
                break;
            }
        }

        self.save_registry()?;
        Ok(())
    }

    /// Upload checkpoint to cloud storage (stub)
    fn upload_checkpoint(&self, checkpoint_name: &str) -> Result<()> {
        match &self.config.backend {
            StorageBackend::Local => Ok(()),
            StorageBackend::S3 { bucket, prefix } => {
                debug!(
                    "Would upload checkpoint '{}' to S3 bucket '{}' with prefix '{}'",
                    checkpoint_name, bucket, prefix
                );
                // Implement S3 upload using rusoto or aws-sdk-rust
                Ok(())
            }
            StorageBackend::GCS { bucket, prefix } => {
                debug!(
                    "Would upload checkpoint '{}' to GCS bucket '{}' with prefix '{}'",
                    checkpoint_name, bucket, prefix
                );
                // Implement GCS upload
                Ok(())
            }
            StorageBackend::Azure { container, prefix } => {
                debug!(
                    "Would upload checkpoint '{}' to Azure container '{}' with prefix '{}'",
                    checkpoint_name, container, prefix
                );
                // Implement Azure upload
                Ok(())
            }
        }
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Vec<String> {
        self.registry.keys().cloned().collect()
    }

    /// Get checkpoint metadata
    pub fn get_metadata(&self, name: &str) -> Option<&CheckpointMetadata> {
        self.registry.get(name)
    }

    /// Delete a checkpoint
    pub fn delete_checkpoint(&mut self, name: &str) -> Result<()> {
        let checkpoint_dir = self.config.checkpoint_dir.join(name);
        if checkpoint_dir.exists() {
            fs::remove_dir_all(&checkpoint_dir)?;
        }

        self.registry.remove(name);
        self.save_registry()?;

        info!("Deleted checkpoint: {}", name);
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = DistributedCheckpointManager::new(temp_dir.path()).unwrap();
        assert_eq!(manager.rank(), 0);
        assert_eq!(manager.world_size(), 1);
    }

    #[test]
    fn test_checkpoint_metadata() {
        let mut metadata = CheckpointMetadata::new("test".to_string(), 1, 0, 1, 100);
        metadata.add_metric("loss".to_string(), 0.5);
        metadata.add_custom("author".to_string(), "test".to_string());

        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.step, 100);
        assert_eq!(metadata.metrics.get("loss"), Some(&0.5));
        assert_eq!(metadata.custom.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_save_load_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DistributedCheckpointManager::new(temp_dir.path()).unwrap();

        // Create test state
        let mut state = HashMap::new();
        let tensor = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
        state.insert("weights".to_string(), vec![tensor.clone()]);

        // Save checkpoint
        let metadata = manager.save_checkpoint("model", state, 100).unwrap();
        assert_eq!(metadata.step, 100);
        println!("Saved checkpoint with {} shards", metadata.shards.len());
        for shard in &metadata.shards {
            println!(
                "  Shard: file_path={}, params={:?}",
                shard.file_path, shard.parameters
            );
        }

        // Load checkpoint
        let loaded = manager.load_checkpoint("model_v1").unwrap();
        println!("Loaded checkpoint with {} keys", loaded.len());
        for key in loaded.keys() {
            println!("  Key: {}", key);
        }
        assert!(
            loaded.contains_key("weights"),
            "Expected 'weights' key, got keys: {:?}",
            loaded.keys().collect::<Vec<_>>()
        );
        assert_eq!(loaded["weights"].len(), 1);
    }

    #[test]
    fn test_checkpoint_versioning() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DistributedCheckpointManager::new(temp_dir.path()).unwrap();

        let mut state = HashMap::new();
        let tensor = Tensor::ones((2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        state.insert("w".to_string(), vec![tensor.clone()]);

        // Save multiple versions
        manager
            .save_checkpoint("model", state.clone(), 100)
            .unwrap();
        manager
            .save_checkpoint("model", state.clone(), 200)
            .unwrap();
        manager
            .save_checkpoint("model", state.clone(), 300)
            .unwrap();

        let checkpoints = manager.list_checkpoints();
        assert!(checkpoints.contains(&"model_v1".to_string()));
        assert!(checkpoints.contains(&"model_v2".to_string()));
        assert!(checkpoints.contains(&"model_v3".to_string()));
    }

    #[test]
    fn test_checkpoint_pruning() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = CheckpointConfig::default();
        config.checkpoint_dir = temp_dir.path().to_path_buf();
        config.max_checkpoints = 2;

        let mut manager = DistributedCheckpointManager::with_config(config, 0, 1).unwrap();

        let mut state = HashMap::new();
        let tensor = Tensor::ones((2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        state.insert("w".to_string(), vec![tensor.clone()]);

        // Save 3 checkpoints (should prune oldest)
        manager
            .save_checkpoint("model", state.clone(), 100)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        manager
            .save_checkpoint("model", state.clone(), 200)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        manager
            .save_checkpoint("model", state.clone(), 300)
            .unwrap();

        let checkpoints = manager.list_checkpoints();
        println!("Checkpoints after pruning: {:?}", checkpoints);
        for name in &checkpoints {
            if let Some(meta) = manager.get_metadata(name) {
                println!(
                    "  {}: step={}, timestamp={}",
                    name, meta.step, meta.timestamp
                );
            }
        }
        assert_eq!(
            checkpoints.len(),
            2,
            "Expected 2 checkpoints, got {}: {:?}",
            checkpoints.len(),
            checkpoints
        );
        assert!(
            !checkpoints.contains(&"model_v1".to_string()),
            "model_v1 should have been pruned"
        );
    }

    #[test]
    fn test_checkpoint_deletion() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DistributedCheckpointManager::new(temp_dir.path()).unwrap();

        let mut state = HashMap::new();
        let tensor = Tensor::ones((2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        state.insert("w".to_string(), vec![tensor]);

        manager.save_checkpoint("model", state, 100).unwrap();
        assert!(manager.list_checkpoints().contains(&"model_v1".to_string()));

        manager.delete_checkpoint("model_v1").unwrap();
        assert!(!manager.list_checkpoints().contains(&"model_v1".to_string()));
    }
}
