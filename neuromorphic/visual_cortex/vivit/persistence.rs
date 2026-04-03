//! Model Persistence and Versioning
//!
//! Provides utilities for saving, loading, and versioning ViViT models.
//! Supports multiple formats and backward compatibility.
//!
//! # Features
//!
//! - Model checkpointing with metadata
//! - Version control and migration
//! - Multiple serialization formats (SafeTensors, Pickle, ONNX)
//! - Automatic backup and recovery
//! - Model registry and catalog
//!
//! # Example
//!
/// ```ignore
/// use janus_neuromorphic::visual_cortex::vivit::{ModelCheckpoint, CheckpointConfig};
///
/// # async fn example() -> common::Result<()> {
/// // Save checkpoint
/// let checkpoint = ModelCheckpoint::new("vivit_v1.0".to_string());
/// checkpoint.save_metadata(&metadata)?;
///
/// // Load checkpoint
/// let loaded = checkpoint.load_metadata("v1")?;
/// # Ok(())
/// # }
/// ```
use chrono::{DateTime, Utc};
use common::{JanusError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Semantic version (e.g., "1.2.3")
    pub version: String,

    /// Model architecture name
    pub architecture: String,

    /// Git commit hash (optional)
    pub commit_hash: Option<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Author/creator
    pub author: String,

    /// Description/notes
    pub description: String,

    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

impl ModelVersion {
    /// Create a new version
    pub fn new(version: String, architecture: String) -> Self {
        Self {
            version,
            architecture,
            commit_hash: None,
            created_at: Utc::now(),
            author: "FKS".to_string(),
            description: String::new(),
            metrics: HashMap::new(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Check if compatible with another version
    pub fn is_compatible_with(&self, other: &ModelVersion) -> bool {
        // Parse semantic versions
        let self_parts: Vec<&str> = self.version.split('.').collect();
        let other_parts: Vec<&str> = other.version.split('.').collect();

        if self_parts.len() < 2 || other_parts.len() < 2 {
            return false;
        }

        // Major version must match
        self_parts[0] == other_parts[0]
    }
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model version
    pub version: ModelVersion,

    /// Model configuration (as JSON)
    pub config: serde_json::Value,

    /// Training state
    pub training_state: TrainingState,

    /// File format
    pub format: CheckpointFormat,

    /// File size in bytes
    pub file_size: u64,

    /// Checksum (SHA256)
    pub checksum: String,

    /// Tags for organization
    pub tags: Vec<String>,
}

/// Training state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,

    /// Total steps
    pub global_step: usize,

    /// Best validation loss
    pub best_val_loss: f32,

    /// Best validation accuracy
    pub best_val_acc: f32,

    /// Learning rate
    pub learning_rate: f64,

    /// Optimizer state (optional)
    pub optimizer_state: Option<String>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            best_val_loss: f32::INFINITY,
            best_val_acc: 0.0,
            learning_rate: 0.001,
            optimizer_state: None,
        }
    }
}

/// Checkpoint file format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// PyTorch pickle format
    Pickle,
    /// ONNX format
    ONNX,
    /// Custom binary format
    Custom,
}

impl CheckpointFormat {
    /// Get file extension
    pub fn extension(&self) -> &str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Pickle => "pt",
            Self::ONNX => "onnx",
            Self::Custom => "ckpt",
        }
    }
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Base directory for checkpoints
    pub checkpoint_dir: PathBuf,

    /// Keep N best checkpoints
    pub keep_best_n: usize,

    /// Keep checkpoint every N epochs
    pub save_every_n_epochs: usize,

    /// Format to use
    pub format: CheckpointFormat,

    /// Enable automatic backups
    pub enable_backups: bool,

    /// Maximum backup age (seconds)
    pub max_backup_age: i64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            keep_best_n: 5,
            save_every_n_epochs: 1,
            format: CheckpointFormat::SafeTensors,
            enable_backups: true,
            max_backup_age: 30 * 24 * 3600, // 30 days
        }
    }
}

/// Model checkpoint manager
pub struct ModelCheckpoint {
    /// Checkpoint name/ID
    name: String,

    /// Configuration
    config: CheckpointConfig,

    /// Metadata cache
    metadata_cache: HashMap<String, CheckpointMetadata>,
}

impl ModelCheckpoint {
    /// Create a new checkpoint manager
    pub fn new(name: String) -> Self {
        Self::with_config(name, CheckpointConfig::default())
    }

    /// Create with custom config
    pub fn with_config(name: String, config: CheckpointConfig) -> Self {
        // Create checkpoint directory if it doesn't exist
        if !config.checkpoint_dir.exists() {
            fs::create_dir_all(&config.checkpoint_dir).ok();
        }

        Self {
            name,
            config,
            metadata_cache: HashMap::new(),
        }
    }

    /// Get checkpoint path
    pub fn get_path(&self, version: &str) -> PathBuf {
        let filename = format!(
            "{}_{}.{}",
            self.name,
            version,
            self.config.format.extension()
        );
        self.config.checkpoint_dir.join(filename)
    }

    /// Get metadata path
    fn get_metadata_path(&self, version: &str) -> PathBuf {
        let filename = format!("{}_{}.meta.json", self.name, version);
        self.config.checkpoint_dir.join(filename)
    }

    /// Save checkpoint metadata
    pub fn save_metadata(&mut self, metadata: &CheckpointMetadata) -> Result<()> {
        let path = self.get_metadata_path(&metadata.version.version);

        info!("Saving checkpoint metadata to {:?}", path);

        let json = serde_json::to_string_pretty(metadata).map_err(|e| {
            JanusError::Serialization(format!("Failed to serialize metadata: {}", e))
        })?;

        fs::write(&path, json).map_err(|e| {
            JanusError::Io(std::io::Error::other(format!(
                "Failed to write metadata: {}",
                e
            )))
        })?;

        // Update cache
        self.metadata_cache
            .insert(metadata.version.version.clone(), metadata.clone());

        Ok(())
    }

    /// Load checkpoint metadata
    pub fn load_metadata(&mut self, version: &str) -> Result<CheckpointMetadata> {
        // Check cache first
        if let Some(metadata) = self.metadata_cache.get(version) {
            return Ok(metadata.clone());
        }

        let path = self.get_metadata_path(version);

        debug!("Loading checkpoint metadata from {:?}", path);

        let json = fs::read_to_string(&path)
            .map_err(|e| JanusError::NotFound(format!("Metadata file not found: {}", e)))?;

        let metadata: CheckpointMetadata = serde_json::from_str(&json).map_err(|e| {
            JanusError::Serialization(format!("Failed to deserialize metadata: {}", e))
        })?;

        // Update cache
        self.metadata_cache
            .insert(version.to_string(), metadata.clone());

        Ok(metadata)
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<String>> {
        let mut versions = Vec::new();

        let entries = fs::read_dir(&self.config.checkpoint_dir).map_err(JanusError::Io)?;

        for entry in entries {
            let entry = entry.map_err(JanusError::Io)?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                // Look for metadata files
                if filename.ends_with(".meta.json") && filename.starts_with(&self.name) {
                    // Extract version from filename
                    if let Some(version) = self.extract_version_from_filename(filename) {
                        versions.push(version);
                    }
                }
            }
        }

        versions.sort();
        Ok(versions)
    }

    /// Extract version from metadata filename
    fn extract_version_from_filename(&self, filename: &str) -> Option<String> {
        // Format: {name}_{version}.meta.json
        let prefix = format!("{}_", self.name);
        let suffix = ".meta.json";

        if filename.starts_with(&prefix) && filename.ends_with(suffix) {
            let start = prefix.len();
            let end = filename.len() - suffix.len();
            Some(filename[start..end].to_string())
        } else {
            None
        }
    }

    /// Get latest checkpoint version
    pub fn get_latest_version(&self) -> Result<String> {
        let versions = self.list_checkpoints()?;

        versions
            .last()
            .cloned()
            .ok_or_else(|| JanusError::NotFound("No checkpoints found".to_string()))
    }

    /// Delete old checkpoints (keep only best N)
    pub fn prune_old_checkpoints(&mut self) -> Result<usize> {
        info!(
            "Pruning old checkpoints (keeping best {})",
            self.config.keep_best_n
        );

        let versions = self.list_checkpoints()?;

        if versions.len() <= self.config.keep_best_n {
            return Ok(0);
        }

        // Load all metadata and sort by validation accuracy
        let mut checkpoints: Vec<(String, CheckpointMetadata)> = versions
            .iter()
            .filter_map(|v| self.load_metadata(v).ok().map(|m| (v.clone(), m)))
            .collect();

        checkpoints.sort_by(|a, b| {
            b.1.training_state
                .best_val_acc
                .partial_cmp(&a.1.training_state.best_val_acc)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Delete checkpoints beyond keep_best_n
        let mut deleted = 0;
        for (version, _) in checkpoints.iter().skip(self.config.keep_best_n) {
            if let Err(e) = self.delete_checkpoint(version) {
                warn!("Failed to delete checkpoint {}: {}", version, e);
            } else {
                deleted += 1;
            }
        }

        info!("Deleted {} old checkpoints", deleted);
        Ok(deleted)
    }

    /// Delete a specific checkpoint
    pub fn delete_checkpoint(&mut self, version: &str) -> Result<()> {
        info!("Deleting checkpoint: {}", version);

        let checkpoint_path = self.get_path(version);
        let metadata_path = self.get_metadata_path(version);

        // Delete checkpoint file
        if checkpoint_path.exists() {
            fs::remove_file(&checkpoint_path).map_err(JanusError::Io)?;
        }

        // Delete metadata file
        if metadata_path.exists() {
            fs::remove_file(&metadata_path).map_err(JanusError::Io)?;
        }

        // Remove from cache
        self.metadata_cache.remove(version);

        Ok(())
    }

    /// Create backup of a checkpoint
    pub fn backup_checkpoint(&self, version: &str) -> Result<PathBuf> {
        if !self.config.enable_backups {
            return Err(JanusError::InvalidInput("Backups are disabled".to_string()));
        }

        let checkpoint_path = self.get_path(version);
        let backup_dir = self.config.checkpoint_dir.join("backups");

        if !backup_dir.exists() {
            fs::create_dir_all(&backup_dir).map_err(JanusError::Io)?;
        }

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let backup_filename = format!(
            "{}_{}_backup_{}.{}",
            self.name,
            version,
            timestamp,
            self.config.format.extension()
        );
        let backup_path = backup_dir.join(backup_filename);

        info!("Creating backup: {:?}", backup_path);

        fs::copy(&checkpoint_path, &backup_path).map_err(JanusError::Io)?;

        Ok(backup_path)
    }

    /// Restore from backup
    pub fn restore_from_backup<P: AsRef<Path>>(&self, backup_path: P, version: &str) -> Result<()> {
        info!(
            "Restoring checkpoint from backup: {:?}",
            backup_path.as_ref()
        );

        let checkpoint_path = self.get_path(version);

        fs::copy(backup_path.as_ref(), &checkpoint_path).map_err(JanusError::Io)?;

        info!("Checkpoint restored successfully");
        Ok(())
    }

    /// Prune old backups
    pub fn prune_old_backups(&self) -> Result<usize> {
        let backup_dir = self.config.checkpoint_dir.join("backups");

        if !backup_dir.exists() {
            return Ok(0);
        }

        let cutoff = Utc::now().timestamp() - self.config.max_backup_age;
        let mut deleted = 0;

        let entries = fs::read_dir(&backup_dir).map_err(JanusError::Io)?;

        for entry in entries {
            let entry = entry.map_err(JanusError::Io)?;
            let path = entry.path();

            if let Ok(metadata) = fs::metadata(&path) {
                if let Ok(modified) = metadata.modified() {
                    let modified_ts = modified
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64;

                    if modified_ts < cutoff && fs::remove_file(&path).is_ok() {
                        deleted += 1;
                    }
                }
            }
        }

        if deleted > 0 {
            info!("Deleted {} old backups", deleted);
        }

        Ok(deleted)
    }

    /// Validate checkpoint integrity
    pub fn validate_checkpoint(&mut self, version: &str) -> Result<bool> {
        debug!("Validating checkpoint: {}", version);

        let checkpoint_path = self.get_path(version);
        let metadata = self.load_metadata(version)?;

        // Check if file exists
        if !checkpoint_path.exists() {
            return Ok(false);
        }

        // Check file size
        let file_metadata = fs::metadata(&checkpoint_path).map_err(JanusError::Io)?;
        if file_metadata.len() != metadata.file_size {
            warn!("Checkpoint file size mismatch");
            return Ok(false);
        }

        // Verify SHA256 checksum
        if !metadata.checksum.is_empty() {
            let mut file = fs::File::open(&checkpoint_path).map_err(JanusError::Io)?;
            let mut hasher = Sha256::new();
            let mut buffer = [0u8; 8192];
            loop {
                let bytes_read = file.read(&mut buffer).map_err(JanusError::Io)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            let computed = hex::encode(hasher.finalize());
            if computed != metadata.checksum {
                warn!(
                    "Checkpoint checksum mismatch: expected {}, got {}",
                    metadata.checksum, computed
                );
                return Ok(false);
            }
            debug!("Checkpoint checksum verified: {}", computed);
        }

        Ok(true)
    }
}

/// Model registry for cataloging all models
pub struct ModelRegistry {
    /// Registry directory
    registry_dir: PathBuf,

    /// Registered models
    models: HashMap<String, Vec<ModelVersion>>,
}

impl ModelRegistry {
    /// Create a new registry
    pub fn new<P: AsRef<Path>>(registry_dir: P) -> Result<Self> {
        let registry_dir = registry_dir.as_ref().to_path_buf();

        if !registry_dir.exists() {
            fs::create_dir_all(&registry_dir).map_err(JanusError::Io)?;
        }

        let mut registry = Self {
            registry_dir,
            models: HashMap::new(),
        };

        // Load existing registry
        registry.load()?;

        Ok(registry)
    }

    /// Register a new model version
    pub fn register(&mut self, model_name: String, version: ModelVersion) -> Result<()> {
        info!("Registering model: {} v{}", model_name, version.version);

        self.models
            .entry(model_name.clone())
            .or_default()
            .push(version);

        self.save()?;

        Ok(())
    }

    /// Get all versions of a model
    pub fn get_versions(&self, model_name: &str) -> Option<&Vec<ModelVersion>> {
        self.models.get(model_name)
    }

    /// Get latest version of a model
    pub fn get_latest(&self, model_name: &str) -> Option<&ModelVersion> {
        self.models.get(model_name)?.last()
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Save registry to disk
    fn save(&self) -> Result<()> {
        let path = self.registry_dir.join("registry.json");

        let json = serde_json::to_string_pretty(&self.models).map_err(|e| {
            JanusError::Serialization(format!("Failed to serialize registry: {}", e))
        })?;

        fs::write(&path, json).map_err(JanusError::Io)?;

        Ok(())
    }

    /// Load registry from disk
    fn load(&mut self) -> Result<()> {
        let path = self.registry_dir.join("registry.json");

        if !path.exists() {
            return Ok(());
        }

        let json = fs::read_to_string(&path).map_err(JanusError::Io)?;

        self.models = serde_json::from_str(&json).map_err(|e| {
            JanusError::Serialization(format!("Failed to deserialize registry: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_compatibility() {
        let v1 = ModelVersion::new("1.2.3".to_string(), "vivit".to_string());
        let v2 = ModelVersion::new("1.3.0".to_string(), "vivit".to_string());
        let v3 = ModelVersion::new("2.0.0".to_string(), "vivit".to_string());

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_checkpoint_format_extension() {
        assert_eq!(CheckpointFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(CheckpointFormat::Pickle.extension(), "pt");
        assert_eq!(CheckpointFormat::ONNX.extension(), "onnx");
    }

    #[test]
    fn test_checkpoint_path() {
        let checkpoint = ModelCheckpoint::new("test_model".to_string());
        let path = checkpoint.get_path("1.0.0");
        assert!(path.to_str().unwrap().contains("test_model_1.0.0"));
    }

    #[test]
    fn test_extract_version_from_filename() {
        let checkpoint = ModelCheckpoint::new("vivit".to_string());
        let version = checkpoint.extract_version_from_filename("vivit_1.2.3.meta.json");
        assert_eq!(version, Some("1.2.3".to_string()));
    }
}
