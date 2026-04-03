//! # Experience Vector Store
//!
//! Persists validated experience data from Arrow IPC batches into Qdrant for
//! similarity-based experience retrieval during training.
//!
//! ## Architecture
//!
//! ```text
//! Arrow RecordBatch (validated)
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  ExperienceStore    │
//! │  - extract vectors  │
//! │  - build payloads   │
//! │  - upsert to Qdrant│
//! └─────────────────────┘
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  Qdrant Collection  │
//! │  "experiences"      │
//! └─────────────────────┘
//! ```
//!
//! Each experience row is stored as a Qdrant point where:
//! - **Vector**: The `state_gaf` binary field decoded as `f32` little-endian values
//! - **Payload**: Metadata fields (action, reward, symbol, timestamp, done flag)
//! - **ID**: A UUID v4 generated per row for deduplication

use anyhow::{Context, Result};
use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array,
};
use arrow::record_batch::RecordBatch;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the experience vector store.
#[derive(Debug, Clone)]
pub struct ExperienceStoreConfig {
    /// Qdrant gRPC endpoint URL (e.g. `http://localhost:6334`).
    pub qdrant_url: String,

    /// Name of the Qdrant collection for experience vectors.
    pub collection_name: String,

    /// Dimensionality of the GAF state vector.
    ///
    /// This must match the number of `f32` values encoded in the `state_gaf`
    /// binary column of the Arrow IPC file.  The default (9) corresponds to
    /// a 3×3 GAF image flattened to a vector.
    pub vector_dim: u64,

    /// Maximum number of points to upsert in a single Qdrant request.
    pub upsert_batch_size: usize,

    /// Whether to use mock mode (in-memory, no real Qdrant connection).
    pub use_mock: bool,

    /// Connection timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for ExperienceStoreConfig {
    fn default() -> Self {
        Self {
            qdrant_url: "http://localhost:6334".to_string(),
            collection_name: "experiences".to_string(),
            vector_dim: 9,
            upsert_batch_size: 256,
            use_mock: true,
            timeout_secs: 10,
        }
    }
}

impl ExperienceStoreConfig {
    /// Load configuration from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        Self {
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6334".to_string()),
            collection_name: std::env::var("QDRANT_EXPERIENCE_COLLECTION")
                .unwrap_or_else(|_| "experiences".to_string()),
            vector_dim: std::env::var("QDRANT_EXPERIENCE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(9),
            upsert_batch_size: std::env::var("QDRANT_UPSERT_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            use_mock: std::env::var("QDRANT_USE_MOCK")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            timeout_secs: std::env::var("QDRANT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
        }
    }
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/// Counters for experience persistence operations.
#[derive(Debug, Clone, Default)]
pub struct PersistenceMetrics {
    /// Total number of points successfully upserted.
    pub points_upserted: usize,
    /// Total number of rows that could not be converted to points.
    pub rows_failed: usize,
    /// Number of upsert batches sent to Qdrant.
    pub upsert_calls: usize,
    /// Number of upsert calls that failed.
    pub upsert_errors: usize,
}

// ─── Extracted experience row ─────────────────────────────────────────────────

/// A single experience extracted from an Arrow record batch, ready for
/// Qdrant upsert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRow {
    /// The GAF state vector (f32 values decoded from binary).
    pub state_vector: Vec<f32>,
    /// Raw supplementary state features (optional).
    pub state_raw: Option<Vec<f32>>,
    /// Action type index (0=Buy, 1=Sell, 2=Hold, 3=Close).
    pub action_type: u8,
    /// Trading symbol.
    pub action_symbol: String,
    /// Order quantity.
    pub action_qty: f32,
    /// Scalar reward.
    pub reward: f32,
    /// Next-state GAF vector.
    pub next_state_vector: Vec<f32>,
    /// Next-state raw features (optional).
    pub next_state_raw: Option<Vec<f32>>,
    /// Episode termination flag.
    pub done: bool,
    /// Unix epoch milliseconds.
    pub timestamp_ms: i64,
}

// ─── Mock in-memory store ─────────────────────────────────────────────────────

/// In-memory mock storage used during testing or when Qdrant is unavailable.
#[derive(Debug, Default)]
struct MockStore {
    points: Vec<(String, Vec<f32>, HashMap<String, serde_json::Value>)>,
}

impl MockStore {
    fn upsert(&mut self, rows: &[ExperienceRow]) -> usize {
        let mut count = 0;
        for row in rows {
            let id = uuid::Uuid::new_v4().to_string();
            let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
            payload.insert("action_type".into(), serde_json::json!(row.action_type));
            payload.insert("action_symbol".into(), serde_json::json!(row.action_symbol));
            payload.insert("action_qty".into(), serde_json::json!(row.action_qty));
            payload.insert("reward".into(), serde_json::json!(row.reward));
            payload.insert("done".into(), serde_json::json!(row.done));
            payload.insert("timestamp_ms".into(), serde_json::json!(row.timestamp_ms));

            self.points.push((id, row.state_vector.clone(), payload));
            count += 1;
        }
        count
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn clear(&mut self) {
        self.points.clear();
    }
}

// ─── ExperienceStore ──────────────────────────────────────────────────────────

/// Client for persisting experience vectors to Qdrant.
///
/// Supports both a real Qdrant backend and an in-memory mock for testing /
/// environments where Qdrant is unavailable.
pub struct ExperienceStore {
    config: ExperienceStoreConfig,
    client: Option<Qdrant>,
    mock: Arc<RwLock<MockStore>>,
    is_mock: bool,
    metrics: Arc<RwLock<PersistenceMetrics>>,
}

impl ExperienceStore {
    /// Create a new experience store.
    ///
    /// If `config.use_mock` is `true`, no real Qdrant connection is attempted.
    /// Otherwise the constructor tries to connect and falls back to mock mode
    /// on failure.
    pub async fn new(config: ExperienceStoreConfig) -> Result<Self> {
        if config.use_mock {
            info!("ExperienceStore running in mock mode (no Qdrant connection)");
            return Ok(Self {
                config,
                client: None,
                mock: Arc::new(RwLock::new(MockStore::default())),
                is_mock: true,
                metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
            });
        }

        match Qdrant::from_url(&config.qdrant_url)
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
        {
            Ok(client) => {
                info!(url = %config.qdrant_url, "Connected to Qdrant for experience persistence");
                Ok(Self {
                    config,
                    client: Some(client),
                    mock: Arc::new(RwLock::new(MockStore::default())),
                    is_mock: false,
                    metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
                })
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "Failed to connect to Qdrant — falling back to mock mode"
                );
                Ok(Self {
                    config,
                    client: None,
                    mock: Arc::new(RwLock::new(MockStore::default())),
                    is_mock: true,
                    metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
                })
            }
        }
    }

    /// Create a mock-only experience store (convenience for tests).
    pub fn mock() -> Self {
        Self {
            config: ExperienceStoreConfig {
                use_mock: true,
                ..Default::default()
            },
            client: None,
            mock: Arc::new(RwLock::new(MockStore::default())),
            is_mock: true,
            metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
        }
    }

    /// Whether this store is operating in mock mode.
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }

    /// Returns a snapshot of the current persistence metrics.
    pub async fn metrics(&self) -> PersistenceMetrics {
        self.metrics.read().await.clone()
    }

    /// Number of points currently stored (mock mode only — returns 0 for
    /// real Qdrant since we'd need a collection info call).
    pub async fn mock_point_count(&self) -> usize {
        self.mock.read().await.len()
    }

    /// Clear all mock storage.
    pub async fn mock_clear(&self) {
        self.mock.write().await.clear();
    }

    // ── Collection management ─────────────────────────────────────────────

    /// Ensure the Qdrant collection exists, creating it if necessary.
    pub async fn ensure_collection(&self) -> Result<()> {
        if self.is_mock {
            debug!("Mock mode — collection always exists");
            return Ok(());
        }

        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;

        let exists = client
            .collection_exists(&self.config.collection_name)
            .await
            .context("Failed to check if collection exists")?;

        if exists {
            debug!(
                collection = %self.config.collection_name,
                "Experience collection already exists"
            );
            return Ok(());
        }

        info!(
            collection = %self.config.collection_name,
            dim = self.config.vector_dim,
            "Creating experience collection in Qdrant"
        );

        client
            .create_collection(
                CreateCollectionBuilder::new(&self.config.collection_name)
                    .vectors_config(VectorParamsBuilder::new(
                        self.config.vector_dim,
                        Distance::Cosine,
                    ))
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await
            .context("Failed to create Qdrant collection")?;

        info!(
            collection = %self.config.collection_name,
            "Experience collection created"
        );

        Ok(())
    }

    // ── Batch persistence ─────────────────────────────────────────────────

    /// Persist a validated Arrow record batch to the vector store.
    ///
    /// Returns the number of rows successfully persisted.
    pub async fn persist_batch(&self, batch: &RecordBatch) -> Result<usize> {
        let rows = self.extract_rows(batch)?;

        if rows.is_empty() {
            debug!("No valid rows extracted from record batch — nothing to persist");
            return Ok(0);
        }

        let total = rows.len();
        let mut persisted = 0;

        // Chunk into upsert batches
        for chunk in rows.chunks(self.config.upsert_batch_size) {
            match self.upsert_chunk(chunk).await {
                Ok(n) => {
                    persisted += n;
                    let mut m = self.metrics.write().await;
                    m.points_upserted += n;
                    m.upsert_calls += 1;
                }
                Err(e) => {
                    error!(error = %e, chunk_size = chunk.len(), "Failed to upsert experience chunk");
                    let mut m = self.metrics.write().await;
                    m.upsert_errors += 1;
                    m.rows_failed += chunk.len();
                }
            }
        }

        debug!(persisted, total, "Persisted experience batch");
        Ok(persisted)
    }

    /// Persist pre-extracted experience rows directly.
    pub async fn persist_rows(&self, rows: &[ExperienceRow]) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }

        let mut persisted = 0;
        for chunk in rows.chunks(self.config.upsert_batch_size) {
            match self.upsert_chunk(chunk).await {
                Ok(n) => {
                    persisted += n;
                    let mut m = self.metrics.write().await;
                    m.points_upserted += n;
                    m.upsert_calls += 1;
                }
                Err(e) => {
                    error!(error = %e, "Failed to upsert experience rows");
                    let mut m = self.metrics.write().await;
                    m.upsert_errors += 1;
                    m.rows_failed += chunk.len();
                }
            }
        }
        Ok(persisted)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Upsert a chunk of experience rows to Qdrant (or mock).
    async fn upsert_chunk(&self, rows: &[ExperienceRow]) -> Result<usize> {
        if self.is_mock {
            let mut store = self.mock.write().await;
            let n = store.upsert(rows);
            return Ok(n);
        }

        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;

        let points: Vec<PointStruct> = rows.iter().map(|row| self.row_to_point(row)).collect();
        let count = points.len();

        client
            .upsert_points(
                UpsertPointsBuilder::new(&self.config.collection_name, points).wait(true),
            )
            .await
            .context("Qdrant upsert failed")?;

        Ok(count)
    }

    /// Convert a single `ExperienceRow` into a Qdrant `PointStruct`.
    fn row_to_point(&self, row: &ExperienceRow) -> PointStruct {
        use qdrant_client::qdrant::Value;
        use qdrant_client::qdrant::value::Kind;

        let id = uuid::Uuid::new_v4().to_string();

        let mut payload: HashMap<String, Value> = HashMap::new();

        payload.insert(
            "action_type".into(),
            Value {
                kind: Some(Kind::IntegerValue(row.action_type as i64)),
            },
        );
        payload.insert(
            "action_symbol".into(),
            Value {
                kind: Some(Kind::StringValue(row.action_symbol.clone())),
            },
        );
        payload.insert(
            "action_qty".into(),
            Value {
                kind: Some(Kind::DoubleValue(row.action_qty as f64)),
            },
        );
        payload.insert(
            "reward".into(),
            Value {
                kind: Some(Kind::DoubleValue(row.reward as f64)),
            },
        );
        payload.insert(
            "done".into(),
            Value {
                kind: Some(Kind::BoolValue(row.done)),
            },
        );
        payload.insert(
            "timestamp_ms".into(),
            Value {
                kind: Some(Kind::IntegerValue(row.timestamp_ms)),
            },
        );

        // Encode next_state_vector as a JSON string in the payload so it can
        // be recovered during training without a separate lookup.
        let next_state_json =
            serde_json::to_string(&row.next_state_vector).unwrap_or_else(|_| "[]".to_string());
        payload.insert(
            "next_state_vector".into(),
            Value {
                kind: Some(Kind::StringValue(next_state_json)),
            },
        );

        PointStruct::new(id, row.state_vector.clone(), payload)
    }

    /// Extract `ExperienceRow`s from a validated Arrow record batch.
    ///
    /// Rows that cannot be fully decoded (missing required columns, bad binary
    /// length, etc.) are skipped with a warning and counted in the metrics.
    pub fn extract_rows(&self, batch: &RecordBatch) -> Result<Vec<ExperienceRow>> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let schema = batch.schema();

        // ── Resolve column indices ────────────────────────────────────────
        let col_state_gaf = schema
            .index_of("state_gaf")
            .context("Missing required column: state_gaf")?;
        let col_state_raw = schema.index_of("state_raw").ok();
        let col_action_type = schema
            .index_of("action_type")
            .context("Missing required column: action_type")?;
        let col_action_symbol = schema
            .index_of("action_symbol")
            .context("Missing required column: action_symbol")?;
        let col_action_qty = schema
            .index_of("action_qty")
            .context("Missing required column: action_qty")?;
        let col_reward = schema
            .index_of("reward")
            .context("Missing required column: reward")?;
        let col_next_state_gaf = schema
            .index_of("next_state_gaf")
            .context("Missing required column: next_state_gaf")?;
        let col_next_state_raw = schema.index_of("next_state_raw").ok();
        let col_done = schema
            .index_of("done")
            .context("Missing required column: done")?;
        let col_timestamp = schema
            .index_of("timestamp_ms")
            .context("Missing required column: timestamp_ms")?;

        // ── Downcast columns ──────────────────────────────────────────────
        let state_gaf_arr = batch
            .column(col_state_gaf)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("state_gaf column is not Binary")?;

        let action_type_arr = batch
            .column(col_action_type)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .context("action_type column is not UInt8")?;

        let action_symbol_arr = batch
            .column(col_action_symbol)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("action_symbol column is not Utf8")?;

        let action_qty_arr = batch
            .column(col_action_qty)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("action_qty column is not Float32")?;

        let reward_arr = batch
            .column(col_reward)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("reward column is not Float32")?;

        let next_state_gaf_arr = batch
            .column(col_next_state_gaf)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("next_state_gaf column is not Binary")?;

        let done_arr = batch
            .column(col_done)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .context("done column is not Boolean")?;

        let timestamp_arr = batch
            .column(col_timestamp)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("timestamp_ms column is not Int64")?;

        // Optional columns
        let state_raw_arr =
            col_state_raw.and_then(|idx| batch.column(idx).as_any().downcast_ref::<BinaryArray>());

        let next_state_raw_arr = col_next_state_raw
            .and_then(|idx| batch.column(idx).as_any().downcast_ref::<BinaryArray>());

        // ── Extract rows ──────────────────────────────────────────────────
        let mut rows = Vec::with_capacity(num_rows);
        let mut failed = 0usize;

        for i in 0..num_rows {
            // Skip rows with null required fields
            if state_gaf_arr.is_null(i)
                || action_type_arr.is_null(i)
                || action_symbol_arr.is_null(i)
                || action_qty_arr.is_null(i)
                || reward_arr.is_null(i)
                || next_state_gaf_arr.is_null(i)
                || done_arr.is_null(i)
                || timestamp_arr.is_null(i)
            {
                debug!(row = i, "Skipping row with null required field");
                failed += 1;
                continue;
            }

            let state_vector = match decode_f32_le(state_gaf_arr.value(i)) {
                Some(v) => v,
                None => {
                    debug!(row = i, "Failed to decode state_gaf binary to f32 vector");
                    failed += 1;
                    continue;
                }
            };

            let next_state_vector = match decode_f32_le(next_state_gaf_arr.value(i)) {
                Some(v) => v,
                None => {
                    debug!(
                        row = i,
                        "Failed to decode next_state_gaf binary to f32 vector"
                    );
                    failed += 1;
                    continue;
                }
            };

            let state_raw = state_raw_arr.and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    decode_f32_le(arr.value(i))
                }
            });

            let next_state_raw = next_state_raw_arr.and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    decode_f32_le(arr.value(i))
                }
            });

            rows.push(ExperienceRow {
                state_vector,
                state_raw,
                action_type: action_type_arr.value(i),
                action_symbol: action_symbol_arr.value(i).to_string(),
                action_qty: action_qty_arr.value(i),
                reward: reward_arr.value(i),
                next_state_vector,
                next_state_raw,
                done: done_arr.value(i),
                timestamp_ms: timestamp_arr.value(i),
            });
        }

        if failed > 0 {
            let mut m_guard = self.metrics.try_write().unwrap_or_else(|_| {
                // If the lock is poisoned (shouldn't happen with tokio RwLock),
                // just skip updating metrics — not worth panicking.
                unreachable!("tokio RwLock cannot be poisoned")
            });
            m_guard.rows_failed += failed;
            warn!(
                failed,
                extracted = rows.len(),
                "Some rows failed extraction"
            );
        }

        Ok(rows)
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Decode a byte slice as a sequence of little-endian `f32` values.
///
/// Returns `None` if the byte length is not a multiple of 4 (i.e. not
/// aligned to `f32` boundaries).
fn decode_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(values)
}

/// Encode a slice of `f32` values as little-endian bytes.
#[cfg(test)]
fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc as StdArc;

    /// Build the expected experience schema.
    fn experience_schema() -> Schema {
        Schema::new(vec![
            Field::new("state_gaf", DataType::Binary, false),
            Field::new("state_raw", DataType::Binary, true),
            Field::new("action_type", DataType::UInt8, false),
            Field::new("action_symbol", DataType::Utf8, false),
            Field::new("action_qty", DataType::Float32, false),
            Field::new("reward", DataType::Float32, false),
            Field::new("next_state_gaf", DataType::Binary, false),
            Field::new("next_state_raw", DataType::Binary, true),
            Field::new("done", DataType::Boolean, false),
            Field::new("timestamp_ms", DataType::Int64, false),
        ])
    }

    /// Create a test record batch with the given number of rows.
    fn create_test_batch(num_rows: usize) -> RecordBatch {
        let schema = StdArc::new(experience_schema());

        // 9 floats × 4 bytes = 36 bytes per GAF vector
        let gaf_bytes: Vec<u8> = encode_f32_le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let raw_bytes: Vec<u8> = encode_f32_le(&[0.1, 0.2, 0.3]);

        let state_gaf: Vec<&[u8]> = (0..num_rows).map(|_| gaf_bytes.as_slice()).collect();
        let state_raw: Vec<Option<&[u8]>> =
            (0..num_rows).map(|_| Some(raw_bytes.as_slice())).collect();
        let next_state_gaf: Vec<&[u8]> = (0..num_rows).map(|_| gaf_bytes.as_slice()).collect();
        let next_state_raw: Vec<Option<&[u8]>> =
            (0..num_rows).map(|_| Some(raw_bytes.as_slice())).collect();

        let action_types: Vec<u8> = (0..num_rows).map(|i| (i % 4) as u8).collect();
        let symbols: Vec<&str> = (0..num_rows).map(|_| "BTCUSD").collect();
        let quantities: Vec<f32> = (0..num_rows).map(|i| (i + 1) as f32 * 0.1).collect();
        let rewards: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let dones: Vec<bool> = (0..num_rows).map(|i| i == num_rows - 1).collect();
        let timestamps: Vec<i64> = (0..num_rows)
            .map(|i| 1_700_000_000_000_i64 + (i as i64) * 1000)
            .collect();

        RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(BinaryArray::from_vec(state_gaf)),
                StdArc::new(BinaryArray::from(state_raw)),
                StdArc::new(UInt8Array::from(action_types)),
                StdArc::new(StringArray::from(symbols)),
                StdArc::new(Float32Array::from(quantities)),
                StdArc::new(Float32Array::from(rewards)),
                StdArc::new(BinaryArray::from_vec(next_state_gaf)),
                StdArc::new(BinaryArray::from(next_state_raw)),
                StdArc::new(BooleanArray::from(dones)),
                StdArc::new(Int64Array::from(timestamps)),
            ],
        )
        .expect("Failed to create test RecordBatch")
    }

    // ── Unit tests ────────────────────────────────────────────────────────

    #[test]
    fn test_decode_f32_le_round_trip() {
        let values = vec![1.0_f32, 2.5, -std::f32::consts::PI, 0.0];
        let encoded = encode_f32_le(&values);
        let decoded = decode_f32_le(&encoded).unwrap();
        assert_eq!(values.len(), decoded.len());
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_decode_f32_le_bad_length() {
        assert!(decode_f32_le(&[0u8; 5]).is_none());
        assert!(decode_f32_le(&[0u8; 3]).is_none());
        assert!(decode_f32_le(&[0u8; 1]).is_none());
    }

    #[test]
    fn test_decode_f32_le_empty() {
        let decoded = decode_f32_le(&[]).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = ExperienceStoreConfig::default();
        assert_eq!(config.collection_name, "experiences");
        assert_eq!(config.vector_dim, 9);
        assert!(config.use_mock);
    }

    #[test]
    fn test_extract_rows_from_batch() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(5);
        let rows = store.extract_rows(&batch).unwrap();

        assert_eq!(rows.len(), 5);

        // Check first row
        assert_eq!(rows[0].state_vector.len(), 9);
        assert!((rows[0].state_vector[0] - 1.0).abs() < 1e-6);
        assert_eq!(rows[0].action_type, 0);
        assert_eq!(rows[0].action_symbol, "BTCUSD");
        assert!(!rows[0].done);
        assert_eq!(rows[0].timestamp_ms, 1_700_000_000_000);

        // Last row should be terminal
        assert!(rows[4].done);
    }

    #[test]
    fn test_extract_rows_empty_batch() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(0);
        let rows = store.extract_rows(&batch).unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn test_mock_persist_batch() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(10);

        let persisted = store.persist_batch(&batch).await.unwrap();
        assert_eq!(persisted, 10);

        assert_eq!(store.mock_point_count().await, 10);

        let m = store.metrics().await;
        assert_eq!(m.points_upserted, 10);
        assert_eq!(m.upsert_calls, 1);
        assert_eq!(m.upsert_errors, 0);
    }

    #[tokio::test]
    async fn test_mock_persist_rows() {
        let store = ExperienceStore::mock();

        let rows = vec![
            ExperienceRow {
                state_vector: vec![1.0; 9],
                state_raw: None,
                action_type: 0,
                action_symbol: "ETHUSD".into(),
                action_qty: 1.0,
                reward: 0.5,
                next_state_vector: vec![2.0; 9],
                next_state_raw: None,
                done: false,
                timestamp_ms: 1_700_000_000_000,
            },
            ExperienceRow {
                state_vector: vec![3.0; 9],
                state_raw: Some(vec![0.1, 0.2]),
                action_type: 1,
                action_symbol: "BTCUSD".into(),
                action_qty: 0.5,
                reward: -0.2,
                next_state_vector: vec![4.0; 9],
                next_state_raw: Some(vec![0.3, 0.4]),
                done: true,
                timestamp_ms: 1_700_000_001_000,
            },
        ];

        let persisted = store.persist_rows(&rows).await.unwrap();
        assert_eq!(persisted, 2);
        assert_eq!(store.mock_point_count().await, 2);
    }

    #[tokio::test]
    async fn test_mock_persist_batch_chunked() {
        let config = ExperienceStoreConfig {
            upsert_batch_size: 3, // force chunking
            ..Default::default()
        };

        let store = ExperienceStore::new(config).await.unwrap();
        let batch = create_test_batch(10);

        let persisted = store.persist_batch(&batch).await.unwrap();
        assert_eq!(persisted, 10);
        assert_eq!(store.mock_point_count().await, 10);

        let m = store.metrics().await;
        // 10 rows / 3 per chunk = 4 upsert calls (3+3+3+1)
        assert_eq!(m.upsert_calls, 4);
    }

    #[tokio::test]
    async fn test_mock_clear() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(5);
        store.persist_batch(&batch).await.unwrap();
        assert_eq!(store.mock_point_count().await, 5);

        store.mock_clear().await;
        assert_eq!(store.mock_point_count().await, 0);
    }

    #[tokio::test]
    async fn test_ensure_collection_mock() {
        let store = ExperienceStore::mock();
        // Should not error in mock mode
        store.ensure_collection().await.unwrap();
    }

    #[test]
    fn test_row_to_point() {
        let store = ExperienceStore::mock();
        let row = ExperienceRow {
            state_vector: vec![1.0, 2.0, 3.0],
            state_raw: None,
            action_type: 2,
            action_symbol: "BTCUSD".into(),
            action_qty: 0.5,
            reward: 1.23,
            next_state_vector: vec![4.0, 5.0, 6.0],
            next_state_raw: None,
            done: false,
            timestamp_ms: 1_700_000_000_000,
        };

        let point = store.row_to_point(&row);

        // Should have a UUID-format ID
        assert!(point.id.is_some());

        // Check payload fields exist
        let payload = &point.payload;
        assert!(payload.contains_key("action_type"));
        assert!(payload.contains_key("action_symbol"));
        assert!(payload.contains_key("reward"));
        assert!(payload.contains_key("done"));
        assert!(payload.contains_key("timestamp_ms"));
        assert!(payload.contains_key("next_state_vector"));
    }

    #[test]
    fn test_experience_row_serde() {
        let row = ExperienceRow {
            state_vector: vec![1.0, 2.0],
            state_raw: Some(vec![0.5]),
            action_type: 1,
            action_symbol: "ETHUSD".into(),
            action_qty: 2.0,
            reward: -0.1,
            next_state_vector: vec![3.0, 4.0],
            next_state_raw: None,
            done: true,
            timestamp_ms: 123_456_789,
        };

        let json = serde_json::to_string(&row).unwrap();
        let deserialized: ExperienceRow = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.action_symbol, "ETHUSD");
        assert_eq!(deserialized.state_vector.len(), 2);
        assert!(deserialized.done);
    }
}
