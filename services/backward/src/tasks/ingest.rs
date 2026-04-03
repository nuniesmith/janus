//! Ingest task - reads from shared memory and persists experience data.
//!
//! This task is responsible for:
//! 1. Opening an Arrow IPC file from a shared memory path
//! 2. Reading and deserializing experience batches
//! 3. Validating the schema and data integrity
//! 4. Storing parsed experiences to Qdrant vector database via [`ExperienceStore`]
//!
//! ## Architecture
//!
//! ```text
//! Shared Memory (Arrow IPC)
//!       │
//!       ▼
//! ┌─────────────────┐
//! │  Open IPC File   │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Validate Schema  │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Parse Batches   │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Persist to      │
//! │  Qdrant VectorDB │
//! └─────────────────┘
//! ```

use anyhow::{Context, Result};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::path::Path;
use tracing::{debug, error, info, warn};

use crate::persistence::ExperienceStore;
use crate::worker::IngestJob;

/// Expected schema for experience Arrow IPC files.
///
/// Fields:
/// - `state_gaf`     : Binary   — flattened GAF feature vector (f32 le bytes)
/// - `state_raw`     : Binary   — raw supplementary features (f32 le bytes)
/// - `action_type`   : UInt8    — 0=Buy, 1=Sell, 2=Hold, 3=Close
/// - `action_symbol` : Utf8     — trading symbol
/// - `action_qty`    : Float32  — quantity
/// - `reward`        : Float32  — scalar reward
/// - `next_state_gaf`: Binary   — next-state GAF features
/// - `next_state_raw`: Binary   — next-state raw features
/// - `done`          : Boolean  — episode termination flag
/// - `timestamp_ms`  : Int64    — unix epoch millis
fn expected_schema() -> Schema {
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

/// Metrics collected during an ingest run.
#[derive(Debug, Clone, Default)]
pub struct IngestMetrics {
    /// Total number of record batches read from the IPC file
    pub batches_read: usize,
    /// Total number of rows (experiences) across all batches
    pub rows_ingested: usize,
    /// Number of rows that failed validation and were skipped
    pub rows_skipped: usize,
    /// Number of schema fields that matched expectations
    pub schema_fields_matched: usize,
    /// Total schema fields expected
    pub schema_fields_expected: usize,
    /// Number of rows successfully persisted to the vector store
    pub rows_persisted: usize,
    /// Number of rows that failed persistence
    pub rows_persist_failed: usize,
}

/// Handle an ingest job dispatched by the worker pool.
///
/// Reads an Arrow IPC file at the path specified in the job, validates its
/// schema, iterates over record batches, and persists valid experiences to
/// the Qdrant vector database via the provided [`ExperienceStore`].
///
/// If `store` is `None`, the function validates and counts rows but does not
/// persist them (legacy behaviour, useful for dry-run / testing).
#[allow(dead_code)]
pub async fn handle_ingest(job: IngestJob, store: Option<&ExperienceStore>) -> Result<()> {
    info!(
        batch_id = %job.batch_id,
        shm_path = %job.shm_path,
        has_store = store.is_some(),
        "Starting ingest job"
    );

    let mut metrics = IngestMetrics::default();

    // ── 1. Open the Arrow IPC file ────────────────────────────────────────
    let ipc_path = Path::new(&job.shm_path);

    if !ipc_path.exists() {
        warn!(
            batch_id = %job.batch_id,
            path = %job.shm_path,
            "IPC file does not exist — skipping ingest"
        );
        return Ok(());
    }

    let file = File::open(ipc_path).with_context(|| {
        format!(
            "Failed to open Arrow IPC file at '{}' for batch {}",
            job.shm_path, job.batch_id
        )
    })?;

    let reader = FileReader::try_new(file, None).with_context(|| {
        format!(
            "Failed to create Arrow IPC reader for batch {}",
            job.batch_id
        )
    })?;

    info!(
        batch_id = %job.batch_id,
        num_batches = reader.num_batches(),
        "Opened Arrow IPC file"
    );

    // ── 2. Validate schema ────────────────────────────────────────────────
    let file_schema = reader.schema();
    let expected = expected_schema();

    metrics.schema_fields_expected = expected.fields().len();
    metrics.schema_fields_matched = validate_schema(&file_schema, &expected);

    if metrics.schema_fields_matched < metrics.schema_fields_expected {
        warn!(
            batch_id = %job.batch_id,
            matched = metrics.schema_fields_matched,
            expected = metrics.schema_fields_expected,
            "Schema mismatch — some expected fields are missing or have wrong types"
        );
    } else {
        debug!(
            batch_id = %job.batch_id,
            "Schema validation passed"
        );
    }

    // ── 3. Ensure vector store collection exists ──────────────────────────
    if let Some(s) = store
        && let Err(e) = s.ensure_collection().await
    {
        warn!(
            batch_id = %job.batch_id,
            error = %e,
            "Failed to ensure Qdrant collection — persistence may fail"
        );
    }

    // ── 4. Read and process record batches ────────────────────────────────
    for batch_result in reader {
        match batch_result {
            Ok(batch) => {
                metrics.batches_read += 1;
                let batch_rows = batch.num_rows();

                debug!(
                    batch_id = %job.batch_id,
                    batch_num = metrics.batches_read,
                    rows = batch_rows,
                    "Processing record batch"
                );

                match process_record_batch(&batch) {
                    Ok(valid_rows) => {
                        let skipped = batch_rows - valid_rows;
                        metrics.rows_ingested += valid_rows;
                        metrics.rows_skipped += skipped;

                        if skipped > 0 {
                            warn!(
                                batch_id = %job.batch_id,
                                batch_num = metrics.batches_read,
                                skipped = skipped,
                                "Some rows failed validation"
                            );
                        }

                        // ── 5. Persist to vector database ─────────────
                        if let Some(s) = store {
                            match s.persist_batch(&batch).await {
                                Ok(persisted) => {
                                    metrics.rows_persisted += persisted;
                                    debug!(
                                        batch_id = %job.batch_id,
                                        batch_num = metrics.batches_read,
                                        persisted,
                                        "Persisted batch to vector store"
                                    );
                                }
                                Err(e) => {
                                    error!(
                                        batch_id = %job.batch_id,
                                        batch_num = metrics.batches_read,
                                        error = %e,
                                        "Failed to persist batch to vector store"
                                    );
                                    metrics.rows_persist_failed += valid_rows;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(
                            batch_id = %job.batch_id,
                            batch_num = metrics.batches_read,
                            error = %e,
                            "Failed to process record batch — skipping entire batch"
                        );
                        metrics.rows_skipped += batch_rows;
                    }
                }
            }
            Err(e) => {
                error!(
                    batch_id = %job.batch_id,
                    error = %e,
                    "Failed to read record batch from IPC file"
                );
            }
        }
    }

    info!(
        batch_id = %job.batch_id,
        batches_read = metrics.batches_read,
        rows_ingested = metrics.rows_ingested,
        rows_skipped = metrics.rows_skipped,
        rows_persisted = metrics.rows_persisted,
        rows_persist_failed = metrics.rows_persist_failed,
        schema_match = format!("{}/{}", metrics.schema_fields_matched, metrics.schema_fields_expected),
        "Ingest job completed"
    );

    Ok(())
}

/// Validate that the file schema contains all expected fields with correct types.
///
/// Returns the number of fields that matched.
fn validate_schema(file_schema: &Schema, expected: &Schema) -> usize {
    let mut matched = 0;
    for expected_field in expected.fields() {
        match file_schema.field_with_name(expected_field.name()) {
            Ok(actual_field) => {
                if actual_field.data_type() == expected_field.data_type() {
                    matched += 1;
                } else {
                    warn!(
                        field = expected_field.name().as_str(),
                        expected_type = ?expected_field.data_type(),
                        actual_type = ?actual_field.data_type(),
                        "Field type mismatch"
                    );
                }
            }
            Err(_) => {
                debug!(
                    field = expected_field.name().as_str(),
                    "Expected field not found in file schema"
                );
            }
        }
    }
    matched
}

/// Process a single record batch, performing per-row validation.
///
/// Returns the count of valid rows. Invalid rows are logged and skipped.
fn process_record_batch(batch: &RecordBatch) -> Result<usize> {
    let num_rows = batch.num_rows();
    let num_cols = batch.num_columns();

    if num_rows == 0 {
        debug!("Empty record batch — nothing to process");
        return Ok(0);
    }

    debug!(
        rows = num_rows,
        columns = num_cols,
        "Validating record batch"
    );

    let mut valid_count = 0;
    let schema = batch.schema();

    // Validate that required columns have no nulls
    for row_idx in 0..num_rows {
        let mut row_valid = true;

        for col_idx in 0..num_cols {
            let column = batch.column(col_idx);
            let field = schema.field(col_idx);

            // Check nullability constraints
            if !field.is_nullable() && column.is_null(row_idx) {
                debug!(
                    row = row_idx,
                    column = field.name().as_str(),
                    "Non-nullable column has null value"
                );
                row_valid = false;
                break;
            }
        }

        if row_valid {
            valid_count += 1;
        }
    }

    debug!(
        valid = valid_count,
        total = num_rows,
        "Record batch validation complete"
    );

    Ok(valid_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array,
    };
    use arrow::datatypes::Schema;
    use arrow::ipc::writer::FileWriter;
    use std::sync::Arc;
    use uuid::Uuid;

    /// Helper to create a valid record batch matching the expected schema.
    fn create_test_batch(num_rows: usize) -> RecordBatch {
        let schema = Arc::new(expected_schema());

        let state_gaf: Vec<&[u8]> = (0..num_rows).map(|_| &[0u8; 36][..]).collect();
        let state_raw: Vec<Option<&[u8]>> = (0..num_rows).map(|_| Some(&[0u8; 12][..])).collect();
        let next_state_gaf: Vec<&[u8]> = (0..num_rows).map(|_| &[0u8; 36][..]).collect();
        let next_state_raw: Vec<Option<&[u8]>> =
            (0..num_rows).map(|_| Some(&[0u8; 12][..])).collect();

        let action_types: Vec<u8> = (0..num_rows).map(|i| (i % 4) as u8).collect();
        let symbols: Vec<&str> = (0..num_rows).map(|_| "BTCUSD").collect();
        let quantities: Vec<f32> = (0..num_rows).map(|i| (i + 1) as f32 * 0.1).collect();
        let rewards: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let dones: Vec<bool> = (0..num_rows).map(|i| i == num_rows - 1).collect();
        let timestamps: Vec<i64> = (0..num_rows)
            .map(|i| 1700000000000_i64 + (i as i64) * 1000)
            .collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BinaryArray::from_vec(state_gaf)),
                Arc::new(BinaryArray::from(state_raw)),
                Arc::new(UInt8Array::from(action_types)),
                Arc::new(StringArray::from(symbols)),
                Arc::new(Float32Array::from(quantities)),
                Arc::new(Float32Array::from(rewards)),
                Arc::new(BinaryArray::from_vec(next_state_gaf)),
                Arc::new(BinaryArray::from(next_state_raw)),
                Arc::new(BooleanArray::from(dones)),
                Arc::new(Int64Array::from(timestamps)),
            ],
        )
        .expect("Failed to create test RecordBatch")
    }

    /// Write a test IPC file to a temp directory and return the path.
    fn write_test_ipc(batches: &[RecordBatch]) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "janus_test_ingest_{}_{}.ipc",
            std::process::id(),
            Uuid::new_v4()
        ));
        let file = File::create(&path).expect("Failed to create temp file");
        let schema = batches[0].schema();

        let mut writer = FileWriter::try_new(file, &schema).unwrap();

        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();

        path
    }

    #[test]
    fn test_expected_schema_has_correct_field_count() {
        let schema = expected_schema();
        assert_eq!(schema.fields().len(), 10);
    }

    #[test]
    fn test_validate_schema_full_match() {
        let schema = expected_schema();
        let matched = validate_schema(&schema, &schema);
        assert_eq!(matched, 10);
    }

    #[test]
    fn test_validate_schema_partial_match() {
        let partial = Schema::new(vec![
            Field::new("state_gaf", DataType::Binary, false),
            Field::new("reward", DataType::Float32, false),
        ]);
        let expected = expected_schema();
        let matched = validate_schema(&partial, &expected);
        assert_eq!(matched, 2);
    }

    #[test]
    fn test_validate_schema_type_mismatch() {
        let wrong = Schema::new(vec![
            Field::new("state_gaf", DataType::Utf8, false), // wrong type
            Field::new("reward", DataType::Float32, false),
        ]);
        let expected = expected_schema();
        let matched = validate_schema(&wrong, &expected);
        // Only reward matches; state_gaf has wrong type
        assert_eq!(matched, 1);
    }

    #[test]
    fn test_process_record_batch_all_valid() {
        let batch = create_test_batch(10);
        let valid = process_record_batch(&batch).unwrap();
        assert_eq!(valid, 10);
    }

    #[test]
    fn test_process_record_batch_empty() {
        let batch = create_test_batch(0);
        let valid = process_record_batch(&batch).unwrap();
        assert_eq!(valid, 0);
    }

    #[tokio::test]
    async fn test_handle_ingest_missing_file() {
        let job = IngestJob {
            batch_id: "test-missing".to_string(),
            shm_path: "/tmp/nonexistent_arrow_file.ipc".to_string(),
        };
        // Should succeed (skip gracefully) when file doesn't exist
        let result = handle_ingest(job, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_ingest_valid_file() {
        let batch = create_test_batch(5);
        let path = write_test_ipc(&[batch]);
        let path_str = path.to_string_lossy().to_string();

        let job = IngestJob {
            batch_id: "test-valid".to_string(),
            shm_path: path_str,
        };

        let result = handle_ingest(job, None).await;
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "Ingest failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_handle_ingest_multiple_batches() {
        let batch1 = create_test_batch(3);
        let batch2 = create_test_batch(7);
        let path = write_test_ipc(&[batch1, batch2]);
        let path_str = path.to_string_lossy().to_string();

        let job = IngestJob {
            batch_id: "test-multi".to_string(),
            shm_path: path_str,
        };

        let result = handle_ingest(job, None).await;
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "Ingest failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_handle_ingest_with_mock_store() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(8);
        let path = write_test_ipc(&[batch]);
        let path_str = path.to_string_lossy().to_string();

        let job = IngestJob {
            batch_id: "test-with-store".to_string(),
            shm_path: path_str,
        };

        let result = handle_ingest(job, Some(&store)).await;
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "Ingest with store failed: {:?}",
            result.err()
        );

        // Verify rows were persisted to mock store
        let count = store.mock_point_count().await;
        assert_eq!(count, 8, "Expected 8 points persisted, got {count}");

        let m = store.metrics().await;
        assert_eq!(m.points_upserted, 8);
        assert_eq!(m.upsert_errors, 0);
    }

    #[tokio::test]
    async fn test_handle_ingest_with_store_multiple_batches() {
        let store = ExperienceStore::mock();
        let batch1 = create_test_batch(4);
        let batch2 = create_test_batch(6);
        let path = write_test_ipc(&[batch1, batch2]);
        let path_str = path.to_string_lossy().to_string();

        let job = IngestJob {
            batch_id: "test-multi-store".to_string(),
            shm_path: path_str,
        };

        let result = handle_ingest(job, Some(&store)).await;
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "Ingest with store failed: {:?}",
            result.err()
        );

        let count = store.mock_point_count().await;
        assert_eq!(count, 10, "Expected 10 points persisted (4+6), got {count}");
    }
}
