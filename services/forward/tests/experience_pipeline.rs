//! End-to-end experience pipeline test (EXPERIENCE_PIPELINE.md §7 item 5):
//! synthetic candles → `ExperienceRecorder` → `ExperienceWriter` → a real
//! spool file → **backward's real `handle_ingest`** (dry-run store) —
//! proving the producer and the already-merged consumer agree on the frozen
//! Arrow contract, so schema drift fails CI instead of corrupting ingestion.

use janus_backward::tasks::ingest::handle_ingest;
use janus_backward::worker::IngestJob;
use janus_forward::experience::{
    DecisionCtx, ExperienceConfig, ExperienceRecorder, GAF_K, GAF_WINDOW, spawn_writer,
};

const MIN_MS: i64 = 60_000;

#[tokio::test]
async fn producer_batches_ingest_through_the_real_consumer() {
    let spool = tempfile::tempdir().expect("tempdir");
    // Deterministic flush: exactly the number of completed rows we drive.
    let completed_rows = 8usize;
    let cfg = ExperienceConfig {
        enabled: true,
        spool_dir: spool.path().to_path_buf(),
        batch_rows: completed_rows,
        flush_secs: 3600,
        spool_max_mb: 16,
    };
    let (tx, writer_task) = spawn_writer(cfg, None); // no Redis: sweep-mode, file is truth

    let mut rec = ExperienceRecorder::new(tx);
    let key = "BTCUSDT:1m";
    let total_bars = GAF_WINDOW + completed_rows;
    let mut t: i64 = 1_700_000_000_000;
    for i in 0..total_bars {
        // A gentle sine keeps the GAF windows non-degenerate; one Buy and one
        // Sell decision land after warmup, the rest are Holds.
        let close = 100.0 + (i as f64 * 0.7).sin() * 5.0;
        let ctx = if i == GAF_WINDOW + 2 {
            DecisionCtx {
                action: janus_core::SignalType::Buy,
                confidence: 0.9,
                blocked: None,
                regime: None,
            }
        } else if i == GAF_WINDOW + 4 {
            DecisionCtx {
                action: janus_core::SignalType::Sell,
                confidence: 0.8,
                blocked: Some("gate: test-block".to_string()),
                regime: None,
            }
        } else {
            DecisionCtx::hold(None)
        };
        rec.observe(key, "BTCUSDT", "1m", t, close, ctx);
        t += MIN_MS;
    }

    // The writer flushes when the batch fills; poll for the finished file.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let spool_file = loop {
        let found = std::fs::read_dir(spool.path())
            .expect("read spool dir")
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().and_then(|e| e.to_str()) == Some("arrow"));
        if let Some(p) = found {
            break p;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "writer never flushed a finished .arrow batch"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };

    // Feed the file to the REAL consumer (dry-run: validate + count).
    let batch_id = spool_file
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("batch id from filename")
        .to_string();
    let job = IngestJob {
        batch_id,
        shm_path: spool_file.display().to_string(),
    };
    let metrics = handle_ingest(job, None).await.expect("ingest succeeds");

    assert_eq!(metrics.batches_read, 1);
    assert_eq!(
        metrics.rows_ingested, completed_rows,
        "every completed decision transition ingests"
    );
    assert_eq!(
        metrics.rows_skipped, 0,
        "no row fails the consumer's validation"
    );
    assert_eq!(
        metrics.schema_fields_matched, metrics.schema_fields_expected,
        "producer schema matches the frozen consumer schema exactly"
    );

    // Content spot-checks straight off the file: vector dims + action codes.
    let file = std::fs::File::open(&spool_file).expect("open spool file");
    let reader = arrow::ipc::reader::FileReader::try_new(file, None).expect("footer complete");
    let schema = reader.schema();
    assert_eq!(
        schema.metadata().get("schema_version").map(String::as_str),
        Some("1")
    );
    // The reward decomposition (§4g) rides the same `rows_meta` side-channel as
    // confidence/blocked and must survive the real batched write → file round
    // trip, so a future verdict script can read gross vs fee per point. Every
    // completed row carries a numeric reward_gross + fee_sigma (legacy scheme
    // here: fee_sigma == 0 and gross == net, but the fields are always present).
    let rows_meta_json = schema
        .metadata()
        .get("rows_meta")
        .expect("producer emits the rows_meta side-channel");
    let rows_meta: Vec<serde_json::Value> =
        serde_json::from_str(rows_meta_json).expect("rows_meta is a JSON array");
    assert_eq!(
        rows_meta.len(),
        completed_rows,
        "one rows_meta entry per completed row"
    );
    assert!(
        rows_meta.iter().all(|e| {
            e.get("reward_gross").and_then(|v| v.as_f64()).is_some()
                && e.get("fee_sigma").and_then(|v| v.as_f64()).is_some()
        }),
        "every row carries a numeric reward_gross + fee_sigma decomposition: {rows_meta:?}"
    );
    let mut action_codes = Vec::new();
    let mut gaf_lens = Vec::new();
    for batch in reader {
        let batch = batch.expect("readable batch");
        let actions = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::UInt8Array>()
            .expect("action_type is u8");
        let gafs = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::BinaryArray>()
            .expect("state_gaf is binary");
        for i in 0..batch.num_rows() {
            action_codes.push(actions.value(i));
            gaf_lens.push(gafs.value(i).len());
        }
    }
    assert!(
        gaf_lens.iter().all(|&l| l == GAF_K * GAF_K * 4),
        "every state_gaf is k²=9 f32-LE values (36 bytes): {gaf_lens:?}"
    );
    assert!(
        action_codes.contains(&janus_core::SignalType::Buy.to_u8()),
        "the Buy decision is in the batch"
    );
    assert!(
        action_codes.contains(&janus_core::SignalType::Sell.to_u8()),
        "the Sell decision is in the batch"
    );

    drop(rec);
    let _ = writer_task.await;
}
