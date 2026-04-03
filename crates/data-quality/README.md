# JANUS Data Quality

Comprehensive data quality pipeline for validating, detecting anomalies,
and ensuring data integrity across all market data sources.

## Status

Files to copy from `docs/WEEK3_SOURCE_CODE.md`:

- [ ] src/validators/orderbook.rs
- [ ] src/anomaly/mod.rs
- [ ] src/anomaly/statistical.rs
- [ ] src/anomaly/sequence.rs
- [ ] src/anomaly/latency.rs
- [ ] src/pipeline.rs
- [ ] src/metrics.rs (if using CNS)
- [ ] src/export/mod.rs
- [ ] src/export/parquet.rs

## Quick Reference

Each file can be found in `docs/WEEK3_SOURCE_CODE.md` by searching for:

```
### crates/data-quality/src/[filename]
```

Then copy the code between the ```rust markers.

## Build Status

After copying all files, run:

```bash
cargo build --package janus-data-quality
cargo test --package janus-data-quality
```
