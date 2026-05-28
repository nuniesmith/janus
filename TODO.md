# fks-janus — TODO

> **Repo:** `github.com/nuniesmith/janus`
> **Scope:** janus-only (Rust ML inference + neuromorphic + signal generation).
> Infrastructure orchestration lives in [fks](https://github.com/nuniesmith/fks);
> rustcode / claw / RC-CRATES items live under `fks/src/rustcode/` and are
> **not** tracked here anymore.
> **Last synced:** 2026-05-24

---

## P0 — Codebase Health

- [ ] **303 `#[allow(dead_code)]` annotations** (down from 318) — audit results (2026-05-25):
  - `services/api/src/grpc.rs` + `services/forward/src/api/grpc.rs` — deleted in #4 (confirmed dead scaffolding).
  - `services/optimizer/src/collector.rs` — consolidated 10 → 6 in #10. Remaining: `datetime`, `DataGap`, `get_date_range`, `detect_gaps`, `vacuum`, and the whole `impl CollectionMetadata` block — all genuinely unused but useful API surface.
  - `services/optimizer/src/scheduler.rs` (7) — empirical sweep confirmed every annotation is needed (`with_failure_config`, `get_state`, `get_stats`, `stop`, `is_running`, `interval`, `interval_str` accessors on `OptimizationScheduler` are real but unused).
  - `services/data/src/api/auth.rs` (7), `services/api/src/rate_limit.rs` (7), `services/data/src/lib.rs` (6) — also empirically confirmed; each annotation guards a genuinely unused item (mostly disabled-mode constructors and serde-only struct fields).
  - `neuromorphic/thalamus/sources/clients/openweathermap.rs` (13) — new hotspot; not yet audited.
  - `services/execution/tests/integration/scenarios.rs` (7) — test scaffolding, low priority.
- [ ] **Tonic version split** — workspace declares `0.14.2` but some crates resolve `0.10.2` transitively (via `apalis`). Track and resolve when `apalis` hits 1.0 stable.
- [ ] Evaluate `shared_memory` IPC in containers — `/dev/shm` size limits may break Forward→Backward zero-copy Arrow IPC (removed crate, but protocol design question remains).
- [ ] **STRUCT-C deferred:** Consolidate stray `services/forward/proto/janus/v1/janus.proto` → `proto/fks/janus/v1/signal_service.proto` once GrpcServer ownership is decided. See `services/forward/build.rs:14-19`.

---

## P1 — Signal Flow (JFLOW)

### JFLOW-A: Session metrics wiring  *(complete 2026-05-24)*
- [x] `SessionMetricsClient` + `SessionMetrics` types in `lib/janus-core/src/session_metrics.rs` (no-op when `JANUS_AI_URL` is unset).
- [x] Session metrics reporter loop in `services/forward/src/lib.rs::start_module` — snapshots `SignalGenerator::metrics()` every `JANUS_AI_PUSH_SECS` (default 60s) and pushes to JanusAI. Session id from `JANUS_SESSION_ID` env or a generated UUID.
- [x] Confidence accumulator + latency ring (1024 sample window) on `SignalMetrics` — both `generate_from_analysis` and `generate_from_indicators` now feed it via `record_generated(confidence, elapsed_us)`. Reporter sends real `avg_confidence` / `p50_latency_us` / `p99_latency_us`.
- [x] `JanusState::{current_regime, set_current_regime}` plumbed in. Signal-rx loop in `start_module` opportunistically captures `signal.metadata["regime"]` and pushes it into state. Reporter reads it for the `regime` field. Producers that want richer behaviour can call `state.set_current_regime(...)` directly from the brain pipeline.

### JFLOW-B: Dynamic asset config from Ruby  *(overlay landed 2026-05-24)*
- [x] Janus startup config overlay from Redis: `Config::load()` now reads key `fks:janus:config` after env overrides (`lib/janus-core/src/config.rs::apply_redis_overlay`).
- [ ] When a JanusAI session starts, write session-specific config to Redis (`fks:janus:config`) — **producer side, lives in fks repo**.
- [ ] Optimizer reads asset list from Ruby's asset registry via gRPC or Redis (currently env-only).

### JFLOW-C: Two-way position feedback (remaining)
- [x] **Receive path foundation (2026-05-25, #13)**: `PositionEvent` wire type in `lib/janus-core/src/position_events.rs`; `POST /api/v1/positions/event` on janus-api validates and logs.
- [x] **Raw event persistence (2026-05-25, #14)**: `PositionEventStore` in `lib/janus-api/src/position_store.rs` writes each received event into the Postgres `janus_position_events` ingest log. Best-effort: missing table or unreachable DB → disabled with warn (mirrors JFLOW-D probe pattern). Schema owned by the JanusAI Python service.
- [x] **Guidance stub (2026-05-25, #15)**: `compute_guidance(event, regime)` in `position_events.rs` returns hold/reduce/exit + reason from current regime + ±5% / -2% unrealized P&L thresholds; handler reads `state.current_regime()` and includes guidance in the 202 response.
- [x] **Guidance ← OptimizedParams (2026-05-25, #16)**: `GuidanceThresholds::from_optimized_params` lets per-asset optimizer-tuned `take_profit_pct` override the default 5%. `ParamManager` plumbed into janus-api as an Extension. Handler extracts the per-asset base symbol with `base_asset(&event.symbol)`. Stop-loss ratio stays at the conservative default until `OptimizedParams` grows an explicit `stop_loss_pct` field (Python-side coordination).
- [x] **Redis bootstrap load (2026-05-25, #17)**: `janus-api::start_module` now best-effort `HGETALL`s `fks:{instance}:optimized_params` at boot so production guidance actually uses tuned thresholds. Missing/empty hash or unreachable Redis → warn + fall back to defaults.
- [x] **Live param-update subscription (2026-05-25)**: `janus-api::param_updates` spawns a tokio task that subscribes to `fks:{instance}:param_updates` and applies notifications to the shared `ParamManager` via `process_notification`. Exponential backoff with reconnect on Redis disconnect; aborts cleanly on shutdown.
- [ ] Dedupe with `services/forward`: both modules now subscribe to the same Redis channel + maintain separate `ParamManager` caches. Promoting `ParamManager` onto `JanusState` would collapse them to one — deferred since it touches forward's `ParamReloadManager` (which also owns appliers).
- [x] **`stop_loss_pct` in `OptimizedParams` (2026-05-28)**: New field with serde default `2.0` (matching the old hardcoded `-2%` threshold). `GuidanceThresholds::from_optimized_params` now maps both `stop_loss_pct` and `take_profit_pct` to their signed ratios. Validation rejects out-of-range values. Python optimizer schema bump needed to start emitting the field; until then serde default keeps existing behaviour.
- [ ] Smarter guidance: volatility from market data, exit urgency from amygdala.
- [ ] Compact `janus_position_events` raw log into closed-trade rows in `janus_memories` (JanusAI side, fks repo).

### JFLOW-D: Startup bootstrap  *(direct Postgres path landed 2026-05-24)*
- [x] `bootstrap_affinity_from_postgres()` queries `janus_memories` via `sqlx` and replays into the strategy-affinity tracker. Probes for the table first so a missing schema isn't reported as an error.
- [x] Wired in `services/forward/src/main.rs` as the preferred path when `DATABASE_URL` is set; falls through to the Redis ring buffer on empty / missing / errored Postgres.
- [x] `persistence` feature added to forward's `default = […]` so sqlx is compiled in by default. Opt out with `--no-default-features`.

---

## P1 — Janus AI (remaining)

- [ ] Wire signal pipeline (JFLOW-A) to actually emit metrics — see JFLOW-A above. Endpoint contract lives in JanusAI service (Python, in fks repo).

---

## P2 — Build & Deployment

- [x] Standalone `Dockerfile` for the unified `janus` binary (multi-stage, no parent-repo clone needed).
- [x] Standalone `docker-compose.yml` with Redis + QuestDB + Postgres so this repo can boot independently of the fks compose tree.
- [x] Vendored `fks-proto` crate at `crates/fks-proto/` — proto schemas reconstructed from janus call-site usage (2026-05-24). Workspace builds without cloning the parent fks repo.
  - **Wire-compat note:** field numbers were chosen locally; talking to legacy upstream services may require schema alignment until the upstream catches up.
- [ ] CI: add a GitHub Actions job that exercises `cargo check --workspace` + the Dockerfile build path + pushes to `nuniesmith/janus` on dockerhub on merges to main.

---

## P2 — Housekeeping

- [ ] Proto: Consolidate dual `ForwardService` — `fks.janus.v1.ForwardService` (4 RPCs) vs `fks.forward.v1.ForwardService` (7 RPCs) — **deferred**: see STRUCT-C above.
- [x] Reduce legacy fields in `Config` (was lines 99-145 of `lib/janus-core/src/config.rs`): `http_port`, `grpc_port`, `enable_forward`, `redis_url`, … — **dropped 2026-05-25 (#12)**, replaced with `#[serde(deny_unknown_fields)]` so stale TOMLs fail loudly. Fixed a latent bug in `janus-api`'s `/status` that read the always-`None` legacy module flags.

---

## P3 — Future

- [ ] Neural architecture: 30-day live trading validation → document public API → stabilize neuromorphic crate for production use.
- [ ] Optimizer reads asset list from Ruby's asset registry via gRPC or Redis (JFLOW-B follow-up).

---

## Out of scope for this repo

The following live in the parent [fks](https://github.com/nuniesmith/fks) repo
and are tracked there, not here:

- `RC-CRATES-*` (rustcode workspace integration: runtime/api/tools/plugins/commands/server/claw-cli/compat-harness/lsp)
- `API-*` (rustcode API security & config: skip-extensions config, ModelRouter tuning, rc-core/rc-api/rc-rag/rc-llm workspace split)
- `OSS-*` evaluation queue (OpenViking, Heretic, Nanochat)
- JanusAI Python service endpoints (`POST /api/janus-ai/sessions/{id}/metrics` etc.)
- Ruby execution decision engine + asset registry
