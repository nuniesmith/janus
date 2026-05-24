# fks-janus — TODO

> **Repo:** `github.com/nuniesmith/janus`
> **Scope:** janus-only (Rust ML inference + neuromorphic + signal generation).
> Infrastructure orchestration lives in [fks](https://github.com/nuniesmith/fks);
> rustcode / claw / RC-CRATES items live under `fks/src/rustcode/` and are
> **not** tracked here anymore.
> **Last synced:** 2026-05-24

---

## P0 — Codebase Health

- [ ] **318 `#[allow(dead_code)]` annotations** — audit results (2026-05-24):
  - `services/forward/src/api/grpc.rs` — confirmed dead per `services/forward/build.rs:9-22` (audit 2026-03-26). **Intentionally retained** to keep impl type-checked until the signal-service ownership question (Option A vs B) is resolved. Do NOT delete without resolving STRUCT-C first.
  - `services/api/src/grpc.rs` (11) — `GrpcClientManager` / `GatewayForwardService` scaffolding, never instantiated outside its own tests. Not benign serde — these are unused types. Decide: wire up or delete.
  - `services/optimizer/src/collector.rs` (10) — mix of unused `OhlcCollector` methods (`fetch_ohlc`, `backfill`, etc.) and the `DataGap` struct. Some are reachable from `scheduler.rs`; verify before pruning.
  - `services/optimizer/src/scheduler.rs` (7), `services/execution/tests/integration/scenarios.rs` (7), `services/data/src/api/auth.rs` (7), `services/api/src/rate_limit.rs` (7) — review each.
- [ ] **Tonic version split** — workspace declares `0.14.2` but some crates resolve `0.10.2` transitively (via `apalis`). Track and resolve when `apalis` hits 1.0 stable.
- [ ] Evaluate `shared_memory` IPC in containers — `/dev/shm` size limits may break Forward→Backward zero-copy Arrow IPC (removed crate, but protocol design question remains).
- [ ] **STRUCT-C deferred:** Consolidate stray `services/forward/proto/janus/v1/janus.proto` → `proto/fks/janus/v1/signal_service.proto` once GrpcServer ownership is decided. See `services/forward/build.rs:14-19`.

---

## P1 — Signal Flow (JFLOW)

### JFLOW-A: Session metrics wiring  *(infrastructure landed 2026-05-24)*
- [x] `SessionMetricsClient` + `SessionMetrics` types in `lib/janus-core/src/session_metrics.rs` (no-op when `JANUS_AI_URL` is unset).
- [ ] Call `SessionMetricsClient::push(session_id, metrics)` from the forward signal pipeline (`services/forward/src/signal/mod.rs`) after each generation cycle.
- [ ] Forward `event_loop.rs` aggregator: roll signal counts / win rate / latency into `SessionMetrics` once per minute.

### JFLOW-B: Dynamic asset config from Ruby  *(overlay landed 2026-05-24)*
- [x] Janus startup config overlay from Redis: `Config::load()` now reads key `fks:janus:config` after env overrides (`lib/janus-core/src/config.rs::apply_redis_overlay`).
- [ ] When a JanusAI session starts, write session-specific config to Redis (`fks:janus:config`) — **producer side, lives in fks repo**.
- [ ] Optimizer reads asset list from Ruby's asset registry via gRPC or Redis (currently env-only).

### JFLOW-C: Two-way position feedback (remaining)
- [ ] Janus receives live position data and provides guidance: take-profit suggestions based on regime changes, stop adjustment based on volatility, exit urgency from amygdala.
- [ ] All feedback stored as execution memories for learning.

### JFLOW-D: Startup bootstrap (remaining)
- [ ] Full Postgres bootstrap path in Rust: query `janus_memories` directly from Rust at startup (currently uses Python endpoint + Redis ring buffer as intermediate via `services/forward/src/main.rs:814 bootstrap_affinity_from_redis_ring`). Requires `sqlx` in forward service `Cargo.toml`.

---

## P1 — Janus AI (remaining)

- [ ] Wire signal pipeline (JFLOW-A) to actually emit metrics — see JFLOW-A above. Endpoint contract lives in JanusAI service (Python, in fks repo).

---

## P2 — Build & Deployment

- [x] Standalone `Dockerfile` for the unified `janus` binary (multi-stage, builds against an upstream `fks-proto` checkout).
- [x] Standalone `docker-compose.yml` with Redis + QuestDB + Postgres so this repo can boot independently of the fks compose tree.
- [ ] Publish a stub `fks-proto` crate (or vendor it under `crates/`) so the workspace builds without cloning the parent repo. Today `Cargo.toml` declares `fks-proto = { path = "../../src/proto" }` which only resolves when janus is nested inside `fks/src/janus`.
- [ ] CI: add a GitHub Actions job that exercises the Dockerfile path on every PR.

---

## P2 — Housekeeping

- [ ] Proto: Consolidate dual `ForwardService` — `fks.janus.v1.ForwardService` (4 RPCs) vs `fks.forward.v1.ForwardService` (7 RPCs) — **deferred**: see STRUCT-C above.
- [ ] Reduce legacy fields in `Config` (lines 99-145 of `lib/janus-core/src/config.rs`): `http_port`, `grpc_port`, `enable_forward`, `redis_url`, … — these duplicate the new nested config. Cleanup blocked on confirming no external TOMLs still rely on them.

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
