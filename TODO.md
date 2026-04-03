# fks-janus — TODO

> **Repo:** `github.com/nuniesmith/fks-janus`
> **Last synced from master todo:** 2026-04-03

---

## P0 — Codebase Health

- [ ] **318 `#[allow(dead_code)]` annotations** — review `services/api/src/grpc.rs` (11 annotations) and `services/optimizer/src/collector.rs` (10 annotations) for actual dead code; most are benign serde deserialization fields
- [ ] **Tonic version split** — workspace declares `0.14.2` but some crates resolve `0.10.2` transitively (via `apalis`). Track and resolve when `apalis` hits 1.0 stable.
- [ ] Evaluate `shared_memory` IPC in containers — `/dev/shm` size limits may break Forward→Backward zero-copy Arrow IPC (removed crate, but protocol design question remains)
- [ ] Centralize stray `forward/proto/janus/v1/janus.proto` → `proto/fks/janus/v1/signal_service.proto` — **deferred** until gRPC endpoint is actually needed (dead code today)
- [ ] Update forward service `build.rs` to use `fks-proto` crate instead of compiling local protos (blocked on above)

---

## P1 — Signal Flow (JFLOW)

### JFLOW-B: Dynamic asset config from Ruby
- [ ] Janus startup config overlay from Redis: read `fks:janus:config` at startup (`janus-core/config.rs`) — higher-level config like which assets to trade currently reads from env vars only
- [ ] When a JanusAI session starts, write session-specific config to Redis (`fks:janus:config`)
- [ ] Optimizer reads asset list from Ruby's asset registry via gRPC or Redis

### JFLOW-C: Two-way position feedback (remaining)
- [ ] Janus receives live position data and provides guidance: take-profit suggestions based on regime changes, stop adjustment based on volatility, exit urgency from amygdala
- [ ] All feedback stored as execution memories for learning

### JFLOW-D: Startup bootstrap (remaining)
- [ ] Full Postgres bootstrap path in Rust: query `janus_memories` directly from Rust at startup (currently uses Python endpoint + Redis ring buffer as intermediate; direct Postgres requires sqlx setup in forward service Cargo.toml)

---

## P1 — RustCode Crates Integration (RC-CRATES)

> `src/rustcode/crates/` contains 9 new crates. Wire them into the existing `rustcode` binary.

- [ ] **RC-CRATES-A:** Add all 9 crates to `src/rustcode/Cargo.toml` workspace members: `runtime`, `api`, `tools`, `plugins`, `commands`, `server`, `claw-cli`, `compat-harness`, `lsp`
- [ ] **RC-CRATES-A:** Run `cargo check --workspace` — resolve dependency version conflicts (`reqwest 0.12`, `axum 0.8` vs workspace `0.7`)
- [ ] **RC-CRATES-A:** Run `cargo test --workspace` — all existing RC tests must still pass
- [ ] **RC-CRATES-B:** Replace ad-hoc `reqwest` LLM calls with `api` crate client: update `src/rustcode/src/llm/grok_client.rs` and `ollama_client.rs`
- [ ] **RC-CRATES-C:** Replace scanner glob/hashing with `runtime` crate: `runtime::glob_files()`, `runtime::hash_file()`, `runtime::FileEntry`
- [ ] **RC-CRATES-D:** Wire `tools` + `plugins` into RC tool execution — load TOML plugin manifests from `infrastructure/config/rustcode/plugins/`, implement `POST /api/v1/tools/run`
- [ ] **RC-CRATES-D:** Create example plugin manifests: `code_review.toml`, `todo_scan.toml`, `file_summary.toml`
- [ ] **RC-CRATES-E:** Add `--server` flag to `rustcode` binary — starts `server` crate's axum router on port 3501 alongside existing RC API on 3500
- [ ] **RC-CRATES-F:** Build `claw-cli` binary and add to Dockerfile; test `claw scan`, `claw plan`, `claw work` subcommands
- [ ] **RC-CRATES-G:** Create `tests/test_grok_integration.rs` — integration tests behind `#[cfg(feature = "integration")]`: complete call, ModelRouter classification (5 prompt types), RAG injection, cache hit
- [ ] **RC-CRATES-G:** Add `./run.sh rc-test-grok` command — sets `XAI_API_KEY` from `.env`, runs integration tests
- [ ] **RC-CRATES-G:** After integration tests pass: validate Claude API switch (flip `XAI_API_KEY` → `ANTHROPIC_API_KEY`, update `api` crate endpoint URL)

---

## P1 — RustCode: API Security & Config

- [ ] **API-C:** Make skip-extensions configurable per-repo — add `skip_extensions: Vec<String>` to repo config struct
- [ ] **API-C:** Routing heuristic tuning — after deployment, measure local vs Grok classification quality and adjust `ModelRouter::llm_classify` system prompt
- [ ] Consider workspace split: `rc-core`, `rc-api`, `rc-rag`, `rc-llm` — 81K LoC single crate; lower priority now that config is centralized and HNSW is implemented

---

## P1 — Janus AI (remaining)

- [ ] Session metrics: wire signal pipeline (JFLOW-A) to call `POST /api/janus-ai/sessions/{id}/metrics`

---

## P2 — Housekeeping

- [ ] Proto: Consolidate dual `ForwardService` — `fks.janus.v1.ForwardService` (4 RPCs) vs `fks.forward.v1.ForwardService` (7 RPCs) — **deferred**: `janus.v1.JanusService` in `forward/proto/` is confirmed dead code (GrpcServer compiled but not wired into main binary)
- [ ] RC: OSS-B OpenViking — evaluate as replacement/supplement to brute-force `vector_index.rs` search (HNSW implemented, but OpenViking provides tier-based context management)

---

## P3 — Future

- [ ] Janus optimizer reads asset list from Ruby's asset registry via gRPC or Redis (JFLOW-B)
- [ ] Neural architecture: 30-day live trading validation → document public API → stabilize neuromorphic crate for production use
- [ ] RC: OSS-F Backlog — Heretic (local model uncensoring), Nanochat (custom LLM training from scratch, significant GPU time)
