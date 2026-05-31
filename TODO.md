# fks-janus — TODO & Roadmap

> **Repo:** `github.com/nuniesmith/janus`
> **Scope:** janus-only (Rust ML inference + neuromorphic + signal generation).
> Infrastructure orchestration lives in [fks](https://github.com/nuniesmith/fks);
> rustcode / claw / RC-CRATES items live under `fks/src/rustcode/` and are
> **not** tracked here.
> **Last synced:** 2026-05-31

---

## Status snapshot (2026-05-31)

- **Build:** `cargo check --workspace --all-targets` is green in CI (needs
  `protobuf-compiler` on the runner for the `fks-proto` build script).
- **CI/CD:** PR checks (`ci.yml`: check + `--lib` tests + Docker build) and
  Docker Hub publish on merge to `main` (`docker-publish.yml`) are both live.
- **Signal flow:** JFLOW-A/B/C/D have all landed on the Janus side (PRs #1–#23).
  Position guidance now blends regime + per-asset optimizer thresholds +
  volatility (ATR) + amygdala fear. Remaining JFLOW work is producer-side
  (fks repo) or the cross-module `ParamManager` dedupe.
- **Size:** ~50 workspace crates, ~583K LOC (≈250K of it neuromorphic),
  9,888+ tests. ~312 `#[allow(dead_code)]` annotations remain (see P0).

> **Heads-up:** `services/data/docs/TODO_IMPLEMENTATION_PLAN.md` is a stale
> 2024 doc that reports ~everything as 0% done. It is **wrong** — circuit
> breaker, backfill lock/throttle, dedup, Prometheus export, and Docker
> secrets all exist today. Reconcile or retire it (P0 item below).

---

## P0 — Codebase health & correctness

- [ ] **`#[allow(dead_code)]` audit (~312 annotations: 143 services / 107 neuromorphic / 60 crates / 2 bin).**
  Continue trimming the genuinely-dead, keep the justified. Prior audit results:
  - `services/api/src/grpc.rs` + `services/forward/src/api/grpc.rs` — deleted in #4 (dead scaffolding).
  - `services/optimizer/src/collector.rs` — consolidated 10 → 6 in #10; remainder is useful-but-unused API surface (`datetime`, `DataGap`, `get_date_range`, `detect_gaps`, `vacuum`, `impl CollectionMetadata`).
  - `services/optimizer/src/scheduler.rs` (7) — empirically confirmed all needed (accessors on `OptimizationScheduler`).
  - `services/data/src/api/auth.rs` (7), `services/api/src/rate_limit.rs` (7), `services/data/src/lib.rs` (6) — disabled-mode constructors + serde-only fields; confirmed.
  - `neuromorphic/thalamus/sources/clients/openweathermap.rs` (12) — serde wire-shape DTOs; `#[allow]` is correct (audited #23, removed the dead `OWM_ONECALL_URL`).
  - `services/execution/tests/integration/scenarios.rs` (7) — test scaffolding, low priority.
  - **Next:** the 107 neuromorphic annotations have never been swept — that's the biggest untouched cluster.
- [ ] **Panic-safety sweep.** Thousands of `.unwrap()` calls outside tests (a rough grep counts ~8K). A signal engine should not panic on a malformed payload or a dropped connection. Triage the hot paths first (signal generation, position-event handler, Redis/Postgres IO, exchange clients) and convert to `Result` + `tracing`. Track as a rolling effort, not one PR.
- [ ] **Triage `cargo audit` advisories, then make the gate blocking.** First run (2026-05-31) found **10 vulnerabilities + 11 warnings**. The gate is non-blocking until these are cleared. Notable:
  - `rustls-webpki 0.103.10` (×3: RUSTSEC-2026-0098/0099/0104 — name-constraint bypass + CRL-parse panic) — **has a fix, bump it.**
  - `rsa 0.9.10` (RUSTSEC-2023-0071 Marvin timing sidechannel) — no upstream fix yet; transitive via a TLS/DB driver. Assess RSA-signing exposure.
  - `astral-tokio-tar 0.5.6` (×4: PAX/symlink extraction) — only relevant if untrusted tars are extracted; confirm it's build/ML-only, not a request path.
  - `pyo3 0.21.2`, `fast-float 0.2.0` — buffer/segfault in specific APIs; transitive, ML-adjacent.
  - Warnings: unmaintained (`paste`, `bincode`, `rustls-pemfile`, `atomic-polyfill`, `core2`-yanked) + unsound (`rand` ×3 custom-logger edge, `rkyv`). Consider a `deny.toml` to triage/allowlist deliberately.
  - Context: Janus generates signals and **never executes orders / isn't a public web service**, so exploitability of most of these is low — but the webpki bump and `rsa` assessment are worth doing.
- [x] **Reconciled `services/data/docs/TODO_IMPLEMENTATION_PLAN.md` (2026-05-31).** Added a "SUPERSEDED" banner mapping its stale "NOT IMPLEMENTED" P0/P1 items to the code that actually implements them (backfill lock/throttle, circuit breaker, Prometheus export, dedup, Docker secrets, rate limiter, QuestDB writer), and pointing at this file as the single source of truth. Kept non-destructively — ~9 sibling docs in that dir cross-link it, and the code templates remain useful for the genuinely-open items.
- [ ] **Tonic version split.** Workspace declares `0.14.2` but some crates resolve `0.10.2` transitively via `apalis`. Track and resolve when `apalis` hits 1.0 stable.
- [ ] **STRUCT-C: proto consolidation.** Fold the stray `services/forward/proto/janus/v1/janus.proto` into `proto/fks/janus/v1/signal_service.proto` once `GrpcServer` ownership is decided (`services/forward/build.rs:14-19`). Also resolves the dual `ForwardService` (`fks.janus.v1` 4 RPCs vs `fks.forward.v1` 7 RPCs).
- [ ] **Evaluate `shared_memory` IPC in containers.** `/dev/shm` size limits may break Forward→Backward zero-copy Arrow IPC. (Crate removed; the protocol-design question remains.)

---

## P1 — Signal flow (JFLOW)

JFLOW-A→D are **complete on the Janus side**. What remains is producer-side
(emitting signals from the real networks, owned in fks) and one cross-module
dedupe.

### Done (Janus side)
- [x] **JFLOW-A** — session-metrics push loop: `SessionMetricsClient`/`SessionMetrics` (`lib/janus-core/src/session_metrics.rs`), reporter loop in `services/forward/src/lib.rs::start_module` pushing real `avg_confidence` / `p50` / `p99` latency + current regime every `JANUS_AI_PUSH_SECS`.
- [x] **JFLOW-B (overlay)** — Redis config overlay at startup (`Config::load` → `apply_redis_overlay`, key `fks:janus:config`).
- [x] **JFLOW-C (guidance)** — `POST /api/v1/positions/event` returns advisory `hold`/`reduce`/`exit`. Blends, in priority order: crisis regime → high fear → elevated fear (bank winners / tighten losers) → per-asset optimizer stop/take-profit (`GuidanceThresholds::from_optimized_params`) widened for ATR volatility. Events persisted to the `janus_position_events` Postgres ingest log. Live `ParamManager` bootstrap + `param_updates` subscription.
- [x] **JFLOW-D** — direct-Postgres affinity bootstrap (`bootstrap_affinity_from_postgres`) with Redis ring-buffer fallback; `persistence` feature on by default.

### Remaining
- [ ] **JFLOW-B:** Optimizer reads its asset list from Ruby's asset registry (gRPC or Redis) instead of env-only.
- [ ] **JFLOW-C — `ParamManager` dedupe.** `services/forward` and `janus-api` each subscribe to `fks:{instance}:param_updates` and keep separate caches. Promote `ParamManager` onto `JanusState` so both share one. Touches forward's `ParamReloadManager` (which also owns appliers) — that's why it's deferred.
- [ ] **JFLOW-C — producer emission (fks/forward pipeline).** Guidance reads `signal.metadata["regime"]` and `["fear"]` opportunistically; today nothing emits them, so `current_regime`/`current_threat` stay `None` and guidance is P&L-only. Wire the real regime detector and amygdala fear network to populate them. *(See "Position-guidance hardening" for the rest of this thread.)*
- [ ] **JFLOW-C — memory compaction (JanusAI side, fks repo).** Compact the `janus_position_events` raw log into closed-trade rows in `janus_memories`.

---

## P1 — Position-guidance hardening

Natural follow-ups now that the guidance engine exists. These make it *learn*
and *close the loop* instead of being a stateless suggestion.

- [x] **Per-position state (2026-05-31).** `PositionTracker` in `lib/janus-core/src/position_events.rs` accumulates per-`position_id` history (peak P&L ratio, sample count, `first_seen`/`last_seen`, last action) with TTL + max-entries eviction. Wired into `janus-api` as an `Extension`. Adds two history-dependent rules `compute_guidance` (still pure/stateless) can't do: **trailing give-back** (bank a fading winner that's surrendered ≥ `giveback_frac` of an armed peak) and **sticky exit** (don't rescind an `Exit` on a one-tick bounce). Id-less events pass through unchanged.
- [x] **Guidance outcome capture (2026-05-31).** `POST /api/v1/positions/close` takes a `PositionClose`, finalizes the position's `PositionTracker` state, and returns + best-effort persists a `PositionOutcome` — realized P&L and win/loss joined with the guidance history (peak P&L ratio, sample count, last advised action, time-in-position) — to the `janus_position_outcomes` table (same probe/disable pattern as the event log).
- [x] **Real-time affinity feedback (2026-05-31).** `JanusState` gained an `AffinityRecorder` trait-object slot (mirrors the `LogLevelController` pattern, so janus-core/janus-api stay free of a `janus-strategies` dep). Forward installs `PipelineAffinityRecorder` (delegates to the `TradingPipeline`'s gate) at startup; the close handler calls `state.record_affinity_outcome(...)` when the close names a `strategy`, updating affinity weights live — complementing the startup `bootstrap_affinity_from_postgres` replay. `PositionClose`/`PositionOutcome` gained `strategy` + `rr_ratio`. No-op (persist-only) when no recorder is installed or the close omits `strategy`.
- [ ] **Close the learning loop (remaining).** JanusAI-side (fks repo): compact `janus_position_outcomes` + `janus_position_events` into `janus_memories` so the live affinity feedback also survives restarts via the existing bootstrap path.
- [ ] **Learnable trailing + fear/volatility constants.** `TrailingConfig` (`arm_ratio` 0.03, `giveback_frac` 0.5) and `FEAR_EXIT_LEVEL` (0.8) / `FEAR_ELEVATED_LEVEL` (0.5) / `STOP_TIGHTEN_FLOOR` (0.25) are hardcoded. Consider moving them into `OptimizedParams` so the optimizer can tune them per-asset, the way stop/take-profit already are.
- [ ] **Optimizer schema bump.** `OptimizedParams` carries `stop_loss_pct` (serde default 2.0) and `take_profit_pct`, but the Python optimizer doesn't emit them yet — until it does, defaults apply. Coordinate the search-space + payload change (fks repo).

---

## P1 — Data service productionization

Most P0/P1 items from the old plan are **already done** — the real remaining
gaps are below.

- [ ] **Cross-exchange price validation.** `crates/data-quality/src/validators/price.rs` does single-exchange spike detection only. Add multi-exchange median comparison with a deviation threshold (data-poisoning / bad-feed defense).
- [ ] **Distributed rate-limiter state.** `crates/rate-limiter` is solid single-instance (token bucket, sliding window, per-exchange configs, circuit breaker). Multi-instance deployments need shared token state in Redis (+ restart persistence).
- [ ] **Gap-queue lifecycle verification.** Confirm the persistent gap queue + backfill scheduler track the full `detected → backfilling → filled → verified` lifecycle, not just submission. Add an integration test.

> **Already implemented (do not re-do):** circuit breaker (`crates/rate-limiter/src/circuit_breaker.rs`, wired in `services/data/src/connectors/circuit_breaker_integration.rs` + `backfill/executor.rs`), backfill Redis lock (`backfill/lock.rs`), backfill throttle + disk monitor (`backfill/throttle.rs`), dedup (`backfill/gap_integration.rs`), Prometheus `/metrics` (`services/data/src/metrics/prometheus_exporter.rs`), Docker secrets (`services/data/src/config.rs::read_secret`), QuestDB ILP writer (`crates/questdb-writer`).

---

## P2 — Observability & quality gates

- [ ] **`cargo fmt --check` in CI.** Intentionally disabled (`ci.yml:38-40`) because the tree has unformatted regions. Land a one-shot reformat PR, then re-enable the step.
- [x] **Dependency vuln scanning in CI (2026-05-31).** `.github/workflows/security.yml` runs `cargo-audit` (RUSTSEC) on dependency-manifest changes, weekly, and on demand. **Informational** for now: it reports advisories in the job summary + a `::warning::` annotation but **exits 0** (green check), so it surfaces findings without red-lining `main` or training people to ignore a by-design-red check. Flip to a hard gate (drop the `exit 0`, mark required) once the advisory backlog above is cleared. Trivy filesystem scan still TODO.
- [ ] **Service-backed integration tests in CI.** `ci.yml` runs `--lib` only; integration/e2e suites that need Postgres/Redis (e.g. `services/forward/tests/param_reload_integration.rs`, the execution scenarios) don't run. Add a job with service containers.
- [ ] **Distributed tracing (OpenTelemetry/Jaeger).** `Config` has a `tracing`/`jaeger_endpoint` field that nothing reads. Either wire OTel export through it or drop the field. Plain `tracing` is used everywhere today.
- [ ] **Grafana dashboards + Alertmanager rules.** The data service exports rich Prometheus metrics but there are no dashboards/alerts in this repo. Decide whether they live here or in fks, then build SLO dashboards (data completeness, ingestion P99, circuit-breaker state) and alert rules.

---

## P2 — Build & deployment

- [x] Standalone multi-stage `Dockerfile` for the unified `janus` binary.
- [x] Standalone `docker-compose.yml` (Redis + QuestDB + Postgres) for independent boot.
- [x] Vendored `fks-proto` crate (`crates/fks-proto/`) — workspace builds without cloning fks.
- [x] CI: PR checks + Docker Hub publish on merge to `main` (`.github/workflows/`).
- [ ] **`fks-proto` wire-compat.** Field numbers were chosen locally when the schemas were reconstructed; talking to legacy upstream services may need schema alignment until upstream catches up.
- [ ] **Container hardening.** Run as non-root, drop capabilities, read-only rootfs, pin base image, scan the published image.

---

## P2 — Documentation

- [ ] **README crate list is out of date.** It omits `crates/models` (`janus-models`: trades/signals/account/performance/**prop-firm validator**). Audit the list against `crates/` and refresh the stats block.
- [ ] **Document the public surface.** Brain REST API (`/api/v1/brain/*`, `/api/v1/risk/evaluate`, `/api/v1/positions/event`), the JanusAI session-metrics contract, and the env-var reference.
- [ ] **Surface prop-firm support.** `crates/models/src/prop_firm.rs` (`PropFirmValidator`, `ChallengeType`) is undocumented — note where/whether it's wired into execution.

---

## P3 — Neural architecture & research

- [ ] **30-day live-trading validation** of the neuromorphic stack (~250K LOC across 10 brain regions) → document the public API → stabilize the `janus-neuromorphic` crate for production use. This is the gate before treating brain output as authoritative.
- [ ] **Optimizer ↔ asset registry** (JFLOW-B follow-up): pull the asset universe from Ruby's registry rather than env.
- [ ] Sweep the 107 neuromorphic `#[allow(dead_code)]` annotations once the regions stabilize (currently churning too fast to audit usefully).

---

## Out of scope for this repo

Tracked in the parent [fks](https://github.com/nuniesmith/fks) repo:

- `RC-CRATES-*` (rustcode workspace: runtime/api/tools/plugins/commands/server/claw-cli/compat-harness/lsp)
- `API-*` (rustcode API security & config; rc-core/rc-api/rc-rag/rc-llm split)
- `OSS-*` evaluation queue (OpenViking, Heretic, Nanochat)
- **JanusAI Python service** endpoints (`POST /api/janus-ai/sessions/{id}/metrics`, memory compaction, optimizer search-space)
- **Ruby execution engine** + asset registry (Janus only *suggests* — Ruby decides and executes)
- Session-start config producer (writes `fks:janus:config` to Redis when a JanusAI session starts)
