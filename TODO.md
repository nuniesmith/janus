# fks-janus — TODO & Roadmap

> **Repo:** `github.com/nuniesmith/janus`
> **Scope:** janus-only (Rust ML inference + neuromorphic + signal generation).
> Infrastructure orchestration lives in [fks](https://github.com/nuniesmith/fks);
> rustcode / claw / RC-CRATES items live under `fks/src/rustcode/` and are
> **not** tracked here.
> **Last synced:** 2026-05-31
>
> 📋 **Consolidation plan:** [`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md) —
> staged plan to consume `indicators-ta` (TA) + `exchange-apiws` (exchanges)
> and retire the internal `jflow-indicators` / `jflow-exchanges` /
> `jflow-bybit-client` duplicates.

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

## P0 — Brain: wire what's built into the live path  ◀ THE HEADLINE GOAL

> **Goal:** janus decides *what to trade and how much* across asset classes
> under explicit risk rules. A 2026-06 cross-repo survey found that **much of
> this already exists on disk but is not in the running binary**. See
> `fks-full/docs/MULTI_ASSET_BRAIN_ROADMAP.md` (Track 3 + 5). These are the
> highest-value items and are **not** otherwise tracked below.

### The orphaned sophisticated pipeline
- [x] **`services/forward/src/event_loop.rs` deleted (#51).** It was dead code —
      compiled out (no `mod event_loop` / `mod actors` in `lib.rs`) — holding a
      single-symbol *reference* pipeline: the 5-strategy suite
      (EMAFlip/MeanReversion/Squeeze/VWAP/ORB), inline `PropFirmValidator`, and
      regime gating. All of that now lives in the **live** multi-symbol loop in
      `lib.rs`, so the orphan was removed (~4.5k LOC incl. `actors/`) rather than
      re-wired. The live loop is the single source of truth — see
      `docs/architecture/RISK_TOPOLOGY.md`.

      **DECISION (2026-06-02): PORT incrementally into the live `lib.rs` loop;
      retire `event_loop.rs` once its capabilities land. Do _not_ re-wire it as a
      parallel entry point.** Rationale: the live loop is already the multi-symbol
      production path; re-wiring `event_loop.rs` (a single-symbol reference) would
      create two divergent loops with double the test/maintenance burden and a
      standing risk of behavioural drift. A single live path keeps one test suite
      and one source of truth. Port in safe, independently-shippable stages, each
      additive and gated behind config so a half-finished port never degrades the
      running service:
        1. ✅ **Feed `RegimeManager` live** (PR #41).
        2. ✅ **`PropFirmValidator` inline** — advisory pre-trade pass (PR #43).
        3. ✅ **Emit `regime` + `fear` into `signal.metadata`** (PRs #41/#42).
        4. ✅ **`RiskManager::apply_risk_management` inline** — advisory, against the
           live `PortfolioState` (PR #44).
        5. ✅ **Strategy suite ported + `event_loop.rs` retired (#51).** The core suite is
           wired into the live loop with per-symbol state: `EMAFlipStrategy` (replacing the
           inline `ema_cross`), `MeanReversionStrategy` (Bollinger-band MR),
           `SqueezeBreakoutStrategy` (BB/Keltner squeeze → breakout), and the session-anchored
           `VwapScalperStrategy` + `OrbStrategy` pair. VWAP/ORB anchor each symbol's session to
           the UTC calendar day (00:00 UTC reset — `reset_session()` / `start_session()`), since
           24/7 crypto has no natural open. Optional extras remain (EmaRibbon / TrendPullback /
           MomentumSurge / MultiTfTrend). **Risk-engine note:** the original 2026-06-02 plan to
           make `logic::ComprehensiveRiskEngine` canonical (and delete `ComplianceSheriff` +
           `PropFirmValidator`) was **investigated and dropped** — `ComprehensiveRiskEngine` is an
           unwired, stateless library validator with no production consumer, and the live loop
           already enforces via `PropFirmValidator` + `RiskManager` (now blocking under
           `JANUS_RISK_ENFORCE`, #49) plus a cross-process Redis kill-switch (#50). The standing
           engine-consolidation item is below; see `docs/architecture/RISK_TOPOLOGY.md` for the
           actual live topology.
      The integration tests that once *mirrored* `event_loop.rs` (e.g.
      `regime_mr_integration.rs`, `kraken_strategies_integration.rs`) now stand as the
      live-path reference; the source is gone. (Minor follow-up: a few stale `event_loop`
      mentions linger in comments — harmless, clean up opportunistically.)

### Risk enforcement is REST-on-demand, not inline
- [x] **Apply risk on each live signal.** **`PropFirmValidator` (Stage 2)** + **`RiskManager`
      (Stage 4)** are both wired inline now. Each prospective entry gets (a) a prop-firm
      validation pass (ATR stop + account-risk size, `prop_firm` metadata; blocking opt-in
      via `JANUS_PROP_FIRM_ENFORCE`) and (b) a portfolio-aware `RiskManager::apply_risk_management`
      check against the live `PortfolioState` (concurrent positions / exposure / daily loss),
      surfaced as `risk_check` metadata. Both **advisory by default** (never block live trades
      on their own, per "no autonomous execution"); the live portfolio is kept current by the
      REST add/close endpoints. **Follow-ups:** ✅ the RiskManager verdict is now wired into the
      enforce gate (`JANUS_RISK_ENFORCE`, #49), and a cross-process Redis kill-switch suppresses
      execution at the `submit_signal_to_execution` choke point (#50). ⏳ Still open: feed
      closed-trade outcomes back so the validators' stateful daily-loss / drawdown rules engage
      (not just per-trade / point-in-time checks).
- [x] **Unify the duplicate prop-firm / risk engines — done (#54).** Consolidated onto the live
      `models::prop_firm::PropFirmValidator` + portfolio `RiskManager`: deleted the orphaned
      `logic::ComprehensiveRiskEngine` (+ `risk`/`constraints`) and `common::RiskEngine`, migrated
      `crates/backtest` off `compliance::ComplianceSheriff` and removed it, and **generalized
      `PropFirmValidator` across firms** (`PropFirm` enum + presets, config-driven prohibited
      symbols; `new()` keeps the HyroTrader preset). See `docs/architecture/RISK_TOPOLOGY.md`.

### The regime detector is built but under-fed
- [x] **Feed the regime detector live.** `RegimeManager` (`services/forward/src/
      regime.rs`) is now instantiated (task-owned) in the live signal loop and fed
      every closed candle via `on_candle(symbol, high, low, close)` — Stage 1 of the
      event_loop port above.
- [x] **Emit `regime` + `fear` into `signal.metadata`** (JFLOW-C producer gap — closed).
      The live loop stamps **`regime`** (`current_regime(symbol)`) and **`fear`** (the
      amygdala threat label from `regime_bridge::bridge_regime_signal` on the routed
      regime) into each emitted signal's metadata, and `submit_signal_to_execution`
      propagates both onto the execution-path `TradingSignal`. Guidance is no longer
      regime/threat-blind. *(Refinement: feed real ADX/BB-width/ATR/rel-volume into the
      bridge instead of `None` for a sharper amygdala read.)*

### The brain itself is rule-based; ML/neuromorphic are unwired
- [ ] **ML story: decided + Phase 1 foundation built — now run the probe.** Documented in
      `docs/architecture/ML_STORY.md` (#55): the live path is rule-based; ~300K LOC of ML is
      unwired; the tract-ONNX `inference.rs` is a dead end (the trained models are Burn, not ONNX)
      and `neuromorphic/` is now CI-quarantined (#56). Direction chosen = the vision/GAF route
      (`ML_VISION_SCOPE.md`, #58). **Phase 1 machinery is built + merged (#59–#62):** GAF feature
      extractor → real experience builder → sliding-window batches → opt-in `backward` seeding.
      **Remaining (needs real data + GPU):** run the go/no-go probe (does GAF beat chance?) per
      `docs/architecture/ML_PHASE1_HANDOFF.md` (#63); enrich the flat features + reward scheme;
      serve into forward only if it shows edge.

### Multi-asset breadth
- [ ] **Futures + equities asset classes.** `crates/optimizer/src/asset.rs`
      `AssetCategory` + `AssetRegistry::with_kraken_defaults()` cover **crypto +
      forex only**. Add futures/equities variants with class params (min spread,
      hold time, ATR mult, TP range), liquidity tiers, and venues.

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
- [ ] **Panic-safety sweep.** ~8K `.unwrap()`/`.expect()` repo-wide — but a 2026-06 survey of the **live hot paths found them already panic-safe**, so this is *not* the urgent fire the raw count implies. forward's per-tick loop (`lib.rs`, `signal/`, `regime.rs`, `risk/`) and the `services/data` ingestion path have **zero unguarded non-test unwraps**: the genuine ones are guarded (peek-then-pop, `is_empty` checks), safe-by-construction (static `Regex::new`, Prometheus metric registration), or set-before-use invariants. The high per-file counts are **test modules + startup boilerplate**, not hot-path risk. Remaining genuine spots are sparse + low-priority: `bybit_compat::BybitClient::new`'s `.expect` (a pub exchange-client ctor, no callers today) and the un-surveyed crates (`persistence/`, `crates/*`, exchange clients). Keep as a rolling cleanup re-scoped to those, not a blanket ~8K sweep.
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

- [x] **`cargo fmt --check` in CI — done (#57).** One-shot `cargo fmt --all` reformat landed and the `cargo fmt --all --check` gate is re-enabled in `ci.yml` (runs before the heavier check). Tree is fmt-clean.
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

- [x] **README crate list refreshed.** Audited the `### Core crates` list
      against `crates/` (39-member workspace): added `models` (prop-firm
      validator) and `fks-proto`, dropped `indicators` (now the crates.io
      `indicators-ta` dep) and `bybit-client` (removed). Stats block corrected
      `50+ crate` → `39-crate`.
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
