# janus — TODO & Roadmap

> **Repo:** `github.com/nuniesmith/janus`
> **Scope:** janus-only (Rust ML inference + neuromorphic + signal generation).
> The workspace is standalone — its own `Dockerfile` + `docker-compose.yml`
> live here. Unrelated rustcode / claw / RC-CRATES items are **not** tracked
> here (see "Out of scope" below).
> **Last synced:** 2026-06-08
>
> ⭐ **2026-06-07 — janus is now the platform.** `src/ruby/` was deleted from
> `fks-full`; the Python data/engine/trainer service is gone. janus no longer
> *suggests to* Ruby — it **is** the trading platform fks-full runs. The forward
> roadmap is therefore the **Ruby→Rust migration** (see *"The migration is the
> roadmap now"* right after the status snapshot, and
> `fks-full/docs/architecture/RUST_MIGRATION.md`).
>
> 📋 **Consolidation plan:** [`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md) —
> staged plan to consume `indicators-ta` (TA) + `exchange-apiws` (exchanges)
> and retire the internal `jflow-indicators` / `jflow-exchanges` /
> `jflow-bybit-client` duplicates.

---

## Status snapshot (2026-06-08)

- **Platform shift (2026-06-07):** `src/ruby/` was deleted from fks-full —
  **janus is the platform now**, not just the brain. The remaining roadmap is
  the **Ruby→Rust migration**: rebuild the capabilities fks-full used to get
  from the Python service as janus crates/services. See *"The migration is the
  roadmap now"* below and `fks-full/docs/architecture/RUST_MIGRATION.md`.
- **Build:** `cargo check --workspace --all-targets` is green in CI (needs
  `protobuf-compiler` on the runner for the `fks-proto` build script). The new
  `crates/execution-gate` is dependency-light and builds/tests standalone
  (`cargo test -p jflow-execution-gate`).
- **CI/CD:** PR checks (`ci.yml`: fmt + check + `--lib` tests + Docker build) and
  Docker Hub publish on merge to `main` (`docker-publish.yml`) are both live.
- **Execution gate (2026-06-08):** Ruby's 9-gate `ExecutionGate` ported to
  `crates/execution-gate` — a faithful, pure/synchronous chain (circuit-breaker
  → risk → vol → quality → AO → fee → CNN-agree → CNN-conf → correlation) plus
  the `ConsecutiveLossBreaker`, `CorrelationGuard`, and `AdaptiveThreshold` it
  needs (37 unit tests + doctest, clippy-clean). **Now wired into the live loop,
  advisory by default** — all 9 gates act on real data, with Redis observability
  + breaker-state persistence; enforcement is one flag away (`JANUS_GATE_ENFORCE`).
  See Track C.
- **Data factory (2026-06-08):** the gap-scan → reconcile → backfill loop is
  complete in Rust — `plan_candle_backfill` (bounded, newest-first windows),
  `GapIntegrationManager::handle_candle_scan`, and an opt-in periodic
  `candle_scan` reconciler wired into the data-service `main` behind
  `JANUS_CANDLE_SCAN` (inert by default). Live lifecycle verification against a
  running QuestDB/Redis stack is the only step left. See Track A.
- **Signal flow:** JFLOW-A/B/C/D landed on the Janus side. Position guidance
  blends regime + per-asset optimizer thresholds + ATR volatility + amygdala
  fear. NB: the "producer-side" JFLOW work the old notes pushed to "the fks
  repo" is **now janus's own job** — janus is the producer.
- **Size:** ~51 workspace crates, ~583K LOC (≈250K of it neuromorphic),
  9,900+ tests. ~312 `#[allow(dead_code)]` annotations remain (see P0).

> **Heads-up:** `services/data/docs/TODO_IMPLEMENTATION_PLAN.md` is a stale
> 2024 doc that reports ~everything as 0% done. It is **wrong** — circuit
> breaker, backfill lock/throttle, dedup, Prometheus export, and Docker
> secrets all exist today. Reconcile or retire it (P0 item below).

---

## ⭐ The Ruby→Rust migration is the roadmap now (2026-06-07)

`src/ruby/` is gone from fks-full; janus must stand alone as the platform. This
section is the **forward plan** — the capabilities to rebuild natively, in
dependency order, each shippable behind a flag so the running service never
degrades. It's the janus-side companion to
`fks-full/docs/architecture/RUST_MIGRATION.md` (phase rationale + parity
strategy). Method throughout: **strangler-fig + golden-vector parity**, never
rewrite-and-pray.

**Sequencing (highest leverage first):**

```
Track A (data SoT) ───┐
Track C (safety/gate) ─┼─► together they unblock a janus-only live path
Track B (ML parity) ──┘    (Track B gated on the user's .pt goldens)
Track D (infra contract) ── coordinates with fks-full's nginx/WebUI repoint
Track E (long tail) ──────── rebuild-if-needed; Rithmic stays Python
```

### Track A — Data layer: make janus the sole source of truth · RUST_MIGRATION Phase 1
The #1-leverage, lowest-ML-risk cut. The Python "Ruby" data service is **gone**
from the deployment, so its client + the dead registry sync are being retired,
leaving janus's native ingestion (exchange WS → QuestDB) as the only data path.
- [x] **Retired `python_data_client.rs` (data service).** Deleted the ~1,095-line
      Ruby data-API client + the dead `DataServiceProvider` wrapper, and rewired
      indicator warmup from a 3-tier (Python → QuestDB → Binance) chain to a
      native 2-tier (QuestDB → Binance) one. This also removes the per-call
      timeout-to-`fks_ruby` that slowed every warmup now that Ruby is gone.
- [x] **Retired the registry's Python sync.** Removed the `sync_from_python_loop`
      background task (+ `sync_once` / `push_annotations_to_python` /
      `sync_python_services` / `parse_python_asset`, ~390 lines) that polled
      `{PYTHON_DATA_SERVICE_URL}/api/registry/*` every 5 min. The registry is now
      janus-native. **`PYTHON_DATA_SERVICE_URL` is gone from all janus code.**
- [~] **Non-crypto data source: scoped to crypto-only (decided).** CME futures
      (MGC, MES, …) were only served by the retired Python service; QuestDB/Binance
      can't backfill them, and there's no Massive API access here to build a native
      source. **Decision: the platform is crypto-only** for now — non-crypto symbols
      warm up empty + fall back to live candles (already graceful, no crash). A
      native non-crypto source (Massive integration) is **deferred, gated on API
      access** — reopen this if/when CME is in scope.
- [ ] **Finish the data factory** (gap-scan → backfill → reconcile). The
      reconcile path is built **end-to-end through tested code** and now **wired
      into startup** (opt-in); only live verification remains:
      - `crates/gap-detection`: sequence/heartbeat/statistical trade-gap
        detectors + parameterized QuestDB gap SQL, pure **`detect_candle_gaps`**
        for candle-level holes, and now **`plan_candle_backfill`** (✅) — chunks
        oversized gaps into bounded windows, filters noise, orders newest-first.
      - `services/data/src/backfill/`: executor/scheduler/lock/throttle/dedup,
        **`GapIntegrationManager::handle_candle_scan`** (✅, detect → plan →
        submit through the existing tested `handle_gap` path), and the opt-in
        **`candle_scan`** reconciler loop (✅, `JANUS_CANDLE_SCAN=1`) — the
        QuestDB read sits behind the `CandleTimestampSource` trait so the
        orchestrator (`run_scan_once`) is unit-tested with a fake;
        `QuestDbCandleSource` is the gated real impl.
      **Now wired (opt-in):** `DataFactory::start_candle_scan` constructs the
      scheduler + `GapIntegrationManager` + `QuestDbCandleSource` and spawns the
      reconciler in the data-service `main`, **entirely behind
      `JANUS_CANDLE_SCAN=1`** — inert by default (no resources acquired when off).
      Set `JANUS_CANDLE_SCAN_SYMBOLS` to the exact `candles_crypto` symbols.
      **Remaining (live-stack only):** flip the flag against a running
      QuestDB/Redis stack and verify the `detected → backfilling → filled →
      verified` lifecycle (integration, not unit). Ruby ref:
      `ruby/src/data/{gap_scanner,backfill_manager}.py`.
- [ ] **Asset registry in janus** (subsumes JFLOW-B's env-only list + the old
      "pull from Ruby's registry" follow-ups): port
      `ruby/src/services/asset_registry.py` → a registry the optimizer + forward
      read. (`services/registry` + `crates/registry` exist — confirm scope.)
- [ ] **Exit:** janus is the sole data writer; all Ruby data calls gone.

### Track B — burn-native ML, no PyTorch champions · RUST_MIGRATION Phase 2 + 4
**Reframed (2026-06-07): there are no `.pt` champion weights**, so there's
nothing to PyTorch→burn parity against — that framing (goldens / weight-transfer
oracle) is dropped. janus already does the right thing without champions: CNN
inference is **off by default** (`ENABLE_CNN_INFERENCE`), and even when enabled a
missing/bad checkpoint just logs a warning and disables — falling back to the
rule-based 9-gate chain (`services/forward/src/cnn_inference.rs:105`). So no
models are required to run. The burn scaffolding in `crates/ml` (PerAssetCnn,
MasterCnn, 20-ch features, labeler, dataset, trainer, `train_champion`) is the
path to *create* champions from scratch later.
- [x] **Sane defaults / graceful-disable verified.** No champion ⇒ CNN inactive
      ⇒ rule-based path; `CnnConfig::default()` is off-by-default. "Run without
      weights" already works — nothing to do.
- [x] **CNN feature contract: 20-channel (decided).** The Ruby `models/` README's
      15-feature / window-60 shape described the *PyTorch* champions — which don't
      exist as weights. Since janus trains from scratch in burn, ground truth is
      janus's own 20-channel `crates/ml` pipeline; the 15-feature contract is
      historical. (Exact window/classes confirmed from `crates/ml` when training
      lands — but the channel count is settled: 20.)
- [ ] **Train champions from scratch in burn** (when data + GPU available):
      `train_champion(ohlcv)` → features → labels → train → save the `.bin` +
      `.json` sidecar. Then cosine LR / train-val split / temperature calibration
      as training polish.
- [ ] **Flip the gate** only once a trained model shows edge on a shadow basis:
      `ENABLE_CNN_INFERENCE` → `ENABLE_BRAIN_RUNTIME`. The CNN-agreement /
      -confidence gates already consume the vote (Track C).

### Track C — Safety: execution gate + risk · RUST_MIGRATION Phase 3 · ◀ all 9 gates live (advisory)
- [x] **Port the 9-gate `ExecutionGate`** → `crates/execution-gate` (faithful
      chain + `ConsecutiveLossBreaker` + `CorrelationGuard` + `AdaptiveThreshold`;
      pure/synchronous core; 37 tests + doctest; clippy-clean). 2026-06-07.
- [x] **Wire into the live loop — Stage 1 (advisory) done.** `services/forward/src/
      gate_integration.rs` (`ForwardGate`) is instantiated in the live loop;
      each prospective entry is evaluated and the verdict stamped as `gate`
      signal metadata. Enforcement is behind `JANUS_GATE_ENFORCE` (default off) —
      a blocking verdict then joins the prop-firm/risk `block_reason` chain and
      suppresses the execution submit (signal still publishes to the bus,
      preserving no-autonomous-execution). Stage 1 feeds the `RiskManager` verdict
      + confidence; the remaining gate inputs are inert pass-throughs until the
      next two items land (so the gate never emits a spurious block).
- [x] **Producer plumbing — Stage 2a (CNN + correlation) done.** The live loop
      now captures the CNN vote (separate from its consensus contribution) and
      feeds it to the gate's `cnn_agreement` + `cnn_confidence` checks, and feeds
      the correlation guard per-candle close-to-close log returns + the open-
      position symbols from `PortfolioState`. With the guard enabled in
      `ForwardGate`, the `correlation` gate is live. (CNN gates stay inert until
      `ENABLE_CNN_INFERENCE`.)
- [x] **Producer plumbing — Stage 2b (AO + vol + quality) done.** Rather than
      hand-roll indicators, the loop runs the published `indicators-ta` signal
      engine per symbol (`gate_integration::GateProducers`, fed each candle) and
      feeds the gate **`ao`** (its Awesome Oscillator), **`vol_pct`** (a
      `VolatilityPercentile` over engine ATR), and a **`quality`** proxy
      (momentum + wave-ratio percentile blend, 0–100). Producers stay `None`
      (gates inert) until the engine warms (~100 bars), so no spurious blocks.
- [x] **Producer plumbing — Stage 2c (fee/TP gate) done.** `GateProducers`
      derives a `tp_pct` (`tp_atr_mult × ATR / close`, default 4.0 ≈ 2R on a
      2×ATR stop, override `JANUS_GATE_TP_ATR_MULT`) and feeds it to the
      fee-viability gate, which measures the round-trip taker + slippage fees
      against it — config-driven via `JANUS_GATE_FEE_TAKER` / `JANUS_GATE_FEE_SLIP`
      (applied by `ForwardGate::apply_fees`; defaults match `GateContext`).
      **All 8 entry gates now act on real data** — only the consecutive-loss
      breaker (close-driven) remains.
- [ ] **Quality refinement (optional)** — replace the momentum/wave quality
      proxy with `indicators-ta`'s `compute_signal` confluence/`bull_score` once
      the `LiquidityProfile` + `ConfluenceEngine` inputs are assembled per bar.
- [x] **Close the loop (breaker close-feed) done — all 9 gates now act on real
      data.** A `GateOutcomeRecorder` trait-object slot on `JanusState` (mirrors
      `AffinityRecorder`) lets the `/api/v1/positions/close` handler feed every
      realized close into the gate's consecutive-loss breaker; the gate is shared
      as `Arc<RwLock<ForwardGate>>` and keyed by base-asset on both sides so the
      close-feed matches the eval key. (The adaptive threshold rides the same
      `record_trade_outcome` path but stays unconfigured for now — a tuning
      follow-up.)
- [x] **Redis observability for the gate counters (Grafana).** Done:
      `GateMetrics::snapshot()` (gate crate, pure) + a best-effort exporter in
      `services/forward/src/gate_metrics_redis.rs` that mirrors the block/eval
      counters to Redis (`{prefix}gate_blocks:{asset}:{reason}`,
      `{prefix}gate_evals:{asset}`) on a timer. Opt-in via
      `JANUS_GATE_METRICS_REDIS=1`; no-ops if Redis is absent; Redis I/O stays out
      of the gate crate. The same exporter now also **persists the consecutive-loss
      breaker state** (`{prefix}gate_breaker_state`, JSON) and restores it on
      startup — a tripped breaker survives a restart instead of silently resetting
      (`ConsecutiveLossBreaker::export`/`import`). (Adaptive-threshold-state
      persistence is the only remaining bit.)
- [x] **Correlation guard: keep both (decided).** The gate's `CorrelationGuard`
      (log-returns, externally fed, parity-faithful to Ruby) and `jflow-risk`'s
      `CorrelationTracker` (price-fed, different defaults) serve different layers;
      the gate crate is intentionally dependency-free, so it keeps its own rather
      than depending on `jflow-risk`. Not merging — the rationale is documented in
      `crates/execution-gate/src/correlation.rs`.
- [ ] **Enforcement flip (runbook).** The gate is **advisory by default** — it
      stamps `gate` metadata on every signal but never blocks. To enforce, set
      `JANUS_GATE_ENFORCE=1`: a blocking verdict then joins the prop-firm/risk
      `block_reason` chain and suppresses the execution *submit* only (the signal
      still publishes to the bus — no-autonomous-execution preserved). Procedure:
      (1) run advisory for N days, watch the `gate` metadata + block reasons;
      (2) confirm the block mix looks right (esp. that warmup-empty producers
      aren't over-blocking); (3) flip `JANUS_GATE_ENFORCE=1`. Related tunables:
      `JANUS_GATE_TP_ATR_MULT` (fee-viability TP target), `ENABLE_CNN_INFERENCE`
      (CNN gates), `JANUS_PROP_FIRM_ENFORCE` / `JANUS_RISK_ENFORCE` (sibling gates).
- [ ] **Exit:** order flow gated entirely in Rust; the human-confirmation
      invariant (`EXECUTION_MODE=paper_trading` default) preserved throughout.

### Track D — Infra contract: serve what fks-full repoints to · RUST_MIGRATION §12-C
fks-full's removal of Ruby left its WebUI + nginx pointing at `fks_ruby` (still
74 refs in nginx, 9 in `vite.config.ts`, 10 `ruby_signal` refs in the WebUI).
janus must serve the contract before fks-full can repoint cleanly — that's the
janus half of the work.
- [~] **Serve the WebUI/data contract from `lib/janus-api`** — the routes the
      SvelteKit dashboard + nginx call. **Chart path done:** `/sse/bars/{symbol}`
      live tail (#106) + REST history `GET /bars/{symbol}` (columnar) and
      `GET /bars/{symbol}/candles` (flat, ms timestamps), both reading
      `candles_crypto` over `QUESTDB_HTTP_URL` with separator-insensitive
      symbol matching. **Front-page trio done:** `/api/pipeline/scores/json`
      (latest signal per symbol → score/`cnn_signal`), `/api/trades/open` (the
      `PositionTracker`'s open positions — `PositionState` now retains the
      latest economics), `/factory/status` (per-module health + uptime). All in
      `lib/janus-api/src/webui_contract.rs`, served truthfully from janus state,
      empty-safe. **Remaining:** the **fks-full side** — repoint nginx +
      `vite.config.ts` off `fks_ruby` onto janus (74 nginx refs / 9 vite refs);
      optional `/factory/coverage/{symbol}` + asset-list endpoints if the UI
      still needs them (the `ruby_signal` field is retired). Tracked in fks-full.
- [x] **Documented the public surface** → [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md):
      the brain / data / forward / execution HTTP+WS routes (`/api/v1/brain/*`,
      `/api/v1/risk/*` incl. `/risk/validate`, `/api/v1/positions/*`, the data
      gaps/indicators/signals endpoints, the WS streams) and the full env-var
      reference. (Serving the contract — above — is still the open half.)

### Track E — Long tail: rebuild only what's needed · RUST_MIGRATION Phase 5
- [ ] News/sentiment, on-chain — port from `ruby/src/data/{news,chain}` *only if
      the demo needs them*; mostly I/O + glue.
- [ ] **Rithmic stays Python** behind a thin gRPC/HTTP sidecar (proprietary
      mTLS+gRPC; no Rust equivalent) — migrate last or never.
- [ ] Multi-account routing, journal, dashboards — rebuild as janus crates if
      required.

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
- [ ] **Triage `cargo audit` advisories, then make the gate blocking.** Re-audited
      2026-06-13 (`cargo-audit 0.22.2`, 1184 deps): **5 vulnerabilities** + unmaintained/
      unsound warnings, down from 10. Cleared:
  - ~~`rustls-webpki 0.103.10` (×3: RUSTSEC-2026-0098/0099/0104)~~ → 0.103.13 (#109).
  - ~~`postgres-protocol 0.6.10` (×2: RUSTSEC-2026-0179/0180 — SCRAM CPU-exhaustion +
    `hstore` panic) · `tokio-postgres 0.7.16` (RUSTSEC-2026-0178 — `DataRow` panic)~~ →
    0.6.12 / 0.7.18 (2026-06-13) — three DoS advisories cleared by a patch-level bump.
  - ~~`pyo3 0.21.2`, `fast-float 0.2.0`~~ — no longer in the lock (dropped by transitive updates).
  Remaining (both already assessed as accept-for-now):
  - `astral-tokio-tar 0.5.6` (×4: RUSTSEC-2026-0066/0112/0113/0145 — PAX/symlink
    extraction) — **dev-only**: pulled solely by `testcontainers 0.26.3` (a dev-dependency
    pinned `^0.5.6`), so it's never in the shipped binary and only touches test-time
    container-image extraction. Clearing it needs a `testcontainers` major bump (to a
    release on tar ≥0.6.2) + adapting the integration tests — deferred, low real risk.
  - `rsa 0.9.10` (RUSTSEC-2023-0071 Marvin timing sidechannel) — no upstream fix;
    transitive via the DB/TLS stack, no RSA signing on a janus hot path.
  - Warnings (no fix; candidates for a `deny.toml` allowlist): unmaintained `paste`,
    `bincode`, `rustls-pemfile`, `atomic-polyfill`, yanked `core2`; unsound `rand`
    (custom-logger edge) / `rkyv`.
  - Context: janus generates signals, **never executes orders / isn't a public web
    service**, so exploitability is low. **To flip the gate blocking:** add a `deny.toml`
    allowlisting the 5 remaining (rsa + the 4 dev-only tar) with rationale, then drop the
    `exit 0` in `security.yml` so *new* advisories red-line `main`.
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
- [ ] **JFLOW-B:** Optimizer reads its asset list from janus's native asset registry (`crates/registry`) instead of env-only. (Ruby is gone — this feeds off the janus-native registry now; same thread as Track A's "Asset registry in janus".)
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
- [x] **Documented the public surface** → [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md).
      Brain / data / forward / execution HTTP+WS routes + the env-var reference
      (core, execution/safety incl. `EXECUTION_MODE`, the gate `JANUS_GATE_*`,
      the candle-scan `JANUS_CANDLE_SCAN_*`, and JanusAI). Path/method/purpose
      level, grounded in the actual routers; payload shapes link to source.
- [ ] **Surface prop-firm support.** `crates/models/src/prop_firm.rs` (`PropFirmValidator`, `ChallengeType`) is undocumented — note where/whether it's wired into execution.

---

## P3 — Neural architecture & research

- [ ] **30-day live-trading validation** of the neuromorphic stack (~250K LOC across 10 brain regions) → document the public API → stabilize the `janus-neuromorphic` crate for production use. This is the gate before treating brain output as authoritative.
- [ ] **Optimizer ↔ asset registry** (JFLOW-B follow-up): pull the asset universe from janus's native registry rather than env. (Ruby's registry is gone; same thread as Track A's "Asset registry in janus".)
- [ ] Sweep the 107 neuromorphic `#[allow(dead_code)]` annotations once the regions stabilize (currently churning too fast to audit usefully).

---

## Out of scope for this repo

Cross-cutting / fks-full-owned items (tracked in
[fks-full](https://github.com/nuniesmith/fks-full)):

- `RC-CRATES-*` (rustcode workspace: runtime/api/tools/plugins/commands/server/claw-cli/compat-harness/lsp)
- `API-*` (rustcode API security & config; rc-core/rc-api/rc-rag/rc-llm split)
- `OSS-*` evaluation queue (OpenViking, Heretic, Nanochat)
- **fks-full infrastructure repoint** — nginx / WebUI / test scripts off
  `fks_ruby` (RUST_MIGRATION §12-C). janus's half (serving the contract) is
  **Track D** above.

> **No longer out of scope (Ruby is gone, 2026-06-07):** the **execution engine
> + asset registry** (janus no longer merely *suggests* — it's the platform; see
> Tracks A + C), the **JanusAI session-metrics / memory compaction / optimizer
> search-space**, and the **session-start config producer** (`fks:janus:config`)
> are now janus's own responsibility or obsolete. Fold the live ones into the
> tracks above as they surface.
