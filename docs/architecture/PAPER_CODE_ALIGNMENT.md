# Paper ↔ Code Alignment Review

> **What this is.** A claim-by-claim audit of *Project JANUS: A Neuromorphic
> Architecture for Autonomous Trading* (the paper, dated 2026-03-29) against the
> actual Rust in this repository. Every row was verified by reading source, not
> READMEs. Paths are `file:line` and clickable.
>
> **How to read the status column.** ✅ implemented as described · ⚠️ partial,
> relocated, or differs in detail · ❌ missing, stubbed, or contradicted.
>
> **Bottom line up front.** The paper describes the *aspirational* system. Most
> of the sophisticated machinery genuinely exists on disk — and several pieces
> are excellent — but (a) the **live trading binary is a rule engine**, not the
> neuromorphic brain, and (b) of the unwired research code, fidelity to the
> paper's equations is uneven: some is exact, some is an "orphaned twin" of a
> simpler wired version, and a handful of headline claims are absent or
> misrepresented. This is consistent with the repo's own
> [`ML_STORY.md`](ML_STORY.md) and [`RISK_TOPOLOGY.md`](RISK_TOPOLOGY.md).

---

## 1. The headline gap: "implemented" vs. "wired"

The paper (§7, §10.3) and its prose throughout present the brain-region pipeline
(OpAL action selection, LTN constraints, DiffGAF/ViViT vision, three-timescale
memory) as the operating system. The running service does **none** of that
neurally:

- **The live path runs zero neural inference.** It is `indicators (EMA/RSI/MACD)
  → RegimeManager → Hypothalamus (ATR+regime → position scale) → Amygdala
  (drawdown/vol → threat filter) → affinity/correlation gating → execution`
  (`docs/architecture/ML_STORY.md:14`, `services/forward/src/brain_wiring.rs:527`).
  The *shape* matches the paper's 6-stage pipeline (§5.2.2); the *implementation*
  is rule-based, not the neural OpAL/LTN/vision stack.
- **~300K LOC of ML/neuromorphic is wired to nothing.** The 251K-LOC
  `neuromorphic/` crate compiles into every build but no service depends on it;
  it is now CI-quarantined (`ML_STORY.md:34`, `TODO.md:122`).
- **No model weights exist.** Zero `.onnx`/`.safetensors`/`.pt` artifacts in the
  repo (`ML_STORY.md:19`). The only "inference" hook is `enable_ml_inference:
  false` by default and its enabling constructor has no callers.

This is the single most important thing to reconcile between paper and code. The
paper's Limitations section (§8) discusses integration *risk* but never states
that the production brain is rule-based.

---

## 2. Tier-1 findings — claims that are absent, fictional at runtime, or contradicted

| Paper claim | Status | Reality | Evidence |
|---|---|---|---|
| **OpAL action selection** (Contribution #4, Innovation #5): Q⁺/Q⁻ pathways, `a = softmax(d_direct − λ·d_indirect)`, OpAL* dynamics (eq 30–34) | ❌ | The `direct_pathway.rs`/`indirect_pathway.rs` files are 1-line threshold stubs. No Q⁺/Q⁻ opponent learning, no `softmax(d_direct − λ·d_indirect)` anywhere, zero references to OpAL/Jaskir/Frank. What exists is a generic actor-critic + tabular Q-learning + a separate NoGo risk-veto. | `neuromorphic/basal_ganglia/.../direct_pathway.rs:15`, `indirect_pathway.rs:15`; `actor_critic.rs:192` |
| **Custom wgpu GPU kernels** (Innovation #10): MatMul, Softmax, LayerNorm, Attention, GELU, Reduce | ❌ | All 6 WGSL kernels exist as *source strings* but `wgpu` is not even a dependency; the runtime is an explicit **CPU simulation** ("these shaders are not actually compiled — the CPU fallback paths execute instead"). `discover_adapter` always returns `Err(NoAdapter)`. | `neuromorphic/gpu/shaders.rs:7-10`, `gpu/runtime.rs:486,635` |
| **GPU-accelerated LOB simulator** (§9.1, JAX-LOB analogue) | ❌ | Real CPU matching engine, but **zero** GPU/parallel code in `crates/lob` — no wgpu/cuda/rayon deps. | `crates/lob/src/matching_engine.rs:634`; `crates/lob/Cargo.toml` (no GPU deps) |
| **Mahalanobis distance + circuit-breaker trigger** `D_M > τ_danger` (eq 35, 36) | ❌ | No Mahalanobis / inverse-covariance anywhere in `neuromorphic/`. Circuit breakers trigger off VPIN thresholds, not eq-36. | grep clean in `neuromorphic/` |
| **Chronos foundation-model forecasting via ONNX** (§4.4.1) | ❌ | Statistical-baseline **mock**. Real tokenizer exists, but no ONNX runtime runs — "ONNX model loading via tract is available but not fully wired. Using statistical baseline forecast." `ort` dep is commented out (yanked). | `neuromorphic/thalamus/sources/chronos.rs:1234-1252`; `neuromorphic/Cargo.toml:39` |
| **Gated cross-attention multimodal fusion** of 4 modalities (eq 12, 13) | ⚠️→❌ | eq-12 scaled-dot attention exists and Wilson-Cowan is real, but the pieces are **never assembled**: eq-13's learnable gate `g = sigmoid(Wg[v;t;s]+bg)` is not implemented (only a per-channel scalar sigmoid), and the 4 "modalities" are single-modality venue aggregators that are never cross-fused. | `thalamus/attention/cross_attention.rs:275`; `gate.rs:320`; `fusion/mod.rs:5-14` |
| **"Candle for training + inference"** + safetensors (§7.1) | ⚠️ | Candle is real for `crates/training`/`vision`/`logic`, but the flagship **Double-DQN/RL path uses Burn**, and the dormant inference path is tract-ONNX. So "end-to-end Candle" overstates. | `crates/ml/src/dqn.rs:49` (burn); `crates/training/src/optimizer.rs:27` (candle) |
| **README**: "Janus NEVER executes orders directly" | ❌ (README) / ✅ (paper) | The execution service **is** fully wired to place real authenticated REST orders (Kraken HMAC-SHA512 `AddOrder`). It is gated only by `dry_run=true` default + credentials + kill-switch, not by architectural inability. Here the **paper is accurate** and the README oversells the safety guarantee. | `orders/mod.rs:199` → `exchanges/router.rs:226` → `exchanges/kraken/rest.rs:425`; `execution/signal_flow.rs:806` |

---

## 3. Tier-2 findings — formula and detail mismatches

| Paper claim | Status | Reality | Evidence |
|---|---|---|---|
| `slightly(x) = √x − x` (eq 24) | ❌ | Coded as `x − x²` (`v - v*v`); even the comment says "x − x²". | `neuromorphic/prefrontal/ltn/fuzzy_logic.rs:186` |
| **Dual-mode LTN**: Product t-norm for training, Łukasiewicz for inference (eq 15/16, 19/20) | ⚠️ | Both norms exist, but the t-norm is fixed at construction — **no train/infer switching** anywhere. `DiffLTN` defaults Łukasiewicz for both. | `crates/logic/src/diff_tnorm.rs:180,187`; `ltn.rs:857` |
| `SAT(KB) = p-mean of axioms` (eq 28) | ⚠️ | A p-mean aggregator exists but the **loss path doesn't use it** — SAT is a weighted Mean/Min/Product. (`L = 1 − SAT` itself ✅.) | `crates/logic/src/ltn.rs:730,691` |
| "**10 market axioms**" (KB: wash-sale, Almgren-Chriss, position/risk limits) | ⚠️ | Exactly 10 axioms exist, but they are **signal-logic** rules (trending→long, prob-sum=1…), not the regulatory KB constraints listed. The regulatory rules live in imperative `conscience/` modules, not as LTN axioms. | `crates/ltn/axioms.rs:92-268` |
| Almgren-Chriss **slippage axiom** `Execute ⇒ Slippage < λ·Volatility` (eq 27) | ❌ | Not an axiom at all. Almgren-Chriss exists only as an execution-cost model. | grep clean in `neuromorphic/prefrontal` |
| Almgren-Chriss impact model (§5.5.1) | ⚠️ | The **wired** cerebellum version is simplified and **drops σ** (`kappa = lambda*eta/(t*eta+gamma)`); the correct `κ = √(λσ²/η)` sinh-form version lives in a different (also-wired) module. | `cerebellum/almgren_chriss.rs:30` (simplified) vs `cerebellum/impact/almgren_chriss.rs:203` (canonical) |
| Volatility spike `σ > 3·σ_baseline` (§5.4.1) | ⚠️ | Wired fear network uses `volatility > avg*2.0` (2σ). | `neuromorphic/amygdala/fear/fear_network.rs:230` |
| VPIN with **bulk-volume classification** (§5.4.2) | ⚠️ | Wired calculator takes a pre-set `is_buy` bool — no BVC. Proper BVC (erf/normal-CDF) exists only in an **orphaned** twin file. | `vpin/calculator.rs:48` vs orphaned `vpin/vpin_calculator.rs:326` |
| **Four-scope kill switch** (§5.4.2) | ⚠️ | A 4-variant `KillSwitchScope` enum is defined but **never used** — trigger logic is global-only. (The 4 *actions* — KillSwitch/PositionFreeze/SafeMode/CancelAll — are real and wired.) | `amygdala/kill_switch.rs:46`; `circuit_breakers/mod.rs:11` |
| Pattern separation `h = tanh(W_rand·[s;a;c])` (eq 40) | ❌ | No random projection / `W_rand` in `hippocampus/`. (Place-cell spatial map ✅.) | grep clean |
| Schema **centroid** `z_k` (eq 47) + **recall-gated** EMA update (eq 48) + **k-means** in deep-sleep (Algorithm 3) | ❌ | Schemas are range-only; no `z_k` centroid, no `recall_success` gating anywhere, no k-means routine anywhere. | `cortex/memory/schemas.rs` (range-only); grep clean for recall/kmeans |
| Episodic tuple `(s,a,r,s',c,e)` with emotional tags `(fear, confidence, surprise)` (eq 39) | ⚠️ | Base tuple is `(s,a,r,s',done)`; context and emotion live in separate structs. Emotional model is valence/arousal/**fear** — only fear is an explicit tag; "confidence"/"surprise" are not. | `hippocampus/buffer.rs:9`; `emotional/emotional_tagging.rs:40` |
| DDPM **Min-SNR loss weighting + EMA averaging** (§6.2.2) | ❌ | Config enums/fields exist (`MinSnr`, `ema_decay`) but are **never read** in training; loss is plain MSE. (DDPM itself is real Candle.) | `hippocampus/diffusion.rs:163,1377` |
| **AlignedUMAP** temporal manifold alignment (§6.5) | ❌ | Not present. (Parametric UMAP + drift detection + Qdrant bridge ✅.) | grep clean |
| Forward Service **HTTP port 7000** (§3.4, §7.2) | ❌ | Actual REST is 8080; brain REST 8180. No `7000` in the repo. | `services/forward/src/main.rs:656` |
| **Six exchange adapters** (Kraken/Bybit/Coinbase/OKX/Binance/Kucoin) (§7.2.2) | ⚠️ | Only **3** execution adapters exist: Kraken, Bybit, Binance. Coinbase/OKX/Kucoin appear only in the data layer, not as `Exchange` impls. | `exchanges/market_data.rs:27` (`ExchangeId = {Bybit,Kraken,Binance}`) |
| **7 regime categories** (Bull…Deflation) (§4.3.2) | ⚠️ | The exact 7-variant enum exists in the **memory schema**, but the active **regime detector uses a different 4-variant enum** (Trending/MeanReverting/Volatile/Uncertain), and several other inconsistent `MarketRegime` enums exist. | `cortex/memory/schemas.rs:33` vs `crates/regime/src/types.rs:19` |
| **Ensemble** = HMM + statistical tests + technical (§4.3.1) | ⚠️ | The actual ensemble combines **2** families (HMM + technical). Variance-ratio/Hurst/fat-tail are real but live in `amygdala`/`dsp`, not wired into the regime ensemble. | `crates/regime/src/ensemble.rs:140`; `amygdala/.../regime_shift.rs:314` |
| **Isolation forest** anomaly scoring (§5.4.1) | ❌ | Doc-comment only ("Isolation Forest-inspired"); no implementation. | `amygdala/threat_detection/anomaly_detector.rs:5` |
| Cerebellar forward model `p̂_{t+1} = f(s,a)` (eq 37) | ❌ | Exported `ForwardModel` is a stub (one field, no predict). Real predictors (Smith predictor etc.) exist in submodules but no single eq-37 function. | `cerebellum/forward_model.rs:3` |
| `indicators` crate (ADX/ATR/Bollinger/EMA/RSI/MACD, incremental) (§7.1) | ⚠️ | No standalone crate (retired → external `indicators-ta`); a copy lives in `crates/regime`. **MACD is not implemented incrementally.** | `crates/regime/src/indicators.rs`; `Cargo.toml:26` |
| 8-D DSP vector O(1)/zero-alloc (§7.1) | ⚠️ | 8 fields exact ✅; FRAMA is O(1) ✅; but Sevcik keeps an O(window) buffer and allocates per tick, contradicting the "zero-alloc" docstring. | `crates/dsp/pipeline.rs:453`; `sevcik.rs:300` |
| AdamW/SGD optimizers (§7.1) | ⚠️ | The Candle wrapper only instantiates AdamW (SGD is a config-only enum); the gradient clipper is a **no-op stub**. | `crates/training/src/optimizer.rs:183,259` |
| `ComplianceSheriff` (§7.1) | ⚠️ | Removed from `crates/compliance`; consolidated into `PropFirmValidator` (which is real). Paper text still names ComplianceSheriff. | `crates/compliance/src/lib.rs:4`; `crates/models/src/prop_firm.rs:71` |
| **8 services run together** (§3.4 deployment diagram) | ⚠️ | The 8 binaries exist, but the standalone `docker-compose.yml` runs a **single unified binary**, not 8 separate services. | `docker-compose.yml:44` |

---

## 4. Tier-3 — claims that hold up (genuinely implemented, often excellent)

These are real, non-stub, and (where applicable) match the math. Credit where due:

- **DiffGAF engine** — analytic Jacobian with numerical-gradient verification at
  1e-4, exact GASF/GADF (eq 5/6), learnable affine (eq 2), polar derivative
  (eq 7). `neuromorphic/visual_cortex/gaf/differentiable.rs:565,1805`;
  `crates/vision/src/diff_gaf.rs:232,267`. *(Caveat: the Burn-LSTM variant uses
  Taylor-series trig; the code computes **dual-GAF**, not GADF-only as the prose
  says nor GASF-only as Algorithm 1 says.)*
- **Wilson-Cowan oscillatory dynamics** (§4.4.2) — fully real (2,534 lines):
  E-I ODEs, RK4, Hilbert-transform amplitude/phase, bifurcation sweep
  (Hopf/saddle-node), fixed-point stability classification.
  `neuromorphic/thalamus/gating/wilson_cowan.rs:676,845,1286,1444`.
- **Quantum-inspired portfolio optimization** (§5.1.4) — the strongest single
  module. Genuine QAOA (2ⁿ statevector, cost unitary + transverse-field mixer),
  VQE, simulated quantum annealing (Suzuki-Trotter), plus real Markowitz,
  Risk-Parity (Spinu), Black-Litterman (Gauss-Jordan). No stubs in 2000+ lines.
  `neuromorphic/prefrontal/.../quantum_portfolio.rs:1546,1951,2445`.
- **Prioritized experience replay** (eq 41–43) — correct `p=|δ|+ε` (ε=1e-6),
  `P(i)=pᵅ/Σ` (α=0.6), IS weight with β annealed 0.4→1.0, **two real sum-tree
  implementations** (O(log n)). Double-DQN Bellman target is correct.
  `services/backward/src/tasks/train.rs:501,642`; `crates/memory/src/replay.rs:11`.
- **Sleep-phase state machine** (eq 44) — Awake→Light→Deep→Integration→Transition
  with per-phase replay scaling. `hippocampus/swr/consolidation_sync.rs:15`.
- **Range-based schemas + Markov transitions** (eq 45, 46) — 7 schemas,
  weighted multi-criteria range matching, transition matrix with stationary
  distribution. `cortex/memory/schemas.rs:193,411`.
- **Parametric UMAP + drift detection** (eq 49, 50) — real attraction/repulsion
  loss with k=5 negatives, Candle MLP encoder, configurable metrics, drift
  severity, Qdrant bridge. `visual_cortex/parametric_umap.rs:630,565,1298,1473`.
- **DDPM diffusion** — real Candle forward/denoise/sample with regime
  conditioning and 3 noise schedules. `hippocampus/diffusion.rs:676,1451`.
- **PID controller + Ziegler-Nichols auto-tuning** (eq 38) — full P/I/D with
  anti-windup, derivative filtering, 6 ZN tuning variants (tested).
  `cerebellum/error_correction/pid_controller.rs:234,477`.
- **Kill switch (dual-layer)** — in-process `AtomicBool` zero-latency halt +
  Redis cross-process coordination with dead-man TTL.
  `amygdala/kill_switch.rs:59`; `services/forward/src/persistence/kill_switch_redis.rs:298`.
- **CNS safety subsystem** — the most faithfully implemented area. 5-phase
  preflight (Infrastructure/Sensory/Regulatory/Strategy/Executive) with
  criticality levels + PreFlightRunner + BootReport; watchdog with auto
  kill-switch on critical death; Closed/Open/HalfOpen circuit breakers; reflex
  actions with allowlist-validated command exec; Brain Coordinator with exactly
  10 regions, topological init ordering, per-region health.
  `crates/cns/src/preflight/mod.rs:52`, `watchdog.rs:741`, `reflexes.rs:16`,
  `neuromorphic.rs:201`.
- **BERT/FinBERT sentiment** (Candle) + **Qdrant** vector store — both real
  (not mocks). `thalamus/sources/bert_sentiment.rs:1196`;
  `crates/memory/src/qdrant_client.rs:75`.
- **Wash-sale detector** — full 30-day lookback **and** lookforward, partial
  disallowed-loss, cost-basis tracking. `crates/compliance/src/wash_sale.rs:301`.
- **Homeostasis controller**, anomaly detector (z-score/MA/percentile/
  multivariate), regime-shift CUSUM, correlation-break, VPIN→kill-switch flow —
  all real and wired. `hypothalamus/homeostasis/controller.rs:189`;
  `amygdala/threat_detection/anomaly_detector.rs:236`.
- **Dual-process services + 6-stage pipeline + 9 strategies + StrategyGate** —
  Forward/Backward split is real; the live pipeline order
  (Regime→Hypothalamus→Amygdala→Gate→Correlation→Execution) matches §5.2.2
  exactly; exactly 9 named strategies; full gating (regime/affinity/
  allow-deny/untested). `services/forward/src/brain_wiring.rs:541-757`;
  `crates/strategies/src/{*.rs,gating.rs}`.
- **TWAP/VWAP/Iceberg** execution algorithms — all three real slicing engines
  (fills currently simulated). `services/execution/src/strategies/`.
- **Double DQN, warmup+cosine LR, safetensors** — all real.
  `crates/ml/src/dqn.rs:481`, `optimizer.rs:421`; `crates/training/src/loop.rs:514`.

---

## 5. Cross-cutting patterns worth knowing

1. **"Orphaned twin" anti-pattern.** Repeatedly, the rigorous implementation
   exists but isn't declared in its `mod.rs`, while a simpler twin is what's
   live: the good Kelly (`position_sizing/`), BVC-VPIN (`vpin_calculator.rs`),
   liquidity-crisis & flash-crash detectors, fear-extinction `FearNetwork`, and
   the canonical Almgren-Chriss. A reader of the paper would find the *good*
   code; a reader of the running service gets the *simple* one.
2. **Fragmentation / duplication.** Three parallel LTN implementations
   (`crates/ltn`, `crates/logic`, `neuromorphic/prefrontal/ltn`); multiple
   regime enums; two AdamW paths; two sum-trees. Consolidation would make the
   paper's single-system narrative true.
3. **Stubs at integration seams.** `RiskManager`, `Regulator`, cerebellar
   `ForwardModel`, the basal-ganglia pathways — the *coordinators* that would
   tie components into the paper's pipelines are placeholders, even where the
   leaf components are real.
4. **Mixed backend reality.** Burn (DQN) + Candle (training/vision) + tract-ONNX
   (dormant) + (claimed) wgpu (CPU-simulated). The paper's "pure Candle,
   end-to-end" is the aspiration, not the state.

---

## 6. Recommended reconciliation

For each gap there are two honest routes: **fix the code** to match the paper, or
**soften the paper** to match the code. Suggested split:

### Soften / reword the paper (lowest effort, restores accuracy)
- Add an explicit **Implementation Status** that states the live path is
  rule-based and the neuromorphic brain is research scaffolding not yet on the
  hot path (the repo already says this in `ML_STORY.md`).
- Reword GPU claims: kernels are **authored** but execute via CPU fallback;
  real GPU is an optional Burn-wgpu backend. Drop "GPU-accelerated LOB."
- Say **Burn + Candle**, not "pure Candle end-to-end."
- Chronos: "tokenizer + statistical baseline; ONNX path scaffolded, not wired."
- Exchanges: list the **3** implemented (Kraken/Bybit/Binance); mark others planned.
- Fix port 7000 → 8080/8180; note the unified-binary compose mode.
- Multimodal fusion: describe the components, not an assembled gated cross-attention.
- Reconcile the README's "never executes directly" with the real (gated) order path.

### Fix the code (if you want the paper to stay as-is)
- **High value:** implement OpAL properly (Q⁺/Q⁻ + `softmax(d_direct − λ·d_indirect)`)
  — it's a named contribution; add Mahalanobis eq-35/36 to the amygdala; wire a
  real eq-37 cerebellar forward model.
- **Correctness:** fix `slightly` → `√x − x`; restore σ in the wired
  Almgren-Chriss; make the vol-spike 3σ; use p-mean in `SAT(KB)`.
- **Wiring:** add the dual-mode (Product/Łukasiewicz) train/infer switch;
  promote the "orphaned twins" (BVC-VPIN, good Kelly, canonical AC) over the
  simple versions; collapse the 4-variant detector enum onto the 7-regime
  `RegimeId`; fold the statistical tests into the regime ensemble.
- **Smaller:** apply Min-SNR/EMA in the diffusion trainer; implement the eq-40
  random projection, eq-47/48 centroid + recall-gating, Algorithm-3 k-means,
  AlignedUMAP; make the grad-clipper real; add incremental MACD.

---

## 7. Minor / cosmetic

- README self-contradicts on size: "39-crate" (line 5, correct) vs "~50
  workspace crates" (line 113). Workspace defines **39** members.
- The paper's Repository box links `github.com/nuniesmith/fks`; the repo is
  `nuniesmith/janus`.
- Reflex action enum is typo'd `RefexAction` (`crates/cns/src/reflexes.rs:229`).

---

*Generated from a six-part source audit (perception, reasoning, risk/safety,
memory, ML-stack/infra, services/strategies). Re-run the verification commands in
`ML_STORY.md:121` to confirm the wiring claims.*
