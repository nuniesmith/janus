# The ML Story — Current State & Decision Memo

> **Status: decision memo, awaiting a call** (the *"decide the ML story"* P0 in
> [`TODO.md`](../../TODO.md)). janus carries **~300K LOC of ML code, almost none
> of it wired into live trading.** This records what exists, the train→serve
> gap, and the realistic directions, so the decision can be made from facts
> rather than re-surveyed. Compiled 2026-06 from a full read-only survey of the
> ML crates + the live signal loop.

---

## TL;DR

- The **live trading path runs zero neural inference.** It is a deterministic
  indicator/rule engine: technical indicators → regime detection → position
  scaling → threat filter → execution gating (`services/forward` →
  `brain_wiring.rs` → `TradingPipeline`).
- Around it sits **~300K LOC of ML research scaffolding wired to nothing**, and
  there are **no model weight files anywhere** in the repo. The gap between
  "code that can train/infer" and "a model that affects a trade" is total.
- The open decision is **which of four directions** to take (§5). This memo is
  *not* an argument to keep building ML on spec.

---

## 1. What exists (verified map)

| Component | LOC | Status | Evidence |
|---|---|---|---|
| `forward/src/inference.rs` (tract-ONNX) | 644 | **Built, dormant** | `enable_ml_inference: false` default (`signal/mod.rs:64`); the `predict()` call is gated behind it (`signal/mod.rs:536`); `with_ml_inference()` (`:153`) has **zero callers** |
| `crates/ml` (Burn; LSTM / MLP / DQN) | 12.8K | **Trains, never served** | trained in `services/backward` (`tasks/train.rs`); **no ONNX/weight export path** |
| `crates/vision` (DiffGAF → ViViT) | 32.8K | **Experimental, unwired** | no service depends on it; examples/benches only. Input is raw time-series→images — **does not match** the live loop's hand-crafted indicators |
| `crates/training` / `crates/ltn` / `crates/logic` | ~9K | **Experimental, unwired** | consumed only by the `vision_ltn_training` example |
| **`neuromorphic/`** | **251K** | **Experimental, unwired** | a workspace member (`Cargo.toml:52`) — so it compiles in workspace/CI builds — but **no service depends on it**; its own `lib.rs` marks the APIs unstable |
| `forward` `brain_runtime` / `brain_wiring` | ~1.8K | **Live (rule-based)** | the actual pipeline: regime + hypothalamus scaling + amygdala filtering — **no model calls** |

**Two clarifications that are easy to misread:**

- `crates/cns` ("central nervous system") is **not ML** — it is the
  health/watchdog/metrics service. The name is a metaphor.
- `crates/logic`'s recent git timestamp is the **risk-engine dead-code deletion**
  (this repo's risk-consolidation work), **not** ML development.

---

## 2. The train→serve gap (why nothing serves)

1. **No export.** `services/backward` trains LSTM/MLP/DQN via `crates/ml` but
   never serializes a model to ONNX or disk.
2. **No artifacts.** Zero `.onnx` / `.safetensors` / `.pt` files exist. The
   `inference.rs` default points at a `models/` dir that isn't present. Only
   checkpoint **metadata** JSON exists under `checkpoints/`.
3. **Feature mismatch.** `crates/vision` consumes raw time-series→images; the
   live loop emits hand-crafted indicators. **Only `crates/ml`'s features align**
   with what `forward` already computes.
4. **Default-off + untested.** Even the wired-up ONNX path is
   `enable_ml_inference: false` and has no production test.

---

## 3. The live path, precisely

`forward`'s "brain" is a rule engine, not a network:

```
indicators (EMA/RSI/MACD)  →  RegimeManager (trending / MR / crisis)
   →  Hypothalamus (ATR + regime → position scale)
   →  Amygdala (drawdown / volatility → threat filter)
   →  affinity / correlation gating
   →  execution decision (scale / block / reduce-only)
```

The ONNX `predict()` step is present in `signal/mod.rs` but lives entirely
inside `if self.config.enable_ml_inference { … }`, which is `false` on every
production construction path.

---

## 4. The four directions

- **A — Own that it's rule-based; cut the dead weight.** Delete/quarantine the
  unwired scaffolding. The prize is large: the **251K-LOC `neuromorphic`
  subsystem compiles into every workspace/CI build while serving nothing** —
  removing it alone meaningfully cuts build/CI time and ends standing confusion.
  *Lowest effort, highest clarity. Cost: writes off the research (recoverable
  from git history).*
- **B — Close the loop minimally on the *aligned* ONNX path.** `backward`
  exports its `crates/ml` model → ONNX → `forward`'s existing `tract` path runs
  it as an **advisory** signal beside the indicators (gated, default-off — the
  same pattern as `JANUS_RISK_ENFORCE`). *Smallest path to "a model influences a
  trade"; reuses `inference.rs` + `param_reload/model_reload.rs`. Moderate
  effort.*
- **C — Commit to vision (DiffGAF → ViViT).** Highest ceiling, but needs a *new*
  raw-series feature pipeline, real training to produce a champion model, and
  latency validation. *Largest, most speculative bet.*
- **D — Neuro-symbolic hedge (LTN as auditor).** Keep the rule-based live path;
  train `crates/ltn` on historical trade labels and run it as a **post-hoc
  validator** of high-confidence signals — zero hot-path latency, purely
  additive. *Leverages the LTN research without betting the live loop on a
  model.*

These are not all mutually exclusive: **A pairs with B or D** (clean house +
one focused bet); **B vs C** is the real "which model serves the live loop"
fork.

---

## 5. Recommendation (non-binding — the call is the owner's)

1. **Do A's `neuromorphic` cleanup regardless of direction.** 251K compiled-but-
   unused LOC is a tax on every build; nothing depends on it. Quarantine it out
   of the default workspace (or delete it) independent of the ML bet.
2. **For the ML bet itself, B is the pragmatic move.** It is the only route
   where the features already align, and it turns existing dormant code
   (`inference.rs`, `model_reload`) into something live with a small, default-off,
   reversible change — instead of building a new pipeline (C) or a new training
   objective (D) from scratch.

---

## 6. How to re-verify

```bash
# ONNX path is gated off by default and the enabling ctor is never called:
grep -rn "enable_ml_inference\|with_ml_inference" services/forward/src/

# No model weights exist:
find . -name '*.onnx' -o -name '*.safetensors' -o -name '*.pt' | grep -v target

# neuromorphic is a compiled workspace member with no service dependency:
grep -n neuromorphic Cargo.toml
grep -rn "neuromorphic" services/*/Cargo.toml   # → (no results)
```
