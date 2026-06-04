# Scoping: vision (GAF) → live advisory signal

> **Status: scoping doc, awaiting a commit decision.** This is the "scope it
> before committing" output for Option C from [`ML_STORY.md`](ML_STORY.md) —
> putting the vision (DiffGAF/ViViT) ML pipeline behind a live advisory trading
> signal. Compiled 2026-06 from a read-only survey. **No code has been written.**

---

## TL;DR

- This is a **~4–6 week, largely greenfield ML initiative.** The hard ML pieces
  (DiffGAF, ViViT) are real and tested, but **nothing connects them to data or
  to the live path**, the "training" is a stub that learns on zeros, and there's
  a framework fork to settle first. *The components exist; the system doesn't.*
- **Recommendation: commit to Phase 1 only, as an explicit go/no-go gate** —
  not the whole thing. No model has ever trained on real GAF features, so there
  is **zero evidence the approach has edge.** Phase 1 (~2 weeks) answers "does a
  model trained on real GAF features beat chance?" If not, stop before building
  any serve infrastructure.

---

## What's real vs. not (verified)

| Piece | State | Reality | Evidence |
|---|---|---|---|
| DiffGAF (time-series→image) | **REAL** | Candle; functional — outputs 2D images, **no flat-feature API** | `crates/vision/src/diff_gaf.rs:304` |
| ViViT transformer | **REAL** | Candle; functional — **no trained weights, no serve plumbing** | `crates/vision/src/vivit.rs:516` |
| Training pipeline | **STUB** | backward trains the LSTM on `gaf_features_flat` = **`[0.0; 9]` zeros** | `services/backward/src/tasks/train.rs:120,151` |
| Live data buffer | **PARTIAL** | forward keeps ~60 **close prices**, not the OHLCV windows DiffGAF needs | `services/forward/src/features.rs` |
| LSTM serve | **~70%** | has checkpoint + Redis hot-reload plumbing… but Tract expects ONNX while the model is Burn | `services/forward/src/param_reload/model_reload.rs`, `inference.rs:66` |
| ViViT serve | **~5%** | no export, no adapter, no reload registration | (search: zero ViViT refs in `services/forward`) |

---

## The 5 real gaps (the actual work)

1. **No training data → DiffGAF** (the biggest): make backward compute real GAF
   features from real OHLCV instead of training on zeros. Until this exists,
   *no model predicts anything*.
2. **Undefined feature contract:** what *are* the 9 features? DiffGAF emits 2D
   images, not a 9-vector — so either define an image→9-scalar pooling, **or**
   drop the 9-vector LSTM and serve ViViT on the full images.
3. **Framework fork (decide first):** serving puts **2 ML frameworks in the live
   binary** + retires Tract. The repo is three-way today: **LSTM = Burn**
   (`crates/ml`), **DiffGAF/ViViT = Candle** (`crates/vision/Cargo.toml:45-47`),
   **forward inference = Tract/ONNX**.
4. **Live OHLCV buffer** in forward (currently close-prices only).
5. **Serve wire:** retire the dead Tract-ONNX `inference.rs`; build a Candle
   inference path + advisory fusion (default-off, like `JANUS_RISK_ENFORCE`).

---

## The fork within Option C

| | **C-LSTM** | **C-ViViT** |
|---|---|---|
| Model | Candle DiffGAF → **Burn LSTM** | **all-Candle** DiffGAF → ViViT |
| Input | 9 pooled scalars (throws away most of the image) | full GAF images (the "real" vision bet) |
| Serve plumbing | reuses LSTM checkpoint/reload (~70%) | greenfield (~5%) |
| Frameworks in binary | Burn + Candle | Candle only |

**Key insight for the *probe*:** Phase 1 is **training only** — it serves
nothing — so C-LSTM's serve-plumbing advantage is **irrelevant to the probe**.
For a clean test of "do GAF features have edge," **C-ViViT uses the full image**
and won't give a falsely-negative result caused by pooling to 9 scalars. The
serve-plumbing question can be deferred to Phase 3 (after the probe passes).

---

## Staged plan — with a kill-gate

| Phase | Work | Est. |
|---|---|---|
| **0 — decide** | C-LSTM vs C-ViViT; define the feature contract | — |
| **1 — make training real** ⛔ **GO/NO-GO** | real OHLCV → DiffGAF → features into backward; train to convergence; **measure edge vs. chance** | ~2 wk |
| 2 — live OHLCV buffer in forward | accumulate candles | ~0.5 wk |
| 3 — serve | DiffGAF + model live, advisory, default-off | ~1.5 wk |
| 4 — validate | train/serve feature parity, latency, paper-trade A/B | ~1 wk |

---

## Recommendation

1. **Commit to Phase 1 only, as a go/no-go gate.** It converts a speculative
   multi-week build into a bounded ~2-week feasibility probe, and forces the
   feature-contract + framework decisions up front where they're cheap. **If
   Phase 1 shows no edge, stop.**
2. **Use C-ViViT for the probe** — Phase 1 doesn't serve, so the full-image test
   is the honest one; revisit C-LSTM only if serve latency later demands it.

**Honest caveat:** this re-introduces a second ML framework (Candle) into the
live binary — the opposite of the weight just trimmed by the `neuromorphic`
quarantine. Worth being sure the edge is real before paying that cost.

---

## How to re-verify

```bash
# Training learns on zeros (the showstopper):
sed -n '118,124p;148,154p' services/backward/src/tasks/train.rs

# vision is Candle for the GAF/ViViT pipeline; ml is Burn:
grep -n candle crates/vision/Cargo.toml
grep -n burn crates/ml/Cargo.toml

# forward has no OHLCV window / no GAF / no ViViT:
grep -rn "gaf_features_flat\|DiffGAF\|ViViT" services/forward/src   # → none
```
