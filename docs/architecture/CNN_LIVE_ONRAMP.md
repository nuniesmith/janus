# CNN Live On-Ramp — First Neural Inference in the Live Signal Loop

> **Status (2026-07-03):** the live wiring is **fully merged and gate-protected**.
> The only missing ingredient is a trained champion `.bin`. This document is the
> de-risked plan to mint one and turn the path on in paper mode.

## Why the CNN vote, not the LSTM fusion

Two neural paths exist in `forward`. They are **not** equivalent on-ramps:

| | LSTM fusion (`generate_from_analysis`) | **Per-asset CNN vote** |
|---|---|---|
| Enters the live loop? | **No** — request-driven side path only; the live loop uses `resolve_consensus` and never calls it | **Yes** — pushed into the same `strategy_votes` the live loop tallies (`lib.rs:1709`) |
| Config wired? | No — `enable_ml_inference` is dead config; `with_ml_inference` has zero prod callers | Yes — `CnnInference::from_env` init'd in the live loop (`lib.rs:1402`) |
| Feature contract | **Broken** — trainer emits 9 GAF features, forward extracts ~20+; input-size check guarantees fallback | Same 20-feature × 60-window spec and same code — **but a warmup-context skew makes the *values* diverge at inference (see ⚠️ below)** |
| Gate inheritance | Would need new routing | Automatic — it's just one vote among the rule strategies |

> **⚠️ Correction (2026-07-03, adversarial verification).** An earlier read of
> this path claimed the feature contract was fully consistent. That is true for
> feature *count, order, and window* (so a champion loads and runs) — but an
> empirical probe found the feature *values* diverge between training and live
> inference. This is a real blocker; see **"Parity break — must fix before
> enable"** below. It does not invalidate the champion-minting binary (it
> correctly mints + roundtrips an artifact), only the readiness to enable.

The CNN vote is the lowest-risk path because a `"per_asset_cnn"` vote is
**structurally indistinguishable** from a rule vote once pushed — it has no
privileged path and inherits every existing control.

## What is already READY (merged, tested, no work needed)

- **Live init:** `services/forward/src/lib.rs:1402` — `CnnInference::from_env()`,
  per-symbol rolling `CandleBuffer` map (`lib.rs:1403-1404`). Runs only under
  `DATA_SOURCE=live` (`lib.rs:1329`, already the compose default).
- **Per-candle vote:** `lib.rs:1687-1711` — gated by `cnn.is_active()`, which is
  true only when `ENABLE_CNN_INFERENCE=true` **and** a champion loads
  (`cnn_inference.rs:120-122`). Pushes the vote into `strategy_votes` tagged
  `"per_asset_cnn"` (`lib.rs:1709`) and feeds the execution-gate context
  (`cnn_gate_vote`, `lib.rs:1700-1708`, `2115`).
- **Model + inference:** `CnnInference` wraps `janus_ml::models::PerAssetCnn<CpuBackend>`
  (Burn-native: Conv1d + BatchNorm + squeeze-excitation → 4-class softmax).
  Input `[1, 20, 60]`, argmax → Buy(1)/Sell(2)/None(0,3), suppressed below
  `CNN_CONFIDENCE_THRESHOLD` (default 0.5). Warm-up needs 85 bars. Well
  unit-tested (`cnn_inference.rs:261-381`, roundtrip/parity `per_asset_cnn.rs:787-918`).
- **Champion-absent safety:** a load failure logs `warn!` and disables the vote —
  never panics, never blocks the rule path (`cnn_inference.rs:105-111`, test at
  `:370-380`).
- **Training *algorithm*:** `crates/ml/src/train_per_asset.rs:103` `train_champion(...)`
  — precompute features → breakout labels → `CnnTrainer` (AdamW, weighted CE,
  LR warmup, early stopping) → savable `PerAssetCnn`. End-to-end tested
  (`train_per_asset.rs:231-264`).

## The gate chain a CNN vote inherits (blast radius: bounded)

`strategy_votes` → `resolve_consensus` (`lib.rs:1935`, needs **≥2 strategies**,
**≥0.6 agreement** — a lone CNN vote cannot trade, it must agree with a rule
strategy) → `min_confidence ≥ 0.6` (`lib.rs:1961`) → regime tag → prop-firm
(`JANUS_PROP_FIRM_ENFORCE`) → risk manager (`JANUS_RISK_ENFORCE`, default on) →
unified execution gate (`JANUS_GATE_ENFORCE`) → publish to bus → **hard submit
floor `avg_confidence ≥ 0.7`** (`lib.rs:2178`) → kill-switch choke point
(`signal/mod.rs`). In fks the janus service defaults to `ENABLE_EXECUTION=false`
+ `EXEC_ACCOUNT_TYPE=paper`, so the CNN can at most alter *published/paper*
signals — it cannot bypass any control or place a live order.

## ⚠️ Parity break — must fix before enable (found by adversarial verification)

Both feature paths call the **same** implementation
(`crates/ml/src/features/per_asset_cnn.rs`) with the same channel set, order,
window, and normalization — so count/order/shape are guaranteed identical and a
champion always loads. **But the input *length* differs, and several channels
are history-dependent, so the feature values diverge:**

- **Warmup-context skew.** Training computes channels over the **full series**
  then slices 60-bar windows with ~94 bars of preceding context
  (`train_per_asset.rs` → `make_windows`, `per_asset_dataset.rs`). Live
  inference **hard-slices to the last `window + 25 = 85` bars**
  (`services/forward/src/features/per_asset_cnn.rs` `extract_features`),
  discarding earlier history even though the `CandleBuffer` holds ~149 bars.
  History-dependent channels then differ for the same terminal bar. Measured on
  a synthetic series (isolating warmup): `hurst` maxdiff **0.50** (16/60 cols
  fall back to the 0.5 default at inference), `market_phase` maxdiff **1.00**
  (29/60 cols are the 0.0 default at inference vs real values in training),
  plus RSI/`norm_velocity`/`price_acceleration`/`norm_close` skew. Those
  channels are out-of-distribution at inference.
- **Live-scalar channels 7–9 skew (acknowledged in-code).** `train_champion`
  passes `LiveState::default()`, so `book_imbalance`/`wave_ratio`/`vol_percentile`
  are trained as **constants**; live, `cnn_inference.rs` feeds the real
  live-varying values into those same channels.

**Fix before enable:** reconcile the warmup contract so both paths produce
identical features for the same terminal bar (either extend `extract_features`
to compute over the full available buffer, or build training samples through
the same limited-warmup tail), **add a golden regression test** asserting
`training-window features == inference features`, and either feed real
`LiveState` at training time or drop channels 7–9. Re-run the divergence check
(require ~0 maxdiff across all 20 channels) before flipping the env.

## The mint blocker — RESOLVED

`train_champion` existed and was tested, **but nothing called it and no champion
`.bin` existed** (the golden dirs hold only READMEs; backward trains an LSTM/DQN,
not a CNN). This is now fixed: `crates/ml/src/bin/train_cnn_champion.rs` reads an
OHLCV CSV, calls `train_champion`, saves the checkpoint, and reloads it through
`PerAssetCnn::load` (the live path) to prove the artifact is valid. A first
champion was minted on 4,521 BTCUSDT 1m bars. **Minting works; enabling is
gated on the parity fix + validation above.**

## Minimal path (ordered)

| # | Step | Class | Detail |
|---|------|-------|--------|
| 1 | **Champion-minting binary** | **NEEDS-CODE** | Add a `crates/ml` `[[bin]]` (e.g. `train-cnn-champion`): load real OHLCV (QuestDB `candles_crypto` is already persisted, `fks/docker-compose.yml:345`), call `train_champion(...)` with `n_features=20, window=60`, then `model.save("models/per_asset_cnn.bin")`. |
| 2 | **Run it** | **NEEDS-TRAINING** | Mint the `.bin` on a real per-asset series; verify with the existing roundtrip/parity tests. |
| 3 | **Fix the feature-parity break** | **NEEDS-CODE** | Reconcile the train/inference warmup contract + fix channels 7–9, and add a golden test asserting identical features (see ⚠️ section). Re-mint the champion after. **This is the true gate to enabling** — without it the model is served out-of-distribution features. |
| 4 | **Validate out-of-sample** | **NEEDS-TRAINING** | Retrain with a train/validation/(walk-forward) holdout and report validation loss + per-class precision/recall on the actionable long/short classes. The first champion's `best_loss=0.09` is **in-sample training loss** (no split) on ~dozens of effectively-independent breakout episodes — overfit, not a quality signal. |
| 5 | **Ship the champion** | READY | Mount the re-minted `.bin` into the janus container; point `CNN_CHECKPOINT_PATH` at it. |
| 6 | **Flip the env** | READY | janus service in `fks/docker-compose.yml`: `ENABLE_CNN_INFERENCE=true` (+ optional `CNN_CHECKPOINT_PATH`, `CNN_CONFIDENCE_THRESHOLD`). On boot, `is_active()` flips true and votes enter consensus. |
| 7 | **Observe in paper** | READY / safe | With `ENABLE_EXECUTION=false` + paper account, zero live-order blast radius. Pair CNN with a rule strategy (needs `min_strategies=2`), watch `source=per_asset_cnn` in published signals. |

**Summary (corrected after adversarial verification):** the live wiring, gates,
model machinery, and the champion-minting binary are **READY** — steps 1–2 are
done (the binary exists and mints + roundtrips a real artifact). But enabling is
**not** env-only: step 3 (feature-parity fix + golden test) is a genuine
NEEDS-CODE blocker, and step 4 (out-of-sample validation) is required before the
model's votes should be trusted even in paper. The first champion is a
**pipeline proof, not a validated model.** Steps 5–7 remain env-only and
reversible once 3–4 are done.

## Decision points before step 1

- **Which asset(s)** to train the first champion on (one model applies to every
  symbol — there are no per-asset toggles). BTC/USDT has the most history.
- **How much history** to pull from QuestDB, and the label horizon
  (`generate_labels_breakout`).
- **Confidence threshold** — default 0.5; raise to reduce vote frequency during
  the trial.

_This plan is grounded in a full read of the live loop, the CNN module, the
training crate, and the fks compose. See also `PAPER_CODE_ALIGNMENT.md` and the
`janus-ml-loop-gap` note._
