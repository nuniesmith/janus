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
| Feature contract | **Broken** — trainer emits 9 GAF features, forward extracts ~20+; input-size check guarantees fallback | Consistent — trainer and inference share the same 20-feature × 60-window spec |
| Gate inheritance | Would need new routing | Automatic — it's just one vote among the rule strategies |

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

## The ONE blocker

`train_champion` exists and is tested, **but nothing calls it and no champion
`.bin` exists anywhere** (the golden dirs hold only READMEs; backward trains an
LSTM/DQN, not a CNN; the `crates/ml` `[[bin]]` slots are stubbed/commented at
`Cargo.toml:82,86`). There is no way to produce a champion today without adding
a training-driver entrypoint.

## Minimal path (ordered)

| # | Step | Class | Detail |
|---|------|-------|--------|
| 1 | **Champion-minting binary** | **NEEDS-CODE** | Add a `crates/ml` `[[bin]]` (e.g. `train-cnn-champion`): load real OHLCV (QuestDB `candles_crypto` is already persisted, `fks/docker-compose.yml:345`), call `train_champion(...)` with `n_features=20, window=60`, then `model.save("models/per_asset_cnn.bin")`. |
| 2 | **Run it** | **NEEDS-TRAINING** | Mint the `.bin` on a real per-asset series; verify with the existing roundtrip/parity tests. |
| 3 | **Ship the champion** | READY | Mount the `.bin` into the janus container; point `CNN_CHECKPOINT_PATH` at it. |
| 4 | **Flip the env** | READY | janus service in `fks/docker-compose.yml`: `ENABLE_CNN_INFERENCE=true` (+ optional `CNN_CHECKPOINT_PATH`, `CNN_CONFIDENCE_THRESHOLD`). On boot, `is_active()` flips true and votes enter consensus. |
| 5 | **Observe in paper** | READY / safe | With `ENABLE_EXECUTION=false` + paper account, zero live-order blast radius. Pair CNN with a rule strategy (needs `min_strategies=2`), watch `source=per_asset_cnn` in published signals. |

**Summary:** wiring, model, features, gates, and the training algorithm are all
**READY**. Steps 1–2 (a champion-minting entrypoint + one training run) are the
entire gap between "off" and a first live paper CNN vote. Everything after is
env-only and reversible.

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
