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

**✅ Warmup skew FIXED (PR #129).** A single `WARMUP = 110` constant now drives
both `extract_features`' slice depth and `min_rows` (the live `CandleBuffer`
auto-sizes off it), so every returned window column is fully warmed. The new
`bounded_warmup_matches_full_series` regression test asserts bounded-inference
features match full-series training features: fixed-window channels <1e-4, the
four EMA channels <1e-2 (was up to 1.0). **Channels 7–9 were a non-issue** —
the live loop already passes `LiveState::default()` (`lib.rs:1698`), matching
training, so no code change was needed there. Remaining gate before enable is
now just **out-of-sample validation** (below).

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
| 3 | **Fix the feature-parity break** | ✅ **DONE (PR #129)** | `WARMUP=110` warms every channel; regression test proves bounded-inference == full-series training features (<1e-4 fixed, <1e-2 EMA). Channels 7–9 needed no change (live loop already passes `LiveState::default()`). |
| 4 | **Validate out-of-sample** | ✅ **DONE — ROBUST DIRECTIONAL EDGE (#128, #130)** | Leakage-safe purged holdout (`--val-frac`) + anchored **walk-forward** (`--walk-forward N`), GPU-accelerated (`--gpu`, #130). 3-day data → fails (acc 0.369 < 0.407 baseline). **130k-bar / 90-day data, 5-fold walk-forward: long precision beats base rate in 5/5 folds, short in 5/5 (mean ~2× base)** — consistent across regimes, not one lucky split. Accuracy doesn't beat the majority-flat baseline (expected for a minority-class signal). The data hypothesis was right. |
| 5 | **PnL backtest** | ⛔ **DONE — NOT TRADABLE AS-IS (#130)** | Walk-forward event-driven backtest (`--backtest`, enter at signal bar, TP 1.5/SL 1.0 ATR bracket, intrabar exits, per-side cost). At 6 bps/side: **net −12.7 over 10.7k trades, 0/5 folds profitable, Sharpe −1.56.** At 0 bps the gross signal is only barely positive (41% win, Sharpe +0.006). Root cause: on 1m BTC, ATR ≈ 5 bps, so a 1.5-ATR target ≈ 7.5 bps < the 12-bps round-trip cost — the per-trade edge (~+0.1 bps) is ~100× smaller than costs. **The robust classification edge does NOT survive transaction costs.** Do not enable. |
| 6 | **Ship the champion** | READY | Mount the (validated, all-data-retrained) `.bin` into the janus container; point `CNN_CHECKPOINT_PATH` at it. |
| 7 | **Flip the env** | READY | janus service in `fks/docker-compose.yml`: `ENABLE_CNN_INFERENCE=true` (+ optional `CNN_CHECKPOINT_PATH`, `CNN_CONFIDENCE_THRESHOLD`). On boot, `is_active()` flips true and votes enter consensus. |
| 8 | **Observe in paper** | READY / safe | With `ENABLE_EXECUTION=false` + paper account, zero live-order blast radius. Pair CNN with a rule strategy (needs `min_strategies=2`), watch `source=per_asset_cnn` in published signals. |

**Summary (updated after walk-forward).** All the training/validation *code* is
done: minting binary (#128), feature-parity fix + regression guard (#129),
leakage-safe holdout + **anchored walk-forward** validation (#128, #130), and a
**GPU backend** (#130) that makes multi-fold sweeps fast. The experiment answered
the paper's core question empirically: on 3 days of data the champion fails
(acc 0.369 < 0.407), but on **~90 days (130k bars), 5-fold walk-forward shows a
robust directional edge — long AND short precision beat their base rate in 5/5
folds (~2× base), across distinct time windows.** The data hypothesis was right:
the CNN+breakout-label approach carries real, regime-consistent directional
signal once it has adequate history.

**But the PnL backtest (step 5) settled it: the edge is real but NOT tradable
as-is.** At realistic 6-bps/side costs the signal loses decisively (net −12.7,
0/5 folds profitable); even at zero cost the gross edge is noise-level
(Sharpe +0.006). On 1m BTC the ATR-scaled profit target (~7.5 bps) is smaller
than the round-trip cost (12 bps), so the ~+0.1 bps/trade edge is ~100× too
small to overcome costs. **The enable gate stays shut — correctly and, now,
conclusively.**

**Honest next directions (open research, NOT queued work):** the labels are
forward-looking breakout outcomes, but the backtest enters at the signal bar —
a breakout-confirmation entry (as the labeler defines) might capture more of the
move; lower frequency / wider brackets so the target exceeds costs; or accept
the per-bar signal is simply too weak (gross Sharpe ≈ 0) and this asset/label/
timeframe combination isn't the one. Steps 6–8 (ship/enable/observe) remain
gated behind a tradable result that does not yet exist. **Nothing is enabled,
which is the right outcome.** The value delivered is the rigorous, reusable
rig — GPU trainer, purged walk-forward, cost-aware backtest — and a decisive,
honest answer reached *before* any money was at risk.

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
