# ML Phase 1 — Handoff & Probe Runbook

> **Status: the Phase 1 *machinery foundation* is built, tested, and merged
> (#59–#62). The go/no-go *probe* has not been run** — it needs real market
> data + a GPU, which the build environment lacked. This doc is the baton: what
> exists, what's left, and exactly how to finish + run the probe on real infra.
> Companion to [`ML_VISION_SCOPE.md`](ML_VISION_SCOPE.md) (the why/how-big).

---

## What's built (merged, tested, default-safe)

A complete **zeros → real-GAF-features** pipeline. Every step is additive and
CPU/synthetic-tested; nothing changes default behaviour.

| # | Piece | Where |
|---|-------|-------|
| #59 | `gaf_features_from_series` / `_from_closes` — OHLCV window → DiffGAF `(1,F,H,W)` image **and** pooled flat vector | `crates/vision/src/gaf_features.rs` |
| #60 | `experience_from_closes` — build one real-feature training `Experience` | `services/backward/src/tasks/train.rs` |
| #61 | `experiences_from_series` — sliding-window batch of experiences (directional action + next-bar-return reward) | `services/backward/src/tasks/train.rs` |
| #62 | `build_seed_experiences` + opt-in seeding: `TrainingConfig.seed_closes` (default `None`) seeds the replay buffer with real features instead of `[0.0; 9]` zeros | `services/backward/src/tasks/train.rs` |

**Data flow today:** `closes → DiffGAF → GafFeatures{image, flat} → State::from_flat_gaf → Experience → replay buffer` (when `seed_closes` is set).

---

## What's NOT done (the real-infra work)

### A. Empirical decisions (don't bake these in blind — measure them)
1. **Feature representation.** `GafFeatures.flat` is a per-channel spatial
   mean → length `num_features` (≈1 for single-feature closes), padded to the
   LSTM's `model_input_size` (9). **This is thin** — most of the GAF image is
   discarded. Try: grid/adaptive-pool to a `KxK` summary, the GASF diagonal,
   or feed the **full image to ViViT** (which is the point of the vision bet).
2. **Reward / label scheme.** Currently directional action + next-bar return —
   a placeholder. Decide supervised next-move classification vs. the DQN reward.
3. **Which head.** ViViT (full image) vs. LSTM (flat) — train both, compare
   (that was the "both" decision). ViViT has **no weight serialization yet**
   (`crates/vision/src/vivit.rs`) — add save/load before it can be a champion.

### B. Wiring + harness
4. **Real OHLCV source** → `seed_closes`. Use `crates/backtest`'s
   `ohlcv_loader` (CSV/Parquet; Kraken/Binance/TradingView presets) or the
   Ruby data service. (No data files are committed.)
5. **Eval / edge harness.** Measure "beats chance?" — held-out accuracy or
   risk-adjusted return vs. a no-skill baseline. **This is the go/no-go gate.**

---

## How to run the probe (on a data + GPU box)

```bash
# 1. Get real OHLCV (e.g. a Kraken/Binance CSV) into the box.
# 2. Load it and feed seed_closes:
#      let closes = ohlcv_loader::load(path)?.iter().map(|b| b.close as f32).collect();
#      let cfg = TrainingConfig { seed_closes: Some(closes), gaf_window: 32,
#                                 gaf_image_size: 16, ..Default::default() };
# 3. Run backward training to convergence (GPU: enable the burn-cuda/wgpu feature on crates/ml + vision).
# 4. Evaluate on a held-out split; compare to a no-skill baseline.
```

**Decision rule:** if a model trained on *real* GAF features doesn't beat chance
on held-out data → **stop** (Option A in `ML_STORY.md`: the vision bet doesn't
pay off; cut the dead ML weight). If it does → proceed to serving (forward).

---

## Why it wasn't run in-container

No committed market data, **no GPU** (4 CPU cores), and a tight disk (the
candle+burn+vision build cycle repeatedly filled `target/`). The build
environment is fine for the *additive, CPU-testable* machinery above but cannot
train a ViViT to convergence or measure edge — hence the hand-off.

## Re-verify the foundation

```bash
cargo test -p jflow-vision gaf_features        # extractor
cargo test -p janus-backward --lib             # experience builder, batches, seeding (87 tests)
```
