# JANUS Consolidation Plan — consume the shared crates

> **Goal:** make janus depend on the published, standalone crates instead of
> reimplementing them internally — `indicators-ta` for TA math and
> `exchange-apiws` for exchange REST/WS — and retire the duplicates
> (`jflow-indicators`, `jflow-exchanges`, `jflow-bybit-client`).
>
> **Status:** scoping only — no code moved yet. This doc is the plan to review
> before any churn.
> **Last updated:** 2026-05-31

---

## TL;DR

| Leg | Target crate | Verdict | Effort |
|-----|--------------|---------|--------|
| **TA** | [`indicators-ta`](https://crates.io/crates/indicators-ta) `0.1.5` | ✅ **DONE** — `jflow-indicators` retired; janus consumes published `indicators-ta` (superset). Cost the workspace a polars 0.44→0.53 bump. | shipped |
| **Bybit client** | [`exchange-apiws`](https://crates.io/crates/exchange-apiws) `0.4.0` | ✅ **DONE** — `jflow-bybit-client` retired; forward consumes 0.4.0's signed Bybit via a thin `bybit_compat` adapter (REST order entry + WS feed) | shipped |
| **Exchange ingestion** | `exchange-apiws` `0.5.0` | ✅ **DONE** — `services/data` bridges now wrap the published Coinbase/Kraken/OKX connectors; the duplicate `jflow-exchanges` adapters are out of the data path. CNS metrics + health stay janus-side. | shipped |
| **Framework** | `rustrade` | ✋ **reframed — not janus's job** (lives in fks-full `bots/`; `JanusBrain` already ties them together) | n/a |

> **Phase 1 shipped** (branch `claude/phase1-indicators-ta`): the workspace
> moved to **polars 0.53** to match `indicators-ta`, which required 0.53 API
> fixes (`with_column(col.into_column())`, `DataFrame::new_infer_height`,
> `LazyFrame::scan_parquet(PlRefPath)`). `cargo check --workspace` clean;
> jflow-strategies (58) + jflow-backtest (310) tests pass.
>
> **Phase 2 shipped** (branch `claude/phase2-bybit`): exchange-apiws 0.4.0 added
> the signed Bybit surface, so `jflow-bybit-client` is retired. A
> `services/forward/src/bybit_compat.rs` adapter re-exposes the old API
> (`BybitTick`/`WsMessage`/`BybitRestClient`/`OrderRequest`/…) on top of
> `BybitPrivateClient` + `BybitConnector`/`run_feed`, so the event loop is
> untouched. `cargo test -p janus-forward` → 540 passing.
>
> **Phase 3 shipped** (branch `claude/phase3-exchanges`): exchange-apiws 0.5.0
> added the Coinbase + OKX connectors, so `services/data`'s bridges now wrap the
> published Coinbase/Kraken/OKX connectors instead of `jflow-exchanges`'s
> adapters. CNS metrics + health monitoring remain janus-side (the only
> `jflow-exchanges` surface still imported). `cargo build -p janus-data` clean;
> 170 lib tests pass (2 pre-existing backfill-throttle failures are
> environment-sensitive, unrelated to this change).
>
> **Consolidation complete** — all four legs (TA, Bybit, ingestion; framework
> reframed to the `bots/` layer) are done. A clean follow-up remains: the now-
> dead adapter code (~2.6k LOC) inside `jflow-exchanges/src/adapters/` can be
> deleted, leaving that crate as a CNS/health/normalizer utility.
>
> ---
> _Historical (pre-0.5.0 gating notes):_
> `exchange-apiws 0.3.2` — see the table. They need new surface *in
> exchange-apiws* (Bybit signing/orders; Coinbase + OKX connectors) before
> janus can retire `jflow-bybit-client` / `jflow-exchanges`.

> **On `rustrade`:** janus is a ~50-crate multi-service ML engine with its own
> lifecycle (it dropped `rustrade-supervisor` at extraction). Forcing it into
> `rustrade::Bot`/`Brain` is a poor fit. `rustrade` belongs in the **`bots/`
> layer** (in `fks-full`) that *consumes* janus's signals — not inside janus.
> A thin `rustrade::Brain` adapter that calls janus's brain REST API can live
> in `bots/` so a rustrade bot can use janus as its brain. Tracked there, not
> here.

---

## Overlap analysis

### TA — `jflow-indicators` ⊂ `indicators-ta`

`crates/indicators` (package `jflow-indicators`) is a single `lib.rs` exporting:

```
ema, sma, atr, rsi, macd, true_range          (batch fns)
EMA, ATR, IncrementalEma, IncrementalAtr       (incremental structs)
IndicatorCalculator, StrategyIndicators         (aggregators)
IndicatorError { InsufficientData, InvalidParameter }
```

`indicators-ta` (lib name `indicators`) **re-exports every one of those names**
from its `functions` module and adds a large superset: `momentum`, `trend`,
`volatility`, `volume`, an 11-layer `signal` engine, and `regime` detection.
Its `IndicatorError` is the same enum + one extra variant (`UnknownIndicator`).

**Seam:** `indicators-ta` does **not** re-export `IndicatorError` at the crate
root (it lives at `indicators::error::IndicatorError`). Either land a one-line
`pub use error::IndicatorError;` in `indicators-ta` (→ 0.1.4) or have janus
import the fully-qualified path. Everything else maps 1:1.

### Exchanges — `exchange-apiws` has gaps janus fills today

`exchange-apiws` (published `0.1.10`) ships clients for KuCoin, Binance, Bybit,
Kraken, Crypto.com. Private/signed surface exists for **KuCoin, Kraken,
Crypto.com**; **Binance and Bybit are public-only**.

| janus internal | What it is | `exchange-apiws` equivalent | Gap |
|---|---|---|---|
| `jflow-bybit-client` | full Bybit v5: REST + public/private WS + HMAC signing + **order entry** | `BybitConnector` + `BybitRestClient` — **public only** | Bybit **private REST + WS + orders** missing in `exchange-apiws` |
| `jflow-exchanges` | janus-coupled **ingestion** layer: adapters for **Kraken / Coinbase / OKX** + `CNSReporter` + `PriceNormalizer` | Kraken (richer) | **Coinbase + OKX** missing in `exchange-apiws`; CNS/normalizer are janus-specific and stay |

So janus's Bybit client is actually **ahead** of `exchange-apiws`, and its
ingestion layer needs `exchange-apiws` to grow Coinbase/OKX before it can fully
retire `jflow-exchanges`.

---

## Blast radius (who depends on what)

| Internal crate | Dependent | Intensity | Symbols used |
|---|---|---|---|
| `jflow-indicators` | `services/forward` (`indicators/mod.rs` + `event_loop.rs`) | **High** | `ATR, EMA, ema, macd, rsi, sma, IndicatorCalculator` |
| | `crates/strategies` | Low | `pub use janus_indicators as indicators;` (pass-through) |
| | `crates/backtest` | Low | `IncrementalEma` (tests) |
| | `services/optimizer` | **Dead** | declared, never imported |
| `jflow-exchanges` | `services/data` (`connectors/{mod,bridge}.rs`) | Medium | `adapters::{CoinbaseAdapter, KrakenAdapter, OkxAdapter}`, `CNSReporter`, `HealthChecker` |
| | `services/optimizer` | **Dead** | declared (`features=["kraken"]`), never imported |
| `jflow-bybit-client` | `services/forward` (`event_loop.rs`) | Medium | `BybitCredentials, BybitRestClient, BybitWebSocket, OrderRequest, OrderSide, OrderType, WsMessage` |

`jflow-exchanges` and `jflow-bybit-client` are **decoupled** (no dep between
them) → each leg migrates independently. The `forward` `indicators/mod.rs` is
630 lines of janus's *own* higher-level logic that merely consumes 6 base
primitives — so even the "high" consumer is an import swap, not a rewrite.

---

## Staged plan

### Phase 0 — Quick wins (no behaviour change) · risk: none
- [ ] Delete the **dead** `janus-indicators` + `janus-exchanges` deps from
      `services/optimizer/Cargo.toml` (verified unused — 0 imports).
- [ ] Land `pub use error::IndicatorError;` in `indicators-ta` (→ 0.1.4) so the
      import seam is clean. _(Or skip and fully-qualify in janus.)_

### Phase 1 — TA → `indicators-ta` · risk: low
Every symbol exists in `indicators-ta 0.1.3` under the same name; this is
purely dependency + import-path swaps.
- [ ] Add `indicators-ta = "0.1"` to root `[workspace.dependencies]`.
- [ ] `crates/strategies`: `pub use janus_indicators as indicators;` →
      `pub use indicators;` (lib name is already `indicators`).
- [ ] `crates/backtest`: swap the `IncrementalEma` import to `indicators::`.
- [ ] `services/forward`: change the two import lines
      (`use janus_indicators::{ATR, EMA, ema, macd, rsi, sma};` and
      `use janus_indicators::IndicatorCalculator;`) to `use indicators::…`.
      The 630-line local logic is untouched.
- [ ] Fix any `IndicatorError` path (per Phase 0 seam).
- [ ] Delete `crates/indicators`; drop it from `members` + the
      `janus-indicators` `[workspace.dependencies]` entry.
- [ ] Verify: `cargo check/test -p janus-forward -p jflow-strategies -p jflow-backtest`.

### Phase 2 — Bybit → `exchange-apiws` · risk: medium · **gated**
> Order-execution critical path. Gated on `exchange-apiws` gaining a signed
> Bybit surface (its own TODO B2 / C3 / C5).
- [ ] **In `exchange-apiws`:** port `jflow-bybit-client`'s HMAC signing,
      private WS, and order entry into the Bybit module (private REST + WS +
      `WsOrderClient`). Release (≥ 0.2 / 0.3).
- [ ] **In janus `services/forward`:** swap `BybitRestClient` / `BybitWebSocket`
      / `OrderRequest` / `OrderSide` / `OrderType` → `exchange-apiws` Bybit
      types; map `BybitCredentials` → `exchange-apiws` credentials.
- [ ] Delete `crates/bybit-client`; drop from workspace.
- [ ] Verify against a Bybit **testnet** before trusting order paths.

### Phase 3 — Exchange ingestion → `exchange-apiws` · risk: medium-high · **gated**
> `services/data`'s `bridge.rs` already adapts exchange events into janus's
> `DataMessage` + CNS metrics. Migration = make the bridge wrap `exchange-apiws`
> connectors instead of `jflow-exchanges` adapters; the CNS reporter +
> `PriceNormalizer` stay janus-side.
- [ ] **Decide (open question):** add **Coinbase + OKX** connectors to
      `exchange-apiws` (its TODO H1/H2) — _recommended, keeps one client layer_
      — **or** keep those two adapters in a slimmed janus crate and migrate
      only Kraken.
- [ ] Rewrite `services/data/src/connectors/bridge.rs` to consume
      `exchange-apiws` connectors (Kraken now; Coinbase/OKX once available).
- [ ] Keep `CNSReporter` + `HealthChecker` + `PriceNormalizer` as the janus
      ingestion layer on top.
- [ ] Delete `crates/exchanges` once all three venues are covered.
- [ ] Verify: data-completeness + health checks against live feeds.

### Phase 4 — Framework, in `bots/` not janus (optional, separate repo)
- [ ] In `fks-full` `bots/`: a `rustrade::Brain` adapter that calls janus's
      brain REST API, so a `rustrade::Bot` can use janus as its brain. Pure
      addition; no janus change.

---

## Sequencing & gating

```
Phase 0 (now) ──► Phase 1 TA (now, against indicators-ta 0.1.x) ──► delete jflow-indicators
                                                                    
Phase 2 Bybit  ── gated on ─► exchange-apiws signed-Bybit release ─► delete jflow-bybit-client
Phase 3 Ingest ── gated on ─► exchange-apiws Coinbase/OKX (or scope to Kraken) ─► delete jflow-exchanges
Phase 4 bots   ── independent, lives in fks-full
```

Phase 1 is unblocked and self-contained — **start there.** Phases 2 and 3 each
depend on shipping new surface in `exchange-apiws` first, so they pace with that
repo's roadmap (and a fresh `exchange-apiws` publish — note the published
`0.1.10` vs local `0.3.2` skew to reconcile).

## Open decisions
1. **Coinbase/OKX:** grow `exchange-apiws` to cover them (one client layer, more
   work in that repo) vs. keep a thin janus exchange crate for just those two.
   _Recommendation: add to `exchange-apiws`._
2. **`IndicatorError` seam:** re-export in `indicators-ta` (0.1.4) vs. fully
   qualify in janus. _Recommendation: re-export — it's one line and helps every
   consumer._
3. **`exchange-apiws` versioning:** reconcile `0.1.10` (crates.io) vs `0.3.2`
   (local) and publish before Phases 2/3 depend on the newer surface.
