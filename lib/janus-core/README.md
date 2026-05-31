# janus-core

Shared types, configuration, and application state for the
[JANUS](https://github.com/nuniesmith/janus) trading engine.

This is the foundational crate the rest of the JANUS workspace builds on. It
deliberately depends on no other workspace crate, so it can be reused
standalone.

## What's here

- **`Config`** — unified TOML + environment configuration with a Redis overlay.
- **`JanusState`** — shared application state (signal/market-data buses, service
  lifecycle, runtime log-level control, regime/threat snapshots, and a
  pluggable affinity recorder).
- **Signals** — `Signal`, `SignalBus`, `SignalType`.
- **Market data** — `MarketDataEvent` and friends (`TradeEvent`, `KlineEvent`,
  `OrderBookEvent`, …) plus the `MarketDataBus`.
- **Position guidance** — `PositionEvent`, `PositionClose`, `compute_guidance`,
  `PositionTracker` (stateful trailing/give-back + sticky-exit), and
  `PositionOutcome` for closed-trade feedback.
- **Optimized params** — `OptimizedParams` / `ParamManager` for hot-reloadable,
  per-asset strategy parameters.
- **Supervisor**, **metrics**, **session metrics**, and shared `Error`/`Result`.

## Features

- `redis` *(default)* — enables the Redis-backed config overlay and
  param-update subscription. Disable with `--no-default-features` for a
  build with no Redis dependency.

## Usage

```toml
[dependencies]
janus-core = "0.1"
```

```rust
use janus_core::{Config, JanusState};

# async fn run() -> janus_core::Result<()> {
let config = Config::load()?;
let state = JanusState::new(config).await?;
# Ok(())
# }
```

## License

MIT
