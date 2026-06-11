# janus — Public API & Configuration Reference

> **Status:** living reference, last synced 2026-06-08.
> **Scope:** the operator-facing **HTTP / WebSocket** surface and the
> **environment variables** that configure the running services. It is a
> *path / method / purpose* map plus a config reference — for exact request /
> response JSON shapes, follow the linked handler source (paths are clickable in
> most editors). gRPC/proto contracts (`proto/fks/…`), example binaries, and the
> fks-full `bots/*` are **out of scope** here.
>
> This is the janus half of the **Track D** infra contract (see
> [`TODO.md`](../TODO.md) and `fks-full/docs/architecture/RUST_MIGRATION.md`
> §12-C): the routes fks-full's nginx + SvelteKit WebUI repoint to once they
> stop pointing at the retired `fks_ruby`.

---

## Services at a glance

| Service | Crate | Role | Default port(s) |
|---------|-------|------|-----------------|
| **Brain API** | `lib/janus-api` | unified dashboard / signals / positions / service control | `JANUS_HTTP_PORT`; metrics on `JANUS_METRICS_PORT` |
| **Data factory** | `services/data` | market-data ingestion, gaps, indicators, signals, WS streams | `8080` (HTTP+WS) |
| **Forward** | `services/forward` | live signal generation, brain runtime, risk REST, WS | service HTTP + `/ws` |
| **Execution** | `services/execution` | order management (gated; paper by default) | `HTTP_PORT` / `GRPC_PORT` |
| **Backward** | `services/backward` | persistence / analytics | health + metrics only |

**Every** service exposes `GET /health` and `GET /metrics` (Prometheus). The
no-autonomous-execution invariant holds throughout: order flow terminates at a
human-confirmation point and `EXECUTION_MODE=paper_trading` is the default.

---

## HTTP / WebSocket surface

### Brain API — `lib/janus-api`

Source: [`lib/janus-api/src/lib.rs`](../lib/janus-api/src/lib.rs). The main
dashboard/control API (and the closest successor to Ruby's API surface).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | root / banner |
| GET | `/health`, `/status` | liveness + service status |
| GET | `/api/dashboard/overview` | dashboard summary |
| GET | `/api/dashboard/performance` | portfolio performance |
| GET | `/api/dashboard/signals/summary` | signal summary for the dashboard |
| GET | `/api/signals/latest` | most recent signals |
| POST | `/api/signals/publish` | publish a signal to the bus |
| GET | `/api/signals/summary` · `/api/signals/categories` | aggregates |
| GET | `/api/signals/generate` | trigger a generation pass |
| GET | `/api/signals/by-id/{signal_id}` | one signal |
| GET | `/api/signals/by-symbol/{symbol}` | signals for a symbol |
| GET | `/api/modules/health` | per-module health |
| GET | `/api/services/status` | service registry status |
| POST | `/api/services/start` · `/api/services/stop` | service lifecycle |
| GET·POST | `/api/log-level` | read / set runtime log level |
| POST | `/api/v1/positions/event` | **position guidance** — advisory `hold`/`reduce`/`exit` (JFLOW-C) |
| POST | `/api/v1/positions/close` | finalize a position; returns/persists the realized outcome + feeds the gate breaker & affinity |
| GET | `/bars/{symbol}` | **WebUI chart history** (columnar `{columns, data}`; `?interval=&days_back=&limit=`) from `candles_crypto` |
| GET | `/bars/{symbol}/candles` | chart history, flat shape (`{candles: [{timestamp(ms), o,h,l,c,v}]}`) |
| GET (SSE) | `/sse/bars/{symbol}` | live closed-candle stream (`event: bar`; `?interval=`, default `1m`) |

Bars read QuestDB over `QUESTDB_HTTP_URL`; symbol matching is
separator-insensitive (`BTCUSDT` ≡ `BTC-USDT`).

Metrics router (separate, `JANUS_METRICS_PORT`): `GET /metrics`, `GET /health`.

### Data factory — `services/data` (`:8080`)

Source: [`services/data/src/api/mod.rs`](../services/data/src/api/mod.rs).
Protected routes sit behind optional JWT auth (`auth_config`); WS routes are
unauthenticated.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health`, `/metrics` | public liveness + Prometheus |
| GET | `/api/v1/gaps` | detected data gaps |
| GET | `/api/v1/metrics` | data metrics (JSON) |
| GET | `/api/v1/indicators` · `/api/v1/indicators/info` | list + descriptors |
| GET | `/api/v1/indicators/{symbol}/{timeframe}` | computed indicators |
| GET | `/api/v1/indicators/{symbol}/{timeframe}/status` | warmup status |
| POST | `/api/v1/indicators/warmup` · `/warmup/deep` | trigger warmup |
| GET | `/api/v1/signals` · `/signals/stats` | signals + stats |
| GET | `/api/v1/signals/{symbol}` · `/{symbol}/{timeframe}` | filtered signals |
| POST | `/api/v1/signals/backtest` | run a signal backtest |
| WS | `/ws/stream` | live normalized market-data stream |
| WS | `/ws/signals` | live signal stream |

### Forward — `services/forward`

Brain runtime + risk. Sources:
[`api/brain_rest.rs`](../services/forward/src/api/brain_rest.rs),
[`api/risk_rest.rs`](../services/forward/src/api/risk_rest.rs),
[`api/server.rs`](../services/forward/src/api/server.rs).

**Brain** (`brain_rest`, mounted only when the brain runtime is enabled):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/brain/health` | brain runtime health |
| GET | `/api/v1/brain/pipeline` | pipeline state |
| GET | `/api/v1/brain/affinity` | export strategy-affinity weights |
| POST | `/api/v1/brain/affinity/record` | record an affinity outcome |
| POST | `/api/v1/brain/kill-switch/activate` · `/deactivate` | 🔒 token-gated kill switch |
| POST | `/api/v1/brain/affinity/reset` | 🔒 token-gated reset |

**Risk** (`risk_rest`):

| Method | Path | Purpose |
|--------|------|---------|
| GET·PUT | `/api/v1/risk/config` | read / update risk config |
| GET | `/api/v1/risk/portfolio` | live portfolio state |
| POST | `/api/v1/risk/portfolio/positions` · `/positions/close` | add / close a position |
| DELETE | `/api/v1/risk/portfolio/positions/{symbol}` | remove a position |
| GET | `/api/v1/risk/metrics` · `/risk/performance` | risk metrics + performance |
| POST | `/api/v1/risk/validate` | validate a prospective signal |
| POST | `/api/v1/risk/calculate/position-size` · `/stop-loss` · `/take-profit` | sizing helpers |

**Signals / meta** (`server`): `POST /api/v1/signals/generate`,
`POST /api/v1/signals/batch`, `GET /api/v1/health`, `GET /api/v1/version`,
`GET /api/v1/metrics`, and `GET /api/v1/account` (read-only, only when a live
account feed is wired). WebSocket: `GET /ws`
([`websocket/server.rs`](../services/forward/src/websocket/server.rs)).

### Execution — `services/execution`

Source: [`services/execution/src/api/http.rs`](../services/execution/src/api/http.rs).
Order flow is gated; **paper by default** (`EXECUTION_MODE`).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health`, `/health/ready`, `/health/live`, `/metrics`, `/sim/metrics` | health + metrics |
| GET | `/api/v1/orders` · `/orders/{order_id}` | list / get orders |
| POST | `/api/v1/orders/cancel-all` | cancel all open orders |
| GET | `/api/v1/stats` | execution stats |
| GET·… | `/api/v1/admin/config` | runtime config (admin) |

---

## Configuration (environment variables)

Names + purpose below; **defaults shown only where verified in source**.
Anything unmarked has its default in the cited file. Secrets come from `.env` /
Docker secrets — never hardcode.

### Core / runtime — [`lib/janus-core/src/config.rs`](../lib/janus-core/src/config.rs)

| Var | Purpose |
|-----|---------|
| `JANUS_CONFIG_PATH` | optional config-file path |
| `JANUS_SERVICE_NAME` · `JANUS_ENVIRONMENT` | service identity / env name |
| `JANUS_HOST` | bind host |
| `JANUS_HTTP_PORT` · `JANUS_GRPC_PORT` · `JANUS_WS_PORT` · `JANUS_METRICS_PORT` | listen ports |
| `JANUS_ENABLE_{FORWARD,BACKWARD,CNS,API,DATA,WEBSOCKET,GRPC,METRICS}` | per-subsystem toggles |
| `REDIS_URL` · `DATABASE_URL` · `QUESTDB_HOST` | backing stores |
| `JANUS_FORWARD_SIGNAL_INTERVAL` · `JANUS_FORWARD_ML_MODEL_PATH` | forward loop cadence / model path |
| `RISK_ACCOUNT_BALANCE` · `RISK_MAX_POSITION_SIZE_PCT` | base risk inputs |
| `OPTIMIZE_ASSETS` · `ENABLED_ASSETS` · `TRADING_ASSETS` · `PRIORITY_ASSETS` | asset universe (env-only today) |
| `DEFAULT_QUOTE_CURRENCY` · `PRIMARY_EXCHANGE` | market defaults |
| `TRADING_MODE` · `REAL_ORDERS_ENABLED` | trading posture |
| `JANUS_CORS_ORIGINS` · `RUST_LOG` · `LOG_FORMAT` | HTTP CORS + logging |
| `JANUS_REDIS_OVERLAY` · `JANUS_REDIS_CONFIG_KEY` | startup Redis config overlay (JFLOW-B; key `fks:janus:config`) |

### Execution & safety — [`services/execution/src/config.rs`](../services/execution/src/config.rs)

| Var | Default | Purpose |
|-----|---------|---------|
| `EXECUTION_MODE` | `paper_trading` | ⚠ **never default to live** — the no-autonomous-execution invariant |
| `ENABLE_RISK_CHECKS` | | toggle pre-trade risk checks |
| `MAX_POSITION_SIZE_USD` · `MAX_PORTFOLIO_EXPOSURE_USD` · `MAX_OPEN_POSITIONS` · `MAX_DAILY_LOSS_USD` | | hard risk caps |
| `KRAKEN_API_KEY` · `KRAKEN_API_SECRET` · `KRAKEN_TESTNET` · `KRAKEN_DRY_RUN` | | broker creds / safe modes (secrets) |
| `HTTP_PORT` · `GRPC_PORT` | | execution listen ports |

### Execution gate — Track C (`services/forward`, `crates/execution-gate`)

| Var | Default | Purpose |
|-----|---------|---------|
| `JANUS_GATE_ENFORCE` | off | enforce blocking verdicts (else advisory-only) |
| `JANUS_GATE_TP_ATR_MULT` | `4.0` | TP target ATR multiple for the fee-viability gate |
| `JANUS_GATE_FEE_TAKER` | `0.0006` | per-side taker fee fraction (fee-viability gate) |
| `JANUS_GATE_FEE_SLIP` | `0.0001` | per-side assumed slippage fraction |
| `JANUS_GATE_METRICS_REDIS` | off | mirror gate eval/block counters + breaker state to Redis |
| `JANUS_GATE_METRICS_PREFIX` | `janus:` | Redis key prefix for the above |
| `JANUS_GATE_METRICS_INTERVAL_SECS` | `15` | flush interval |
| `ENABLE_CNN_INFERENCE` | off | enable CNN-agreement / -confidence gates |
| `JANUS_PROP_FIRM_ENFORCE` · `JANUS_RISK_ENFORCE` | off | sibling enforce flags |

### Candle-scan reconciler — Track A (`services/data`)

Opt-in; off unless `JANUS_CANDLE_SCAN=1`. See
[`services/data/src/backfill/candle_scan.rs`](../services/data/src/backfill/candle_scan.rs).

| Var | Default | Purpose |
|-----|---------|---------|
| `JANUS_CANDLE_SCAN` | off | enable the periodic gap-scan → backfill reconciler |
| `JANUS_CANDLE_SCAN_SYMBOLS` | — | comma-separated symbols (exact `candles_crypto` symbols; **required** when enabled) |
| `JANUS_CANDLE_SCAN_EXCHANGE` | `binance` | exchange label on enqueued gaps |
| `JANUS_CANDLE_SCAN_INTERVAL` | `1m` | bar interval (timeframe string) |
| `JANUS_CANDLE_SCAN_LOOKBACK_SECS` | `86400` | trailing window scanned per pass |
| `JANUS_CANDLE_SCAN_EVERY_SECS` | `300` | pass cadence (min 5) |
| `JANUS_CANDLE_SCAN_MIN_MISSING` | `1` | drop gaps smaller than this |
| `JANUS_CANDLE_SCAN_MAX_WINDOW` | `1000` | max candles per backfill window (chunking) |
| `JANUS_CANDLE_SCAN_MAX_WINDOWS` | `0` (unlimited) | cap windows per pass |
| `QUESTDB_HTTP_URL` | `http://questdb:9000` | QuestDB HTTP `/exec` endpoint (also used by indicator warmup) |

### JanusAI integration — [`session_metrics.rs`](../lib/janus-core/src/session_metrics.rs) / [`checkpoint_notify.rs`](../lib/janus-core/src/checkpoint_notify.rs)

| Var | Purpose |
|-----|---------|
| `JANUS_AI_URL` · `JANUS_AI_TIMEOUT_SECS` · `JANUS_AI_PUSH_SECS` | session-metrics push loop target / timeout / cadence |
| `FKS_INSTANCE_ID` | instance id for per-instance Redis channels (`fks:{instance}:…`) |
| `ENABLE_CHECKPOINT_NOTIFY` | enable model-checkpoint notifications |

---

## See also

- [`TODO.md`](../TODO.md) — roadmap; Tracks A (data) / C (gate) / D (infra contract).
- [`docs/architecture/`](architecture/) — `RISK_TOPOLOGY.md`, `ML_STORY.md`, and the rest.
- `fks-full/docs/architecture/RUST_MIGRATION.md` — cross-repo migration + the §12-C repoint.
