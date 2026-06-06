# janus

**Rust ML inference engine and neuromorphic trading brain.**

A self-contained 39-crate Rust workspace implementing ML inference, neuromorphic GPU compute, signal generation, gRPC services, QuestDB data pipelines, and risk management. The workspace builds and runs **standalone** — its own [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) live in this repo. It can also be consumed as a service by the [fks-full](https://github.com/nuniesmith/fks-full) orchestrator.

---

## What's here

### Services (binaries)

| Service | Role |
|---------|------|
| `services/forward/` | Forward pass / live inference + brain REST server + regime bridge |
| `services/backward/` | Backward pass / training / experience store |
| `services/execution/` | Signal output → Alertmanager push + exchange execution |
| `services/optimizer/` | Hyperparameter optimization (Optuna-style) |
| `services/api/` | Unified HTTP/gRPC API gateway |
| `services/cns/` | Central Nervous System — watchdog, preflight, shutdown coordinator |
| `services/data/` | Data service — connectors (Binance, Bybit, KuCoin), QuestDB ILP, backfill |
| `services/registry/` | Asset registry service |

### Neuromorphic modules (`neuromorphic/`)

Brain-region-mapped Rust modules — experimental (30-day live validation required before stabilization):

`prefrontal` · `cortex` · `amygdala` · `hippocampus` · `thalamus` · `hypothalamus` · `basal_ganglia` · `cerebellum` · `visual_cortex` · `distributed`

### Core crates (`crates/`)

`ml` · `lob` (limit order book simulator) · `logic` (LTN fuzzy logic) · `dsp` (FRAMA signal processing) · `strategies` · `backtest` · `vision` (DiffGAF/ViViT) · `regime` · `cns` · `ltn` · `optimizer` · `memory` · `training` · `exchanges` · `models` (trades/signals/account/performance + prop-firm validator) · `data-quality` · `gap-detection` · `compliance` · `rate-limiter` · `risk` · `health` · `common` · `apalis-redis` · `registry` · `questdb-writer` · `fks-proto` (Janus-side protobuf surface)

### Proto definitions (`services/forward/proto/`)

gRPC service definitions. The shared protobuf surface is the `fks-proto` crate at `crates/fks-proto/`.

## Architecture

```
Kill Switch → Regime → Hypothalamus → Amygdala → Strategy Gate → Correlation
                                                                        │
                                                               Signal Output
                                                                        │
                                                    ┌───────────────────┤
                                                    ▼                   ▼
                                             Alertmanager          Redis pub/sub
                                                    │
                                                    ▼
                                             Ruby (execution decision)
```

**Key principle: Janus NEVER executes orders directly.** It generates signals and pushes them to Alertmanager and Ruby pub/sub. Ruby decides whether and how to execute based on account type and risk state.

**Brain REST server** runs on port `http_port + 100` (default: 8180) and exposes:
- `GET /api/v1/brain/health` — watchdog health + boot status
- `POST /api/v1/brain/affinity/record` — record trade outcome for learning
- `GET /api/v1/risk/evaluate` — risk gate evaluation

## Quick start (Docker)

```bash
docker compose up -d           # build + run janus with Redis, Postgres, QuestDB
docker compose logs -f janus   # tail janus
docker compose down            # stop & remove
```

`docker-compose.yml` boots the janus binary together with the backing services it needs to come up cleanly (Redis, Postgres, QuestDB) on its own bridge network. Downstream consumers (Alertmanager, JanusAI, a Ruby executor) are **not** included — janus tolerates their absence (signal pushes log a warning and continue); fks-full provides them when janus runs as part of that stack.

## Building (from source)

```bash
# Full workspace build
cargo build --workspace

# Just the forward service (most commonly iterated)
cargo build -p janus-forward

# Run tests
cargo test --workspace

# Check only (fast feedback)
cargo check --workspace
```

Requires Rust stable (edition 2024). GPU features require CUDA toolkit + matching NVIDIA drivers.

## Key env vars

| Var | Default | Description |
|-----|---------|-------------|
| `ENABLE_BRAIN_RUNTIME` | `true` | Start brain REST server |
| `JANUS_BOOTSTRAP_DAYS` | `30` | Days of memories to load on cold start |
| `JANUS_BOOTSTRAP_LIMIT` | `500` | Max memory records to bootstrap |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL (compose points this at the bundled service) |
| `DATABASE_URL` | `postgres://janus:janus@postgres:5432/janus` | Postgres URL (compose-bundled) |
| `QUESTDB_HOST` | `questdb` | QuestDB host (compose-bundled) |
| `ALERTMANAGER_URL` | `http://fks_alertmanager:9093` | Signal push endpoint — optional; if unreachable (standalone), pushes warn and continue |
| `JANUS_FORWARD_URL` | `http://fks_janus:8180` | Brain REST URL an external executor calls |

The compose file sets the backing-service URLs above; see `config/janus.toml` and `docker-compose.yml` for the full set.

## Deployment

- **Standalone** — `docker compose up -d` in this repo builds the image (multi-stage [`Dockerfile`](Dockerfile), unified `janus` binary) and runs it with its backing services.
- **As part of [fks-full](https://github.com/nuniesmith/fks-full)** — the orchestrator builds this repo's image and runs janus alongside the shared infrastructure (Alertmanager, Ruby executor, Grafana, …). It's the same self-contained workspace in both cases.

## Stats

- ~583K lines of Rust
- 9,888+ tests
- 1,020 `.rs` files
- ~50 workspace crates
- 8 Janus services (unified binary mode)
