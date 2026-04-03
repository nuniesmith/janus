# fks-janus

**Rust ML inference engine and neuromorphic trading brain — source code only.**

This repo contains the Janus engine: a 50+ crate Rust workspace implementing ML inference, neuromorphic GPU compute, signal generation, gRPC services, QuestDB data pipelines, and risk management. Infrastructure (Docker, compose, CI/CD) lives in [fks](https://github.com/nuniesmith/fks).

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

`ml` · `lob` (limit order book simulator) · `logic` (LTN fuzzy logic) · `dsp` (FRAMA signal processing) · `strategies` · `backtest` · `vision` (DiffGAF/ViViT) · `regime` · `cns` · `ltn` · `optimizer` · `memory` · `training` · `exchanges` · `indicators` · `data-quality` · `gap-detection` · `compliance` · `rate-limiter` · `risk` · `health` · `common` · `apalis-redis` · `registry` · `questdb-writer` · `bybit-client`

### Proto definitions (`services/forward/proto/`)

gRPC service definitions — see `src/proto/` in the root for the shared `fks-proto` crate.

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

## Building

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
| `ALERTMANAGER_URL` | `http://fks_alertmanager:9093` | Signal push endpoint |
| `JANUS_FORWARD_URL` | `http://fks_janus:8180` | Brain REST (for Ruby to call) |

Full env reference in [fks/.env.example](https://github.com/nuniesmith/fks/blob/main/.env.example).

## Deployment

Deployed via [fks](https://github.com/nuniesmith/fks). The Dockerfile clones this repo at build time. No Docker config lives here.

## Stats

- ~583K lines of Rust
- 9,888+ tests
- 1,020 `.rs` files
- ~50 workspace crates
- 8 Janus services (unified binary mode)
