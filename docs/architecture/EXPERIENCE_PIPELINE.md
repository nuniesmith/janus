# Experience Pipeline — forward → spool → backward → Qdrant

> **Status: design doc, nothing here is implemented yet.** This designs the
> pipeline that turns live paper-trading decisions into persisted experience
> vectors: the prerequisite for (a) the "what is janus thinking" UMAP view and
> (b) the paper's learning loop (experience replay feeding `backward` training).
> Grounded in a read-only survey (2026-07) of `services/forward`,
> `services/backward`, `crates/vision`, `crates/ml`, and the companion docs
> [`ML_STORY.md`](ML_STORY.md), [`ML_VISION_SCOPE.md`](ML_VISION_SCOPE.md),
> [`ML_PHASE1_HANDOFF.md`](ML_PHASE1_HANDOFF.md),
> [`CNN_LIVE_ONRAMP.md`](CNN_LIVE_ONRAMP.md).

---

## TL;DR

- **Both ends of the pipe exist; the pipe does not.** `backward` has a fully
  implemented, tested ingest path (`tasks/ingest.rs` → `ExperienceStore` →
  Qdrant) that **nothing calls** (`handle_ingest` is `#[allow(dead_code)]`),
  and `forward` has a shared-memory writer (`shm.rs`) that is a **no-op stub
  with zero callers**. No experience has ever flowed.
- Three honest gaps shape the design: (1) `forward` computes **no GAF features
  today** (GAF lives in Candle-based `crates/vision`, a dependency `forward`
  does not have); (2) `forward` does **not observe trade closes** — realized
  PnL arrives only via an externally-posted HTTP endpoint — so per-trade reward
  attribution is not available as a passive signal; (3) `backward`'s job intake
  is **also unimplemented** (workers idle-poll; the Redis queue implied by
  `IngestJob` + the unused `apalis-redis` dependency was never wired), and the
  `ExperienceStore` **defaults to mock mode**, with no Qdrant service in any
  compose file.
- **Phase 1** (smallest honest slice): per-closed-candle experiences with a
  next-bar mark-to-market reward, written as complete Arrow IPC files to a
  `/dev/shm` spool, signalled over a Redis list, ingested by a minimal worker
  intake into a real Qdrant collection. Per-trade PnL attribution and training
  consumption are Phase 2 — explicitly *not* blocked on Phase 1's reward being
  the final scheme.

---

## 1. What exists today (verified)

### 1.1 The producer stub — `services/forward/src/shm.rs`

`SharedMemoryBuffer::{new, write, flush}` exists, takes a `common::Experience`,
and does nothing but log. **Zero callers** anywhere in `forward`
(`grep -rn "SharedMemoryBuffer" services/forward/src` → only the module
itself). Its doc comment sketches the intent: Arrow IPC via
`arrow::ipc::writer::FileWriter`, memmap, ring buffer.

`common::Experience` (`crates/common/src/types.rs:389`) is the in-memory shape:
`state: State { gaf_features: Array2<f32>, gaf_features_flat, raw_features,
metadata }`, `action`, `reward: f32`, `next_state`, `done`, `priority`,
`timestamp`, `episode_id`.

### 1.2 The consumer — `services/backward/src/tasks/ingest.rs`

Fully implemented and unit-tested, **never called in production**
(`#[allow(dead_code)]`). `handle_ingest(job: IngestJob, store:
Option<&ExperienceStore>)`:

1. Opens `job.shm_path` as an **Arrow IPC file** via `FileReader` — note this
   requires a *finished* file with an IPC footer, which constrains the writer
   design (§4.2).
2. Validates against `expected_schema()` (the contract, §3).
3. Iterates record batches, per-row null validation.
4. Persists via `ExperienceStore::persist_batch`.

Missing files are skipped gracefully (`Ok(())` + warn) — good for at-least-once
job delivery.

### 1.3 The store — `services/backward/src/persistence/experience_store.rs`

Qdrant client with an in-memory mock. Per row: point ID = **random UUIDv4**,
vector = `state_gaf` decoded as f32-LE, payload = `action_type`,
`action_symbol`, `action_qty`, `reward`, `done`, `timestamp_ms`,
`next_state_vector` (JSON string). Collection: `experiences`, Cosine distance,
scalar quantization, dim `QDRANT_EXPERIENCE_DIM` (default **9**), upserts
chunked at 256.

Sharp edges (all addressed in this design):

- **`QDRANT_USE_MOCK` defaults to `"true"`**, and a failed Qdrant connection
  *silently falls back to mock*. As shipped, a "successful" deployment can
  persist everything into a process-local `Vec` and lose it on restart.
- **No Qdrant service exists** in `docker-compose.yml` or `config/`.
- `state_raw` / `next_state_raw` are decoded into `ExperienceRow` but
  **dropped from the Qdrant payload**.
- **No read API whatsoever** — the store can only upsert. The UMAP view and
  Phase 2 training both need a sample/scroll surface (§8).
- Random point IDs mean **re-ingesting a batch duplicates every row** (no
  idempotency).

### 1.4 Job intake — designed but not wired

`worker.rs` defines `IngestJob { batch_id, shm_path }` (serde-serializable,
with a test asserting JSON round-trip — clearly written for a queue) and
`BackwardServiceConfig` carries a `redis_url`; `services/backward/Cargo.toml`
depends on `apalis` + `apalis-redis`. **None of it is used**: the worker loop
in `lib.rs::start()` literally sleeps 5s and runs a DB health check —
`// in production this would read from a Redis/channel-based job queue.`
The backward's intake is **also unimplemented**; §4.3 designs the minimal one.

### 1.5 Where forward could observe an experience (the live loop)

The live signal loop (`services/forward/src/lib.rs`, live-mode task spawned
around `:1329`) processes each **closed candle** per symbol:

```
closed kline → IndicatorAnalyzer → strategy_votes
   (EMA-flip, mean-reversion, squeeze, VWAP, ORB + optional CNN vote, lib.rs:1689)
→ resolve_consensus (≥2 strategies, ≥0.6 agreement, lib.rs:1935)
→ min_confidence filter → prop-firm / RiskManager / execution-gate checks
→ signal_bus.publish(signal)                                  (lib.rs:~2140)
→ if !Hold && conf ≥ 0.7 && not blocked: submit_signal_to_execution (paper gRPC)
```

Everything needed to *describe a decision* is in scope inside this one task at
the moment a candle closes: the candle window (`cnn_buffers` /
`CandleBuffer`), the votes, the consensus outcome (including "no consensus ⇒
Hold" — currently a bare `continue`), the regime/fear labels, the gate
outcome, and the proposed size. **This loop is the writer's hook point** (§4.1).

What is *not* observable in forward:

- **Fills and position closes from the paper exchange.** Execution happens in
  `services/execution` over gRPC; forward gets a submit ACK, not an outcome.
  Realized PnL re-enters janus only when something POSTs
  `/api/v1/positions/close` (janus-api → `AffinityRecorder` /
  `GateOutcomeRecorder` → forward's affinity tracker + gate breaker) or
  `/api/v1/risk/portfolio/positions/close` (forward risk REST). These are
  **externally-driven push endpoints** — nothing guarantees they fire for a
  paper trade. The Bybit private-feed `LiveAccount` (`account_state.rs`) folds
  real fills, but only with live credentials, not paper.
- **GAF features.** `gaf_features_from_series` lives in `crates/vision`
  (Candle); forward depends on neither. Forward's only live neural features
  are the CNN's 20-channel × 60-bar window
  (`janus_ml::features::per_asset_cnn::extract_features`, warmup 110 bars).

### 1.6 Deployment topology

The unified `bin/janus` binary runs forward and backward as **modules of one
process** (the `fks_janus` container). Standalone `services/{forward,backward}`
binaries also exist. The transport must work for both: a filesystem spool +
Redis signal does (both topologies already share Redis); an in-process channel
would not survive a service split, and a true shared-memory ring buffer is
incompatible with the already-implemented `FileReader` consumer anyway.

---

## 2. Data flow (target design)

```
┌────────────────────────────── forward (live loop) ─────────────────────────────┐
│  closed candle (symbol S, t)                                                    │
│    ├─ indicators / votes / consensus / gates   (existing, unchanged)            │
│    └─ ExperienceRecorder::observe(S, t, features, decision)                     │
│         │  holds 1 pending decision per symbol                                  │
│         ▼  at candle t+1: reward = next-bar return → completed Experience       │
│    ExperienceWriter (per-process, off hot path via mpsc)                        │
│         │  buffers rows; every N rows or T secs:                                │
│         ▼                                                                       │
│    write  /dev/shm/janus/experience/<batch_id>.arrow.tmp                        │
│    finish + fsync + rename → <batch_id>.arrow      (atomic, footer complete)    │
│         │                                                                       │
│         ▼                                                                       │
│    LPUSH janus:experience:ingest  {"batch_id":..,"shm_path":..}   (IngestJob)   │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Redis list (at-least-once)
┌──────────────────────────────────────▼──────────────── backward ────────────────┐
│  intake worker: BRPOP → handle_ingest(job, Some(&store))   (existing fn)        │
│    ├─ on success: delete <batch_id>.arrow                                       │
│    ├─ on failure: retry w/ backoff, then park file (sweep retries later)        │
│    └─ startup + periodic sweep of spool dir (recovers lost signals)             │
│  ExperienceStore::persist_batch → Qdrant "experiences" (real mode)              │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                     ┌─────────────────┴───────────────────┐
                     ▼                                     ▼
        Phase 2: training loop samples            UMAP view: GET
        replay batches (train.rs replay           /api/v1/experiences/sample
        buffer seeded from Qdrant)                → project → "what is janus
                                                    thinking" scatter
```

---

## 3. The experience record

The Arrow IPC schema is **already frozen** by `expected_schema()` in
`ingest.rs`; the design keeps it byte-compatible and defines what each field
honestly contains. One record = one *decision transition*: the state at candle
`t`, the action janus's consensus took at `t`, and the reward observable by
`t+1`.

| Field | Arrow type | Contents (Phase 1) |
|---|---|---|
| `state_gaf` | `Binary` (f32-LE), non-null | Flat pooled GAF vector of the close-price window at `t`, length = `QDRANT_EXPERIENCE_DIM` (see §3.1) |
| `state_raw` | `Binary` (f32-LE), nullable | The 20 per-channel CNN feature values at the decision bar (last column of the live `(20, 60)` window) when the CNN buffer is warm; null otherwise. Aligns the record with what the *live* model actually saw |
| `action_type` | `UInt8`, non-null | `0=Buy, 1=Sell, 2=Hold, 3=Close` — canonical mapping of `janus_core::SignalType` (must be defined once in `crates/common`, see §3.2) |
| `action_symbol` | `Utf8`, non-null | e.g. `"BTCUSDT"` |
| `action_qty` | `Float32`, non-null | Proposed position size from the risk check; `0.0` for Hold / blocked entries |
| `reward` | `Float32`, non-null | Signed next-bar return aligned with the action (§3.3) |
| `next_state_gaf` | `Binary`, non-null | Same as `state_gaf`, computed at `t+1` |
| `next_state_raw` | `Binary`, nullable | Same as `state_raw`, at `t+1` |
| `done` | `Boolean`, non-null | `false` in continuous paper trading; `true` only when the writer closes an episode on a data-gap/stream-restart boundary (§3.4) |
| `timestamp_ms` | `Int64`, non-null | Close time of candle `t` (epoch ms) |

Additional context that does not fit the frozen schema (consensus contributor
list, regime, fear label, gate outcome, confidence) goes into a **custom
Arrow schema metadata map** on the file (`FileWriter` supports it; the current
`validate_schema` only checks fields, so this is forward-compatible) *and*,
for the fields the UMAP view needs (confidence, regime), into the Qdrant
payload in Phase 1.5 (§5). Do **not** widen the field list in Phase 1 — the
whole point is to light up the existing, tested consumer unchanged.

### 3.1 The `state_gaf` honesty problem (flagged mismatch)

The schema and store were built for a **9-dim flattened GAF** ("3×3 GAF
image"). Reality:

- Forward computes **no GAF at all** today. The only real GAF code is
  `crates/vision::gaf_features_from_series` — Candle-based, used solely by
  backward's opt-in replay-buffer seeding (`train.rs`, #59–#62).
- The "flat" GAF is a per-channel spatial mean of the DiffGAF image — length
  `num_features` (=1 for a closes-only series), **zero-padded to 9** in the
  seed path. The 9-dim default is therefore mostly padding today
  (acknowledged as "thin" in `ML_PHASE1_HANDOFF.md` §A.1).
- The live model's actual features are the CNN's `20 × 60` window (1200
  values) — a *different* feature family from the paper's vision/GAF path.

**Decision:** keep `state_gaf` as a true GAF vector (it is what the paper's
learning loop trains on) but fix the two problems:

1. **Extract a dependency-free GAF-flat function.** A pooled GASF over a
   normalized close window is ~50 lines of pure `f32`/ndarray math (min-max →
   arccos angles → mean of `cos(φi+φj)` per pooling cell); it does not need
   Candle. Add `gaf_flat(closes: &[f32], k: usize) -> Vec<f32>` to
   `crates/common` (or a tiny new `crates/gaf-lite`), producing a `k×k`
   pooled GASF flattened to `k²` values. **Both** the forward writer and
   backward's trainer/seeder use this one function — train/serve feature
   parity by construction (the exact lesson of the CNN parity break in
   `CNN_LIVE_ONRAMP.md`), and no Candle in the live binary (the cost
   `ML_VISION_SCOPE.md` explicitly warned about).
2. **Make the dimension real, not padding.** Recommend `k=3` ⇒ dim **9**,
   matching `QDRANT_EXPERIENCE_DIM`'s default and the existing collection
   contract — but now all 9 values carry signal (9 pooled regions of the
   GASF), instead of 1 real value + 8 zeros. If Phase 1 experimentation wants
   richer vectors, `k=4` (16) or `k=8` (64) only requires bumping the env var
   *before the collection is first created* (§6).

Equivalence with `crates/vision`'s DiffGAF output should be asserted by a unit
test (fixed input → both paths within tolerance) *if* backward keeps the
Candle path for images; the flat path should switch to the shared function.

### 3.2 `action_type` mapping (flagged mismatch)

`ingest.rs` documents `0=Buy, 1=Sell, 2=Hold, 3=Close`, but no code anywhere
defines this mapping — `janus_core::SignalType` has no stable `u8` repr.
Define `impl From<SignalType> for u8` (and back) in one place
(`crates/common` or `janus-core`) and use it in the writer; a silent enum
reorder would otherwise corrupt every stored experience.

### 3.3 Reward: what is honest today

Candidates, in decreasing honesty-per-effort:

| Signal | Available? | Verdict for Phase 1 |
|---|---|---|
| **Next-bar mark-to-market return** | Yes — the live loop sees candle `t+1` one interval later | ✅ **Use this.** Dense (every bar), self-contained, zero attribution machinery. `reward = dir(action) × log(close_{t+1}/close_t)`, `dir = +1` Buy, `−1` Sell, `0` Hold. Optionally subtract the gate's configured round-trip fee for non-Hold actions so Buy/Sell aren't free |
| Per-trade realized PnL | Only via externally-POSTed `/api/v1/positions/close` (janus-api) or `/api/v1/risk/portfolio/positions/close` (forward). Forward does **not** observe paper-exchange exits on its own; nothing guarantees these fire | ❌ Not Phase 1. Phase 2 designs the join (§7) |
| Exchange-fill PnL (`LiveAccount`) | Only with live Bybit credentials; not paper | ❌ Out of scope |

This matches the placeholder scheme already used by backward's seed builder
(`experiences_from_series`: directional action + next-bar return), so Phase 1
live experiences and Phase 1 seeded experiences are **the same distribution**
— the probe in `ML_PHASE1_HANDOFF.md` and the live pipeline stay comparable.
`ML_PHASE1_HANDOFF.md` §A.2 already marks the reward scheme as an open
empirical decision; this design does not pretend to settle it, it just picks
the one honest dense signal that exists.

**Consequence:** an experience is only *complete* at `t+1`. The recorder holds
one pending decision per symbol and emits it when the next candle for that
symbol closes (also supplying `next_state_*`). A pending decision older than
`max(2 × interval, 5 min)` when a new candle arrives (data gap) is emitted
with `done = true` and reward computed across the gap — or dropped if the gap
exceeds the interval by >10×; both counted in metrics.

### 3.4 `done` semantics

Continuous 24/7 paper trading has no natural episodes. Phase 1: `done = false`
everywhere except the gap/restart boundary above. Document this loudly — DQN
target math in `train.rs` special-cases `done`, and pretending session
boundaries are episode ends would inject fake terminal states. Revisit in
Phase 2 (a trade-close could become a natural `done` for trade-scoped
episodes).

### 3.5 Which bars produce a record

**Every closed candle for every live symbol once the recorder is warm** (needs
the GAF window, ~64 bars of closes; `state_raw` additionally needs the CNN
buffer's 110-bar warmup and is null before that). Not just published signals:

- Hold is the overwhelmingly common action and exactly what the UMAP view
  needs to show ("janus is mostly deciding *not* to act, here's where the
  Buys live in state space").
- Volume is trivial: 1-minute bars × a handful of symbols ≈ 10k rows/day ≈
  a few MB/day of IPC at dim 9 + 20 raw floats.

The recorded `action_type` is the **post-gate outcome**: consensus Hold or no
consensus ⇒ Hold; consensus Buy/Sell that was risk/gate-*blocked* from
execution submit is still recorded as Buy/Sell with a `blocked` marker in
file metadata (Phase 1) — the decision is what the brain wanted; the gate is
part of the environment. (Open question §9.2 revisits this.)

---

## 4. Transport: spool file + Redis signal

### 4.1 Writer hook in forward

New module `services/forward/src/experience.rs` (replacing the `shm.rs` stub —
delete it or rewrite it in place) with two pieces:

- **`ExperienceRecorder`** — plain struct owned by the live signal-loop task
  (no locks; same pattern as `cnn_buffers` / `regime_manager`). Hook point:
  the end of the per-candle processing in the live loop
  (`services/forward/src/lib.rs`, the task spawned at `:1380`), at three
  sites:
  1. after `resolve_consensus`' `None ⇒ continue` (records the Hold),
  2. after the confidence filter `continue`s (also Hold),
  3. after the gate/submit block (records Buy/Sell/Close + qty + blocked flag).
  The `continue`s must be restructured to fall through a common
  `recorder.observe(...)` call — the only touch to existing control flow.
  It maintains per-symbol close windows (reuse `CandleBuffer` or a small
  `VecDeque<f32>`) and the one-pending-decision-per-symbol slot (§3.3).
- **`ExperienceWriter`** — a spawned task receiving completed rows over a
  bounded `tokio::mpsc` channel (capacity ~1024; **`try_send` + drop-and-count
  on full** — the hot path must never await the writer). It buffers rows and
  flushes a batch when **256 rows** or **60 s** elapse, whichever first.

Config (env, all with safe defaults): `JANUS_EXPERIENCE_ENABLED` (default
**false** — same default-off discipline as `ENABLE_CNN_INFERENCE` /
`JANUS_RISK_ENFORCE`), `JANUS_EXPERIENCE_SPOOL_DIR` (default
`/dev/shm/janus/experience`), `JANUS_EXPERIENCE_BATCH_ROWS` (256),
`JANUS_EXPERIENCE_FLUSH_SECS` (60), `JANUS_EXPERIENCE_SPOOL_MAX_MB` (256).

### 4.2 Spool format: complete IPC files, not a ring buffer

The stub's doc comment imagines a memmapped ring buffer. **Rejected**: the
already-implemented consumer uses `arrow::ipc::reader::FileReader`, which
requires the IPC *footer*, i.e. a finished file. A ring buffer would mean
rewriting the tested consumer for zero Phase-1 benefit at ~10k rows/day.
`/dev/shm` gives the "shared memory" property (tmpfs, no disk I/O) with plain
file semantics; `IngestJob.shm_path` and its own doc example
(`/dev/shm/janus_batch_001.ipc`) already assume exactly this.

Batch protocol:

1. `batch_id = <unix_ms>-<uuid-v4-short>` (sortable, unique across restarts).
2. Write to `<spool>/<batch_id>.arrow.tmp` via `FileWriter` (schema §3 +
   file-level metadata: `schema_version=1`, producer git sha, interval,
   symbol set, blocked/regime side-channel as JSON).
3. `finish()` → `fsync` → **atomic `rename`** to `<batch_id>.arrow`. A crash
   mid-write leaves only a `.tmp`, which no consumer ever reads.
4. Enqueue the job (§4.3). The file is the source of truth; the queue entry is
   only a doorbell.

Rotation = one file per batch; there is no long-lived segment to rotate.
**Spool bound:** before writing, if the spool exceeds
`JANUS_EXPERIENCE_SPOOL_MAX_MB`, delete oldest `.arrow` files until under the
limit (drop-oldest, counted in a Prometheus counter). Experience collection
must never be able to fill `/dev/shm` and destabilize the live loop —
experiences are droppable; trading is not.

### 4.3 Signalling: Redis list, plus a sweep (the minimal intake)

Backward's intake must be built either way (§1.4), so pick the mechanism its
code was already shaped for: **a Redis list carrying `IngestJob` JSON**
(`IngestJob` already derives Serialize/Deserialize and has a JSON round-trip
test; `BackwardServiceConfig.redis_url` already exists; Redis is already a
hard dependency of the deployment). Skip `apalis` for Phase 1 — a
`LPUSH`/`BRPOP` pair is ~30 lines and has no framework surface; promote to
apalis only if job types multiply.

- Forward, after rename: `LPUSH janus:experience:ingest <json(IngestJob)>`.
  Redis errors are logged + counted, **not retried in the hot path** — the
  sweep (below) is the retry.
- Backward: replace the idle-poll body of one worker (`lib.rs::start()`
  workers loop) with `BRPOP janus:experience:ingest 5` → on job:
  `handle_ingest(job, Some(&store))` → on `Ok` **delete the spool file**; on
  `Err` re-`LPUSH` with an attempt counter in a wrapper envelope, park in
  `<spool>/failed/` after 3 attempts.
- **Sweep (crash/downtime recovery):** on backward startup and every 5 min,
  list `<spool>/*.arrow` older than 2 min whose batch_id is not in-flight and
  synthesize jobs for them. This makes the queue merely an optimization:
  losing Redis, losing a job, or backward being down for an hour all converge
  to "the sweep ingests it later". Duplicate delivery is possible
  (sweep + late queue entry) — which is why point IDs must be idempotent (§5).

Works identically in the unified binary (same tmpfs, same Redis) and in
split-service deployments (mount the same tmpfs volume into both containers —
add `/dev/shm/janus` as a shared mount in compose; document that a
cross-*host* split requires replacing the spool with object storage, out of
scope).

---

## 5. Qdrant collection lifecycle

- **Deploy Qdrant.** Add a `qdrant` service to `docker-compose.yml` (image
  `qdrant/qdrant`, gRPC 6334, volume for `/qdrant/storage`) and set
  `QDRANT_URL=http://qdrant:6334`, `QDRANT_USE_MOCK=false` on the janus
  service. Today no compose/config mentions Qdrant at all.
- **Fix the silent-mock failure mode.** `ExperienceStore::new` falling back to
  mock on connection failure means "everything persisted" while writing to a
  `Vec` that dies with the process. Change: when `use_mock=false`, a failed
  connection is an **error** (retry with backoff at startup; while
  disconnected, ingest jobs fail → files stay parked in the spool → nothing
  is silently lost). Mock stays available for tests via explicit
  `QDRANT_USE_MOCK=true`.
- **Creation:** keep lazy `ensure_collection()` on first ingest (already
  implemented). Dim comes from `QDRANT_EXPERIENCE_DIM`; the writer's GAF `k²`
  and this env var must agree — the ingest path should **hard-fail a batch
  whose vector length ≠ collection dim** (today `extract_rows` would happily
  upsert mismatched vectors and Qdrant would reject them row by row).
- **Idempotent point IDs:** replace UUIDv4 with **UUIDv5 of
  `(batch_id, row_index)`** so at-least-once delivery (sweep + queue) upserts
  the same point rather than duplicating. This is a one-line change in
  `row_to_point` / `MockStore::upsert`.
- **Payload additions (Phase 1.5, needed by the UMAP view):** `confidence`
  (f64), `regime` (string), `blocked` (bool), plus payload **indexes** on
  `action_symbol`, `timestamp_ms`, `action_type` for filtered sampling.
  `state_raw` should also be persisted (JSON string like `next_state_vector`)
  instead of being dropped.
- **Versioning:** collection name from `QDRANT_EXPERIENCE_COLLECTION`; on
  breaking schema/dim changes create `experiences_v2` and leave v1 readable —
  never mutate a collection's dim in place.
- **Retention:** unbounded growth is fine for months at Phase 1 volume
  (~10k points/day). Add a scheduled backward job (the 60s scheduler tick
  already exists) that enforces `QDRANT_EXPERIENCE_MAX_POINTS` (default 5M)
  by deleting oldest via a `timestamp_ms` range filter, off by default until
  needed.

---

## 6. Failure modes

| Failure | Behaviour | Bound |
|---|---|---|
| Backward down / slow | Files accumulate in spool; queue grows | Spool capped at `SPOOL_MAX_MB` (drop-oldest + counter); Redis list capped via `LTRIM` to ~10k jobs (files are truth; sweep recovers trimmed jobs) |
| Redis down | LPUSH fails (logged); ingest continues via periodic sweep | Latency degrades to sweep period (≤5 min); no loss |
| Qdrant down | Ingest job errors → retry ×3 → file parked in `failed/`; sweep re-tries parked files hourly | No silent loss (**requires** removing the mock fallback, §5) |
| Forward crash mid-batch | `.tmp` file has no footer, never ingested; cleaned by sweep after 1h | Loses ≤ one in-memory batch (≤256 rows) — acceptable; experiences are not transactional data |
| Writer channel full (hot-path backpressure) | `try_send` drops the row, increments `experience_rows_dropped` | Live loop never blocks on the pipeline, by construction |
| Duplicate delivery (sweep + queue race) | UUIDv5 point IDs make re-upsert idempotent | Exactly-once *effect* without exactly-once delivery |
| Partial batch (some rows invalid) | Existing `ingest.rs` behaviour: skip invalid rows, persist valid, count both | Already implemented + tested |
| Schema drift (old backward, new forward) | `validate_schema` warns on mismatch; add: reject the batch (park) when a **required** field is missing, rather than persisting partial nonsense | `schema_version` in file metadata makes this explicit |
| `/dev/shm` pressure from other tenants | Writer checks spool size before write; on `ENOSPC` drops batch + counter | Never propagates a panic into the signal loop |

Observability: Prometheus counters on both ends (`experience_rows_written`,
`_dropped`, `_batches_spooled`, `_spool_bytes`, backward's existing
`IngestMetrics` exported instead of just logged), so "the pipeline is quietly
dead" is visible on the dashboard rather than discovered at training time.

---

## 7. Phased implementation plan

### Phase 1 — real vectors in Qdrant during paper trading (the smallest honest slice)

Goal: `docker compose up` + `JANUS_EXPERIENCE_ENABLED=true` ⇒ Qdrant's
`experiences` collection visibly grows while paper trading, with honest
per-bar decision records. **~1–1.5 weeks.**

1. `crates/common`: `gaf_flat(closes, k)` (pure, no Candle) + canonical
   `SignalType ↔ u8` mapping + unit tests (incl. parity test against
   `crates/vision` DiffGAF-flat on fixed input). *(~1 day)*
2. `services/forward`: `ExperienceRecorder` + `ExperienceWriter` (Arrow
   `FileWriter` matching `expected_schema()` — reuse the test-batch builder in
   `ingest.rs` tests as the reference), spool + atomic rename + LPUSH; hook
   the three live-loop sites; env-gated off by default. *(~3 days)*
3. `services/backward`: BRPOP intake in one worker → `handle_ingest` (drop the
   `#[allow(dead_code)]`) → delete-on-success; startup/periodic sweep;
   `ExperienceStore` fixes: no silent mock fallback, UUIDv5 ids, dim
   hard-check. *(~2 days)*
4. Compose: add `qdrant` service; janus env (`QDRANT_USE_MOCK=false`,
   `JANUS_EXPERIENCE_ENABLED=true` in the paper profile). *(~0.5 day)*
5. Metrics + an integration test: synthetic candles through recorder → writer
   → real `handle_ingest` → mock store, asserting row counts and vector
   contents end-to-end. *(~1 day)*

Explicitly **not** in Phase 1: trade-attributed rewards, training consumption,
any change to `expected_schema()`, apalis, ViViT images (only the flat vector
is stored; full GAF images would need a different store layout — open question
§9.4).

### Phase 2 — the learning loop consumes it

1. **Read API on `ExperienceStore`**: `scroll(filter, limit)` +
   `sample_random(n)` (Qdrant scroll with random offset or match-any +
   `with_vectors`). Feeds both training and the UMAP endpoint (§8).
2. **Training from Qdrant**: replace/augment `TrainingConfig.seed_closes` —
   `train.rs` fills its replay buffer from `scroll` (time-range filtered), so
   the DQN/LSTM trains on *live-collected* experiences. PER priorities remain
   process-local (Qdrant payload update per TD-error is possible but deferred).
3. **Trade-attributed reward (the honest upgrade):** producers of
   `/api/v1/positions/close` carry `position_id`/`strategy`; janus-api's
   handler (which already fans out to `AffinityRecorder` + gate breaker) gains
   a third recorder that emits a **trade-outcome experience** — state = the
   entry-decision state (recorder keeps a small entry-state cache keyed by
   symbol/position_id), reward = `pnl_realized` normalized by risk, `done =
   true`. This is *additive*: per-bar MTM rows and per-trade rows coexist,
   distinguished by `done`/payload tag. **Gap, stated plainly:** if nothing
   POSTs position closes for paper trades, no trade-attributed rows exist —
   closing that loop requires the execution service (or the paper-exchange
   wrapper) to emit close events, which is outside forward and outside this
   design's Phase 1/2 code. The design degrades gracefully to per-bar rewards.
4. Reward-scheme experiments per `ML_PHASE1_HANDOFF.md` §A.2 (fees,
   vol-normalization, horizon >1 bar) — all writer-side constants once the
   pipe exists.

### Phase 1.5 (parallel, small) — UMAP payload enrichment

`confidence`, `regime`, `blocked` in the Qdrant payload + payload indexes
(§5), since the UMAP view is the first consumer.

---

## 8. What the UMAP view reads

The "what is janus thinking" view is a 2-D projection of recent `state_gaf`
(or `state_raw`) vectors, colored by action/reward/regime. It needs one
endpoint, on **backward's existing HTTP server** (`http.rs`, which today only
has `/health` + `/metrics`):

```
GET /api/v1/experiences/sample?limit=2000&symbol=BTCUSDT&since_ms=...&vector=gaf|raw

200 {
  "count": 2000,
  "dim": 9,
  "points": [
    { "id": "…", "vector": [0.12, …],           // 9 floats (gaf) or 20 (raw)
      "action_type": 2, "reward": -0.0003,
      "symbol": "BTCUSDT", "timestamp_ms": 1782000000000,
      "confidence": 0.0, "regime": "trending", "done": false },
    …
  ]
}
```

Implementation: `ExperienceStore::scroll` (Phase 2.1) with `with_vectors:
true`, payload filter from query params, `limit` capped at ~5k. The projection
itself (UMAP/t-SNE/PCA) runs **client-side or in a viz sidecar** — backward
serves raw vectors + payload only; at dim 9 × 2k points the response is
~100 KB, no server-side reduction needed. (If dim grows to 64+, revisit with a
server-side PCA-to-16 pre-step.) A `?projection=umap` server-side option is
explicitly deferred — no Rust UMAP dependency is worth it before the view
proves out.

Freshness: the view polls the endpoint; end-to-end latency is bounded by
`FLUSH_SECS` (60 s) + queue latency (~instant) — comfortably "live" for a
thinking-dashboard.

---

## 9. Open questions

1. **GAF `k` (vector dim).** 9 keeps every default aligned but is coarse; 64
   (8×8) is likely nearer what a UMAP needs to show structure. Cheap to decide
   empirically in week 1 — but it must be decided **before** the collection is
   first created in a long-lived environment (§5 versioning otherwise).
2. **Record pre-gate or post-gate action?** §3.5 records the brain's intent
   (pre-block) with a `blocked` flag. The alternative — record what was
   actually submitted — is more honest about *behaviour* but hides the
   interesting decisions the UMAP exists to show. Revisit when the first
   projections are looked at.
3. **Multi-symbol windows / cross-asset state.** Phase 1 state is
   single-symbol closes. The regime/portfolio context that the gate uses is
   richer; whether cross-asset state belongs in `state_raw` or a wider schema
   v2 is a paper-level modelling question.
4. **Full GAF images for ViViT.** The flat vector deliberately discards most
   of the image (`ML_PHASE1_HANDOFF.md` §A.1). If the ViViT bet proceeds,
   images (`k×k×F` ≫ 9) probably want Arrow files retained in object storage
   with Qdrant holding only the pooled vector + a file reference — a Phase 3
   storage design, not a Phase 1 blocker.
5. **Should Hold rows get an opportunity-cost reward** (e.g. `−|r|` when a
   strong move was missed) instead of 0? Writer-side constant; decide with
   data.
6. **Unified-binary fast path.** Forward and backward share one process in
   `bin/janus`; an in-process mpsc could skip Redis entirely. Rejected for
   Phase 1 (one mechanism for both topologies beats two), but if the split
   services are ever retired, the queue can collapse to a channel without
   touching the file format.

---

## Re-verify the ground truth

```bash
# Producer stub, zero callers:
grep -rn "SharedMemoryBuffer" services/forward/src/
# Consumer is dead code:
grep -n "allow(dead_code)" services/backward/src/tasks/ingest.rs
# Worker intake is an idle poll:
sed -n '/Workers idle-poll/,+3p' services/backward/src/lib.rs
# Store defaults to mock; no Qdrant deployed:
grep -n "QDRANT_USE_MOCK" services/backward/src/persistence/experience_store.rs
grep -rn qdrant docker-compose.yml config/   # → nothing
# Forward computes no GAF; realized PnL only arrives via HTTP POST:
grep -rn "gaf" services/forward/src/         # → nothing
grep -rn "positions/close" services/forward/src/affinity_recorder.rs
```
