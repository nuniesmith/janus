# Risk Topology — Forward (Live) Path

> **Why this doc exists.** A task arrived framed as *"unify the live trading path
> onto `logic::ComprehensiveRiskEngine`."* That framing was wrong in three ways,
> and untangling it cost real time. This note records the **actual** risk
> topology of the forward service so the next person doesn't re-derive it from
> scratch. Captured after PRs #49 / #50 / #51 (Tracks A1, A2, B).

---

## TL;DR

- The forward service has **exactly one** production signal → execution loop
  (in `services/forward/src/lib.rs`). The "other" loop (`event_loop.rs` + the
  `actors/` `StrategyActor` path) was **dead code** — never compiled into the
  binary — and was deleted in Track B (#51).
- Every execution submit funnels through a **single choke point**:
  [`SignalGenerator::submit_signal_to_execution`](../../services/forward/src/signal/mod.rs)
  (`signal/mod.rs:366`). The brain-gated path, the direct path, **and** the
  signal-bus consumer path all call it.
- Two **independent, default-off-but-enforceable** guards protect that choke
  point. Both default to *advisory* (log + metadata, never block) to preserve
  the platform's **no autonomous execution** principle; each is opt-in via its
  own env flag.
- `logic::ComprehensiveRiskEngine` is a **stateless library validator with no
  production consumer**. It is **not** on the live path and never was.

---

## The choke point

```
 live loop (lib.rs)  ─┐
 signal-bus consumer ─┼─►  submit_signal_to_execution()   ← Guard 2: kill-switch
 (direct callers)    ─┘            (signal/mod.rs:366)        (is_killed_cached)
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                            ▼
                   brain-gated client            direct execution client
                          │                            │
                          ▼                            ▼
                                  broker / exchange
```

Guard 1 (per-entry risk) runs **upstream**, in the live loop, and decides
whether to *call* the choke point at all. Guard 2 (kill-switch) runs **inside**
the choke point, so it covers callers that bypass the live loop's own checks
(notably the bus consumer). Defense in depth.

---

## Guard 1 — per-entry risk enforcement (Track A1, #49)

Two symmetric, opt-in gates evaluated per prospective entry in the live loop:

| Gate | Validator | Env flag | Default | Source |
|------|-----------|----------|---------|--------|
| Prop-firm rules | `PropFirmValidator` | `JANUS_PROP_FIRM_ENFORCE` | advisory | `lib.rs:1032`, `:1038` |
| Portfolio risk | `RiskManager` verdict | `JANUS_RISK_ENFORCE` | advisory | `lib.rs:1055` |

**Behaviour** (`lib.rs:1575`–`1596`): for an actionable entry (`final_type !=
Hold && avg_confidence >= 0.7`), if the relevant flag is enabled **and** the
verdict is adverse (`prop_firm_label` starts with `"violation"`, or
`risk_check` starts with `"rejected"`), the loop logs a `risk ENFORCE: blocking`
warning and **skips the execution submit only**. The signal is *still published
to the bus* above that point for observability. So enforcement removes the
*order*, never the *signal* — consistent with "no autonomous execution."

With both flags unset (the default), behaviour is identical to before #49:
verdicts are logged and attached as signal metadata (`prop_firm`, `risk_check`),
nothing is blocked.

---

## Guard 2 — Redis kill-switch (Track A2, #50)

Type: [`RedisKillSwitch`](../../services/forward/src/persistence/kill_switch_redis.rs)
(`persistence/kill_switch_redis.rs`). A cross-process emergency stop, checked at
the choke point (`signal/mod.rs:372`) via `is_killed_cached()`:

```rust
if let Some(ks) = &self.kill_switch
    && ks.is_killed_cached().await
{
    // suppress execution submit for this signal
    return Ok(());
}
```

Because the check sits in `submit_signal_to_execution`, it halts **all** submit
paths — brain-gated, direct, and the bus consumer — independently of whether a
brain is attached.

**Semantics (read these before relying on it):**

- **Cross-process.** State is a Redis key (`"1"` = tripped). Any process can
  trip it; all foward instances observe it within their poll interval.
- **Hot path is cache-only.** `is_killed_cached()` reads a local `RwLock<bool>`,
  never Redis — no round-trip latency on the execution path. A background task
  (`spawn_sync_task`, interval `BRAIN_KILL_SWITCH_POLL_MS`) polls Redis and
  updates that cache.
- **Startup requires Redis.** `RedisKillSwitch::new()` returns `Err` if Redis
  is unreachable/times out; the initial cache state is seeded from the key.
- **Last-known-state wins on errors.** If polling fails mid-run, the cache is
  **not** reset — a *tripped* switch stays tripped across a Redis outage (it
  never auto-clears on error). This is deliberately *not* a blanket
  "fail-closed": an untripped switch with a flaky poll stays untripped. Treat it
  as "the switch latches; outages don't silently release it."

**Relationship to the in-brain switch.** There is a separate, *local*
`PipelineKillSwitch` (a synchronous stage inside the brain pipeline,
`brain_runtime.rs` / `brain_wiring.rs`). `RedisKillSwitch::sync_once` pushes the
cross-process state *into* that local switch for the brain-gated path. Track A2
additionally wired the cache check at the shared choke point so the non-brain
paths are covered too — the two are complementary, not duplicates.

---

## What is **not** on the live path

- **`logic::ComprehensiveRiskEngine`** (`crates/logic/src/risk_engine.rs:45`) —
  a stateless validator. Grepping the tree, it has **no consumer outside
  `crates/logic`'s own tests**. The forward service does **not** depend on
  `crates/logic` at all (Track B removed the orphaned `jflow-logic` dep). The
  crate is retained only because `crates/training` references it. Do not wire
  this into the live path expecting it to be "the" risk engine — the live
  guards above are `PropFirmValidator` + `RiskManager`.
- **`event_loop.rs` + `actors/` (`StrategyActor`, etc.)** — deleted in Track B
  (#51, ~4.5k lines). It was never wired into `main.rs`; the live loop is the
  one in `lib.rs`. If you find a reference to it in old notes, it's stale.

---

## Toggle quick-reference

| Env var | Default | Effect when set (`1`/`true`) |
|---------|---------|------------------------------|
| `JANUS_PROP_FIRM_ENFORCE` | advisory | Block execution submit on prop-firm `violation` |
| `JANUS_RISK_ENFORCE` | advisory | Block execution submit on `RiskManager` `rejected` |
| `BRAIN_KILL_SWITCH_POLL_MS` | `DEFAULT_POLL_INTERVAL_MS` | Redis kill-switch poll interval |

Kill-switch activation itself is operational (set the Redis key), not an env
toggle — see `RedisKillSwitch::activate` / `deactivate`.

---

## Provenance

| PR | Track | Change |
|----|-------|--------|
| #49 | A1 | `JANUS_RISK_ENFORCE` — portfolio `RiskManager` verdict made blocking (opt-in) |
| #50 | A2 | Redis kill-switch wired into `submit_signal_to_execution` (covers all paths) |
| #51 | B | Deleted dead `event_loop.rs` + `actors/` path; dropped orphaned `jflow-logic` dep |

The original "unify onto `ComprehensiveRiskEngine`" framing was discarded after
investigation showed (a) the loop it referenced was dead, (b) the engine is an
unwired library validator, and (c) the live loop already had `PropFirmValidator`.
The work above closed the *real* gap: the one live loop went from advisory-only
risk with no emergency stop → an enforceable verdict plus a cross-process
kill-switch, both default-safe.
