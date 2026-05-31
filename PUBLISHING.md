# Publishing JANUS crates to crates.io

This workspace is a multi-member **virtual** workspace, not a single crate.
Running `cargo publish` at the root **cannot work** (no root `[package]`, and
crates.io rejects path-only internal dependencies). Crates must be published
**individually, bottom-up in dependency order**.

## The `janus` name is taken — published crates are rebranded `jflow-*`

The `janus` ecosystem name on crates.io is **already owned by unrelated,
established projects**:

| Wanted name | Owner on crates.io | Status |
|---|---|---|
| `janus` | uw-labs (an API gateway), v0.2.0 | 🔴 taken |
| `janus-core` | **divviup** — the ISRG/DAP "Janus" privacy project, v0.7.x | 🔴 taken |

A `cargo publish -p janus-core` fails with:

```
error: failed to publish ... (status 403 Forbidden): this crate exists but you
don't seem to be an owner.
```

The **other 27 `janus-*` names we checked are free**, but to avoid living in a
contested namespace (and being confused with the DAP project), the published
crates are **rebranded under the `jflow-*` prefix** — matching the `jflow`
naming already used internally. `jflow-*` names were confirmed free on
crates.io.

### How the rebrand works (zero source-code churn)

We change each crate's **published** name without touching any Rust source or
the internal dependency *names*, using Cargo's dependency renaming:

1. In the crate, set `[package] name = "jflow-<x>"`. The library target then
   builds as `jflow_<x>` — clean for external consumers (`use jflow_core;`).
2. Everywhere that crate is depended on, keep the dependency **key** as
   `janus-<x>` and add `package = "jflow-<x>"`. The dependent still refers to it
   as `janus_<x>` in code, so **no `use` statements change**:

   ```toml
   # root [workspace.dependencies] — covers every `janus-core.workspace = true`:
   janus-core = { path = "lib/janus-core", package = "jflow-core" }

   # a direct path dep (there are ~56 of these across the workspace):
   janus-core = { path = "../../lib/janus-core", package = "jflow-core" }
   ```

> ⚠️ Internal deps are declared **two ways** here: ~26 via
> `janus-x = { workspace = true }` (fix once in the root manifest) and **~56 via
> direct `path = ...`** (each needs `package = "jflow-x"` added individually). A
> full-workspace `cargo build` will fail loudly on any site you miss, so it is
> self-verifying.

The first crate, **`jflow-core`** (`lib/janus-core`), has already been rebranded
this way and is publish-ready (see Status). Replicate the pattern for the rest.

### Rename mapping (published name)

All publishable libraries map `janus-<x>` → `jflow-<x>` (e.g. `janus-indicators`
→ `jflow-indicators`). The crates that **don't** use the `janus-` prefix are
**also taken on crates.io** and must be rebranded too:

| Current crate | Publish as |
|---|---|
| `common` | `jflow-common` |
| `memory` | `jflow-memory` |
| `vision` | `jflow-vision` |
| `logic` | `jflow-logic` |
| `training` | `jflow-training` |
| `apalis-redis` (vendored fork) | keep `publish = false`, or `jflow-apalis-redis` |
| `fks-proto` (vendored schema) | keep `publish = false`, or `jflow-proto` |

(Verify each `jflow-*` target is still free with `cargo search` before relying
on it — names are first-come and permanent.)

---

## Hard rules crates.io enforces

1. **Every internal dependency needs a `version`,** not just a `path`. Add
   `version = "0.1.0"` alongside the `path`/`package` on each internal dep
   before publishing that crate (only required at publish time, not for local
   builds).
2. **Each crate needs `description` and `license`.** Inherit where possible:
   `license.workspace = true`.
3. **`readme` must point to a file that exists** in the crate dir.
4. **Names are global and permanent** — a version can be *yanked*, never deleted.
5. **Publish bottom-up** — a crate can't be published until every crate it
   depends on is already live at the required version.

## Per-crate prep checklist

- [ ] `[package] name = "jflow-<x>"`; confirm free (`cargo search jflow-<x>`).
- [ ] Add `package = "jflow-<x>"` to **every** dependency site (root workspace
      alias + all direct `path` deps). `cargo metadata` must resolve.
- [ ] `[package]` inherits `license.workspace`/`repository.workspace`/
      `authors.workspace`/`rust-version.workspace`.
- [ ] `description = "..."`, `readme = "README.md"` (+ the file), optional
      `documentation = "https://docs.rs/jflow-<x>"`, `keywords`, `categories`.
- [ ] Every internal dep also has `version = "0.1.0"`.
- [ ] `cargo publish --dry-run -p jflow-<x>` is clean (no metadata warnings).
- [ ] All internal deps already published.

---

## Publish order (bottom-up, from the real dependency DAG)

Names below are the **current** package names; publish each as its `jflow-*`
rename.

**Tier 0 — true leaves** (no internal deps): `janus-core` ✅(→`jflow-core`),
`janus-api`, `janus-indicators`, `janus-dsp`, `janus-health`, `janus-risk`,
`janus-models`, `janus-regime`, `janus-rate-limiter`, `janus-gap-detection`,
`janus-compliance`, `janus-ltn`, `janus-bybit-client`, `janus-questdb-writer`,
`janus-cns`, `janus-registry-lib`, `common` *(+ `apalis-redis`, `fks-proto` if
publishing the vendored crates)*.

**Tier 1:** `janus-strategies` (←indicators, models, regime); `janus-lob` (←core);
`janus-exchanges` (←cns, core); `janus-data-quality` (←cns, core, gap-detection);
`janus-execution` (←fks-proto, compliance, models, questdb-writer, risk);
`logic`/`memory` (←common).

**Tier 2:** `janus-backtest` (←compliance, indicators, models, regime, risk,
strategies); `janus-ml` (←core, data-quality); `janus-data` (←cns, core,
exchanges, gap-detection, questdb-writer, rate-limiter); `vision` (←common).

**Tier 3+:** `janus-optimizer` (←backtest, core, indicators, models, strategies);
`training` (←common, logic, vision); `janus-neuromorphic` (←common, fks-proto,
memory); then the services.

**Binaries / services — do NOT publish** (deployed as Docker images): `janus`
(already `publish = false`), `janus-backtest-cli`, and the `services/*` members
(`janus-gateway`, `janus-forward`, `janus-backward`, `janus-cns-service`,
`janus-data`, `janus-optimizer-service`, `janus-registry-service`,
`janus-execution`). Add `publish = false` to each before any bulk publish run so
a stray `cargo publish` can't try to upload an application crate.

---

## Authentication & the actual publish

```bash
cargo login                 # paste a token from https://crates.io/settings/tokens
# or in CI:  export CARGO_REGISTRY_TOKEN=...

cargo publish -p jflow-core              # from a CLEAN committed tree
# wait for the index to update before publishing a crate that depends on it
```

## `scripts/release.sh`

Fixed: it now bumps off `Cargo.toml` (not absent git tags), runs
`cargo update --workspace` so `Cargo.lock` stays valid for `--locked` builds,
and no longer attempts the impossible root `cargo publish`. It is a **git-tag**
release helper only; crates.io publishing is the manual, per-crate process above.

---

## Status

| Crate | Prepped | Published |
|---|---|---|
| `jflow-core` (`lib/janus-core`) | ✅ renamed + metadata + README + clean `--dry-run` | ❌ not yet (needs token) |
| everything else | ❌ (rename pending) | ❌ |

`jflow-core` is ready: authenticate, then from a clean tree run
`cargo publish -p jflow-core`.
