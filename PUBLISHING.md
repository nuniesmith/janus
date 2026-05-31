# Publishing JANUS crates to crates.io

This workspace is a **44-member virtual workspace**, not a single crate. The
`scripts/release.sh` step `cargo publish` (run at the root) **cannot work** —
the root manifest has no `[package]`, and crates.io rejects path-only internal
dependencies. Crates must be published **individually, bottom-up in dependency
order**, after their internal dependencies are already live.

This document is the checklist for doing that correctly. As of 2026-05-31 only
**`janus-core`** has been prepped (see "Status" below).

---

## Hard rules crates.io enforces

1. **Every dependency needs a version requirement.** Internal deps declared as
   `{ path = "..." }` (or a workspace dep that is path-based) must become
   `{ path = "...", version = "0.1" }`. The `version` is what users actually
   resolve from crates.io; `path` is only used for local builds.
2. **Each crate needs `description` and `license`** (or `license-file`).
   Inherit from the workspace where possible: `license.workspace = true`.
3. **A `readme` field must point to a file that exists** in the crate dir.
4. **Names are global and permanent.** A published version can be *yanked* but
   never deleted. Pick names deliberately.
5. **Publish bottom-up.** You cannot publish a crate until every crate it
   depends on is already on crates.io at the version it requires.

---

## Name collisions (must rename before publishing)

These crate names are **already taken** on crates.io and cannot be reused. Any
crate keeping these names is unpublishable; any crate depending on them is
blocked until they are renamed (suggest the `janus-` prefix the rest already
use):

| Current name | crates.io owner | Suggested rename |
|---|---|---|
| `common` | a generic "buffett common lib" | `janus-common` |
| `memory` | a HashMap-with-forget crate | `janus-memory` |
| `vision` | computer-vision datasets | `janus-vision` |
| `logic` | propositional-logic crate | `janus-logic` |
| `apalis-redis` | the real apalis project (vendored fork here) | `janus-apalis-redis` |

Renaming means: change `name =` in the crate's `Cargo.toml`, update the
`[workspace.dependencies]` alias + every dependent's reference, and (for a fork
like `apalis-redis`) keep `[lib] name = "apalis_redis"` if you don't want to
touch `use` statements. `fks-proto` and `training` appear free but `fks-proto`
is a reconstructed/vendored schema — confirm you want it public at all.

---

## Per-crate prep checklist

For each crate, before `cargo publish`:

- [ ] Name is unique on crates.io (`cargo search <name>`), renamed if needed.
- [ ] `[package]` inherits `license.workspace = true`, `repository.workspace = true`,
      `authors.workspace = true`, `rust-version.workspace = true`.
- [ ] `description = "..."` added (one sentence).
- [ ] `readme = "README.md"` **and the file exists** (write a short one).
- [ ] Optional but nice: `documentation = "https://docs.rs/<name>"`,
      `keywords` (≤5, ≤20 chars each), `categories` (from the fixed
      [crates.io slug list](https://crates.io/category_slugs)).
- [ ] Every internal dependency has both `path` **and** `version`.
- [ ] `cargo publish --dry-run -p <name>` is clean (no metadata warnings).
- [ ] All internal deps are already published at the required version.

---

## Publish order (bottom-up, from the real dependency DAG)

**Tier 0 — true leaves** (no internal deps; publish in any order):
`janus-core`, `janus-api`, `janus-indicators`, `janus-dsp`, `janus-health`,
`janus-risk`, `janus-models`, `janus-regime`, `janus-rate-limiter`,
`janus-gap-detection`, `janus-compliance`, `janus-ltn`, `janus-bybit-client`,
`janus-questdb-writer`, `janus-cns`, `janus-registry-lib`, `fks-proto`
*(+ `common`, `apalis-redis` once renamed)*.

**Tier 1 — depend only on Tier 0:**
- `janus-strategies` ← `janus-indicators`, `janus-models`, `janus-regime`
- `janus-lob` ← `janus-core`
- `janus-exchanges` ← `janus-cns`, `janus-core`
- `janus-data-quality` ← `janus-cns`, `janus-core`, `janus-gap-detection`
- `janus-execution` ← `fks-proto`, `janus-compliance`, `janus-models`, `janus-questdb-writer`, `janus-risk`
- `janus-cns-service` ← `janus-cns`, `janus-core`
- `janus-registry-service` ← `janus-core`, `janus-registry-lib`
- `logic`/`memory` ← `common` *(rename first)*

**Tier 2:**
- `janus-backtest` ← `janus-compliance`, `janus-indicators`, `janus-models`, `janus-regime`, `janus-risk`, `janus-strategies`
- `janus-ml` ← `janus-core`, `janus-data-quality`
- `janus-data` ← `janus-cns`, `janus-core`, `janus-exchanges`, `janus-gap-detection`, `janus-questdb-writer`, `janus-rate-limiter`
- `vision` ← `common` *(rename first)*

**Tier 3+:**
- `janus-optimizer` ← `janus-backtest`, `janus-core`, `janus-indicators`, `janus-models`, `janus-strategies`
- `janus-backtest-cli` ← `janus-backtest`
- `training` ← `common`, `logic`, `vision` *(renames first)*
- `janus-neuromorphic` ← `common`, `fks-proto`, `memory` *(renames first)*
- `janus-optimizer-service` ← `janus-cns`, `janus-core`, `janus-exchanges`, `janus-indicators`, `janus-optimizer`
- `janus-forward` ← (15 internal crates incl. `logic`)
- `janus-backward` ← `apalis-redis`, `common`, `fks-proto`, `janus-core`, `janus-health`, `janus-ml`, `janus-models`, `memory`

**Top binary:** `janus` ← `janus-backward`, `janus-cns-service`, `janus-data`, `janus-forward`.

> **Binaries vs libraries:** the `services/*` and `bin/*` members are
> applications, not libraries. You usually do **not** publish those to
> crates.io — they're deployed as Docker images (see `docker-publish.yml`).
> Mark them `publish = false` to be safe. Publishing is for the reusable
> `lib/*` and `crates/*` libraries.

---

## Authentication & the actual publish

`cargo publish` needs a crates.io API token (not configured in this repo):

```bash
cargo login                 # paste a token from https://crates.io/settings/tokens
# or, non-interactively / in CI:
export CARGO_REGISTRY_TOKEN=...
```

Then, per crate (from a **clean** committed tree):

```bash
cargo publish -p <crate>    # omit --dry-run to actually upload
# wait for the index to update before publishing a crate that depends on it
```

---

## `scripts/release.sh` — what to fix

The current script (on `main`) does: bump patch → commit → tag → **push to
`main`** → `cargo publish` at the root. Problems:

- It pushes the version bump + tag to `main` **before** publishing, so a failed
  publish (which is guaranteed today) leaves `main` tagged for a release that
  never shipped.
- `cargo publish` at the root targets a virtual manifest → immediate failure
  (`all dependencies must have a version requirement specified`).
- It bases the next version on `git tag` (none exist → `v0.0.0 → v0.0.1`),
  contradicting the `0.1.0` already in `Cargo.toml`.

If you want to keep it as a **git-tag-only** release helper (no crates.io),
drop step 6 (`cargo publish`) and have it bump `[workspace.package] version`,
commit, tag, and push — that part is sound and pairs with the Docker publish
workflow. If you want it to publish libraries, it must iterate the tiers above
in order with a `--dry-run` gate and publish each `-p <crate>` individually.

---

## Status

| Crate | Prepped | Published |
|---|---|---|
| `janus-core` | ✅ metadata + README + clean `--dry-run` (2026-05-31) | ❌ not yet (needs token) |
| everything else | ❌ | ❌ |

To publish `janus-core` (the only ready crate): authenticate, then from a clean
tree run `cargo publish -p janus-core`.
