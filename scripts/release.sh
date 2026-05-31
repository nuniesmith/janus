#!/usr/bin/env bash
# release.sh — bump the workspace patch version, sync the lockfile, commit, tag,
# and push.
#
# This is a GIT-TAG release helper. It deliberately does NOT run `cargo publish`:
# JANUS is a multi-crate *virtual* workspace, so `cargo publish` at the root
# fails ("all dependencies must have a version requirement specified"). Crates
# must be published individually, bottom-up in dependency order — see
# PUBLISHING.md for the procedure and ordering.
#
# Usage: ./release.sh [--dry-run]
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "==> DRY RUN — no changes will be made"
fi

run() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

# ── 1. Make sure the working tree is clean ────────────────────────────────────
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is dirty — commit or stash changes first"
    exit 1
fi

# ── 2. Read the CURRENT version from Cargo.toml ───────────────────────────────
CARGO="Cargo.toml"
if [[ ! -f "$CARGO" ]]; then
    echo "ERROR: $CARGO not found — run this script from the workspace root"
    exit 1
fi

# The source of truth is the version already in Cargo.toml — NOT git tags.
# Deriving the version from tags regresses it when no tags exist yet
# (v0.0.0 -> v0.0.1), which is how 0.1.0 got clobbered to 0.0.2.
CURRENT_VERSION=$(grep -E '^version\s*=' "$CARGO" | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
if [[ -z "$CURRENT_VERSION" ]]; then
    echo "ERROR: could not parse '[workspace.package] version' from $CARGO"
    exit 1
fi

MAJOR="${CURRENT_VERSION%%.*}"
REST="${CURRENT_VERSION#*.}"
MINOR="${REST%%.*}"
PATCH="${REST##*.}"
NEW_VERSION="${MAJOR}.${MINOR}.$(( PATCH + 1 ))"
NEW_TAG="v${NEW_VERSION}"
echo "==> Version: $CURRENT_VERSION -> $NEW_VERSION  (tag: $NEW_TAG)"

if git rev-parse -q --verify "refs/tags/${NEW_TAG}" >/dev/null; then
    echo "ERROR: tag $NEW_TAG already exists"
    exit 1
fi

# ── 3. Bump Cargo.toml and sync Cargo.lock ────────────────────────────────────
if ! $DRY_RUN; then
    sed -i "0,/^\(version\s*=\s*\)\"[^\"]*\"/s//\1\"${NEW_VERSION}\"/" "$CARGO"
    # Re-pin the workspace crates in Cargo.lock to the new version. Without this,
    # `cargo build --locked` (used by the Docker build and CI) fails because the
    # lockfile still references the old version. --workspace touches ONLY
    # workspace members, never third-party dependency versions.
    cargo update --workspace
fi

# ── 4. Commit and tag (Cargo.toml + Cargo.lock together) ──────────────────────
run git add Cargo.toml Cargo.lock
run git commit -m "chore: bump version to ${NEW_VERSION}"
run git tag -a "$NEW_TAG" -m "Release ${NEW_TAG}"

# ── 5. Push commit and tag ────────────────────────────────────────────────────
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "==> Pushing branch '$BRANCH' and tag '$NEW_TAG'"
run git push origin "$BRANCH"
run git push origin "$NEW_TAG"

# ── 6. Publishing to crates.io is intentionally NOT done here ──────────────────
# Root `cargo publish` cannot work on this virtual workspace. Publish the
# reusable libraries individually, bottom-up in dependency order, following
# PUBLISHING.md (renaming the crates that collide with existing crates.io names
# first).
echo ""
echo "✓ Released ${NEW_TAG} (git tag only)"
echo "  To publish crates to crates.io, follow PUBLISHING.md."
