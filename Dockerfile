# syntax=docker/dockerfile:1.7
#
# fks-janus standalone Dockerfile
#
# The workspace is now self-contained — `fks-proto` lives at
# `crates/fks-proto` and the binary `janus` (under `bin/janus`) is the
# unified supervisor entry point. No parent-repo clone is needed.
#
# Build:
#   docker build -t fks-janus:dev .
#
# Run (with the companion services from docker-compose.yml):
#   docker compose up -d

# 1.94.1 is required by exchange-apiws 0.4.0 (MSRV), pulled in via janus-forward.
ARG RUST_VERSION=1.94.1
ARG DEBIAN_RELEASE=bookworm

# ─────────────────────────────────────────────────────────────
# Stage 1 — build the janus workspace
# ─────────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS builder

# Build deps: protoc for tonic build scripts, pkg-config + openssl for crates
# that pull in native TLS, cmake for a few transitive deps.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        cmake \
        pkg-config \
        libssl-dev \
        clang \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# Build only the `janus` binary in release mode. BuildKit cache mounts make
# iterative rebuilds fast for developers; the resulting binary is copied out
# of the cached target dir into a fixed path before stage 2 picks it up.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked -p janus \
 && cp target/release/janus /usr/local/bin/janus

# ─────────────────────────────────────────────────────────────
# Stage 2 — slim runtime image
# ─────────────────────────────────────────────────────────────
FROM debian:${DEBIAN_RELEASE}-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1000 janus \
 && useradd  --system --uid 1000 --gid janus --home /opt/janus --shell /usr/sbin/nologin janus \
 && mkdir -p /opt/janus/logs /opt/janus/checkpoints /opt/janus/config \
 && chown -R janus:janus /opt/janus

WORKDIR /opt/janus

COPY --from=builder /usr/local/bin/janus /usr/local/bin/janus
COPY --chown=janus:janus config/janus.toml config/janus.toml

USER janus

ENV JANUS_CONFIG_PATH=/opt/janus/config/janus.toml \
    JANUS_HOST=0.0.0.0 \
    JANUS_HTTP_PORT=8080 \
    JANUS_GRPC_PORT=50051 \
    JANUS_WS_PORT=8081 \
    JANUS_METRICS_PORT=9090 \
    RUST_LOG=info

EXPOSE 8080 8081 9090 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${JANUS_HTTP_PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/janus"]
