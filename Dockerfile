# syntax=docker/dockerfile:1.7
#
# fks-janus standalone Dockerfile
#
# Builds the unified `janus` binary (bin/janus) which supervises the
# forward / backward / cns / api / data modules in-process.
#
# The workspace declares `fks-proto = { path = "../../src/proto" }` —
# i.e. it normally lives nested inside the parent fks repo at
# `fks/src/janus`. To keep this Dockerfile self-contained we shallow-clone
# the proto crate from the parent repo into the position the workspace
# expects. Pin the ref via --build-arg FKS_PROTO_REF=<sha|tag|branch>.
#
# Build:
#   docker build -t fks-janus:dev .
#   docker build -t fks-janus:dev --build-arg FKS_PROTO_REF=v0.4.0 .
#
# Run (with companion services from docker-compose.yml):
#   docker compose up -d

ARG RUST_VERSION=1.92.0
ARG DEBIAN_RELEASE=bookworm
ARG FKS_PROTO_REPO=https://github.com/nuniesmith/fks.git
ARG FKS_PROTO_REF=main

# ─────────────────────────────────────────────────────────────
# Stage 1 — fetch fks-proto from the parent repo
# ─────────────────────────────────────────────────────────────
FROM debian:${DEBIAN_RELEASE}-slim AS proto-fetch
ARG FKS_PROTO_REPO
ARG FKS_PROTO_REF
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /fks
RUN git init -q . \
 && git remote add origin "${FKS_PROTO_REPO}" \
 && git fetch --depth=1 origin "${FKS_PROTO_REF}" \
 && git checkout FETCH_HEAD -- src/proto

# ─────────────────────────────────────────────────────────────
# Stage 2 — build the janus workspace
# ─────────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS builder

# Build deps: protoc for tonic build scripts, pkg-config + openssl for
# crates that pull in native TLS, cmake for a few transitive deps.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        cmake \
        pkg-config \
        libssl-dev \
        clang \
 && rm -rf /var/lib/apt/lists/*

# Position the working copy so `../../src/proto` from the workspace
# resolves to the proto crate fetched in stage 1.
WORKDIR /fks/src
COPY --from=proto-fetch /fks/src/proto ./proto

WORKDIR /fks/src/janus
COPY . .

# Build only the `janus` binary in release mode. Use BuildKit cache mounts
# for the cargo registry/git db and the target dir so iterative rebuilds
# are fast for developers.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/fks/src/janus/target \
    cargo build --release --locked -p janus \
 && cp target/release/janus /usr/local/bin/janus

# ─────────────────────────────────────────────────────────────
# Stage 3 — slim runtime image
# ─────────────────────────────────────────────────────────────
FROM debian:${DEBIAN_RELEASE}-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 janus \
 && useradd  --system --uid 10001 --gid janus --home /opt/janus --shell /usr/sbin/nologin janus \
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
