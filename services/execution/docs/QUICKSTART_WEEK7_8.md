# 🚀 Quick Start Guide - Week 7-8

Get the Execution Service running in 5 minutes!

## Prerequisites

- Rust 1.75+ (`rustup update`)
- Docker & Docker Compose (optional, for containerized deployment)
- Bybit testnet account (optional, for integration testing)

---

## 1️⃣ Local Development (No Docker)

### Build & Test

```bash
cd fks/src/execution

# Run unit tests (no external dependencies needed)
cargo test --lib

# Build release binary
cargo build --release

# Binary location:
# ./target/release/execution-service
```

### Run the Service

```bash
# Set environment variables
export RUST_LOG=info,execution=debug
export GRPC_PORT=50052
export HTTP_PORT=8081

# Run (note: you'll need Redis and QuestDB running separately)
cargo run --release
```

---

## 2️⃣ Docker Compose (Recommended)

### One-Command Start

```bash
cd fks/src/execution

# Start all services (Execution + Redis + QuestDB + Prometheus + Grafana)
docker-compose up -d

# View logs
docker-compose logs -f execution-service

# Stop everything
docker-compose down
```

### Access Services

- **Execution gRPC**: `localhost:50052`
- **Execution HTTP/Metrics**: `http://localhost:8081/metrics`
- **QuestDB Console**: `http://localhost:9000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

---

## 3️⃣ Integration Testing with Bybit Testnet

### Get Testnet Credentials

1. Visit https://testnet.bybit.com/
2. Sign up for testnet account
3. Create API keys (read + trade permissions)

### Configure

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your testnet credentials:
# BYBIT_TESTNET_API_KEY=your_key_here
# BYBIT_TESTNET_API_SECRET=your_secret_here
# RUN_INTEGRATION_TESTS=1
```

### Run Integration Tests

```bash
# Load env vars
source .env

# Run integration tests
cargo test --test integration -- --ignored --nocapture

# Run specific test
cargo test --test integration test_single_limit_order_bybit -- --ignored --nocapture
```

---

## 4️⃣ Kubernetes (Local with Minikube)

### Setup Minikube

```bash
# Install minikube (if not already installed)
# See: https://minikube.sigs.k8s.io/docs/start/

# Start cluster
minikube start

# Enable metrics
minikube addons enable metrics-server
```

### Build & Load Image

```bash
cd fks/src/execution

# Build Docker image
docker build -t execution-service:latest .

# Load into minikube
minikube image load execution-service:latest
```

### Deploy

```bash
# Create namespace
kubectl apply -f deploy/k8s/namespace.yaml

# Configure secrets (edit first!)
# Edit deploy/k8s/secrets.yaml with your API keys
kubectl apply -f deploy/k8s/secrets.yaml

# Deploy config and service
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml

# Check status
kubectl get pods -n execution
kubectl get svc -n execution

# View logs
kubectl logs -f deployment/execution-service -n execution

# Port-forward to access locally
kubectl port-forward svc/execution-service 50052:50052 -n execution
```

---

## 5️⃣ Verify Everything Works

### Test gRPC Endpoint

```bash
# Install grpcurl (if not already installed)
# https://github.com/fullstorydev/grpcurl

# List services
grpcurl -plaintext localhost:50052 list

# Example: Submit order
grpcurl -plaintext -d '{
  "symbol": "BTCUSD",
  "side": "BUY",
  "order_type": "LIMIT",
  "quantity": "0.001",
  "price": "30000"
}' localhost:50052 execution.ExecutionService/SubmitOrder
```

### Check HTTP Endpoints

```bash
# Health check
curl http://localhost:8081/health

# Metrics
curl http://localhost:8081/metrics

# Version
curl http://localhost:8081/version
```

### Check QuestDB

Visit `http://localhost:9000` and run:

```sql
SELECT * FROM orders ORDER BY timestamp DESC LIMIT 10;
SELECT * FROM positions;
SELECT * FROM fills;
```

---

## 📊 Monitoring

### Prometheus Queries

Visit `http://localhost:9090` and try:

```promql
# Total orders
execution_orders_total

# Order latency (p95)
histogram_quantile(0.95, rate(execution_order_latency_seconds_bucket[5m]))

# WebSocket reconnections
rate(execution_websocket_reconnects_total[1m])
```

### Grafana Dashboards

1. Visit `http://localhost:3000` (admin/admin)
2. Add Prometheus data source: `http://prometheus:9090`
3. Import dashboard (ID: 1860 for Node Exporter, or create custom)

---

## 🐛 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs execution-service

# Check if ports are in use
lsof -i :50052
lsof -i :8081

# Restart services
docker-compose restart
```

### Tests failing

```bash
# Ensure dependencies are up
cargo clean
cargo build

# Check Redis is running
redis-cli ping

# Check QuestDB is running
curl http://localhost:9000
```

### Kubernetes pods not starting

```bash
# Check pod status
kubectl describe pod -n execution

# Check logs
kubectl logs -n execution -l app=execution-service

# Check image pull
minikube image ls | grep execution-service
```

---

## 🎯 Next Steps

1. **Run integration tests** with Bybit testnet
2. **Submit your first order** via gRPC
3. **Monitor in Grafana** - watch orders flow through
4. **Load test** - see how it handles 100+ orders/sec
5. **Read full docs** - `WEEK7_8_INTEGRATION_DEPLOYMENT.md`

---

## 📚 Quick Reference

### Useful Commands

```bash
# Build
cargo build --release

# Test
cargo test --lib                          # Unit tests
cargo test --test integration -- --ignored  # Integration tests

# Docker
docker-compose up -d                      # Start all
docker-compose down                       # Stop all
docker-compose logs -f execution-service  # View logs

# Kubernetes
kubectl get pods -n execution             # List pods
kubectl logs -f deployment/execution-service -n execution  # Logs
kubectl port-forward svc/execution-service 50052:50052 -n execution  # Access

# Monitoring
curl http://localhost:8081/metrics        # Prometheus metrics
curl http://localhost:8081/health         # Health check
```

### Project Structure

```
fks/src/execution/
├── src/
│   ├── api/           # gRPC & HTTP APIs
│   ├── exchanges/     # Exchange integrations (Bybit)
│   ├── orders/        # Order management
│   ├── positions/     # Position tracking & P&L
│   ├── strategies/    # TWAP, VWAP, Iceberg
│   └── lib.rs
├── tests/
│   └── integration/   # Integration test scenarios
├── deploy/
│   ├── k8s/          # Kubernetes manifests
│   └── prometheus.yml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

**Happy Coding! 🚀**
