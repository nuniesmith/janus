# Week 7-8: Integration Testing & Production Deployment

## Overview

This phase focuses on production readiness through:
- Integration testing with Bybit testnet
- Real fill/callback wiring
- Docker containerization
- Kubernetes deployment
- CI/CD pipeline

## Status: Framework Complete ✅

All infrastructure and testing frameworks are in place. Integration tests are ready to be populated with real testnet scenarios.

---

## 📁 New Files & Directories

### Integration Testing
- `tests/integration/config.rs` - Testnet configuration loader
- `tests/integration/mod.rs` - Integration test module
- `tests/integration/scenarios.rs` - End-to-end test scenarios

### Docker & Deployment
- `Dockerfile` - Multi-stage optimized container
- `docker-compose.yml` - Local development stack
- `deploy/k8s/namespace.yaml` - Kubernetes namespace
- `deploy/k8s/configmap.yaml` - Configuration
- `deploy/k8s/secrets.yaml` - Secrets (template)
- `deploy/k8s/deployment.yaml` - K8s deployment + services
- `deploy/prometheus.yml` - Prometheus scraping config

### CI/CD
- `.github/workflows/ci.yml` - GitHub Actions pipeline

---

## 🧪 Integration Testing

### Configuration

Integration tests load configuration from environment variables:

**Required for Bybit testnet:**
```/dev/null/bash#L1-3
export BYBIT_TESTNET_API_KEY="your_testnet_api_key"
export BYBIT_TESTNET_API_SECRET="your_testnet_api_secret"
export RUN_INTEGRATION_TESTS=1
```

**Optional overrides:**
```/dev/null/bash#L1-7
export BYBIT_TESTNET_REST_URL="https://api-testnet.bybit.com"
export BYBIT_TESTNET_WS_URL="wss://stream-testnet.bybit.com/v5/private"
export REDIS_URL="redis://localhost:6379"
export QUESTDB_HOST="localhost"
export QUESTDB_PORT=9009
export GRPC_PORT=50052
export TEST_TIMEOUT_SECS=30
```

### Running Integration Tests

```/dev/null/bash#L1-10
# Run all unit tests (default)
cargo test --lib

# Run integration tests (requires testnet credentials)
cargo test --test integration -- --ignored --nocapture

# Run specific integration scenario
cargo test --test integration test_single_limit_order_bybit -- --ignored --nocapture

# Run with detailed logging
RUST_LOG=debug cargo test --test integration -- --ignored --nocapture
```

### Test Scenarios Implemented

1. **Single Limit Order** - Submit, verify, cancel
2. **TWAP Execution** - Multi-slice execution with timing
3. **VWAP Execution** - Volume profile-based execution
4. **Iceberg Order** - Hidden quantity management
5. **WebSocket Reconnection** - Connection resilience
6. **Position P&L Tracking** - Accuracy verification
7. **Account Margin Management** - Risk calculation
8. **Load Testing** - 100+ orders/second throughput

*Note: Scenarios are currently placeholders. Real implementations will be added as testnet credentials become available.*

---

## 🐳 Docker Deployment

### Building the Image

```/dev/null/bash#L1-5
# Build locally
docker build -t execution-service:latest .

# Build with specific tag
docker build -t execution-service:v1.0.0 .
```

### Running with Docker Compose

```/dev/null/bash#L1-16
# Start all services (Execution + Redis + QuestDB + Prometheus + Grafana)
docker-compose up -d

# View logs
docker-compose logs -f execution-service

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Access services:
# - Execution gRPC: localhost:50052
# - Execution HTTP/metrics: localhost:8081
# - QuestDB console: http://localhost:9000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Environment Variables for Docker

Create a `.env` file:

```/dev/null/env#L1-10
RUST_LOG=info,execution=debug
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET_API_KEY=your_testnet_key
BYBIT_TESTNET_API_SECRET=your_testnet_secret
REDIS_URL=redis://redis:6379
QUESTDB_HOST=questdb
QUESTDB_PORT=9009
GRPC_PORT=50052
HTTP_PORT=8081
```

Then:
```/dev/null/bash#L1-2
# Use .env file
docker-compose --env-file .env up -d
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (local: minikube, k3s; cloud: GKE, EKS, AKS)
- `kubectl` configured
- Docker image pushed to registry

### Deployment Steps

```/dev/null/bash#L1-22
# 1. Create namespace
kubectl apply -f deploy/k8s/namespace.yaml

# 2. Configure secrets (edit first!)
# Edit deploy/k8s/secrets.yaml with your API keys
kubectl apply -f deploy/k8s/secrets.yaml

# 3. Apply configuration
kubectl apply -f deploy/k8s/configmap.yaml

# 4. Deploy service
kubectl apply -f deploy/k8s/deployment.yaml

# 5. Verify deployment
kubectl get pods -n execution
kubectl get svc -n execution

# 6. Check logs
kubectl logs -f deployment/execution-service -n execution

# 7. Port-forward for local access
kubectl port-forward svc/execution-service 50052:50052 -n execution
```

### Scaling

```/dev/null/bash#L1-5
# Scale to 5 replicas
kubectl scale deployment execution-service --replicas=5 -n execution

# Autoscaling (HPA)
kubectl autoscale deployment execution-service --cpu-percent=70 --min=3 --max=10 -n execution
```

### Rolling Updates

```/dev/null/bash#L1-8
# Update image
kubectl set image deployment/execution-service execution-service=execution-service:v1.1.0 -n execution

# Monitor rollout
kubectl rollout status deployment/execution-service -n execution

# Rollback if needed
kubectl rollout undo deployment/execution-service -n execution
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The pipeline includes:

1. **Lint** - `cargo fmt` and `clippy`
2. **Test** - Unit tests + optional integration tests
3. **Build** - Release binary compilation
4. **Docker** - Build and push to registry (on `main`/`develop`)
5. **Deploy** - Staging (develop) and Production (main)

### Required Secrets

Configure in GitHub Settings → Secrets:

```/dev/null/text#L1-11
# Optional: Enable integration tests in CI
RUN_INTEGRATION_TESTS=true
BYBIT_TESTNET_API_KEY=<your_testnet_key>
BYBIT_TESTNET_API_SECRET=<your_testnet_secret>

# Docker Hub (for image push)
DOCKER_USERNAME=<your_docker_username>
DOCKER_PASSWORD=<your_docker_token>

# Kubernetes (for deployments)
KUBE_CONFIG_STAGING=<base64_encoded_kubeconfig>
KUBE_CONFIG_PRODUCTION=<base64_encoded_kubeconfig>
```

### Workflow Triggers

- **Push to `main`** → Lint, Test, Build, Docker, Deploy to Production
- **Push to `develop`** → Lint, Test, Build, Docker, Deploy to Staging
- **Pull Request** → Lint, Test, Build only

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Execution service exposes metrics at `http://localhost:8081/metrics`

**Key metrics to monitor:**
- `execution_orders_total{status="submitted|filled|cancelled|rejected"}`
- `execution_order_latency_seconds{exchange="bybit"}`
- `execution_position_pnl{symbol="BTCUSD",type="realized|unrealized"}`
- `execution_strategy_fills{strategy="twap|vwap|iceberg"}`
- `execution_websocket_reconnects_total{exchange="bybit"}`
- `execution_rate_limit_hits_total{exchange="bybit"}`

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin)

**Recommended dashboards:**
1. **Order Flow** - Orders/sec, fill rates, latencies
2. **Position Tracking** - Open positions, P&L, margin usage
3. **Exchange Health** - WebSocket status, API latencies, rate limits
4. **Strategy Performance** - TWAP/VWAP/Iceberg metrics
5. **System Health** - CPU, memory, network

---

## 🧪 Load Testing

### Target Performance

- **Throughput**: 100+ orders/second
- **Latency**: 
  - p50 < 10ms (gRPC order submission)
  - p95 < 50ms
  - p99 < 100ms
- **WebSocket**: < 100ms event processing
- **Position updates**: Real-time (< 10ms after fill)

### Load Test Plan

```/dev/null/bash#L1-8
# TODO: Create load test script
# 1. Spin up execution service
# 2. Generate 1000 orders over 10 seconds (100/sec)
# 3. Monitor metrics:
#    - Order submission latency
#    - Fill confirmation latency
#    - Position update latency
#    - Resource usage (CPU, memory)
```

---

## ✅ Production Readiness Checklist

### Core Functionality
- [x] Order management (submit, cancel, amend)
- [x] WebSocket integration (Bybit)
- [x] Position tracking with P&L
- [x] Account balance & margin management
- [x] Advanced strategies (TWAP, VWAP, Iceberg)
- [x] Rate limiting
- [x] Error handling & recovery

### Testing
- [x] Unit tests (117 passing)
- [x] Integration test framework
- [ ] End-to-end testnet scenarios (pending credentials)
- [ ] Load testing (100+ orders/sec)
- [ ] Chaos testing (network failures, exchange downtime)

### Deployment
- [x] Docker containerization
- [x] docker-compose for local dev
- [x] Kubernetes manifests
- [x] CI/CD pipeline
- [ ] Health checks (implement /health and /ready endpoints)
- [ ] Resource limits tuned

### Security
- [ ] Secrets management (Vault integration)
- [ ] mTLS for gRPC
- [ ] API authentication (JWT/OAuth)
- [ ] Rate limiting per client
- [ ] Audit logging

### Observability
- [x] Prometheus metrics (basic)
- [ ] Enhanced metrics (histograms, percentiles)
- [ ] Grafana dashboards
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Log aggregation (ELK/Loki)
- [ ] Alerting rules

### Documentation
- [x] Integration test guide
- [x] Docker deployment guide
- [x] Kubernetes deployment guide
- [x] CI/CD setup
- [ ] Runbooks (incident response)
- [ ] API documentation (gRPC + HTTP)

---

## 🚀 Next Steps

### Immediate (Week 7)
1. **Get Bybit testnet credentials** and run integration tests
2. **Implement health/readiness endpoints** for K8s probes
3. **Run load tests** to identify bottlenecks
4. **Create Grafana dashboards** for monitoring

### Short-term (Week 8)
1. **Vault integration** for secrets management
2. **Enhanced Prometheus metrics** (histograms, percentiles)
3. **Distributed tracing** with OpenTelemetry
4. **Alerting rules** for critical failures

### Medium-term (Week 9-10)
1. **Multi-exchange support** (Binance, etc.)
2. **Advanced strategies** (POV, Implementation Shortfall)
3. **Strategy backtesting** framework
4. **Production deployment** to cloud (GCP/AWS/Azure)

---

## 📝 Integration Test Example

Here's how to add a real testnet scenario:

```/dev/null/path.rs#L1-50
#[tokio::test]
#[ignore]
async fn test_single_limit_order_bybit() {
    // Load config
    let config = TestnetConfig::from_env().expect("Config failed");
    if !config.bybit.enabled {
        eprintln!("Bybit not configured, skipping");
        return;
    }
    
    // Create exchange client
    let bybit = BybitClient::new(
        config.bybit.api_key,
        config.bybit.api_secret,
        config.bybit.rest_url,
    );
    
    // Submit limit order
    let order = Order {
        symbol: "BTCUSD".to_string(),
        side: Side::Buy,
        order_type: OrderType::Limit,
        quantity: Decimal::from_str("0.001").unwrap(),
        price: Some(Decimal::from_str("30000").unwrap()),
        time_in_force: Some(TimeInForce::GTC),
        ..Default::default()
    };
    
    let result = bybit.submit_order(order).await;
    assert!(result.is_ok(), "Order submission failed: {:?}", result.err());
    
    let order_id = result.unwrap();
    println!("✓ Order submitted: {}", order_id);
    
    // Wait for exchange confirmation
    sleep(Duration::from_secs(2)).await;
    
    // Query order status
    let status = bybit.get_order(&order_id).await;
    assert!(status.is_ok());
    println!("✓ Order status: {:?}", status.unwrap());
    
    // Cancel order
    let cancel_result = bybit.cancel_order(&order_id).await;
    assert!(cancel_result.is_ok(), "Cancel failed");
    println!("✓ Order cancelled");
    
    // Verify final status
    sleep(Duration::from_secs(1)).await;
    let final_status = bybit.get_order(&order_id).await;
    assert!(final_status.is_ok());
    assert_eq!(final_status.unwrap().status, OrderStatus::Cancelled);
    println!("✓ Test complete");
}
```

---

## 🎯 Success Criteria

### Week 7
- [ ] Integration tests run successfully on Bybit testnet
- [ ] Docker containers build and run locally
- [ ] Kubernetes deployment successful (minikube/local cluster)
- [ ] CI/CD pipeline passes all stages

### Week 8
- [ ] Load test achieves 100+ orders/sec
- [ ] Prometheus metrics captured and visualized
- [ ] Grafana dashboards created
- [ ] Health checks passing in K8s
- [ ] Documentation complete

### Week 9-10 (Production)
- [ ] Secrets managed via Vault
- [ ] Production cluster deployed
- [ ] Monitoring and alerting operational
- [ ] Runbooks written
- [ ] Production traffic validated

---

## 📚 References

- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **GitHub Actions**: https://docs.github.com/en/actions
- **Bybit Testnet**: https://testnet.bybit.com/

---

**Week 7-8 Framework Complete! Ready for testnet integration.**
