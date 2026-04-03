# 🚀 FKS Data Service: 10-Week Development Plan

**Service**: Market Data Ingestion Service  
**Goal**: Production-ready standalone microservice for high-throughput cryptocurrency data ingestion  
**Timeline**: Weeks 1-10  
**Current Status**: Foundation exists, needs production hardening  

---

## 📊 Current State Assessment

### ✅ What Exists
- Actor-based architecture (Router, WebSocket, Poller)
- Exchange connectors (Binance, Bybit, Kucoin)
- Alternative metrics (Fear & Greed, ETF flows, DVOL)
- QuestDB ILP writer
- Redis state management
- Basic backfill support
- Docker Compose configuration
- **50/53 tests passing** (3 failing: throttle, metrics export)

### ❌ What's Missing
- Production-ready error handling (Circuit Breaker)
- Backfill throttling & orchestration
- Comprehensive metrics export
- Monitoring dashboards
- Integration tests
- Load testing infrastructure
- CI/CD pipeline
- Production deployment manifests (Kubernetes)
- Health check endpoints
- Gap detection automation
- Advanced observability

---

## 🎯 10-Week Roadmap

### **Week 1: Core Reliability & Error Handling**
**Focus**: Circuit breakers, retry logic, graceful degradation

**Deliverables**:
- ✅ Circuit Breaker implementation (rate-limiter crate)
- ✅ Enhanced error handling across all connectors
- ✅ Exponential backoff with jitter
- ✅ Graceful shutdown handling
- ✅ Connection pool management

**Files to Create/Modify**:
- `src/janus/crates/rate-limiter/src/circuit_breaker.rs`
- `src/connectors/{binance,bybit,kucoin}.rs` (error handling)
- `src/actors/websocket.rs` (reconnection logic)

**Tests**: Circuit breaker unit tests, failover integration tests

**Success Criteria**:
- Circuit opens after 5 consecutive failures
- Automatic recovery after cooldown
- No data loss during exchange outages
- All tests passing

---

### **Week 2: Backfill Orchestration**
**Focus**: Distributed backfill coordination, throttling, resource management

**Deliverables**:
- ✅ Backfill throttling (concurrent limit, batch size)
- ✅ Disk usage monitoring
- ✅ Automatic backfill scheduler
- ✅ Gap detection automation
- ✅ Backfill priority queue

**Files to Create/Modify**:
- `src/backfill/throttle.rs` (fix failing tests)
- `src/backfill/scheduler.rs`
- `src/backfill/disk_monitor.rs`
- `src/backfill/priority_queue.rs`

**Tests**: Concurrent backfill tests, disk monitoring tests

**Success Criteria**:
- Max 2 concurrent backfills enforced
- Backfill stops at 90% disk usage
- Gaps auto-detected and queued
- Priority-based backfill execution

---

### **Week 3: Observability & Monitoring**
**Focus**: Prometheus metrics, Grafana dashboards, alerting

**Deliverables**:
- ✅ Prometheus metrics exporter (fix failing test)
- ✅ `/metrics` HTTP endpoint with authentication
- ✅ Grafana dashboard templates
- ✅ Alerting rules (Prometheus Alertmanager)
- ✅ Structured logging with correlation IDs

**Files to Create**:
- `src/metrics/prometheus_exporter.rs` (fix test)
- `src/api/metrics.rs` (HTTP endpoint)
- `monitoring/dashboards/overview.json`
- `monitoring/dashboards/rate-limiter.json`
- `monitoring/dashboards/gap-detection.json`
- `monitoring/alerts/data-service.yml`

**Metrics to Export**:
- Trades ingested/sec by exchange
- WebSocket latency (P50, P95, P99)
- Gap detection rate
- Backfill queue depth
- Circuit breaker state
- Redis connection pool stats

**Success Criteria**:
- Prometheus scrapes `/metrics` endpoint
- Grafana dashboards visualize all metrics
- Alerts fire on SLO violations
- Logs include correlation IDs

---

### **Week 4: API & Integration Layer**
**Focus**: gRPC/HTTP APIs for external consumption, Python UMAP integration

**Deliverables**:
- ✅ gRPC service definition
- ✅ Data query API (historical trades, candles)
- ✅ Real-time WebSocket API (pub/sub)
- ✅ Python UMAP service integration
- ✅ API authentication (JWT/API keys)
- ✅ Rate limiting on API endpoints

**Files to Create**:
- `proto/data_service.proto`
- `src/api/grpc.rs`
- `src/api/query.rs`
- `src/api/stream.rs`
- `src/api/auth.rs`

**APIs**:
```protobuf
service DataService {
  rpc GetTrades(TradesRequest) returns (stream Trade);
  rpc GetCandles(CandlesRequest) returns (stream Candle);
  rpc SubscribeMarketData(SubscribeRequest) returns (stream MarketDataUpdate);
  rpc GetMarketMetrics(MetricsRequest) returns (MetricsResponse);
}
```

**Success Criteria**:
- gRPC API functional with authentication
- WebSocket pub/sub working
- Python UMAP can query historical data
- API rate limiting enforced

---

### **Week 5: WebSocket Enhancements**
**Focus**: Advanced WebSocket features, multi-exchange aggregation

**Deliverables**:
- ✅ WebSocket compression (gzip/deflate)
- ✅ Heartbeat/ping-pong mechanism
- ✅ Multi-exchange aggregation
- ✅ Order book snapshots + deltas
- ✅ Funding rate ingestion

**Files to Create/Modify**:
- `src/actors/websocket.rs` (compression)
- `src/connectors/orderbook.rs`
- `src/connectors/funding_rates.rs`
- `src/actors/aggregator.rs`

**Data Types**:
- Order book snapshots (L2)
- Order book deltas (incremental updates)
- Funding rates (perpetual futures)
- Liquidations (if available)

**Success Criteria**:
- WebSocket compression reduces bandwidth by 60%+
- Order books updated in real-time
- Funding rates stored in QuestDB
- Multi-exchange data aggregated correctly

---

### **Week 6: Data Quality & Validation**
**Focus**: Data integrity, validation, anomaly detection

**Deliverables**:
- ✅ Trade data validation (price/amount ranges)
- ✅ Anomaly detection (price spikes, volume surges)
- ✅ Duplicate detection (enhanced)
- ✅ Latency monitoring (exchange → QuestDB)
- ✅ Data completeness SLI dashboard

**Files to Create**:
- `src/validation/mod.rs`
- `src/validation/price_validator.rs`
- `src/validation/anomaly_detector.rs`
- `src/validation/duplicate_detector.rs`

**Validation Rules**:
- Price within ±10% of previous 1-minute average
- Trade amount < max exchange limit
- No duplicate trade IDs
- Timestamp within ±5 seconds of current time

**Success Criteria**:
- Invalid trades rejected before storage
- Anomalies logged and alerted
- Data completeness SLI > 99.9%
- Latency P99 < 1000ms

---

### **Week 7: Integration Testing**
**Focus**: End-to-end integration tests, exchange testnet validation

**Deliverables**:
- ✅ Integration test framework
- ✅ Exchange testnet tests (Binance, Bybit)
- ✅ QuestDB integration tests
- ✅ Redis cluster tests
- ✅ Multi-service integration tests

**Files to Create**:
- `tests/integration/exchange_tests.rs`
- `tests/integration/questdb_tests.rs`
- `tests/integration/redis_tests.rs`
- `tests/integration/end_to_end.rs`
- `.env.testnet.example`

**Test Scenarios**:
1. WebSocket → Router → QuestDB (full pipeline)
2. Exchange failover (primary → secondary)
3. Backfill with locking (multi-instance)
4. Gap detection → auto-backfill
5. Circuit breaker → recovery
6. Metrics export → Prometheus scrape

**Success Criteria**:
- All integration tests passing
- Testnet data successfully ingested
- Multi-instance coordination works
- Zero data loss during failover

---

### **Week 8: Docker & Kubernetes Deployment**
**Focus**: Production containerization, orchestration, scalability

**Deliverables**:
- ✅ Multi-stage Dockerfile
- ✅ docker-compose.yml (full stack)
- ✅ Kubernetes manifests (Deployment, Services)
- ✅ Helm chart
- ✅ Horizontal Pod Autoscaler (HPA)
- ✅ ConfigMaps and Secrets

**Files to Create**:
- `Dockerfile`
- `docker-compose.yml`
- `deploy/k8s/namespace.yaml`
- `deploy/k8s/configmap.yaml`
- `deploy/k8s/secrets.yaml`
- `deploy/k8s/deployment.yaml`
- `deploy/k8s/service.yaml`
- `deploy/k8s/hpa.yaml`
- `deploy/helm/data-service/` (Helm chart)

**Infrastructure**:
- 3 replicas (high availability)
- Redis Sentinel for HA
- QuestDB cluster (if needed)
- Prometheus + Grafana stack
- Ingress for HTTP/gRPC

**Success Criteria**:
- Docker image < 100MB (multi-stage)
- K8s deployment successful
- HPA scales based on CPU/memory
- Zero-downtime rolling updates
- Health probes passing

---

### **Week 9: CI/CD & Automation**
**Focus**: GitHub Actions, automated testing, deployment pipelines

**Deliverables**:
- ✅ GitHub Actions workflow
- ✅ Automated testing (unit + integration)
- ✅ Docker build & push
- ✅ Deployment automation (staging/prod)
- ✅ Load testing pipeline
- ✅ Security scanning

**Files to Create**:
- `.github/workflows/ci.yml`
- `.github/workflows/integration.yml`
- `.github/workflows/deploy.yml`
- `.github/workflows/load-test.yml`
- `scripts/load-test.sh`

**CI/CD Pipeline**:
1. **Lint**: `cargo fmt`, `clippy`
2. **Test**: Unit tests + integration tests
3. **Build**: Release binary + Docker image
4. **Security**: Dependency audit, container scan
5. **Deploy**: Staging → manual approval → Production

**Load Testing**:
- Target: 100K trades/second ingestion
- Simulate 10 exchanges simultaneously
- Monitor resource usage (CPU, memory, network)
- Stress test QuestDB write throughput

**Success Criteria**:
- CI/CD pipeline fully automated
- All tests passing in CI
- Docker image published to registry
- Load test achieves target throughput
- Security scans pass

---

### **Week 10: Production Readiness & Documentation**
**Focus**: Production hardening, runbooks, documentation

**Deliverables**:
- ✅ Production runbooks
- ✅ Incident response playbooks
- ✅ Architecture diagrams
- ✅ API documentation
- ✅ Deployment guide
- ✅ Operations manual
- ✅ SLA/SLO definitions

**Files to Create**:
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `docs/DEPLOYMENT_GUIDE.md`
- `docs/OPERATIONS_MANUAL.md`
- `docs/RUNBOOKS.md`
- `docs/INCIDENT_RESPONSE.md`
- `docs/SLA_SLO.md`

**Runbooks**:
- Exchange outage response
- QuestDB disk full
- Redis cluster failover
- Gap detection spike
- Circuit breaker open
- High latency investigation

**Production Checklist**:
- [ ] All P0 items complete (7/7)
- [ ] All tests passing (53/53 or more)
- [ ] Load testing successful
- [ ] Security audit passed
- [ ] Documentation complete
- [ ] Monitoring & alerting operational
- [ ] Backup & disaster recovery plan
- [ ] Runbooks validated
- [ ] Team training complete
- [ ] Production deployment successful

---

## 📊 Success Metrics by Week

| Week | Tests | Coverage | Performance | Docs |
|------|-------|----------|-------------|------|
| 1 | 60+ | 70%+ | N/A | 20% |
| 2 | 70+ | 75%+ | N/A | 30% |
| 3 | 75+ | 80%+ | Metrics exported | 40% |
| 4 | 85+ | 85%+ | API functional | 50% |
| 5 | 95+ | 85%+ | WS optimized | 60% |
| 6 | 100+ | 90%+ | Validation working | 70% |
| 7 | 120+ | 90%+ | Integration tests pass | 80% |
| 8 | 125+ | 90%+ | K8s deployment | 85% |
| 9 | 130+ | 90%+ | 100K trades/sec | 90% |
| 10 | 135+ | 90%+ | Production ready | 100% |

---

## 🎯 Performance Targets

| Metric | Target | Week Validated |
|--------|--------|----------------|
| Ingestion Throughput | 100K+ trades/sec | Week 9 |
| WebSocket Latency (P99) | < 100ms | Week 5 |
| Storage Latency (P99) | < 1000ms | Week 6 |
| Data Completeness | 99.9%+ | Week 6 |
| System Uptime | 99.5%+ | Week 10 |
| Gap Detection Time | < 1 minute | Week 2 |
| Backfill Speed | 10K trades/sec | Week 2 |

---

## 🔄 Dependencies & Integrations

### Internal Services
- **Execution Service** ← Real-time market data feed
- **JANUS** ← Historical data queries
- **Python UMAP** → Dimensionality reduction analysis

### External Services
- **Binance** - Primary exchange (WebSocket + REST)
- **Bybit** - Secondary exchange (WebSocket + REST)
- **Kucoin** - Tertiary exchange (WebSocket + REST)
- **Alternative.me** - Fear & Greed Index
- **Farside Investors** - ETF net flows
- **Deribit** - DVOL volatility index

### Infrastructure
- **QuestDB** - Time-series database
- **Redis** - State management & caching
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **Kubernetes** - Orchestration

---

## 🚨 Risk Mitigation

| Risk | Impact | Mitigation | Week |
|------|--------|------------|------|
| Exchange API changes | High | Circuit breaker, failover | 1 |
| QuestDB disk full | Critical | Disk monitoring, auto-stop | 2 |
| Redis cluster failure | High | Sentinel HA, connection pooling | 8 |
| Network partition | Medium | Distributed locking, TTL | 2 |
| High latency spikes | Medium | Circuit breaker, backpressure | 1 |
| Memory leaks | Medium | Resource limits, monitoring | 9 |

---

## 📚 Reference Documentation

Existing documentation (to be enhanced):
- `README.md` - Service overview
- `docs/IMPLEMENTATION_PROGRESS.md` - Current status (2/7 P0 items)
- `docs/CRITICAL_TODOS.md` - P0 requirements
- `docs/SLI_SLO.md` - Operational metrics
- `docs/INTEGRATION_GUIDE.md` - Usage guide

To be created:
- Architecture diagrams
- API reference
- Deployment guide
- Operations manual
- Runbooks

---

## 🎉 Week 1 Preview: Core Reliability

**Next up**: Let's start with Week 1 to fix the 3 failing tests and implement production-grade error handling!

**Week 1 Goals**:
1. Fix failing tests (throttle, metrics export)
2. Implement circuit breaker
3. Enhanced error handling
4. Graceful shutdown
5. All tests passing (60+)

Ready to begin Week 1? 🚀
