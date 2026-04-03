# JANUS Central Nervous System (CNS)

The **Central Nervous System (CNS)** is the health monitoring and auto-recovery backbone for the JANUS trading system. Inspired by the biological nervous system, it provides comprehensive observability, metrics collection, and intelligent reflexes to maintain system health.

## 🧠 Architecture

The CNS is designed following the biological nervous system model:

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                        BRAIN (Coordinator)                   │
│  - Aggregates health signals from all components            │
│  - Orchestrates probes and reflexes                          │
│  - Exposes Prometheus metrics                                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   PROBES     │      │   METRICS    │      │   REFLEXES   │
│ (Sensors)    │      │ (Signals)    │      │ (Actions)    │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      MONITORED COMPONENTS                    │
│  - Forward Service (Wake State)                              │
│  - Backward Service (Sleep State)                            │
│  - Gateway Service                                           │
│  - Redis (Job Queue)                                         │
│  - Qdrant (Vector DB)                                        │
│  - Shared Memory (IPC)                                       │
└─────────────────────────────────────────────────────────────┘
```

### Biological Analogy

| Biological Component | CNS Component | Function |
|---------------------|---------------|----------|
| **Brain** | `brain::Brain` | Central coordinator, decision making |
| **Spinal Cord** | Metrics pipeline | Message highway between brain and body |
| **Sensory Neurons** | `probes::HealthProbe` | Detect component status |
| **Motor Neurons** | `reflexes::Reflex` | Execute recovery actions |
| **Reflex Arcs** | Circuit breakers | Automatic responses without brain involvement |
| **Nerve Signals** | `signals::HealthSignal` | Information transmission |

## 📊 Key Features

### 1. Health Monitoring
- **Comprehensive Probes**: HTTP, gRPC, Redis, Qdrant, Shared Memory
- **Configurable Intervals**: Adjust check frequency per environment
- **Timeout Protection**: Prevents hanging on unresponsive services
- **Concurrent Execution**: All probes run in parallel for efficiency

### 2. Prometheus Metrics
- **System-Level**: Overall health score, status, uptime
- **Component-Level**: Per-component health and response times
- **Service-Specific**: Trading metrics, training iterations, etc.
- **Dependency Metrics**: Redis commands, Qdrant searches, etc.
- **Resource Metrics**: Memory, CPU, active tasks

### 3. Circuit Breakers
- **Automatic Protection**: Prevent cascading failures
- **Configurable Thresholds**: Tune per component
- **Half-Open State**: Test recovery before fully closing
- **Metrics Integration**: Track trips and state changes

### 4. Auto-Recovery Reflexes
- **Rule-Based Actions**: Define conditions and responses
- **Cooldown Protection**: Prevent action spam
- **Alert Integration**: Slack, PagerDuty (extensible)
- **Command Execution**: Restart services, throttle requests

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
janus-cns = { path = "../crates/cns" }
```

### Basic Usage

```rust
use janus_cns::{Brain, BrainConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = BrainConfig::default();
    
    // Create and start the brain
    let brain = Brain::new(config);
    
    // Start monitoring in background
    tokio::spawn(async move {
        brain.start().await.unwrap();
    });
    
    // Your application logic here...
    
    Ok(())
}
```

### Configuration

Create `config/cns.toml`:

```toml
[brain]
health_check_interval_secs = 10
enable_reflexes = true

[endpoints]
forward_service = "http://localhost:8081"
backward_service = "http://localhost:8082"
gateway_service = "http://localhost:8080"
redis = "redis://localhost:6379"
qdrant = "http://localhost:6333"
shared_memory_path = "/dev/shm/janus_forward_backward"
```

## 📈 Metrics Endpoints

### Health Check
```bash
curl http://localhost:9090/health
```

Response:
```json
{
  "signal": {
    "system_status": "Healthy",
    "components": [
      {
        "component_type": "forward_service",
        "status": "Up",
        "message": "OK",
        "last_check": "2025-01-15T10:30:00Z",
        "response_time_ms": 42
      }
    ],
    "timestamp": "2025-01-15T10:30:00Z",
    "uptime_seconds": 3600,
    "version": "0.1.0"
  }
}
```

### Prometheus Metrics
```bash
curl http://localhost:9090/metrics
```

Key metrics:
- `janus_cns_system_health_score` - Overall health (0.0 to 1.0)
- `janus_cns_system_status` - System status code
- `janus_cns_component_health{component="..."}` - Per-component health
- `janus_forward_orders_submitted_total` - Trading activity
- `janus_backward_training_iterations_total` - Training activity

## 📊 Grafana Dashboard

Import the pre-built dashboard from `config/grafana/janus_cns_dashboard.json`.

### Dashboard Sections

1. **System Overview**
   - System status indicator
   - Overall health score gauge
   - System uptime
   - Active tasks

2. **Component Health**
   - Real-time component status
   - Health trends over time
   - Response time graphs

3. **Services Performance**
   - Forward service: Orders submitted/filled/rejected
   - Backward service: Training, memory consolidation
   - Gateway service: HTTP request rates

4. **Dependencies**
   - Redis: Command rates and latency
   - Qdrant: Search performance and vector count

5. **Circuit Breakers**
   - Current states (Closed/Open/Half-Open)
   - Trip count over time

6. **Communication Channels**
   - Shared memory message rates
   - gRPC request performance

7. **Resource Utilization**
   - Memory usage
   - CPU usage

## 🔧 Advanced Usage

### Custom Health Probes

```rust
use janus_cns::probes::{HealthProbe, ProbeResult};
use async_trait::async_trait;

struct CustomProbe;

#[async_trait]
impl HealthProbe for CustomProbe {
    fn component_type(&self) -> ComponentType {
        ComponentType::Custom { name: "my_service" }
    }
    
    async fn check(&self) -> Result<ProbeResult> {
        // Your custom health check logic
        Ok(ProbeResult {
            component_type: self.component_type(),
            status: ProbeStatus::Up,
            message: "Custom check passed".to_string(),
            response_time_ms: 10,
        })
    }
}

// Add to brain
let mut brain = Brain::new(config);
brain.add_probe(Box::new(CustomProbe));
```

### Custom Reflex Rules

```rust
use janus_cns::reflexes::{ReflexRule, ReflexCondition, RefexAction, AlertSeverity};

let rule = ReflexRule {
    id: "custom_rule".to_string(),
    description: "Custom recovery action".to_string(),
    condition: ReflexCondition::ComponentDown {
        component: ComponentType::ForwardService,
    },
    action: RefexAction::SendAlert {
        severity: AlertSeverity::Critical,
        message: "Critical service down!".to_string(),
    },
    cooldown_secs: 300,
};

brain.add_reflex_rule(rule).await;
```

### Circuit Breaker Usage

```rust
use janus_cns::reflexes::{CircuitBreaker, CircuitBreakerConfig};

let config = CircuitBreakerConfig {
    failure_threshold: 5,
    failure_window_secs: 60,
    recovery_timeout_secs: 30,
    success_threshold: 3,
};

let breaker = CircuitBreaker::new(ComponentType::Redis, config);

// Check if call is permitted
if breaker.is_call_permitted() {
    match make_redis_call().await {
        Ok(_) => breaker.record_success(),
        Err(_) => breaker.record_failure(),
    }
} else {
    // Circuit is open, fail fast
    return Err("Circuit breaker open");
}
```

## 🎯 Health Signal Types

### System Status
- `Healthy` - All systems operational
- `Degraded` - Non-critical issues detected
- `Critical` - Critical issues requiring attention
- `Shutdown` - System shutting down
- `Starting` - System initializing

### Probe Status
- `Up` - Component fully operational (score: 1.0)
- `Degraded` - Operational but degraded (score: 0.5)
- `Down` - Component not operational (score: 0.0)
- `Unknown` - Status cannot be determined (score: 0.25)

## 🔄 Auto-Recovery Actions

### Available Actions

1. **LogWarning** - Log a warning message
2. **SendAlert** - Send alert to external systems
3. **RestartComponent** - Attempt component restart
4. **ThrottleComponent** - Rate limit requests
5. **OpenCircuitBreaker** - Open circuit breaker
6. **ExecuteCommand** - Run custom command
7. **GracefulShutdown** - Initiate system shutdown

### Alert Severity Levels
- `Info` - Informational
- `Warning` - Warning condition
- `Error` - Error condition
- `Critical` - Critical issue requiring immediate attention

## 📝 Configuration Reference

### Brain Settings
```toml
[brain]
health_check_interval_secs = 10  # How often to check health
enable_reflexes = true           # Enable auto-recovery
verbose_logging = false          # Detailed debug logs
```

### Circuit Breaker Settings
```toml
[circuit_breakers.redis]
failure_threshold = 5            # Failures before opening
failure_window_secs = 60         # Time window for counting failures
recovery_timeout_secs = 30       # Wait before attempting recovery
success_threshold = 3            # Successes needed to close
```

### Metrics Settings
```toml
[metrics]
enabled = true                   # Enable Prometheus metrics
endpoint = "/metrics"            # Metrics endpoint path
include_metadata = true          # Include detailed metadata
```

## 🧪 Testing

Run the test suite:

```bash
cargo test -p janus-cns
```

Run with logging:

```bash
RUST_LOG=janus_cns=debug cargo test -p janus-cns -- --nocapture
```

## 🐛 Troubleshooting

### High Response Times
- Check network latency to services
- Verify service load and resource usage
- Adjust probe timeouts if necessary

### False Positives
- Increase failure thresholds
- Extend failure window
- Add probe retry logic

### Circuit Breaker Stuck Open
- Check recovery timeout settings
- Verify underlying service is actually healthy
- Review success threshold requirements

## 🔮 Future Enhancements

- [ ] gRPC health check protocol implementation
- [ ] Advanced alerting integrations (Slack, PagerDuty)
- [ ] Automatic component restart logic
- [ ] Distributed tracing integration
- [ ] Machine learning for anomaly detection
- [ ] Predictive health scoring
- [ ] Auto-scaling triggers based on health
- [ ] Historical health data storage

## 📚 Related Modules

- `janus-common` - Shared types and utilities
- `janus-proto` - gRPC definitions
- `services/janus-forward` - Wake state service
- `services/janus-backward` - Sleep state service
- `services/janus-gateway` - Orchestration layer

## 🤝 Contributing

When adding new components to monitor:

1. Add component type to `signals::ComponentType`
2. Create appropriate probe in `probes` module
3. Add metrics to `metrics::MetricsRegistry`
4. Update Grafana dashboard
5. Add default reflex rules if needed
6. Update documentation

## 📄 License

MIT OR Apache-2.0