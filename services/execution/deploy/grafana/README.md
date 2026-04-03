# FKS DataRecorder Monitoring

This directory contains Grafana dashboards and Prometheus alerting rules for monitoring the FKS DataRecorder component.

## Overview

The monitoring setup provides visibility into:

- **Health Status**: Overall recorder health, connection status, and fallback state
- **Event Recording**: Recording rates, throughput, and event type breakdown
- **Buffer & Channel Health**: Utilization metrics to detect backpressure
- **Errors & Reliability**: Drop rates, write errors, and reconnection events
- **Fallback Storage**: Local disk fallback usage when QuestDB is unavailable

## Quick Start

### 1. Deploy Prometheus Rules

Copy the alerting rules to your Prometheus rules directory:

```bash
cp prometheus/alert-rules.yaml /etc/prometheus/rules/fks-recorder-alerts.yaml
```

Add to your `prometheus.yml`:

```yaml
rule_files:
  - /etc/prometheus/rules/fks-recorder-alerts.yaml
```

Reload Prometheus:

```bash
curl -X POST http://localhost:9090/-/reload
```

### 2. Deploy Grafana Dashboard

#### Option A: Manual Import

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `dashboards/data-recorder-health.json`
4. Select your Prometheus datasource

#### Option B: Provisioning (Recommended)

Copy provisioning configs:

```bash
# Copy dashboard provisioning config
cp provisioning/dashboards.yaml /etc/grafana/provisioning/dashboards/

# Copy datasource config
cp provisioning/datasources.yaml /etc/grafana/provisioning/datasources/

# Copy dashboard JSON
mkdir -p /var/lib/grafana/dashboards/fks
cp dashboards/data-recorder-health.json /var/lib/grafana/dashboards/fks/
```

Restart Grafana:

```bash
systemctl restart grafana-server
```

### 3. Docker Compose Setup

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/alert-rules.yaml:/etc/prometheus/rules/fks-alerts.yaml
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards/fks
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Dashboard Panels

### Overview Row

| Panel | Description |
|-------|-------------|
| Health Status | Current health state (Healthy/Unhealthy) |
| QuestDB Connection | Connection status to QuestDB |
| Fallback Status | Whether fallback storage is active |
| Events/sec | Current event recording rate |
| Total Events | Total events recorded since start |
| Bytes Written | Total data written to QuestDB |

### Event Recording Row

| Panel | Description |
|-------|-------------|
| Events Recording Rate | Rate of events by type (ticks, trades, candles) |
| Write Throughput | Bytes/second written to QuestDB |

### Buffer & Channel Health Row

| Panel | Description |
|-------|-------------|
| Buffer Utilization | Gauge showing buffer fill percentage |
| Channel Utilization | Gauge showing channel fill percentage |
| Utilization Over Time | Historical view of buffer/channel usage |

### Errors & Reliability Row

| Panel | Description |
|-------|-------------|
| Drop Rate % | Percentage of events dropped |
| Error Rate % | Percentage of write operations that failed |
| Reconnections | Total reconnection attempts |
| Errors Over Time | Historical error/drop/reconnection events |
| Total Dropped | Cumulative dropped event count |
| Total Write Errors | Cumulative write error count |
| Uptime | Time since recorder started |

### Fallback Storage Row

| Panel | Description |
|-------|-------------|
| Fallback Status Timeline | State timeline showing when fallback is active |
| Fallback Events | Events written to fallback storage |
| Fallback Bytes | Disk space used by fallback |
| Fallback Errors | Errors writing to fallback |
| Fallback Activity | Rate of fallback writes over time |

## Alert Rules

### Critical Alerts (Immediate Response Required)

| Alert | Condition | Duration |
|-------|-----------|----------|
| `RecorderUnhealthy` | Health status is 0 | 1 minute |
| `RecorderDisconnected` | Not connected to QuestDB | 2 minutes |
| `RecorderHighDropRate` | Drop rate > 1% | 2 minutes |
| `RecorderFallbackErrors` | Any fallback write errors | Immediate |

### Warning Alerts (Attention Needed)

| Alert | Condition | Duration |
|-------|-----------|----------|
| `RecorderHighBufferUtilization` | Buffer > 80% full | 5 minutes |
| `RecorderHighChannelUtilization` | Channel > 75% full | 5 minutes |
| `RecorderHighErrorRate` | Error rate > 5% | 5 minutes |
| `RecorderFrequentReconnections` | > 5 reconnections in 5m | Immediate |
| `RecorderLowThroughput` | < 1 event/sec when healthy | 10 minutes |
| `RecorderFallbackActive` | Using fallback storage | 1 minute |
| `RecorderFallbackGrowing` | Fallback growing > 10MB/min | 5 minutes |
| `RecorderFallbackDiskUsageHigh` | Fallback > 1GB | 5 minutes |

### Informational Alerts

| Alert | Condition | Duration |
|-------|-----------|----------|
| `RecorderHighWriteVolume` | Writing > 100MB/s | 10 minutes |
| `RecorderNoEventsRecorded` | No events in 10 min when healthy | 10 minutes |

## Metrics Reference

### Counters (Monotonically Increasing)

| Metric | Description |
|--------|-------------|
| `sim_recorder_events_recorded_total` | Total events recorded |
| `sim_recorder_ticks_recorded_total` | Tick events recorded |
| `sim_recorder_trades_recorded_total` | Trade events recorded |
| `sim_recorder_orderbooks_recorded_total` | Order book snapshots recorded |
| `sim_recorder_candles_recorded_total` | Candle events recorded |
| `sim_recorder_events_dropped_total` | Events dropped (buffer overflow) |
| `sim_recorder_write_errors_total` | Write errors to QuestDB |
| `sim_recorder_reconnections_total` | Reconnection attempts |
| `sim_recorder_bytes_written_total` | Bytes written to QuestDB |
| `sim_recorder_flush_count_total` | Successful flush operations |
| `sim_recorder_fallback_events_total` | Events written to fallback |
| `sim_recorder_fallback_bytes_total` | Bytes written to fallback |
| `sim_recorder_fallback_errors_total` | Fallback write errors |

### Gauges (Current Value)

| Metric | Description |
|--------|-------------|
| `sim_recorder_buffer_depth` | Current buffer depth |
| `sim_recorder_buffer_capacity` | Buffer capacity |
| `sim_recorder_buffer_utilization_pct` | Buffer utilization % |
| `sim_recorder_channel_depth` | Current channel depth |
| `sim_recorder_channel_capacity` | Channel capacity |
| `sim_recorder_channel_utilization_pct` | Channel utilization % |
| `sim_recorder_connected` | Connection status (0/1) |
| `sim_recorder_healthy` | Health status (0/1) |
| `sim_recorder_fallback_active` | Fallback active (0/1) |
| `sim_recorder_events_per_second` | Current events/sec rate |
| `sim_recorder_drop_rate_pct` | Current drop rate % |
| `sim_recorder_error_rate_pct` | Current error rate % |
| `sim_recorder_uptime_seconds` | Uptime in seconds |

## Alertmanager Configuration

Example Alertmanager config for routing FKS alerts:

```yaml
route:
  group_by: ['alertname', 'instance']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    - match:
        component: data-recorder
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    - match:
        component: data-recorder
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default'
    # Default receiver config

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<your-pagerduty-key>'
        description: '{{ .CommonAnnotations.summary }}'

  - name: 'slack-warnings'
    slack_configs:
      - api_url: '<your-slack-webhook>'
        channel: '#fks-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
```

## Troubleshooting

### No Metrics Showing

1. Verify the execution service is exposing metrics:
   ```bash
   curl http://localhost:8080/metrics | grep sim_recorder
   ```

2. Check Prometheus is scraping the target:
   ```bash
   curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "fks-execution")'
   ```

3. Verify the instance label matches the dashboard variable.

### Alerts Not Firing

1. Check Prometheus rule evaluation:
   ```bash
   curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name == "fks_recorder_health")'
   ```

2. Verify alert conditions manually:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=sim_recorder_healthy'
   ```

### Dashboard Not Loading

1. Check Grafana datasource configuration
2. Verify Prometheus URL is correct
3. Check browser console for errors

## Contributing

When adding new metrics:

1. Update `metrics.rs` with the new metric
2. Add to `RecorderStats` if applicable
3. Update the dashboard JSON
4. Add alerting rules if the metric is alertable
5. Update this README

## License

MIT License - See repository root for details.