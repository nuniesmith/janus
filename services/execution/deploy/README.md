# Deployment Configurations

This directory contains deployment configurations for the Execution Service.

## 📁 Contents

### Kubernetes (`k8s/`)

Production-ready Kubernetes manifests:

- `namespace.yaml` - Creates `execution` namespace
- `configmap.yaml` - Environment configuration
- `secrets.yaml` - API keys and secrets (template - EDIT BEFORE DEPLOYING!)
- `deployment.yaml` - Service deployment with 3 replicas

**Deploy**:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml  # Edit first!
kubectl apply -f k8s/deployment.yaml
```

### Prometheus

- `prometheus.yml` - Prometheus scrape configuration

Scrapes:
- Execution Service metrics (port 8081)
- Redis
- QuestDB

### Grafana (Coming Soon)

Dashboard JSON definitions will be added to:
- `grafana/dashboards/`
- `grafana/datasources/`

## 🚀 Quick Deploy

### Local (Docker Compose)

From project root:
```bash
docker-compose up -d
```

### Kubernetes

```bash
# Deploy everything
kubectl apply -f k8s/

# Check status
kubectl get pods -n execution
kubectl get svc -n execution

# View logs
kubectl logs -f deployment/execution-service -n execution
```

## 📝 Notes

- **Secrets**: Never commit real API keys! Use K8s secrets or Vault.
- **Resources**: Adjust CPU/memory limits based on load testing.
- **Replicas**: Start with 3, scale based on traffic.
- **Health Probes**: Ensure `/health` and `/ready` endpoints are implemented.

## 🔗 See Also

- [Quick Start Guide](../QUICKSTART_WEEK7_8.md)
- [Full Deployment Guide](../WEEK7_8_INTEGRATION_DEPLOYMENT.md)
