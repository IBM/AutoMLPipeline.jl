# Rust Argo AutoML Web UI

Rust-native prototype of `../webui` focused on fast Prometheus plotting and anomaly detection without spawning Julia.

## Run

```sh
cd webui-rust
cargo run
```

Defaults to `http://127.0.0.1:8091`.

Useful env vars:

- `PORT` / `HOST`
- `PROMETHEUS_API_URL`
- `PROMETHEUS_METRICS_URL`
- `ARGO_SERVER`, `GRAFANA_URL`, `MLFLOW_URL`

## Scope

Implemented:

- static Command Deck UI
- `/api/config`
- `/api/prometheus/metrics`
- `/api/prometheus/range`
- Rust anomaly detection for quick detector choices
- local prompt fallback summarizing plot anomalies
- agent prompt context defaults to plots only; mention Argo or MLflow to include those systems

Stubbed on purpose:

- Argo template loading/submission
- MLflow query
- remote LLM calls
- MCP

ponytail: this is a speed-first Rust slice, not a full Node feature clone. Add routes only when needed.
