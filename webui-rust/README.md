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
- `BASE_URL` and `ETE_API_KEY` for the remote LLM agent. If they are not in the process environment, the Rust server reads simple assignments from `~/.zshenv` such as `export BASE_URL='https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1'` and `export ETE_API_KEY='...'`.
- `LLM_MODEL` to override the model.

## Scope

Implemented:

- static Command Deck UI
- `/api/config`
- `/api/prometheus/metrics`
- `/api/prometheus/range`
- Rust anomaly detection for quick detector choices
- remote LLM agent calls with bounded retry for transient transport failures and visible errors for missing credentials, unreachable endpoints, HTTP failures, invalid JSON, empty responses, and timeouts
- LLM credential save verifies the submitted endpoint/model/key with the same `/chat/completions` path used by prompts before persisting changes, and displays the structured agent error when verification fails
- no local prompt fallback: if the remote agent is not configured, unreachable, timed out, or returns bad data, the prompt request fails with `agent_error`
- plot prompt context is compacted and plot questions are normalized before remote LLM calls so explanations use the agent without raw time-series arrays or timeout-prone prompt wording
- agent prompt context defaults to plots only; mention Argo or MLflow to include those systems

Stubbed on purpose:

- Argo template loading/submission
- MLflow query
- MCP

ponytail: this is a speed-first Rust slice, not a full Node feature clone. Add routes only when needed.
