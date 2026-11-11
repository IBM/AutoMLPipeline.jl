# KAIROS-6G

KAIROS-6G is a Rust-native operations command deck for SUNRISE-6G experiments. It combines Prometheus KPI plotting, anomaly detection, LLM-assisted reasoning, and Argo workflow submission so operators can see KPI drift, explain incidents, and launch remediation workflows from one local portal. The current prototype is intentionally local-first; access hardening is required before binding it beyond a trusted workstation.

## Start

From this repository root:

```sh
HOST=127.0.0.1 \
PORT=8092 \
PROMETHEUS_API_URL=http://prom2.prometheus.home:8080 \
PROMETHEUS_METRICS_URL=http://prom1.prometheus.home:8080/metrics \
ARGO_SERVER=http://argo.home:8080 \
ARGO_TEMPLATE_URL=http://argo.home:8080/cluster-workflow-templates \
ARGO_WEBUI_TOKEN=$(openssl rand -hex 24) \
GRAFANA_URL=http://grafana.home:8080 \
MLFLOW_URL=http://mlflow.home:8080 \
BASE_URL=https://your-llm-endpoint.example/v1 \
ETE_API_KEY=your-llm-api-key \
LLM_MODEL=azure/gpt-5.5 \
cargo run
```

Then open `http://127.0.0.1:8092`.

Health check:

```sh
curl -fsS http://127.0.0.1:8092/api/health
# {"status":"ok"}
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address. Non-loopback requires explicit `ARGO_WEBUI_TOKEN` and should sit behind a trusted proxy. |
| `PORT` | `8092` | HTTP port. |
| `ARGO_NAMESPACE` | `argo` | Argo namespace used by workflow operations. |
| `ARGO_SERVER` | `http://argo.home:8080` | Argo UI/API base URL shown in portal links. |
| `ARGO_TEMPLATE_URL` | `http://argo.home:8080/cluster-workflow-templates` | Template source endpoint. |
| `ARGO_BIN` | `argo` | Argo CLI path for local workflow commands. |
| `GRAFANA_URL` | `http://grafana.home:8080` | Grafana link. |
| `MLFLOW_URL` | `http://mlflow.home:8080` | MLflow API/link. |
| `PROMETHEUS_METRICS_URL` | `http://prom1.prometheus.home:8080/metrics` | Prometheus metrics endpoint. |
| `PROMETHEUS_API_URL` | `http://prom2.prometheus.home:8080` | Prometheus HTTP API base URL. |
| `BASE_URL` | unset | LLM `/chat/completions`-compatible endpoint. Falls back to simple assignments in `~/.zshenv`. |
| `ETE_API_KEY` | unset | LLM bearer token. Falls back to simple assignments in `~/.zshenv`. |
| `LLM_MODEL` | `azure/gpt-5.5` | LLM model name. |
| `LLM_TIMEOUT_SECS` | `20` | LLM request timeout. |
| `LLM_MAX_TOKENS` | `512` | LLM max tokens; set `0` to omit. |
| `ARGO_WEBUI_TOKEN` | random per process | API token required for non-health `/api/*` routes. If unset, server prints generated session token at startup. |

## Current access-control state

KAIROS binds to `127.0.0.1:8092` by default, which limits access to callers on the same host. Every non-health `/api/*` route now requires either `Authorization: Bearer <ARGO_WEBUI_TOKEN>` or `x-api-key: <ARGO_WEBUI_TOKEN>`, while `/api/health` remains public for probes. If `ARGO_WEBUI_TOKEN` is not set on loopback, the server generates a random token for that process and prints it to the terminal; if `HOST` is non-loopback, startup requires an explicit `ARGO_WEBUI_TOKEN`. The browser stores a pasted token in session storage only.

## Access-hardening plan

### Goal

Make KAIROS safe to run on an operator workstation and ready for controlled deployment behind a portal or reverse proxy. Preserve local development speed, but make unsafe exposure hard to do by accident.

### Scope

In scope:

- HTTP API authentication and clear default access policy.
- Secret handling for LLM credentials and portal tokens.
- Safe bind/listen defaults and deployment guidance.
- Basic security tests and operator-facing documentation.

Out of scope for the first hardening pass:

- Full user accounts/RBAC.
- Multi-tenant authorization.
- Replacing upstream Argo/Grafana/MLflow/Prometheus auth.
- Public internet exposure without a reverse proxy or identity layer.

### Implementation units

1. **Enforce KAIROS API token** — implemented.
   - Axum middleware checks `Authorization: Bearer <ARGO_WEBUI_TOKEN>` or `x-api-key: <ARGO_WEBUI_TOKEN>` for `/api/*` except `/api/health`.
   - If `ARGO_WEBUI_TOKEN` is unset on loopback, server generates a random per-process token and prints it at startup; non-loopback binds require explicit `ARGO_WEBUI_TOKEN`.
   - Missing credentials return `401`; wrong credentials return `403`.
   - Tests in `src/main.rs` cover health-public, config-protected, and workflow-submit-protected paths.

2. **Update browser client to send token** — implemented.
   - `public/app.js` includes the selected token in same-origin API fetches.
   - Operator pastes the token once; browser stores it in session storage only.
   - Token is not placed in static HTML, query strings, logs, localStorage, or `/api/config` responses.

3. **Protect credential mutation**
   - Keep masking `ETE_API_KEY` in `/api/config` responses.
   - Require auth before `/api/llm` can update credentials.
   - Keep sanitizing downstream LLM errors so API keys do not leak in UI or logs.
   - Reject empty `BASE_URL`/`ETE_API_KEY` on save, as current code already does.

4. **Keep safe network defaults**
   - Keep `HOST=127.0.0.1` as default.
   - Require explicit `ARGO_WEBUI_TOKEN` when binding non-loopback.
   - Document that `HOST=0.0.0.0` should still sit behind an authenticated reverse proxy for real deployments.

5. **Add operator verification checklist**
   - `curl http://127.0.0.1:8092/api/health` succeeds without auth.
   - `curl http://127.0.0.1:8092/api/config` fails without auth after hardening.
   - `curl -H "Authorization: Bearer $ARGO_WEBUI_TOKEN" http://127.0.0.1:8092/api/config` succeeds.
   - UI still loads, plots Prometheus data, verifies LLM credentials, and handles prompt failures without leaking secrets.

### Acceptance criteria

- `/api/health` remains public and returns `{"status":"ok"}`.
- Every non-health `/api/*` route rejects unauthenticated callers.
- Authenticated UI calls keep working with same-origin requests.
- API keys never appear unmasked in `/api/config`, browser-rendered errors, or logs.
- README start/config/health/access-control docs match actual behavior.

## Scope

Implemented:

- static Command Deck UI
- `/api/config`
- `/api/prometheus/metrics`
- `/api/prometheus/range`
- `/api/health`
- Rust anomaly detection for quick detector choices
- remote LLM agent calls with bounded retry, bounded timeout/output, and visible errors for missing credentials, unreachable endpoints, HTTP failures, invalid JSON, empty responses, and timeouts
- LLM credential save verifies the submitted endpoint/model/key with the same `/chat/completions` path used by prompts before persisting changes, and displays the structured agent error when verification fails
- no local prompt fallback: if the remote agent is not configured, unreachable, timed out, or returns bad data, the prompt request fails with `agent_error`
- plot prompt context uses the full-detector plot by default, compacts raw points, and normalizes plot questions before remote LLM calls so explanations do not use quick-detector stream data unless explicitly requested
- agent prompt context defaults to plots only; mention Argo or MLflow to include those systems

Stubbed on purpose:

- Argo template loading/submission (template names load first; selected template parameters load lazily)
- MLflow query
- MCP
