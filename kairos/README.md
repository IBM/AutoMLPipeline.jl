# KAIROS-6G

KAIROS-6G is a Rust-native operations command deck for SUNRISE-6G experiments. It combines Prometheus KPI plotting, anomaly detection, LLM-assisted reasoning, and Argo workflow submission so operators can see KPI drift, explain incidents, and launch remediation workflows from one local portal. The current prototype is local-first and enforces an API token on all non-health API routes before any non-loopback bind is allowed.

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
| `PATH` | shell default | Must include the `argo` CLI; workflow commands always invoke `argo` directly. |
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

## Access control implementation

KAIROS protects every non-health `/api/*` route with a single operator API token. The token is read from `ARGO_WEBUI_TOKEN` when set. On loopback binds (`127.0.0.1`, `localhost`, or `::1`), an unset or empty token generates a random per-process local token and prints it to the server terminal. On non-loopback binds such as `0.0.0.0`, startup exits unless `ARGO_WEBUI_TOKEN` is explicitly set.

Accepted request credentials:

```http
Authorization: Bearer <ARGO_WEBUI_TOKEN>
x-api-key: <ARGO_WEBUI_TOKEN>
```

Response behavior:

- `/api/health` stays public and returns `{"status":"ok"}` for probes.
- Missing credentials on protected API routes return `401` with JSON error `Missing API token`.
- Wrong credentials return `403` with JSON error `Invalid API token`.
- Static UI assets remain public, but browser API calls need the token.

Browser behavior:

- `public/app.js` adds `x-api-key` to same-origin `/api/*` calls except `/api/health`.
- On a `401` or `403`, the UI clears any stale session token, prompts once for `ARGO_WEBUI_TOKEN`, stores the pasted value in `sessionStorage`, and retries the request once.
- The token is not embedded in HTML, added to query strings, written to `localStorage`, returned by `/api/config`, or intentionally logged by the browser.

Credential handling:

- `/api/config` masks the LLM API key and does not serialize the KAIROS session token.
- `/api/llm` is protected by the same API middleware before it can update LLM credentials.
- Downstream LLM errors are sanitized so submitted API keys do not appear in UI-visible errors.

Network policy:

- Keep `HOST=127.0.0.1` for local operator use.
- For `HOST=0.0.0.0`, set `ARGO_WEBUI_TOKEN` explicitly and put KAIROS behind a trusted authenticated reverse proxy for real deployments.

Operator verification:

```sh
curl -fsS http://127.0.0.1:8092/api/health
# {"status":"ok"}

curl -i http://127.0.0.1:8092/api/config
# HTTP/1.1 401 Unauthorized

curl -fsS -H "Authorization: Bearer $ARGO_WEBUI_TOKEN" \
  http://127.0.0.1:8092/api/config
# succeeds with masked LLM config and no session token field
```

Tests in `src/main.rs` cover public health access, missing/wrong/correct API tokens, bearer-token auth, `/api/llm` protection, workflow-submit protection, local-only generated tokens, and token/key non-leakage in config and LLM update errors.

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
