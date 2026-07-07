# Argo AutoML Command Deck

Local web UI and MCP surface for Argo AutoML workflows, logs, MLflow, and Prometheus plots.

## Run

```bash
cd argo-workflow/webui
npm install
npm start
```

The server binds to `127.0.0.1:8090` by default. Browser UI calls receive a same-site local session cookie from `/api/config`; external MCP/API clients can still use the startup token as `Authorization: Bearer <token>`. Do not expose this app directly on a LAN or public interface.

## Configuration

| Variable | Default |
|---|---|
| `HOST` | `127.0.0.1` |
| `PORT` | `8090` |
| `ARGO_NAMESPACE` | `argo` |
| `ARGO_SERVER` | `http://argo.home:8080` |
| `ARGO_TEMPLATE_URL` | `http://argo.isiath.duckdns.org:8080/cluster-workflow-templates` |
| `GRAFANA_URL` | `http://grafana.home:8080` |
| `MLFLOW_URL` | `http://mlflow.home:8080` |
| `PROMETHEUS_METRICS_URL` | `http://prom1.prometheus.home:8080/metrics` |
| `PROMETHEUS_API_URL` | `http://prom2.prometheus.home:8080` |
| `ARGO_WEBUI_TOKEN` | random per start |
| `ARGO_MCP_ALLOWED_OPERATIONS` | `submit` |

LLM defaults are loaded from Pi's external provider in `~/.pi/agent/models.json`, preferring `azure/gpt-5.5`. A stale global `OPENAI_API_KEY` does not override that by itself. To intentionally use another OpenAI-compatible endpoint, set `OPENAI_BASE_URL` and optionally `OPENAI_API_KEY`/`OPENAI_MODEL`, or set `ARGO_WEBUI_USE_OPENAI_ENV=1`.

## Prometheus plots

The dashboard exposes `/api/prometheus/metrics` and `/api/prometheus/range?metric=cpu|memory|network|latency`. Range queries cover the last 24 hours with a 10 minute step. `PROMETHEUS_API_URL` must point at the Prometheus HTTP API root, not a scrape-only `/metrics` ingress.

Built-in queries use common Prometheus/node-exporter names:

- CPU: `node_cpu_seconds_total`
- Memory: `node_memory_MemAvailable_bytes` / `node_memory_MemTotal_bytes`
- Network: `node_network_receive_bytes_total` + `node_network_transmit_bytes_total`
- Latency: `prober_probe_duration_seconds_bucket` (network probe latency)

If your cluster uses different metric names, edit `webui/lib/prometheus.js`.

## MCP Argo tools

`/mcp` exposes safe Argo workflow tools modeled on the official Argo Workflows CLI concepts: Workflows, WorkflowTemplates, ClusterWorkflowTemplates, CronWorkflows, logs, watch, and operations.

Read tools:

- `list_workflows`, `get_workflow_status`, `get_workflow_logs`, `watch_workflow`
- `list_workflow_templates` for UI-enriched ClusterWorkflowTemplates
- `list_argo_templates`, `get_argo_template`
- `list_cron_workflows`, `get_cron_workflow`
- `query_mlflow_results`, `get_mcp_permissions`
- MLflow: `mlflow_search_experiments`, `mlflow_get_experiment`, `mlflow_search_runs`, `mlflow_get_run`, `mlflow_summarize_run`, `mlflow_compare_runs`, `mlflow_get_metric_history`, `mlflow_list_artifacts`

Mutating tools are gated by `ARGO_MCP_ALLOWED_OPERATIONS`:

- `submit_template_workflow`, `submit_yaml_workflow` use `submit`
- `operate_workflow` supports `suspend,resume,retry,terminate,stop,delete`
- `resubmit_workflow` uses `resubmit`
- `operate_cron_workflow` supports `cron:suspend,cron:resume,cron:delete`
- `operate_archived_workflow` supports `archive:retry,archive:resubmit,archive:delete`

Example:

```bash
ARGO_MCP_ALLOWED_OPERATIONS=submit,retry,stop,cron:suspend npm start
```

Additional Argo v3.4.11-inspired tools:

- `argo_version`
- `lint_argo_yaml`
- `resubmit_workflow`
- `get_workflow_nodes`
- Archive: `list_archived_workflows`, `get_archived_workflow`, `list_archive_label_keys`, `list_archive_label_values`, `operate_archived_workflow`

`lint_argo_yaml` is read-only. Archive retry/resubmit/delete and workflow resubmit require explicit permission.

## MLflow MCP query tools

MLflow tools are read-only and follow MLflow v2.21.3 Tracking REST API endpoints:

- experiments: `/mlflow/experiments/search`, `/mlflow/experiments/get`, `/mlflow/experiments/get-by-name`
- runs: `/mlflow/runs/search`, `/mlflow/runs/get`
- metrics: `/mlflow/metrics/get-history`
- artifacts: `/mlflow/artifacts/list`

Useful examples:

- `mlflow_search_experiments` with `filter`, `orderBy`, `viewType`
- `mlflow_search_runs` with `experimentIds`, `filter`, `orderBy`
- `mlflow_summarize_run` for params/metrics/tags maps
- `mlflow_compare_runs` for selected metric keys across runs
- `mlflow_get_metric_history` for charts over one metric
- `mlflow_list_artifacts` to inspect run outputs

## Safety

- UI/MCP never exposes raw mutating `kubectl` or `argo` commands.
- Uploaded `Workflow` YAML is validated before submit.
- Uploaded `ClusterWorkflowTemplate` YAML is preview metadata unless it matches an existing cluster template name.
- Logs and MLflow data are redacted before LLM context.
