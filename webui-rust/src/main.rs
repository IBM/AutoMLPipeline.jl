use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env, fs,
    net::SocketAddr,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::RwLock,
    time::{Duration, timeout},
};
use tower_http::services::ServeDir;
use url::Url;

struct AppState {
    config: RwLock<Config>,
    client: reqwest::Client,
}

#[derive(Clone, Serialize)]
struct Config {
    host: String,
    port: u16,
    namespace: String,
    argo_server: String,
    template_url: String,
    grafana_url: String,
    mlflow_url: String,
    prometheus_metrics_url: String,
    prometheus_api_url: String,
    argo_bin: String,
    session_token: String,
    llm: LlmConfig,
}

#[derive(Clone, Serialize)]
struct LlmConfig {
    base_url: String,
    model: String,
    source: String,
    #[serde(skip_serializing)]
    api_key: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicLlm {
    base_url: String,
    model: String,
    has_key: bool,
    api_key: String,
}

impl Config {
    fn from_env() -> Self {
        Self {
            host: env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            port: env::var("PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8091),
            namespace: env::var("ARGO_NAMESPACE").unwrap_or_else(|_| "argo".into()),
            argo_server: env::var("ARGO_SERVER").unwrap_or_else(|_| "http://argo.home:8080".into()),
            template_url: env::var("ARGO_TEMPLATE_URL")
                .unwrap_or_else(|_| "http://argo.home:8080/cluster-workflow-templates".into()),
            grafana_url: env::var("GRAFANA_URL")
                .unwrap_or_else(|_| "http://grafana.home:8080".into()),
            mlflow_url: env::var("MLFLOW_URL").unwrap_or_else(|_| "http://mlflow.home:8080".into()),
            prometheus_metrics_url: env::var("PROMETHEUS_METRICS_URL")
                .unwrap_or_else(|_| "http://prom1.prometheus.home:8080/metrics".into()),
            prometheus_api_url: env::var("PROMETHEUS_API_URL")
                .unwrap_or_else(|_| "http://prom2.prometheus.home:8080".into()),
            argo_bin: env::var("ARGO_BIN").unwrap_or_else(|_| "argo".into()),
            session_token: env::var("ARGO_WEBUI_TOKEN").unwrap_or_else(|_| random_hex(24)),
            llm: load_llm_config(),
        }
    }
}

fn load_llm_config() -> LlmConfig {
    let shell = load_zshenv_vars();
    let api_key = env_or_zshenv("ETE_API_KEY", &shell).unwrap_or_default();
    let base_url = env_or_zshenv("BASE_URL", &shell).unwrap_or_default();
    let model = env_or_zshenv("LLM_MODEL", &shell).unwrap_or_else(|| "azure/gpt-5.5".into());
    let source = if api_key.is_empty() || base_url.is_empty() {
        "missing-env"
    } else {
        "env"
    };
    LlmConfig {
        // BASE_URL and ETE_API_KEY from the process environment or ~/.zshenv are the only default agent credentials.
        api_key,
        base_url,
        model,
        source: source.into(),
    }
}

fn env_or_zshenv(name: &str, zshenv: &HashMap<String, String>) -> Option<String> {
    env::var(name)
        .ok()
        .or_else(|| zshenv.get(name).cloned())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn load_zshenv_vars() -> HashMap<String, String> {
    let path = PathBuf::from(env::var("HOME").unwrap_or_default()).join(".zshenv");
    fs::read_to_string(path)
        .map(|text| parse_shell_assignments(&text))
        .unwrap_or_default()
}

fn parse_shell_assignments(text: &str) -> HashMap<String, String> {
    text.lines().filter_map(parse_shell_assignment).collect()
}

fn parse_shell_assignment(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").unwrap_or(line).trim();
    let (name, value) = line.split_once('=')?;
    let name = name.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some((
        name.to_string(),
        strip_shell_value(value.trim()).to_string(),
    ))
}

fn strip_shell_value(value: &str) -> &str {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        if matches!(
            (bytes[0], bytes[value.len() - 1]),
            (b'\'', b'\'') | (b'"', b'"')
        ) {
            return &value[1..value.len() - 1];
        }
    }
    value
}

fn public_llm(llm: &LlmConfig) -> PublicLlm {
    PublicLlm {
        base_url: llm.base_url.clone(),
        model: llm.model.clone(),
        has_key: !llm.api_key.is_empty(),
        api_key: mask_key(&llm.api_key),
    }
}

fn mask_key(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        "*".repeat(value.len().clamp(8, 24))
    }
}

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::rng().fill_bytes(&mut buf);
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicConfig {
    host: String,
    port: u16,
    namespace: String,
    argo_server: String,
    template_url: String,
    grafana_url: String,
    mlflow_url: String,
    prometheus_metrics_url: String,
    prometheus_api_url: String,
    mcp_url: &'static str,
    llm: PublicLlm,
}

#[derive(Serialize, Clone)]
struct MetricOption {
    id: &'static str,
    label: &'static str,
    unit: &'static str,
}

#[derive(Clone)]
struct MetricDef {
    opt: MetricOption,
    query: &'static str,
}

#[derive(Serialize)]
struct DetectorOption {
    id: &'static str,
    label: &'static str,
}

const DETECTORS: &[DetectorOption] = &[
    DetectorOption {
        id: "sk:LocalOutlierFactor",
        label: "LocalOutlierFactor",
    },
    DetectorOption {
        id: "sk:EllipticEnvelope",
        label: "EllipticEnvelope",
    },
    DetectorOption {
        id: "caret:pca",
        label: "PCA",
    },
    DetectorOption {
        id: "caret:mcd",
        label: "MCD",
    },
];

fn metrics() -> Vec<MetricDef> {
    vec![
        MetricDef {
            opt: MetricOption {
                id: "cpu",
                label: "CPU usage",
                unit: "%",
            },
            query: "100 * avg(rate(node_cpu_seconds_total{mode!=\"idle\"}[10m]))",
        },
        MetricDef {
            opt: MetricOption {
                id: "memory",
                label: "Memory usage",
                unit: "%",
            },
            query: "100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))",
        },
        MetricDef {
            opt: MetricOption {
                id: "network",
                label: "Network throughput",
                unit: "B/s",
            },
            query: "sum(rate(node_network_receive_bytes_total[10m]) + rate(node_network_transmit_bytes_total[10m]))",
        },
        MetricDef {
            opt: MetricOption {
                id: "latency",
                label: "Network probe p95 latency",
                unit: "s",
            },
            query: "histogram_quantile(0.95, sum(rate(prober_probe_duration_seconds_bucket[10m])) by (le))",
        },
    ]
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RangeQuery {
    metric: Option<String>,
    hours: Option<f64>,
    step_minutes: Option<f64>,
    votepercent: Option<f64>,
    anomaly_mode: Option<String>,
    detector: Option<String>,
}

#[derive(Serialize, Clone)]
struct Point {
    ts: f64,
    value: f64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    anomaly: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RangeResponse {
    status: &'static str,
    metric: String,
    label: &'static str,
    unit: &'static str,
    query: &'static str,
    start: i64,
    end: i64,
    hours: f64,
    step: i64,
    step_minutes: f64,
    x_range: XRange,
    y_range: Option<YRange>,
    anomaly_count: usize,
    votepercent: f64,
    anomaly_mode: String,
    detector: String,
    warnings: Vec<String>,
    points: Vec<Point>,
}

#[derive(Serialize)]
struct XRange {
    start: f64,
    end: f64,
}
#[derive(Serialize)]
struct YRange {
    min: f64,
    max: f64,
}

#[derive(Deserialize)]
struct PrometheusResponse {
    status: String,
    data: Option<PromData>,
    error: Option<String>,
}
#[derive(Deserialize)]
struct PromData {
    result: Vec<PromSeries>,
}
#[derive(Deserialize)]
struct PromSeries {
    values: Vec<(f64, String)>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let config = Config::from_env();
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("HOST/PORT must form socket address");
    let state = Arc::new(AppState {
        config: RwLock::new(config),
        client: reqwest::Client::new(),
    });
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public");
    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/config", get(public_config))
        .route("/api/llm", post(update_llm))
        .route("/api/templates", get(templates))
        .route("/api/workflows", get(workflows))
        .route("/api/workflows/submit", post(submit_workflow))
        .route("/api/mlflow", get(mlflow))
        .route("/api/prometheus/metrics", get(metric_options))
        .route("/api/prometheus/range", get(metric_range))
        .route("/api/prompt", post(prompt))
        .fallback_service(ServeDir::new(public_dir))
        .with_state(state);

    println!("Rust Argo AutoML Command Deck listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind webui-rust");
    axum::serve(listener, app).await.expect("serve webui-rust");
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn public_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let c = state.config.read().await;
    Json(PublicConfig {
        host: c.host.clone(),
        port: c.port,
        namespace: c.namespace.clone(),
        argo_server: c.argo_server.clone(),
        template_url: c.template_url.clone(),
        grafana_url: c.grafana_url.clone(),
        mlflow_url: c.mlflow_url.clone(),
        prometheus_metrics_url: c.prometheus_metrics_url.clone(),
        prometheus_api_url: c.prometheus_api_url.clone(),
        mcp_url: "/mcp",
        llm: public_llm(&c.llm),
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmUpdate {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
}

async fn update_llm(
    State(state): State<Arc<AppState>>,
    Json(input): Json<LlmUpdate>,
) -> impl IntoResponse {
    let current = state.config.read().await.llm.clone();
    let mut candidate = current;
    if let Some(base_url) = input.base_url.filter(|s| !s.trim().is_empty()) {
        candidate.base_url = base_url.trim().to_string();
    }
    if let Some(model) = input.model.filter(|s| !s.trim().is_empty()) {
        candidate.model = model.trim().to_string();
    }
    if let Some(api_key) = input
        .api_key
        .filter(|s| !s.trim().is_empty() && !s.chars().all(|ch| ch == '*'))
    {
        candidate.api_key = api_key;
    }
    candidate.source = "ui".into();

    if candidate.base_url.trim().is_empty() || candidate.api_key.trim().is_empty() {
        let missing: Vec<&str> = [
            (candidate.base_url.trim().is_empty(), "BASE_URL"),
            (candidate.api_key.trim().is_empty(), "ETE_API_KEY"),
        ]
        .into_iter()
        .filter_map(|(is_missing, name)| is_missing.then_some(name))
        .collect();
        return agent_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &candidate,
            &format!(
                "Agent not configured: missing {} in the submitted credentials.",
                missing.join(" and ")
            ),
        );
    }

    if let Err(message) = probe_llm(&state.client, &candidate).await {
        return agent_error_response(
            StatusCode::BAD_GATEWAY,
            &candidate,
            &format!("Agent not reachable: {message}"),
        );
    }

    let public = public_llm(&candidate);
    state.config.write().await.llm = candidate;
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "ok", "llm": public })),
    )
        .into_response()
}

async fn templates(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let c = state.config.read().await.clone();
    let names = argo_names(&c, &["cluster-template", "list", "-o", "name"]);
    Json(serde_json::json!({
        "status": "ok",
        "templates": names.into_iter().map(|name| serde_json::json!({
            "name": name,
            "source": "argo",
            "parameters": template_parameters(&c, &name),
        })).collect::<Vec<_>>()
    }))
}

async fn workflows(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let c = state.config.read().await.clone();
    let result = run_argo_json(&c, &["-n", &c.namespace, "list", "--output", "json"]);
    Json(result)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubmitInput {
    template_name: String,
    parameters: Option<HashMap<String, String>>,
}

async fn submit_workflow(
    State(state): State<Arc<AppState>>,
    Json(input): Json<SubmitInput>,
) -> impl IntoResponse {
    let c = state.config.read().await.clone();
    if !safe_name(&input.template_name) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "status": "error", "message": "Invalid template name" })),
        );
    }
    let mut args = vec![
        "-n".to_string(),
        c.namespace.clone(),
        "submit".into(),
        "--from".into(),
        format!("clusterworkflowtemplate/{}", input.template_name),
    ];
    for (key, value) in input.parameters.unwrap_or_default() {
        if safe_name(&key) {
            args.push("-p".into());
            args.push(format!("{key}={value}"));
        }
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_argo(&c, &refs);
    let status = if result["status"] == "ok" {
        StatusCode::OK
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status, Json(result))
}

async fn mlflow(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let c = state.config.read().await.clone();
    Json(query_mlflow(&state.client, &c, &q).await)
}

async fn prompt(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let c = state.config.read().await.clone();
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let mut ctx = agent_context(prompt, body.get("context").cloned().unwrap_or_default());
    enrich_context(&state.client, &c, prompt, &mut ctx).await;
    let llm_ctx = compact_context_for_llm(&ctx);

    if c.llm.base_url.trim().is_empty() || c.llm.api_key.trim().is_empty() {
        let missing: Vec<&str> = [
            (c.llm.base_url.trim().is_empty(), "BASE_URL"),
            (c.llm.api_key.trim().is_empty(), "ETE_API_KEY"),
        ]
        .into_iter()
        .filter_map(|(is_missing, name)| is_missing.then_some(name))
        .collect();
        return agent_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &c.llm,
            &format!(
                "Agent not configured: missing {} in process environment or ~/.zshenv.",
                missing.join(" and ")
            ),
        );
    }

    let llm_prompt = prompt_for_llm(prompt, &llm_ctx);
    match call_llm(&state.client, &c.llm, &llm_prompt, &llm_ctx).await {
        Ok(message) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "type": "answer",
                "message": message
            })),
        )
            .into_response(),
        Err(message) => agent_error_response(
            StatusCode::BAD_GATEWAY,
            &c.llm,
            &format!("Agent not reachable: {message}"),
        ),
    }
}

fn mentions(prompt: &str, needle: &str) -> bool {
    prompt.to_lowercase().contains(needle)
}

fn agent_context(prompt: &str, ctx: serde_json::Value) -> serde_json::Value {
    if mentions(prompt, "argo") || mentions(prompt, "mlflow") {
        return ctx;
    }
    let mut out = serde_json::Map::new();
    out.insert(
        "plot".into(),
        ctx.get("plot").cloned().unwrap_or(serde_json::Value::Null),
    );
    if wants_stream_plot(prompt) {
        out.insert(
            "streamPlot".into(),
            ctx.get("streamPlot")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        );
    }
    serde_json::Value::Object(out)
}

fn wants_stream_plot(prompt: &str) -> bool {
    ["quick", "stream", "uni-detector", "quick detector"]
        .iter()
        .any(|needle| mentions(prompt, needle))
}

fn compact_context_for_llm(ctx: &serde_json::Value) -> serde_json::Value {
    let mut compact = ctx.clone();
    if let Some(obj) = compact.as_object_mut() {
        if let Some(plot) = obj.get("plot").cloned() {
            obj.insert("plot".into(), compact_plot_for_llm(&plot));
        }
        if let Some(plot) = obj.get("streamPlot").cloned() {
            obj.insert("streamPlot".into(), compact_plot_for_llm(&plot));
        }
    }
    compact
}

fn compact_plot_for_llm(plot: &serde_json::Value) -> serde_json::Value {
    let Some(points) = plot.get("points").and_then(|v| v.as_array()) else {
        return plot.clone();
    };
    let values: Vec<f64> = points
        .iter()
        .filter_map(|p| finite_number(p.get("value")))
        .collect();
    let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
    let value_min = values.iter().copied().reduce(f64::min);
    let value_max = values.iter().copied().reduce(f64::max);
    let start = plot
        .get("xRange")
        .and_then(|r| r.get("start"))
        .or_else(|| plot.get("start"));
    let end = plot
        .get("xRange")
        .and_then(|r| r.get("end"))
        .or_else(|| plot.get("end"));
    let anomalies: Vec<serde_json::Value> = points
        .iter()
        .filter(|p| p.get("anomaly").and_then(|v| v.as_bool()).unwrap_or(false))
        .take(8)
        .map(|p| {
            serde_json::json!({
                "utc": unix_utc(p.get("ts").unwrap_or(&serde_json::Value::Null)),
                "value": p.get("value").cloned().unwrap_or(serde_json::Value::Null),
            })
        })
        .collect();
    let label = plot.get("label").and_then(|v| v.as_str()).unwrap_or("plot");
    let unit = plot.get("unit").and_then(|v| v.as_str()).unwrap_or("");
    let query = plot
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown query");
    let summary = format!(
        "{label} ({unit}), query {query}, samples {}, min {}, max {}, mean {}, anomalies {}.",
        points.len(),
        value_min
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a".into()),
        value_max
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a".into()),
        mean.map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a".into()),
        anomalies.len()
    );
    serde_json::json!({
        "summary": summary,
        "metric": plot.get("metric").cloned().unwrap_or(serde_json::Value::Null),
        "label": plot.get("label").cloned().unwrap_or(serde_json::Value::Null),
        "unit": plot.get("unit").cloned().unwrap_or(serde_json::Value::Null),
        "query": plot.get("query").cloned().unwrap_or(serde_json::Value::Null),
        "hours": plot.get("hours").cloned().unwrap_or(serde_json::Value::Null),
        "stepMinutes": plot.get("stepMinutes").cloned().unwrap_or(serde_json::Value::Null),
        "startUtc": start.map(unix_utc).unwrap_or_else(|| "unknown".into()),
        "endUtc": end.map(unix_utc).unwrap_or_else(|| "unknown".into()),
        "valueMin": value_min,
        "valueMax": value_max,
        "mean": mean,
        "anomalyCount": plot.get("anomalyCount").cloned().unwrap_or(serde_json::Value::Null),
        "votepercent": plot.get("votepercent").cloned().unwrap_or(serde_json::Value::Null),
        "anomalyMode": plot.get("anomalyMode").cloned().unwrap_or(serde_json::Value::Null),
        "detector": plot.get("detector").cloned().unwrap_or(serde_json::Value::Null),
        "pointCount": points.len(),
        "anomalies": anomalies,
        "pointsOmitted": true,
        "omissionReason": "raw time-series points omitted before LLM call; summary fields and capped anomaly points preserved"
    })
}

fn prompt_for_llm(prompt: &str, ctx: &serde_json::Value) -> String {
    if has_plot_context(ctx) && mentions(prompt, "plot") {
        return "Analyze supplied full-detector plot context. Return concise bullets: metric, query, time window, samples, values, anomalies. Use quick/stream context only if provided. Max 180 words.".into();
    }
    prompt.to_string()
}

fn has_plot_context(ctx: &serde_json::Value) -> bool {
    ["plot", "streamPlot"].iter().any(|key| {
        ctx.get(*key)
            .filter(|v| !v.is_null())
            .and_then(|v| v.as_object())
            .map(|obj| !obj.is_empty())
            .unwrap_or(false)
    })
}

async fn enrich_context(
    client: &reqwest::Client,
    c: &Config,
    prompt: &str,
    ctx: &mut serde_json::Value,
) {
    if !ctx.is_object() {
        *ctx = serde_json::json!({});
    }
    if mentions(prompt, "argo") {
        ctx["argo"] = serde_json::json!({
            "templates": argo_names(c, &["cluster-template", "list", "-o", "name"]),
            "workflows": run_argo_json(c, &["-n", &c.namespace, "list", "--output", "json"]),
        });
    }
    if mentions(prompt, "mlflow") {
        ctx["mlflow"] = query_mlflow(client, c, &HashMap::new()).await;
    }
}

async fn query_mlflow(
    client: &reqwest::Client,
    c: &Config,
    q: &HashMap<String, String>,
) -> serde_json::Value {
    let path = if let Some(run_id) = q.get("runId") {
        format!("/api/2.0/mlflow/runs/get?run_id={run_id}")
    } else {
        "/api/2.0/mlflow/experiments/search".into()
    };
    let Ok(url) = Url::parse(&c.mlflow_url).and_then(|base| base.join(&path)) else {
        return serde_json::json!({ "status": "error", "message": "invalid MLflow URL" });
    };
    let req = if q.get("runId").is_some() {
        client.get(url)
    } else {
        client
            .post(url)
            .json(&serde_json::json!({ "max_results": 20 }))
    };
    match timeout(Duration::from_secs(5), req.send()).await {
        Ok(Ok(res)) => match res.json::<serde_json::Value>().await {
            Ok(v) => serde_json::json!({ "status": "ok", "data": v }),
            Err(e) => serde_json::json!({ "status": "error", "message": e.to_string() }),
        },
        Ok(Err(e)) => serde_json::json!({ "status": "error", "message": e.to_string() }),
        Err(_) => serde_json::json!({ "status": "error", "message": "MLflow timeout" }),
    }
}

fn safe_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

fn argo_names(c: &Config, args: &[&str]) -> Vec<String> {
    run_argo(c, args)
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .lines()
        .map(|s| {
            s.trim()
                .trim_start_matches("clusterworkflowtemplate/")
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn template_parameters(c: &Config, name: &str) -> Vec<serde_json::Value> {
    if !safe_name(name) {
        return vec![];
    }
    let template = run_argo_json(c, &["cluster-template", "get", name, "-o", "json"]);
    template_parameters_from_data(template.get("data").unwrap_or(&template))
}

fn template_parameters_from_data(template: &serde_json::Value) -> Vec<serde_json::Value> {
    template
        .pointer("/spec/arguments/parameters")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|p| {
            let name = p.get("name")?.as_str()?;
            Some(serde_json::json!({
                "name": name,
                "value": p.get("value").and_then(|v| v.as_str()).unwrap_or(""),
                "default": p.get("default").and_then(|v| v.as_str()).unwrap_or(""),
                "required": p.get("value").is_none() && p.get("default").is_none(),
            }))
        })
        .collect()
}

fn run_argo_json(c: &Config, args: &[&str]) -> serde_json::Value {
    let out = run_argo(c, args);
    if out["status"] == "ok"
        && let Some(text) = out.get("output").and_then(|v| v.as_str())
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(text)
    {
        return serde_json::json!({ "status": "ok", "data": json });
    }
    out
}

fn run_argo(c: &Config, args: &[&str]) -> serde_json::Value {
    match Command::new(&c.argo_bin).args(args).output() {
        Ok(out) => serde_json::json!({
            "status": if out.status.success() { "ok" } else { "error" },
            "output": String::from_utf8_lossy(&out.stdout),
            "message": String::from_utf8_lossy(&out.stderr),
        }),
        Err(e) => serde_json::json!({ "status": "error", "message": e.to_string() }),
    }
}

async fn probe_llm(client: &reqwest::Client, llm: &LlmConfig) -> Result<(), String> {
    let ctx = serde_json::json!({
        "plot": {
            "summary": "CPU usage (%), query average non-idle CPU, 24h window, 145 samples, min 2.35, max 2.67, mean 2.38, 4 anomalies.",
            "label": "CPU usage",
            "unit": "%",
            "pointCount": 145,
            "valueMin": 2.35,
            "valueMax": 2.67,
            "mean": 2.38,
            "anomalyCount": 4
        }
    });
    let prompt = prompt_for_llm("Explain this plot.", &ctx);
    call_llm(client, llm, &prompt, &ctx)
        .await
        .map(|_| ())
        .map_err(|message| format!("LLM chat probe failed: {message}"))
}

async fn call_llm(
    client: &reqwest::Client,
    llm: &LlmConfig,
    prompt: &str,
    ctx: &serde_json::Value,
) -> Result<String, String> {
    let mut last_error = String::new();
    for attempt in 1..=3 {
        match call_llm_once(client, llm, prompt, ctx).await {
            Ok(message) => return Ok(message),
            Err(message) if is_retryable_llm_error(&message) && attempt < 3 => {
                last_error = message;
                tokio::time::sleep(Duration::from_millis(250 * attempt)).await;
            }
            Err(message) => return Err(message),
        }
    }
    Err(format!("LLM request failed after 3 attempts: {last_error}"))
}

async fn call_llm_once(
    client: &reqwest::Client,
    llm: &LlmConfig,
    prompt: &str,
    ctx: &serde_json::Value,
) -> Result<String, String> {
    let url = llm_chat_url(&llm.base_url).map_err(|e| format!("invalid LLM base URL: {e}"))?;
    let body = serde_json::json!({
        "model": llm.model,
        "messages": [
            { "role": "system", "content": "You are an Argo AutoML assistant. By default use only the full-detector Prometheus plot. Use quick detector or stream context only when user mentions quick, stream, or uni-detector. Use Argo context only when user mentions Argo. Use MLflow context only when user mentions MLflow. Convert Unix timestamps in answers to YYYY-MM-DD HH:MM:SS UTC. Explain plots with metric, query, time window, sample count, min/max/mean if present, anomaly timestamps, likely cause, and key behavior. State hypotheses as hypotheses. Never claim mutations. Do not use Markdown tables. Use clear bullets and enough detail for debugging." },
            { "role": "user", "content": serde_json::json!({ "prompt": prompt, "context": ctx }).to_string() }
        ]
    });
    let res = timeout(
        Duration::from_secs(60),
        client
            .post(url)
            .bearer_auth(&llm.api_key)
            .json(&body)
            .send(),
    )
    .await
    .map_err(|_| "LLM request timed out after 60s".to_string())?
    .map_err(|e| sanitize_llm_error(&e.to_string(), &llm.api_key))?;
    let status = res.status();
    let text = res
        .text()
        .await
        .map_err(|e| sanitize_llm_error(&e.to_string(), &llm.api_key))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        format!(
            "LLM returned invalid JSON: {e}; body: {}",
            sanitize_llm_error(&text, &llm.api_key)
        )
    })?;
    if !status.is_success() {
        return Err(format!(
            "LLM HTTP {status}: {}",
            sanitize_llm_error(&json.to_string(), &llm.api_key)
        ));
    }
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim();
    if content.is_empty() {
        return Err(format!(
            "LLM returned no message content: {}",
            sanitize_llm_error(&json.to_string(), &llm.api_key)
        ));
    }
    Ok(content.to_string())
}

fn is_retryable_llm_error(message: &str) -> bool {
    [
        "error sending request",
        "timed out",
        "connection",
        "dns",
        "tcp",
        "tls",
    ]
    .iter()
    .any(|needle| message.to_lowercase().contains(needle))
}

fn agent_error_response(
    status: StatusCode,
    llm: &LlmConfig,
    message: &str,
) -> axum::response::Response {
    (
        status,
        Json(serde_json::json!({
            "status": "error",
            "type": "agent_error",
            "message": sanitize_llm_error(message, &llm.api_key),
            "agent": {
                "baseUrl": llm.base_url,
                "model": llm.model,
                "hasKey": !llm.api_key.is_empty(),
            },
        })),
    )
        .into_response()
}

fn llm_chat_url(base_url: &str) -> Result<Url, url::ParseError> {
    llm_endpoint_url(base_url, "chat/completions")
}

fn llm_endpoint_url(base_url: &str, suffix: &str) -> Result<Url, url::ParseError> {
    let mut url = Url::parse(base_url.trim())?;
    let mut path = url.path().trim_end_matches('/').to_string();
    for endpoint in ["/chat/completions", "/models"] {
        if path.ends_with(endpoint) {
            path.truncate(path.len() - endpoint.len());
        }
    }
    if path.is_empty() {
        path = "/v1".into();
    }
    url.set_path(&format!("{path}/{suffix}"));
    Ok(url)
}

fn finite_number(value: Option<&serde_json::Value>) -> Option<f64> {
    value.and_then(|v| v.as_f64()).filter(|v| v.is_finite())
}

fn sanitize_llm_error(message: &str, api_key: &str) -> String {
    if api_key.is_empty() {
        message.to_string()
    } else {
        message.replace(api_key, "<redacted>")
    }
}

fn unix_utc(ts: &serde_json::Value) -> String {
    let Some(seconds) = ts.as_f64().filter(|v| v.is_finite()) else {
        return ts.to_string();
    };
    let secs = seconds.floor() as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += if month <= 2 { 1 } else { 0 };
    (year, month, day)
}

async fn metric_options() -> Json<serde_json::Value> {
    let opts: Vec<_> = metrics().into_iter().map(|m| m.opt).collect();
    Json(serde_json::json!({ "status": "ok", "metrics": opts, "quickDetectors": DETECTORS }))
}

async fn metric_range(
    State(state): State<Arc<AppState>>,
    Query(q): Query<RangeQuery>,
) -> Result<Json<RangeResponse>, (StatusCode, Json<serde_json::Value>)> {
    let metric_id = q.metric.unwrap_or_else(|| "cpu".into());
    let metric = metrics()
        .into_iter()
        .find(|m| m.opt.id == metric_id)
        .ok_or_else(|| bad_request(format!("Unknown metric: {metric_id}")))?;
    let hours = bounded(q.hours, 24.0, 1.0, 24.0 * 14.0);
    let step_minutes = bounded(q.step_minutes, 10.0, 1.0, 24.0 * 60.0);
    let votepercent = bounded(q.votepercent, 0.5, 0.1, 1.0);
    let mode = if q.anomaly_mode.as_deref() == Some("full") {
        "full"
    } else {
        "quick"
    }
    .to_string();
    let detector = detector_id(q.detector.as_deref()).unwrap_or("").to_string();
    let end = now_seconds();
    let start = end - (hours * 3600.0).round() as i64;
    let step = (step_minutes * 60.0).round() as i64;
    let mut url = {
        let c = state.config.read().await;
        prometheus_base(&c).map_err(|e| bad_request(e.to_string()))?
    };
    url.set_path("/api/v1/query_range");
    url.query_pairs_mut()
        .append_pair("query", metric.query)
        .append_pair("start", &start.to_string())
        .append_pair("end", &end.to_string())
        .append_pair("step", &step.to_string());

    let prom: PrometheusResponse = state
        .client
        .get(url)
        .send()
        .await
        .map_err(internal)?
        .json()
        .await
        .map_err(internal)?;
    if prom.status == "error" {
        return Err(internal_msg(
            prom.error.unwrap_or_else(|| "Prometheus error".into()),
        ));
    }
    let series = prom
        .data
        .and_then(|d| d.result.into_iter().next())
        .map(|s| s.values)
        .unwrap_or_default();
    let mut points: Vec<Point> = series
        .into_iter()
        .filter_map(|(ts, v)| {
            v.parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .map(|value| Point {
                    ts,
                    value,
                    anomaly: false,
                })
        })
        .collect();
    let anomaly = detect_anomalies(&points, votepercent, &mode, &detector);
    for idx in &anomaly.indexes {
        if let Some(p) = points.get_mut(*idx) {
            p.anomaly = true;
        }
    }
    let (min, max) = points
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), p| {
            (mn.min(p.value), mx.max(p.value))
        });
    let y_range = if points.is_empty() {
        None
    } else {
        Some(YRange { min, max })
    };
    let x_start = points.first().map(|p| p.ts).unwrap_or(start as f64);
    let x_end = points.last().map(|p| p.ts).unwrap_or(end as f64);

    Ok(Json(RangeResponse {
        status: "ok",
        metric: metric_id,
        label: metric.opt.label,
        unit: metric.opt.unit,
        query: metric.query,
        start,
        end,
        hours,
        step,
        step_minutes,
        x_range: XRange {
            start: x_start,
            end: x_end,
        },
        y_range,
        anomaly_count: anomaly.indexes.len(),
        votepercent,
        anomaly_mode: anomaly.mode,
        detector,
        warnings: anomaly.warnings,
        points,
    }))
}

struct AnomalyResult {
    indexes: Vec<usize>,
    mode: String,
    warnings: Vec<String>,
}

fn detect_anomalies(
    points: &[Point],
    votepercent: f64,
    mode: &str,
    detector: &str,
) -> AnomalyResult {
    if points.len() < 25 {
        return AnomalyResult {
            indexes: vec![],
            mode: mode.into(),
            warnings: vec!["Need at least 25 points for Rust anomaly detection.".into()],
        };
    }
    let values: Vec<f64> = points.iter().map(|p| p.value).collect();
    let indexes = if detector == "caret:mcd" || detector == "sk:EllipticEnvelope" {
        modified_zscore(&values, 3.5)
    } else if detector == "sk:LocalOutlierFactor" {
        neighbor_delta(&values, 3.0)
    } else if detector == "caret:pca" {
        moving_residual(&values, 3.0)
    } else if mode == "full" {
        vote(
            &[
                modified_zscore(&values, 3.5),
                neighbor_delta(&values, 3.0),
                moving_residual(&values, 3.0),
            ],
            values.len(),
            votepercent,
        )
    } else {
        vote(
            &[neighbor_delta(&values, 3.0), moving_residual(&values, 3.0)],
            values.len(),
            votepercent,
        )
    };
    AnomalyResult {
        indexes,
        mode: mode.into(),
        warnings: vec![],
    }
}

fn vote(cols: &[Vec<usize>], len: usize, votepercent: f64) -> Vec<usize> {
    let mut counts = vec![0usize; len];
    for col in cols {
        for &idx in col {
            if idx < len {
                counts[idx] += 1;
            }
        }
    }
    counts
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| ((c as f64 / cols.len() as f64) >= votepercent).then_some(i))
        .collect()
}

fn modified_zscore(values: &[f64], threshold: f64) -> Vec<usize> {
    let med = median(values.to_vec());
    let mad = median(values.iter().map(|v| (v - med).abs()).collect());
    let scale = if mad == 0.0 {
        stddev(values).max(1e-9)
    } else {
        mad / 0.6745
    };
    values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (((v - med).abs() / scale) > threshold).then_some(i))
        .collect()
}

fn neighbor_delta(values: &[f64], threshold: f64) -> Vec<usize> {
    let deltas: Vec<f64> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let prev = i
                .checked_sub(1)
                .and_then(|j| values.get(j))
                .copied()
                .unwrap_or(*v);
            let next = values.get(i + 1).copied().unwrap_or(*v);
            (v - ((prev + next) / 2.0)).abs()
        })
        .collect();
    let med = median(deltas.clone());
    let sd = stddev(&deltas).max(1e-9);
    deltas
        .into_iter()
        .enumerate()
        .filter_map(|(i, d)| (d > med + threshold * sd).then_some(i))
        .collect()
}

fn moving_residual(values: &[f64], threshold: f64) -> Vec<usize> {
    let residuals: Vec<f64> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(values.len());
            let avg = values[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            (v - avg).abs()
        })
        .collect();
    let med = median(residuals.clone());
    let sd = stddev(&residuals).max(1e-9);
    residuals
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| (r > med + threshold * sd).then_some(i))
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|v| v.is_finite());
    values.sort_by(|a, b| a.total_cmp(b));
    match values.len() {
        0 => 0.0,
        n if n % 2 == 1 => values[n / 2],
        n => (values[n / 2 - 1] + values[n / 2]) / 2.0,
    }
}

fn stddev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

fn detector_id(value: Option<&str>) -> Option<&'static str> {
    value.and_then(|v| DETECTORS.iter().find(|d| d.id == v).map(|d| d.id))
}

fn prometheus_base(config: &Config) -> Result<Url, url::ParseError> {
    let raw = if !config.prometheus_api_url.is_empty() {
        &config.prometheus_api_url
    } else {
        &config.prometheus_metrics_url
    };
    let mut url = Url::parse(raw)?;
    if url.path().ends_with("/metrics") {
        let path = url.path().trim_end_matches("/metrics").to_string();
        url.set_path(if path.is_empty() { "/" } else { &path });
    }
    Ok(url)
}

fn bounded(value: Option<f64>, fallback: f64, min: f64, max: f64) -> f64 {
    value
        .filter(|n| n.is_finite())
        .unwrap_or(fallback)
        .clamp(min, max)
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn bad_request(message: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "status": "error", "message": message })),
    )
}

fn internal(err: impl std::fmt::Display) -> (StatusCode, Json<serde_json::Value>) {
    internal_msg(err.to_string())
}

fn internal_msg(message: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "status": "error", "message": message })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_spike_without_julia() {
        let mut points: Vec<Point> = (0..30)
            .map(|i| Point {
                ts: i as f64,
                value: i as f64,
                anomaly: false,
            })
            .collect();
        points[15].value = 999.0;
        let out = detect_anomalies(&points, 0.5, "quick", "caret:pca");
        assert!(out.indexes.contains(&15));
    }

    #[test]
    fn rejects_unknown_detector() {
        assert_eq!(detector_id(Some("sk:OneClassSVM")), None);
        assert_eq!(detector_id(Some("caret:pca")), Some("caret:pca"));
    }

    #[test]
    fn bounds_numbers() {
        assert_eq!(bounded(Some(0.0), 24.0, 1.0, 48.0), 1.0);
        assert_eq!(bounded(Some(99.0), 24.0, 1.0, 48.0), 48.0);
        assert_eq!(bounded(Some(f64::NAN), 24.0, 1.0, 48.0), 24.0);
    }

    #[test]
    fn converts_unix_time_to_utc() {
        assert_eq!(unix_utc(&serde_json::json!(0)), "1970-01-01 00:00:00 UTC");
        assert_eq!(
            unix_utc(&serde_json::json!(1_700_000_000)),
            "2023-11-14 22:13:20 UTC"
        );
    }

    #[test]
    fn parses_zshenv_assignments_for_llm_defaults() {
        let parsed = parse_shell_assignments(
            r#"
            export BASE_URL='https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1'
            ETE_API_KEY="secret-key"
            LLM_MODEL=azure/gpt-5.5
            # ignored comment
            not valid=value
            "#,
        );
        assert_eq!(
            parsed.get("BASE_URL").map(String::as_str),
            Some("https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1")
        );
        assert_eq!(
            parsed.get("ETE_API_KEY").map(String::as_str),
            Some("secret-key")
        );
        assert_eq!(
            parsed.get("LLM_MODEL").map(String::as_str),
            Some("azure/gpt-5.5")
        );
        assert!(!parsed.contains_key("not valid"));
    }

    #[test]
    fn llm_config_ignores_openai_and_pi_fallbacks() {
        let home = env::temp_dir().join(format!("webui-rust-test-{}", random_hex(8)));
        fs::create_dir_all(home.join(".pi/agent")).unwrap();
        fs::write(
            home.join(".pi/agent/models.json"),
            r#"{"providers":{"external":{"apiKey":"pi-key","baseUrl":"https://pi.example/v1","models":[{"id":"pi/model"}]}}}"#,
        )
        .unwrap();
        let vars = vec![
            ("HOME", Some(home.display().to_string())),
            ("BASE_URL", None),
            ("ETE_API_KEY", None),
            ("LLM_MODEL", None),
            (
                "OPENAI_BASE_URL",
                Some("https://openai.example/v1".to_string()),
            ),
            ("OPENAI_API_KEY", Some("openai-key".to_string())),
            ("OPENAI_MODEL", Some("openai/model".to_string())),
        ];
        temp_env::with_vars(vars, || {
            let config = load_llm_config();
            assert_eq!(config.base_url, "");
            assert_eq!(config.api_key, "");
            assert_eq!(config.model, "azure/gpt-5.5");
            assert_eq!(config.source, "missing-env");
        });
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn llm_chat_url_preserves_v1_path() {
        assert_eq!(
            llm_chat_url("https://example.test").unwrap().as_str(),
            "https://example.test/v1/chat/completions"
        );
        assert_eq!(
            llm_chat_url("https://example.test/v1").unwrap().as_str(),
            "https://example.test/v1/chat/completions"
        );
        assert_eq!(
            llm_chat_url("https://example.test/v1/").unwrap().as_str(),
            "https://example.test/v1/chat/completions"
        );
        assert_eq!(
            llm_chat_url("https://example.test/v1/chat/completions")
                .unwrap()
                .as_str(),
            "https://example.test/v1/chat/completions"
        );
    }

    #[test]
    fn plot_prompts_are_normalized_for_remote_llm() {
        let ctx = serde_json::json!({
            "plot": { "label": "cpu", "pointCount": 145 }
        });
        let prompt = prompt_for_llm(
            "Explain this plot in detail. Focus on metric, query, time window, min/max/mean, and potential anomalies.",
            &ctx,
        );
        assert!(prompt.contains("Analyze supplied full-detector plot context"));
        assert!(prompt.contains("Max 180 words"));
    }

    #[test]
    fn compact_context_for_llm_omits_raw_plot_points() {
        let ctx = serde_json::json!({
            "plot": {
                "label": "cpu",
                "unit": "%",
                "query": "cpu_query",
                "points": [
                    { "ts": 1_700_000_000.0, "value": 1.0, "anomaly": false },
                    { "ts": 1_700_000_600.0, "value": 99.0, "anomaly": true }
                ]
            },
            "streamPlot": {
                "label": "cpu stream",
                "points": [
                    { "ts": 1_700_000_000.0, "value": 2.0, "anomaly": false }
                ]
            }
        });
        let compact = compact_context_for_llm(&ctx);
        assert!(compact["plot"].get("points").is_none());
        assert_eq!(compact["plot"]["pointCount"], 2);
        assert_eq!(compact["plot"]["pointsOmitted"], true);
        assert_eq!(compact["plot"]["anomalies"].as_array().unwrap().len(), 1);
        assert!(compact["streamPlot"].get("points").is_none());
        assert_eq!(compact["streamPlot"]["pointCount"], 1);
    }

    #[test]
    fn sanitize_llm_error_redacts_api_key() {
        assert_eq!(
            sanitize_llm_error("bad key secret-key", "secret-key"),
            "bad key <redacted>"
        );
    }

    #[test]
    fn retryable_llm_errors_cover_transient_transport_failures() {
        assert!(is_retryable_llm_error(
            "error sending request for url (https://example.test/v1/chat/completions)"
        ));
        assert!(is_retryable_llm_error("LLM request timed out after 60s"));
        assert!(!is_retryable_llm_error("LLM returned invalid JSON"));
    }

    fn test_config(llm: LlmConfig) -> Config {
        Config {
            host: "127.0.0.1".into(),
            port: 0,
            namespace: "argo".into(),
            argo_server: "http://argo.test".into(),
            template_url: "http://argo.test/templates".into(),
            grafana_url: "http://grafana.test".into(),
            mlflow_url: "http://mlflow.test".into(),
            prometheus_metrics_url: "http://prometheus.test/metrics".into(),
            prometheus_api_url: "http://prometheus.test".into(),
            argo_bin: "argo".into(),
            session_token: "token".into(),
            llm,
        }
    }

    #[tokio::test]
    async fn prompt_returns_visible_agent_error_when_config_missing() {
        let state = Arc::new(AppState {
            config: RwLock::new(test_config(LlmConfig {
                base_url: String::new(),
                model: "azure/gpt-5.5".into(),
                source: "missing-env".into(),
                api_key: String::new(),
            })),
            client: reqwest::Client::new(),
        });
        let response = prompt(
            State(state),
            Json(serde_json::json!({ "prompt": "explain plot", "context": {} })),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["type"], "agent_error");
        assert!(json["message"].as_str().unwrap().contains("BASE_URL"));
        assert!(json["message"].as_str().unwrap().contains("ETE_API_KEY"));
        assert!(json.get("localAnswer").is_none());
    }

    #[tokio::test]
    async fn update_llm_rejects_unreachable_agent_and_keeps_current_config() {
        let state = Arc::new(AppState {
            config: RwLock::new(test_config(LlmConfig {
                base_url: "https://old.example/v1".into(),
                model: "old/model".into(),
                source: "env".into(),
                api_key: "old-secret".into(),
            })),
            client: reqwest::Client::new(),
        });
        let response = update_llm(
            State(state.clone()),
            Json(LlmUpdate {
                api_key: Some("new-secret".into()),
                base_url: Some("http://127.0.0.1:9/v1".into()),
                model: Some("bad/model".into()),
            }),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "agent_error");
        assert!(
            json["message"]
                .as_str()
                .unwrap()
                .contains("Agent not reachable")
        );
        assert!(!json.to_string().contains("new-secret"));
        let saved = state.config.read().await.llm.clone();
        assert_eq!(saved.base_url, "https://old.example/v1");
        assert_eq!(saved.model, "old/model");
        assert_eq!(saved.api_key, "old-secret");
    }

    #[test]
    fn prompt_context_defaults_to_full_detector_plot_only() {
        let ctx = serde_json::json!({
            "plot": { "label": "cpu", "anomalyMode": "full" },
            "streamPlot": { "label": "stream", "anomalyMode": "quick" },
            "logs": "secret logs",
            "argo": { "workflow": "w" },
            "mlflow": { "run": "r" }
        });
        let out = agent_context("explain this plot", ctx);
        assert_eq!(
            out.get("plot")
                .and_then(|v| v.get("anomalyMode"))
                .and_then(|v| v.as_str()),
            Some("full")
        );
        assert!(out.get("streamPlot").is_none());
        assert!(out.get("logs").is_none());
        assert!(out.get("argo").is_none());
        assert!(out.get("mlflow").is_none());
    }

    #[test]
    fn prompt_context_keeps_quick_stream_when_requested() {
        let ctx = serde_json::json!({
            "plot": { "label": "cpu", "anomalyMode": "full" },
            "streamPlot": { "label": "stream", "anomalyMode": "quick" }
        });
        let out = agent_context("explain quick detector stream", ctx);
        assert_eq!(
            out.get("streamPlot")
                .and_then(|v| v.get("anomalyMode"))
                .and_then(|v| v.as_str()),
            Some("quick")
        );
    }

    #[test]
    fn prompt_context_keeps_all_when_argo_or_mlflow_mentioned() {
        let ctx = serde_json::json!({ "logs": "keep", "argo": {}, "mlflow": {} });
        assert_eq!(agent_context("show argo", ctx.clone()), ctx);
        assert_eq!(agent_context("show mlflow", ctx.clone()), ctx);
    }

    #[test]
    fn extracts_template_parameters() {
        let template = serde_json::json!({
            "spec": { "arguments": { "parameters": [
                { "name": "dataset", "value": "iris" },
                { "name": "epochs", "default": "10" },
                { "name": "model" }
            ] } }
        });
        let params = template_parameters_from_data(&template);
        assert_eq!(params.len(), 3);
        assert_eq!(params[0]["name"], "dataset");
        assert_eq!(params[1]["default"], "10");
        assert_eq!(params[2]["required"], true);
    }
}
