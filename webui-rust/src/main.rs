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
    let pi = load_pi_external();
    let has_pi = pi.api_key.is_some() || pi.base_url.is_some() || pi.model.is_some();
    LlmConfig {
        // Pi models.json is the default; env vars only fill gaps.
        api_key: pi
            .api_key
            .or_else(|| env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty()))
            .unwrap_or_default(),
        base_url: pi
            .base_url
            .or_else(|| env::var("OPENAI_BASE_URL").ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "https://api.openai.com/v1".into()),
        model: pi
            .model
            .or_else(|| env::var("OPENAI_MODEL").ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "gpt-4o-mini".into()),
        source: if has_pi { "pi-models" } else { "env" }.into(),
    }
}

#[derive(Default)]
struct PiExternal {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
}

fn load_pi_external() -> PiExternal {
    let path = env::var("PI_MODELS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("HOME").unwrap_or_default()).join(".pi/agent/models.json")
        });
    let Ok(text) = fs::read_to_string(path) else {
        return PiExternal::default();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return PiExternal::default();
    };
    let provider = &json["providers"]["external"];
    let api_key = provider["apiKey"]
        .as_str()
        .map(resolve_secret)
        .filter(|s| !s.is_empty());
    let base_url = provider["baseUrl"]
        .as_str()
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let model = provider["models"].as_array().and_then(|models| {
        models
            .iter()
            .find(|m| m["id"].as_str() == Some("azure/gpt-5.5"))
            .or_else(|| models.first())
            .and_then(|m| m["id"].as_str())
            .map(str::to_string)
    });
    PiExternal {
        api_key,
        base_url,
        model,
    }
}

fn resolve_secret(value: &str) -> String {
    let trimmed = value.trim();
    let name = trimmed
        .strip_prefix("${")
        .and_then(|s| s.strip_suffix('}'))
        .or_else(|| trimmed.strip_prefix('$'));
    name.and_then(|n| env::var(n).ok())
        .unwrap_or_else(|| trimmed.to_string())
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
) -> Json<serde_json::Value> {
    let mut c = state.config.write().await;
    if let Some(base_url) = input.base_url.filter(|s| !s.trim().is_empty()) {
        c.llm.base_url = base_url.trim().to_string();
    }
    if let Some(model) = input.model.filter(|s| !s.trim().is_empty()) {
        c.llm.model = model.trim().to_string();
    }
    if let Some(api_key) = input
        .api_key
        .filter(|s| !s.trim().is_empty() && !s.chars().all(|ch| ch == '*'))
    {
        c.llm.api_key = api_key;
    }
    c.llm.source = "ui".into();
    Json(serde_json::json!({ "status": "ok", "llm": public_llm(&c.llm) }))
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
) -> Json<serde_json::Value> {
    let c = state.config.read().await.clone();
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let mut ctx = agent_context(prompt, body.get("context").cloned().unwrap_or_default());
    enrich_context(&state.client, &c, prompt, &mut ctx).await;
    if !c.llm.api_key.is_empty() {
        if let Ok(message) = call_llm(&state.client, &c.llm, prompt, &ctx).await {
            return Json(serde_json::json!({ "type": "answer", "message": message }));
        }
    }
    Json(local_answer(prompt, &ctx))
}

fn mentions(prompt: &str, needle: &str) -> bool {
    prompt.to_lowercase().contains(needle)
}

fn agent_context(prompt: &str, ctx: serde_json::Value) -> serde_json::Value {
    if mentions(prompt, "argo") || mentions(prompt, "mlflow") {
        return ctx;
    }
    serde_json::json!({
        "plot": ctx.get("plot").cloned().unwrap_or(serde_json::Value::Null),
        "streamPlot": ctx.get("streamPlot").cloned().unwrap_or(serde_json::Value::Null),
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
    if out["status"] == "ok" {
        if let Some(text) = out.get("output").and_then(|v| v.as_str()) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                return serde_json::json!({ "status": "ok", "data": json });
            }
        }
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

async fn call_llm(
    client: &reqwest::Client,
    llm: &LlmConfig,
    prompt: &str,
    ctx: &serde_json::Value,
) -> Result<String, String> {
    let url = Url::parse(&llm.base_url)
        .and_then(|base| base.join("chat/completions"))
        .map_err(|e| e.to_string())?;
    let body = serde_json::json!({
        "model": llm.model,
        "messages": [
            { "role": "system", "content": "You are an Argo AutoML assistant. By default use only Prometheus plots and metric anomaly stream. Use Argo context only when user mentions Argo. Use MLflow context only when user mentions MLflow. Convert Unix timestamps in answers to YYYY-MM-DD HH:MM:SS UTC. Be concise. State hypotheses as hypotheses. Never claim mutations. Do not use Markdown tables. Use bullet points for metrics; align related metric fields in the same bullet." },
            { "role": "user", "content": serde_json::json!({ "prompt": prompt, "context": ctx }).to_string() }
        ]
    });
    let res = client
        .post(url)
        .bearer_auth(&llm.api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = res.status();
    let json: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(json.to_string());
    }
    Ok(json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("No LLM content returned.")
        .to_string())
}

fn local_answer(prompt: &str, ctx: &serde_json::Value) -> serde_json::Value {
    let lower = prompt.to_lowercase();
    let message = if lower.contains("plot") || lower.contains("metric") || lower.contains("anomal")
    {
        local_plot_answer(ctx)
    } else if lower.contains("mlflow") {
        serde_json::to_string_pretty(&ctx["mlflow"]).unwrap_or_else(|_| "No MLflow context.".into())
    } else if lower.contains("argo") || lower.contains("workflow") {
        serde_json::to_string_pretty(&ctx["argo"]).unwrap_or_else(|_| "No Argo context.".into())
    } else {
        "## Rust webui\n\nAgent context defaults to plots only. Mention Argo or MLflow to include that context.".into()
    };
    serde_json::json!({ "type": "local_answer", "message": message })
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

fn local_plot_answer(ctx: &serde_json::Value) -> String {
    fn section(name: &str, plot: Option<&serde_json::Value>) -> String {
        let Some(plot) = plot else {
            return String::new();
        };
        let label = plot.get("label").and_then(|v| v.as_str()).unwrap_or("plot");
        let unit = plot.get("unit").and_then(|v| v.as_str()).unwrap_or("");
        let point_count = plot
            .get("points")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let anomalies: Vec<String> = plot
            .get("points")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter(|p| p.get("anomaly").and_then(|v| v.as_bool()).unwrap_or(false))
            .take(12)
            .map(|p| {
                format!(
                    "- {}: {} {unit}",
                    unix_utc(p.get("ts").unwrap_or(&serde_json::Value::Null)),
                    p.get("value").unwrap_or(&serde_json::Value::Null)
                )
            })
            .collect();
        format!(
            "### {name}\n\n{label}: {}/{} anomalous points.\n\n{}",
            anomalies.len(),
            point_count,
            if anomalies.is_empty() {
                "- none".into()
            } else {
                anomalies.join("\n")
            }
        )
    }
    format!(
        "## Plot anomalies\n\n{}\n\n{}\n\nPossible reason: correlate timestamps with workflow logs, node/pod restarts, traffic spikes, memory pressure, network saturation, or latency spikes.",
        section("Prometheus plot", ctx.get("plot")),
        section("Metric anomaly stream", ctx.get("streamPlot"))
    )
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
        assert_eq!(unix_utc(&serde_json::json!(1_700_000_000)), "2023-11-14 22:13:20 UTC");
    }

    #[test]
    fn local_plot_answer_uses_utc_time() {
        let ctx = serde_json::json!({
            "plot": { "label": "cpu", "unit": "%", "points": [
                { "ts": 1_700_000_000.0, "value": 99, "anomaly": true }
            ] }
        });
        assert!(local_plot_answer(&ctx).contains("2023-11-14 22:13:20 UTC"));
    }

    #[test]
    fn prompt_context_defaults_to_plots_only() {
        let ctx = serde_json::json!({
            "plot": { "label": "cpu" },
            "streamPlot": { "label": "stream" },
            "logs": "secret logs",
            "argo": { "workflow": "w" },
            "mlflow": { "run": "r" }
        });
        let out = agent_context("what happened?", ctx);
        assert_eq!(out.get("plot").and_then(|v| v.get("label")).and_then(|v| v.as_str()), Some("cpu"));
        assert_eq!(out.get("streamPlot").and_then(|v| v.get("label")).and_then(|v| v.as_str()), Some("stream"));
        assert!(out.get("logs").is_none());
        assert!(out.get("argo").is_none());
        assert!(out.get("mlflow").is_none());
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
