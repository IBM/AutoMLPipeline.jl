import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';

function resolveSecret(value, env) {
  const match = String(value || '').match(/^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$/);
  if (!match) return value || '';
  return env[match[1]] || '';
}

function loadPiExternal(modelsFile, env) {
  try {
    const file = modelsFile || path.join(os.homedir(), '.pi', 'agent', 'models.json');
    const json = JSON.parse(fs.readFileSync(file, 'utf8'));
    const provider = json.providers?.external;
    const model = provider?.models?.find?.((m) => m.id === 'azure/gpt-5.5') || provider?.models?.[0];
    return { apiKey: resolveSecret(provider?.apiKey, env), baseUrl: provider?.baseUrl, model: model?.id };
  } catch {
    return {};
  }
}

function maskSecret(value) {
  return value ? '*'.repeat(Math.min(String(value).length, 24)) : '';
}

export function llmFromRequest(current, body = {}) {
  return {
    ...current,
    apiKey: body.apiKey ? String(body.apiKey) : current.apiKey,
    baseUrl: body.baseUrl ? String(body.baseUrl) : current.baseUrl,
    model: body.model ? String(body.model) : current.model,
    source: 'ui'
  };
}

export function buildConfig(env = process.env) {
  const pi = loadPiExternal(env.PI_MODELS_FILE, env);
  const sessionToken = env.ARGO_WEBUI_TOKEN || crypto.randomBytes(24).toString('hex');
  const host = env.HOST || '127.0.0.1';
  // ponytail: OPENAI_API_KEY is often a stale global env var; require base-url/model override before it beats Pi ETE config.
  const useEnvLlm = Boolean(env.OPENAI_BASE_URL || env.OPENAI_MODEL || env.ARGO_WEBUI_USE_OPENAI_ENV === '1');
  const llm = useEnvLlm ? {
    apiKey: env.OPENAI_API_KEY || pi.apiKey || '',
    baseUrl: env.OPENAI_BASE_URL || pi.baseUrl || 'https://api.openai.com/v1',
    model: env.OPENAI_MODEL || pi.model || 'gpt-4o-mini',
    source: 'env'
  } : {
    apiKey: pi.apiKey || env.OPENAI_API_KEY || '',
    baseUrl: pi.baseUrl || 'https://api.openai.com/v1',
    model: pi.model || 'gpt-4o-mini',
    source: pi.apiKey ? 'pi-external' : 'env-fallback'
  };
  return {
    host,
    port: Number(env.PORT || 8090),
    namespace: env.ARGO_NAMESPACE || 'argo',
    argoServer: env.ARGO_SERVER || 'http://argo.home:8080',
    templateUrl: env.ARGO_TEMPLATE_URL || 'http://argo.home:8080/cluster-workflow-templates',
    grafanaUrl: env.GRAFANA_URL || 'http://grafana.home:8080',
    mlflowUrl: env.MLFLOW_URL || 'http://mlflow.home:8080',
    prometheusMetricsUrl: env.PROMETHEUS_METRICS_URL || 'http://prom1.prometheus.home:8080/metrics',
    prometheusApiUrl: env.PROMETHEUS_API_URL || 'http://prom2.prometheus.home:8080',
    argoBin: env.ARGO_BIN || 'argo',
    mcpAllowedOperations: (env.ARGO_MCP_ALLOWED_OPERATIONS || 'submit').split(',').map((s) => s.trim()).filter(Boolean),
    sessionToken,
    llm
  };
}

export function publicConfig(config) {
  return {
    host: config.host,
    port: config.port,
    namespace: config.namespace,
    argoServer: config.argoServer,
    templateUrl: config.templateUrl,
    grafanaUrl: config.grafanaUrl,
    mlflowUrl: config.mlflowUrl,
    prometheusMetricsUrl: config.prometheusMetricsUrl,
    prometheusApiUrl: config.prometheusApiUrl,
    mcpUrl: '/mcp',
    llm: { baseUrl: config.llm.baseUrl, model: config.llm.model, source: config.llm.source, hasKey: Boolean(config.llm.apiKey), apiKeyMasked: maskSecret(config.llm.apiKey) }
  };
}
