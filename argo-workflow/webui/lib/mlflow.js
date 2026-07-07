import { redact } from './redact.js';

export async function mlflowFetch(config, apiPath, body) {
  const res = await fetch(new URL(apiPath, config.mlflowUrl), {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'content-type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(8000)
  });
  const text = await res.text();
  if (!res.ok) throw new Error(redact(`MLflow HTTP ${res.status}: ${text}`));
  return JSON.parse(redact(text || '{}'));
}

function qs(params = {}) {
  const search = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => [k, String(v)]));
  return search.toString();
}

function pairs(items = []) {
  return Object.fromEntries(items.map(({ key, value }) => [key, value]));
}

export const searchExperiments = (config, body = {}) => mlflowFetch(config, '/api/2.0/mlflow/experiments/search', { max_results: 100, ...body });
export const listExperiments = searchExperiments;
export const getExperiment = (config, experimentId) => mlflowFetch(config, `/api/2.0/mlflow/experiments/get?${qs({ experiment_id: experimentId })}`);
export const getExperimentByName = (config, name) => mlflowFetch(config, `/api/2.0/mlflow/experiments/get-by-name?${qs({ experiment_name: name })}`);
export const searchRuns = (config, body = {}) => mlflowFetch(config, '/api/2.0/mlflow/runs/search', { max_results: 50, ...body });
export const listRuns = searchRuns;
export const getRun = (config, runId) => mlflowFetch(config, `/api/2.0/mlflow/runs/get?${qs({ run_id: runId })}`);
export const getMetricHistory = (config, { runId, metricKey } = {}) => mlflowFetch(config, `/api/2.0/mlflow/metrics/get-history?${qs({ run_id: runId, metric_key: metricKey })}`);
export const listArtifacts = (config, { runId, path } = {}) => mlflowFetch(config, `/api/2.0/mlflow/artifacts/list?${qs({ run_id: runId, path })}`);

export async function summarizeRun(config, { runId } = {}) {
  const { run } = await getRun(config, runId);
  return {
    info: run?.info || {},
    params: pairs(run?.data?.params),
    metrics: pairs(run?.data?.metrics),
    tags: pairs(run?.data?.tags),
    inputs: run?.inputs || {}
  };
}

export async function compareRuns(config, { experimentIds, filter, metricKeys = [], maxResults = 20, orderBy } = {}) {
  const result = await searchRuns(config, {
    experiment_ids: experimentIds,
    filter,
    max_results: maxResults,
    order_by: orderBy
  });
  return {
    runs: (result.runs || []).map((run) => ({
      run_id: run.info?.run_id,
      name: run.info?.run_name,
      status: run.info?.status,
      start_time: run.info?.start_time,
      end_time: run.info?.end_time,
      metrics: Object.fromEntries((run.data?.metrics || []).filter((m) => !metricKeys.length || metricKeys.includes(m.key)).map(({ key, value }) => [key, value])),
      params: pairs(run.data?.params),
      tags: pairs(run.data?.tags)
    })),
    next_page_token: result.next_page_token
  };
}

export async function queryMlflowResults(config, query = {}) {
  if (query.runId) return { run: await getRun(config, query.runId) };
  if (query.experimentId) return { runs: await searchRuns(config, { experiment_ids: [query.experimentId] }) };
  return { experiments: await listExperiments(config) };
}
