import { redact } from './redact.js';
import { detectAnomalies } from './anomaly.js';

export const METRICS = {
  cpu: {
    label: 'CPU usage',
    unit: '%',
    query: '100 * avg(rate(node_cpu_seconds_total{mode!="idle"}[10m]))'
  },
  memory: {
    label: 'Memory usage',
    unit: '%',
    query: '100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))'
  },
  network: {
    label: 'Network throughput',
    unit: 'B/s',
    query: 'sum(rate(node_network_receive_bytes_total[10m]) + rate(node_network_transmit_bytes_total[10m]))'
  },
  latency: {
    label: 'Network probe p95 latency',
    unit: 's',
    query: 'histogram_quantile(0.95, sum(rate(prober_probe_duration_seconds_bucket[10m])) by (le))'
  }
};

export function prometheusBase(configOrUrl) {
  const raw = typeof configOrUrl === 'string' ? configOrUrl : (configOrUrl.prometheusApiUrl || configOrUrl.prometheusMetricsUrl);
  const url = new URL(raw);
  if (url.pathname.endsWith('/metrics')) url.pathname = url.pathname.slice(0, -'/metrics'.length) || '/';
  return url;
}

export function metricOptions() {
  return Object.entries(METRICS).map(([id, m]) => ({ id, label: m.label, unit: m.unit }));
}

function boundedNumber(value, fallback, min, max) {
  const n = Number(value ?? fallback);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

export async function queryMetricRange(config, { metric = 'cpu', hours = 24, stepMinutes = 10, votepercent = 0.5, anomalyMode = 'quick' } = {}) {
  const selected = METRICS[metric];
  if (!selected) throw new Error(`Unknown metric: ${metric}`);

  const windowHours = boundedNumber(hours, 24, 1, 24 * 14);
  const step = Math.round(boundedNumber(stepMinutes, 10, 1, 24 * 60) * 60);
  const anomalyVotepercent = boundedNumber(votepercent, 0.5, 0.1, 1);
  const detectorMode = anomalyMode === 'full' ? 'full' : 'quick';
  const end = Math.floor(Date.now() / 1000);
  const start = end - Math.round(windowHours * 60 * 60);
  const url = new URL('/api/v1/query_range', prometheusBase(config));
  url.searchParams.set('query', selected.query);
  url.searchParams.set('start', String(start));
  url.searchParams.set('end', String(end));
  url.searchParams.set('step', String(step));

  const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
  const json = await res.json().catch(() => ({}));
  if (!res.ok || json.status === 'error') throw new Error(redact(json.error || `Prometheus HTTP ${res.status}`));

  const series = json.data?.result?.[0]?.values || [];
  const points = series.map(([ts, value]) => ({ ts: Number(ts), value: Number(value) })).filter((p) => Number.isFinite(p.value));
  const values = points.map((p) => p.value);
  const anomalies = await detectAnomalies(points, { votepercent: anomalyVotepercent, mode: detectorMode });
  for (const index of anomalies.indexes) points[index].anomaly = true;
  return {
    metric,
    label: selected.label,
    unit: selected.unit,
    query: selected.query,
    start,
    end,
    hours: windowHours,
    step,
    stepMinutes: step / 60,
    xRange: { start: points[0]?.ts || start, end: points.at(-1)?.ts || end },
    yRange: values.length ? { min: Math.min(...values), max: Math.max(...values) } : null,
    anomalyCount: anomalies.indexes.length,
    votepercent: anomalyVotepercent,
    anomalyMode: anomalies.mode,
    warnings: anomalies.warnings,
    points
  };
}
