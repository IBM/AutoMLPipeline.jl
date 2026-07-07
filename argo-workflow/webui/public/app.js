import { markdownToHtml, promptResultMarkdown } from './markdown.js';

const $ = (id) => document.getElementById(id);
const state = { config: null, templates: [], current: null, metrics: [] };

async function json(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || res.statusText);
  return body;
}

function endpointLinks(config) {
  const links = [
    ['Argo', config.argoServer], ['Grafana', config.grafanaUrl], ['MLflow', config.mlflowUrl], ['Prometheus metrics', config.prometheusMetricsUrl], ['MCP', config.mcpUrl]
  ];
  $('endpointLinks').innerHTML = links.map(([name, href]) => `<a href="${href}" target="_blank" rel="noreferrer">${name}</a>`).join('');
}

function renderTemplates() {
  const select = $('templateSelect');
  select.innerHTML = state.templates.map((t, i) => `<option value="${i}">${t.name || 'unnamed'} · ${t.source || 'unknown'}</option>`).join('');
  state.current = state.templates[0] || null;
  renderParams();
}

function renderMetricOptions() {
  $('metricSelect').innerHTML = state.metrics.map((m) => `<option value="${m.id}">${m.label} (${m.unit})</option>`).join('');
}

function renderParams() {
  const params = state.current?.parameters || [];
  $('params').innerHTML = params.length ? params.map((p) => `<div class="field"><label><span>${p.name}</span><span>${p.required ? 'required' : 'default'}</span></label><input data-param="${p.name}" value="${p.value ?? p.default ?? ''}" /></div>`).join('') : '<p class="muted">No parameters loaded yet.</p>';
  $('deployBtn').disabled = !state.current;
}

async function boot() {
  state.config = await json('/api/config');
  endpointLinks(state.config);
  $('llmStatus').textContent = state.config.llm.hasKey ? state.config.llm.model : 'parser fallback';
  try {
    const data = await json('/api/templates');
    state.templates = data.templates || [];
    $('templateStatus').textContent = state.templates.length ? `${state.templates.length} found` : data.status;
    renderTemplates();
  } catch (err) {
    $('templateStatus').textContent = 'error';
    $('templateStatus').className = 'badge bad';
    $('logs').textContent = err.message;
  }
  try {
    const data = await json('/api/prometheus/metrics');
    state.metrics = data.metrics || [];
    renderMetricOptions();
  } catch (err) {
    $('metricStatus').textContent = 'error';
    $('metricStatus').className = 'badge bad';
    $('metricSummary').textContent = err.message;
  }
}

$('templateSelect').addEventListener('change', (event) => { state.current = state.templates[Number(event.target.value)]; renderParams(); });
$('yamlFile').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  $('yamlPreview').textContent = file ? await file.text() : 'Upload a Workflow YAML to validate before submit.';
});
$('deployBtn').addEventListener('click', async () => {
  $('deployBtn').disabled = true;
  const parameters = Object.fromEntries([...document.querySelectorAll('[data-param]')].map((el) => [el.dataset.param, el.value]));
  $('logs').textContent = 'Submitting...';
  try {
    const data = await json('/api/workflows/submit', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ templateName: state.current?.name, parameters, requestId: crypto.randomUUID() }) });
    $('workflowStatus').textContent = data.status;
    $('logs').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $('workflowStatus').textContent = 'error';
    $('workflowStatus').className = 'badge bad';
    $('logs').textContent = err.message;
  } finally { $('deployBtn').disabled = false; }
});
$('refreshMlflow').addEventListener('click', async () => {
  $('mlflowStatus').textContent = 'querying';
  try { $('mlflow').textContent = JSON.stringify(await json('/api/mlflow'), null, 2); $('mlflowStatus').textContent = 'ok'; }
  catch (err) { $('mlflow').textContent = err.message; $('mlflowStatus').textContent = 'error'; }
});

function drawMetric(data) {
  const canvas = $('metricChart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height, pad = 36;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#020813'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#23415f'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, 12); ctx.lineTo(pad, h - pad); ctx.lineTo(w - 12, h - pad); ctx.stroke();
  const points = data.points || [];
  if (!points.length) { ctx.fillStyle = '#93a9c7'; ctx.fillText('No data returned', pad + 8, 40); return; }
  const min = Math.min(...points.map((p) => p.value));
  const max = Math.max(...points.map((p) => p.value));
  const span = max - min || 1;
  ctx.strokeStyle = '#7cf7d4'; ctx.lineWidth = 2; ctx.beginPath();
  points.forEach((p, i) => {
    const x = pad + (i / Math.max(points.length - 1, 1)) * (w - pad - 16);
    const y = 12 + (1 - ((p.value - min) / span)) * (h - pad - 24);
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#b9d7ff';
  ctx.fillText(`${data.label} (${data.unit})`, pad + 8, 24);
  ctx.fillText(`min ${min.toFixed(2)}  max ${max.toFixed(2)}`, pad + 8, h - 10);
}

$('plotMetric').addEventListener('click', async () => {
  $('metricStatus').textContent = 'querying';
  try {
    const data = await json(`/api/prometheus/range?metric=${encodeURIComponent($('metricSelect').value)}`);
    drawMetric(data);
    $('metricSummary').textContent = `${data.label}: ${data.points.length} points, 24h window, 10m step.\nQuery: ${data.query}`;
    $('metricStatus').textContent = 'ok';
  } catch (err) {
    $('metricStatus').textContent = 'error';
    $('metricStatus').className = 'badge bad';
    $('metricSummary').textContent = err.message;
  }
});
$('askBtn').addEventListener('click', async () => {
  $('answer').classList.add('markdown');
  $('answer').textContent = 'Thinking...';
  const parameters = Object.fromEntries([...document.querySelectorAll('[data-param]')].map((el) => [el.dataset.param, el.value]));
  try {
    const data = await json('/api/prompt', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: $('prompt').value, context: { template: state.current, parameters, logs: $('logs').textContent, mlflow: $('mlflow').textContent } }) });
    $('answer').innerHTML = markdownToHtml(promptResultMarkdown(data));
  } catch (err) { $('answer').innerHTML = markdownToHtml(`**Error:** ${err.message}`); }
});

boot().catch((err) => { document.body.innerHTML = `<pre>${err.stack || err.message}</pre>`; });
