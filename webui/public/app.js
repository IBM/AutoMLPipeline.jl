import { markdownToHtml, promptResultMarkdown } from './markdown.js';

const $ = (id) => document.getElementById(id);
const state = { config: null, templates: [], current: null, metrics: [], workflowName: '', plot: null };

async function json(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || res.statusText);
  return body;
}

function endpointLinks(config) {
  const links = [
    ['Argo', config.argoServer], ['Grafana', config.grafanaUrl], ['Prometheus metrics', config.prometheusMetricsUrl], ['MCP', config.mcpUrl]
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
$('deployBtn').addEventListener('click', async () => {
  $('deployBtn').disabled = true;
  const parameters = Object.fromEntries([...document.querySelectorAll('[data-param]')].map((el) => [el.dataset.param, el.value]));
  $('logs').textContent = 'Submitting...';
  try {
    const data = await json('/api/workflows/submit', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ templateName: state.current?.name, parameters, requestId: crypto.randomUUID() }) });
    state.workflowName = data.name || '';
    $('workflowStatus').textContent = data.status;
    $('logs').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $('workflowStatus').textContent = 'error';
    $('workflowStatus').className = 'badge bad';
    $('logs').textContent = err.message;
  } finally { $('deployBtn').disabled = false; }
});
function drawMetric(data) {
  const canvas = $('metricChart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const left = 56, top = 30, right = 12, bottom = 58;
  const plotW = w - left - right, plotH = h - top - bottom;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#020813'; ctx.fillRect(0, 0, w, h);
  const points = data.points || [];
  if (!points.length) { ctx.fillStyle = '#93a9c7'; ctx.fillText('No data returned', left + 8, 40); return; }
  const min = data.yRange?.min ?? Math.min(...points.map((p) => p.value));
  const max = data.yRange?.max ?? Math.max(...points.map((p) => p.value));
  const span = max - min || 1;
  const xOf = (i) => left + (i / Math.max(points.length - 1, 1)) * plotW;
  const yOf = (value) => top + (1 - ((value - min) / span)) * plotH;
  ctx.font = '12px Inter, ui-sans-serif, system-ui';
  ctx.strokeStyle = '#23415f'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, top + plotH); ctx.lineTo(left + plotW, top + plotH); ctx.stroke();
  ctx.fillStyle = '#93a9c7';
  for (let i = 0; i < 5; i += 1) {
    const t = i / 4;
    const x = left + t * plotW;
    const idx = Math.round(t * (points.length - 1));
    const yv = min + (1 - t) * span;
    const y = yOf(yv);
    ctx.strokeStyle = '#12304a';
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke();
    ctx.fillStyle = '#93a9c7';
    ctx.fillText(new Date(points[idx].ts * 1000).toLocaleTimeString(), Math.min(x, w - 80), top + plotH + 18);
    ctx.fillText(yv.toFixed(2), 6, y + 4);
  }
  ctx.strokeStyle = '#7cf7d4'; ctx.lineWidth = 2; ctx.beginPath();
  points.forEach((p, i) => {
    const x = xOf(i), y = yOf(p.value);
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#ff7a90';
  points.filter((p) => p.anomaly).forEach((p) => {
    const x = xOf(points.indexOf(p)), y = yOf(p.value);
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
  });
  const start = new Date((data.xRange?.start ?? points[0].ts) * 1000).toLocaleString();
  const end = new Date((data.xRange?.end ?? points.at(-1).ts) * 1000).toLocaleString();
  ctx.fillStyle = '#b9d7ff';
  ctx.fillText(`${data.label} (${data.unit})`, left + 8, 20);
  ctx.fillText(`x ${start} → ${end}`, left + 8, h - 22);
  ctx.fillText(`y ${min.toFixed(2)} → ${max.toFixed(2)}  anomalies ${data.anomalyCount || 0}`, left + 8, h - 8);
}

$('plotMetric').addEventListener('click', async () => {
  $('metricStatus').textContent = 'querying';
  try {
    const params = new URLSearchParams({ metric: $('metricSelect').value, hours: $('metricHours').value || '24', stepMinutes: $('metricStepMinutes').value || '10', votepercent: $('metricVotepercent').value || '0.5' });
    const data = await json(`/api/prometheus/range?${params}`);
    state.plot = data;
    drawMetric(data);
    const xr = data.xRange ? `${new Date(data.xRange.start * 1000).toLocaleString()} → ${new Date(data.xRange.end * 1000).toLocaleString()}` : 'none';
    const yr = data.yRange ? `${data.yRange.min.toFixed(2)} → ${data.yRange.max.toFixed(2)} ${data.unit}` : 'none';
    $('metricSummary').textContent = `${data.label}: ${data.points.length} points, ${data.hours}h window, ${data.stepMinutes}m step, votepercent ${data.votepercent}, ${data.anomalyCount || 0} anomalies.\nx-range: ${xr}\ny-range: ${yr}\n${(data.warnings || []).join('\n')}\nQuery: ${data.query}`.trim();
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
    const data = await json('/api/prompt', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: $('prompt').value, context: { template: state.current, parameters, logs: $('logs').textContent, plot: state.plot } }) });
    $('answer').innerHTML = markdownToHtml(promptResultMarkdown(data));
  } catch (err) { $('answer').innerHTML = markdownToHtml(`**Error:** ${err.message}`); }
});

boot().catch((err) => { document.body.innerHTML = `<pre>${err.stack || err.message}</pre>`; });
