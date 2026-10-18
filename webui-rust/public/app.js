import { markdownToHtml, promptResultMarkdown } from './markdown.js';

const $ = (id) => document.getElementById(id);
const state = { config: null, templates: [], current: null, metrics: [], quickDetectors: [], workflowName: '', plot: null, cpuStream: null };

async function json(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.message || res.statusText);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

function endpointLinks(config) {
  const links = [
    ['Argo', config.argoServer], ['Grafana', config.grafanaUrl], ['Mlflow', config.mlflowUrl], ['Prometheus metrics', config.prometheusMetricsUrl], ['MCP', config.mcpUrl]
  ];
  const nav = $('endpointLinks');
  nav.replaceChildren();
  links.forEach(([name, href]) => {
    const a = document.createElement('a');
    a.href = href;
    a.target = '_blank';
    a.rel = 'noreferrer';
    a.textContent = name;
    nav.append(a);
  });
  const llm = document.createElement('a');
  llm.href = '#llmSettings';
  llm.textContent = 'LLM credentials';
  nav.append(llm);
}

function renderLlm(config) {
  $('llmApiKey').value = config.llm.apiKey || '';
  $('llmBaseUrl').value = config.llm.baseUrl || '';
  $('llmModel').value = config.llm.model || '';
  $('llmStatus').className = config.llm.hasKey && config.llm.baseUrl ? 'badge' : 'badge bad';
  if (!config.llm.baseUrl && !config.llm.hasKey) {
    $('llmStatus').textContent = 'missing BASE_URL + ETE_API_KEY';
  } else if (!config.llm.baseUrl) {
    $('llmStatus').textContent = 'missing BASE_URL';
  } else if (!config.llm.hasKey) {
    $('llmStatus').textContent = 'missing ETE_API_KEY';
  } else {
    $('llmStatus').textContent = config.llm.model;
  }
}

function renderTemplates() {
  const select = $('templateSelect');
  const currentName = state.current?.name;
  select.replaceChildren(...state.templates.map((t, i) => {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = `${t.name || 'unnamed'} · ${t.source || 'unknown'}`;
    return option;
  }));
  const index = Math.max(0, state.templates.findIndex((t) => t.name === currentName));
  select.value = String(index);
  state.current = state.templates[index] || null;
  renderParams();
}

async function loadTemplates() {
  $('templateStatus').textContent = 'loading';
  $('templateStatus').className = 'badge';
  const data = await json('/api/templates');
  state.templates = data.templates || [];
  $('templateStatus').textContent = state.templates.length ? `${state.templates.length} found` : data.status;
  renderTemplates();
}

function renderMetricOptions() {
  $('metricSelect').replaceChildren(...state.metrics.map((m) => {
    const option = document.createElement('option');
    option.value = m.id;
    option.textContent = `${m.label} (${m.unit})`;
    return option;
  }));
  $('cpuStreamDetector').replaceChildren(...state.quickDetectors.map((d) => {
    const option = document.createElement('option');
    option.value = d.id;
    option.textContent = d.label;
    return option;
  }));
  $('cpuStreamDetector').value = 'caret:pca';
}

function renderParams() {
  const params = state.current?.parameters || [];
  const container = $('params');
  if (!params.length) {
    const empty = document.createElement('p');
    empty.className = 'muted';
    empty.textContent = 'No parameters loaded yet.';
    container.replaceChildren(empty);
  } else {
    container.replaceChildren(...params.map((p) => {
      const field = document.createElement('div');
      field.className = 'field';
      const label = document.createElement('label');
      const name = document.createElement('span');
      name.textContent = p.name;
      const status = document.createElement('span');
      status.textContent = p.required ? 'required' : 'default';
      label.append(name, status);
      const input = document.createElement('input');
      input.dataset.param = p.name;
      input.value = p.value ?? p.default ?? '';
      field.append(label, input);
      return field;
    }));
  }
  $('deployBtn').disabled = !state.current;
}

async function boot() {
  state.config = await json('/api/config');
  endpointLinks(state.config);
  renderLlm(state.config);
  try {
    await loadTemplates();
  } catch (err) {
    $('templateStatus').textContent = 'error';
    $('templateStatus').className = 'badge bad';
    $('logs').textContent = err.message;
  }
  try {
    const data = await json('/api/prometheus/metrics');
    state.metrics = data.metrics || [];
    state.quickDetectors = data.quickDetectors || [];
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
function compactNumber(value) {
  const text = value.toFixed(2);
  return text.length > 8 ? value.toExponential(2) : text;
}

function drawMetric(data, canvasId = 'metricChart') {
  const canvas = $(canvasId);
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
    ctx.fillText(compactNumber(yv), 6, y + 4);
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
  ctx.fillStyle = '#b9d7ff';
  ctx.fillText(`${data.label} (${data.unit}) · anomalies ${data.anomalyCount || 0}`, left + 8, 20);
}

function plotSummary(data) {
  const xr = data.xRange ? `${new Date(data.xRange.start * 1000).toLocaleString()} → ${new Date(data.xRange.end * 1000).toLocaleString()}` : 'none';
  const yr = data.yRange ? `${data.yRange.min.toFixed(2)} → ${data.yRange.max.toFixed(2)} ${data.unit}` : 'none';
  const detector = data.detector ? `, detector ${data.detector}` : '';
  return `${data.label}: ${data.points.length} points, ${data.hours}h window, ${data.stepMinutes}m step, ${data.anomalyMode || 'quick'} mode${detector}, votepercent ${data.votepercent}, ${data.anomalyCount || 0} anomalies.\nx-range: ${xr}\ny-range: ${yr}\n${(data.warnings || []).join('\n')}\nQuery: ${data.query}`.trim();
}

async function plotMetricChart() {
  $('metricStatus').textContent = 'querying';
  const params = new URLSearchParams({ metric: $('metricSelect').value, hours: $('metricHours').value || '24', stepMinutes: $('metricStepMinutes').value || '10', votepercent: $('metricVotepercent').value || '0.5', anomalyMode: 'full' });
  const data = await json(`/api/prometheus/range?${params}`);
  state.plot = data;
  drawMetric(data);
  $('metricSummary').textContent = plotSummary(data);
  $('metricStatus').textContent = 'ok';
}

async function plotCpuStreamChart() {
  $('cpuStreamStatus').textContent = 'querying';
  const params = new URLSearchParams({ metric: $('metricSelect').value || 'cpu', hours: $('metricHours').value || '24', stepMinutes: $('metricStepMinutes').value || '10', anomalyMode: 'quick', detector: $('cpuStreamDetector').value });
  const data = await json(`/api/prometheus/range?${params}`);
  state.cpuStream = data;
  drawMetric(data, 'cpuStreamChart');
  $('cpuStreamSummary').textContent = plotSummary(data);
  $('cpuStreamStatus').textContent = 'ok';
}

$('plotMetric').addEventListener('click', async () => {
  try {
    await Promise.all([plotMetricChart(), plotCpuStreamChart()]);
  } catch (err) {
    $('metricStatus').textContent = 'error';
    $('metricStatus').className = 'badge bad';
    $('metricSummary').textContent = err.message;
    $('cpuStreamStatus').textContent = 'error';
    $('cpuStreamStatus').className = 'badge bad';
    $('cpuStreamSummary').textContent = err.message;
  }
});
$('saveLlm').addEventListener('click', async () => {
  $('llmSaveStatus').textContent = 'verifying';
  $('llmSaveStatus').className = 'badge muted';
  $('llmSaveError').hidden = true;
  $('llmSaveError').replaceChildren();
  try {
    const data = await json('/api/llm', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ apiKey: $('llmApiKey').value, baseUrl: $('llmBaseUrl').value, model: $('llmModel').value }) });
    state.config.llm = data.llm;
    renderLlm(state.config);
    $('llmStatus').textContent = 'agent ok';
    $('llmStatus').className = 'badge';
    await loadTemplates();
    $('llmSaveStatus').textContent = 'verified + saved';
    $('llmSaveStatus').className = 'badge';
  } catch (err) {
    $('llmSaveStatus').textContent = 'verify failed';
    $('llmSaveStatus').className = 'badge bad';
    $('llmSaveStatus').title = err.message;
    $('llmStatus').textContent = 'agent error';
    $('llmStatus').className = 'badge bad';
    $('llmSaveError').hidden = false;
    renderMarkdown($('llmSaveError'), err.body ? promptResultMarkdown(err.body) : `## Credential check failed\n\n${err.message}`);
  }
});

function renderMarkdown(target, markdown) {
  const parsed = new DOMParser().parseFromString(markdownToHtml(markdown), 'text/html');
  target.replaceChildren(...parsed.body.childNodes);
}

$('askBtn').addEventListener('click', async () => {
  $('answer').classList.add('markdown');
  $('answer').textContent = 'Thinking...';
  const parameters = Object.fromEntries([...document.querySelectorAll('[data-param]')].map((el) => [el.dataset.param, el.value]));
  try {
    const data = await json('/api/prompt', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: $('prompt').value, context: { template: state.current, parameters, logs: $('logs').textContent, plot: state.plot, streamPlot: state.cpuStream } }) });
    $('llmStatus').textContent = 'agent ok';
    $('llmStatus').className = 'badge';
    renderMarkdown($('answer'), promptResultMarkdown(data));
  } catch (err) {
    $('llmStatus').textContent = 'agent error';
    $('llmStatus').className = 'badge bad';
    renderMarkdown($('answer'), err.body ? promptResultMarkdown(err.body) : `## Error\n\n${err.message}`);
  }
});

boot().catch((err) => { document.body.textContent = err.stack || err.message; });
