import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildConfig, publicConfig } from './lib/config.js';
import { redact } from './lib/redact.js';
import { listTemplates } from './lib/templates.js';
import { submitTemplateWorkflow, submitYamlWorkflow, listWorkflows, getWorkflow, getWorkflowLogs } from './lib/argo.js';
import { queryMlflowResults } from './lib/mlflow.js';
import { metricOptions, queryMetricRange } from './lib/prometheus.js';
import { answerPrompt } from './lib/llm.js';
import { handleMcpRequest } from './lib/mcp.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const config = buildConfig();
export const app = express();

app.disable('x-powered-by');
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

function cookies(req) {
  return Object.fromEntries((req.get('cookie') || '').split(';').map((part) => part.trim().split('=').map(decodeURIComponent)).filter(([k]) => k));
}

function isSameOrigin(req) {
  const origin = req.get('origin');
  if (!origin) return true;
  try {
    return new URL(origin).host === req.get('host');
  } catch {
    return false;
  }
}

export function requireMutationAuth(req, res, next) {
  const auth = req.get('authorization') || '';
  const candidates = [
    cookies(req).argo_webui_token,
    auth.startsWith('Bearer ') ? auth.slice(7) : '',
    req.get('x-argo-webui-token') || ''
  ];
  if (!candidates.includes(config.sessionToken) || !isSameOrigin(req)) {
    return res.status(403).json({ status: 'error', message: 'Mutating route requires local session and same-origin request. Reload the UI, then retry.' });
  }
  next();
}

app.get('/api/config', (_req, res) => {
  res.cookie('argo_webui_token', config.sessionToken, { httpOnly: true, sameSite: 'strict', secure: false, path: '/' });
  res.json(publicConfig(config));
});
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));

app.get('/api/templates', async (_req, res, next) => {
  try { res.json({ status: 'ok', ...(await listTemplates(config)) }); } catch (err) { next(err); }
});

app.get('/api/workflows', async (req, res, next) => {
  try { res.json(await listWorkflows(config, { limit: Number(req.query.limit || 20) })); } catch (err) { next(err); }
});

app.get('/api/workflows/:name/status', async (req, res, next) => {
  try { res.json(await getWorkflow(config, req.params.name)); } catch (err) { next(err); }
});

app.get('/api/workflows/:name/logs', async (req, res, next) => {
  try { res.json(await getWorkflowLogs(config, req.params.name, { tailLines: Number(req.query.tailLines || 500) })); } catch (err) { next(err); }
});

app.post('/api/workflows/submit', requireMutationAuth, async (req, res, next) => {
  try {
    const result = req.body?.yaml ? await submitYamlWorkflow(config, req.body) : await submitTemplateWorkflow(config, req.body);
    res.json(result);
  } catch (err) { next(err); }
});

app.get('/api/mlflow', async (req, res, next) => {
  try { res.json(await queryMlflowResults(config, req.query)); } catch (err) { next(err); }
});

app.get('/api/prometheus/metrics', (_req, res) => res.json({ status: 'ok', metrics: metricOptions() }));

app.get('/api/prometheus/range', async (req, res, next) => {
  try { res.json({ status: 'ok', ...(await queryMetricRange(config, req.query)) }); } catch (err) { next(err); }
});

app.post('/api/prompt', requireMutationAuth, async (req, res, next) => {
  try { res.json(await answerPrompt(config, req.body)); } catch (err) { next(err); }
});

app.all('/mcp', async (req, res, next) => {
  try {
    if (req.method !== 'GET') return requireMutationAuth(req, res, () => handleMcpRequest(config, req, res).catch(next));
    await handleMcpRequest(config, req, res);
  } catch (err) { next(err); }
});

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ status: 'error', message: redact(err.message || String(err)) });
});

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  app.listen(config.port, config.host, () => {
    console.log(`Argo AutoML Command Deck listening on http://${config.host}:${config.port}`);
    console.log(`Mutation token: ${config.sessionToken}`);
  });
}
