import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { validateParameters, validateUploadYaml } from './yaml.js';
import { redact } from './redact.js';

export const deployments = new Map();
const pendingKeys = new Set();

export function runCommand(cmd, args, { timeoutMs = 120000, input } = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { stdio: [input ? 'pipe' : 'ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    if (input) child.stdin.end(input);
    const timer = setTimeout(() => child.kill('SIGTERM'), timeoutMs);
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout: redact(stdout), stderr: redact(stderr), args });
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      resolve({ code: 127, stdout: '', stderr: redact(error.message), args });
    });
  });
}

function workflowNameFrom(text) {
  return text.match(/Name:\s*([^\s]+)/)?.[1] || text.match(/workflow\/["']?([^\s"']+)/)?.[1] || text.match(/@latest\s+([^\s]+)/)?.[1] || '';
}

function safeName(name, label = 'name') {
  if (!/^[A-Za-z0-9_.-]+$/.test(name || '')) throw new Error(`Invalid ${label}`);
  return name;
}

function can(config, operation) {
  return (config.mcpAllowedOperations || []).includes(operation);
}

export function requireOperation(config, operation) {
  if (!can(config, operation)) throw new Error(`Operation ${operation} is not allowed. Set ARGO_MCP_ALLOWED_OPERATIONS=${operation} or include it in the comma list.`);
}

export function buildTemplateSubmitArgs({ namespace, templateName, parameters = {}, watch = false }) {
  const errors = validateParameters(parameters);
  if (Object.keys(errors).length) throw Object.assign(new Error('Invalid parameters'), { details: errors });
  safeName(templateName, 'template name');
  const args = ['-n', namespace, 'submit', '--from', `clusterworkflowtemplate/${templateName}`];
  for (const [key, value] of Object.entries(parameters)) args.push('-p', `${key}=${value}`);
  if (watch) args.push('--watch', '--log');
  return args;
}

export async function submitTemplateWorkflow(config, input) {
  requireOperation(config, 'submit');
  const key = input.requestId || `${input.templateName}:${JSON.stringify(input.parameters || {})}`;
  if (pendingKeys.has(key)) throw new Error('Duplicate deployment already pending');
  pendingKeys.add(key);
  try {
    const args = buildTemplateSubmitArgs({ namespace: input.namespace || config.namespace, templateName: input.templateName, parameters: input.parameters || {} });
    const result = await runCommand(config.argoBin, args);
    const id = randomUUID();
    const name = workflowNameFrom(result.stdout + '\n' + result.stderr);
    const status = result.code === 0 ? 'submitted' : 'failed';
    const record = { id, name, status, templateName: input.templateName, parameters: input.parameters || {}, createdAt: new Date().toISOString(), result };
    deployments.set(id, record);
    if (name) deployments.set(name, record);
    if (result.code !== 0) throw Object.assign(new Error(result.stderr || 'Argo submit failed'), { record });
    return record;
  } finally {
    pendingKeys.delete(key);
  }
}

export async function submitYamlWorkflow(config, { yaml, namespace, dryRun = true } = {}) {
  if (!dryRun) requireOperation(config, 'submit');
  const parsed = validateUploadYaml(yaml);
  if (parsed.mode !== 'submit-workflow') throw new Error('Uploaded ClusterWorkflowTemplate YAML is preview-only; select an existing cluster template to submit.');
  const tmp = path.join(os.tmpdir(), `argo-webui-${randomUUID()}.yaml`);
  await fs.writeFile(tmp, yaml);
  try {
    const args = ['-n', namespace || config.namespace, 'submit', tmp];
    if (dryRun) args.push('--dry-run');
    const result = await runCommand(config.argoBin, args);
    const id = randomUUID();
    const name = workflowNameFrom(result.stdout + '\n' + result.stderr) || parsed.template.name;
    const record = { id, name, status: result.code === 0 ? (dryRun ? 'dry-run' : 'submitted') : 'failed', yamlMode: parsed.mode, createdAt: new Date().toISOString(), result };
    deployments.set(id, record);
    if (name) deployments.set(name, record);
    if (result.code !== 0) throw Object.assign(new Error(result.stderr || 'Argo YAML submit failed'), { record });
    return record;
  } finally {
    fs.unlink(tmp).catch(() => {});
  }
}

export async function listWorkflows(config, { limit = 20, namespace } = {}) {
  const result = await runCommand(config.argoBin, ['-n', namespace || config.namespace, 'list', '--output', 'json']);
  if (result.code !== 0) return { status: 'error', message: result.stderr, workflows: recentDeployments(limit) };
  try { return { status: 'ok', workflows: JSON.parse(result.stdout).items?.slice(0, limit) || [] }; }
  catch { return { status: 'ok', raw: result.stdout, workflows: recentDeployments(limit) }; }
}

export async function getWorkflow(config, name, { namespace } = {}) {
  safeName(name, 'workflow name');
  const result = await runCommand(config.argoBin, ['-n', namespace || config.namespace, 'get', name, '--output', 'json']);
  if (result.code !== 0) return { status: 'error', message: result.stderr, record: deployments.get(name) };
  try { return { status: 'ok', workflow: JSON.parse(result.stdout) }; }
  catch { return { status: 'ok', raw: result.stdout }; }
}

export async function getWorkflowLogs(config, name, { tailLines = 500, namespace } = {}) {
  safeName(name, 'workflow name');
  const result = await runCommand(config.argoBin, ['-n', namespace || config.namespace, 'logs', name, '--tail', String(tailLines)]);
  return { status: result.code === 0 ? 'ok' : 'error', logs: result.stdout || result.stderr || 'waiting for pod logs...', args: result.args };
}

export async function watchWorkflow(config, { name, namespace } = {}) {
  safeName(name, 'workflow name');
  const result = await runCommand(config.argoBin, ['-n', namespace || config.namespace, 'watch', name], { timeoutMs: 30000 });
  return { status: result.code === 0 ? 'ok' : 'error', output: result.stdout || result.stderr, args: result.args };
}

export async function workflowOperation(config, { name, operation, namespace } = {}) {
  safeName(name, 'workflow name');
  const allowed = new Set(['suspend', 'resume', 'retry', 'terminate', 'stop', 'delete']);
  if (!allowed.has(operation)) throw new Error('Unsupported workflow operation');
  requireOperation(config, operation);
  const result = await runCommand(config.argoBin, ['-n', namespace || config.namespace, operation, name]);
  return { status: result.code === 0 ? 'ok' : 'error', operation, output: result.stdout || result.stderr, args: result.args };
}

export async function resubmitWorkflow(config, { name, namespace, parameters = {}, memoized = false } = {}) {
  safeName(name, 'workflow name');
  requireOperation(config, 'resubmit');
  const args = ['-n', namespace || config.namespace, 'resubmit', name, '-o', 'json'];
  for (const [key, value] of Object.entries(parameters)) args.push('-p', `${key}=${value}`);
  if (memoized) args.push('--memoized');
  const result = await runCommand(config.argoBin, args);
  return { status: result.code === 0 ? 'ok' : 'error', output: result.stdout || result.stderr, args: result.args };
}

export async function workflowNodes(config, { name, namespace } = {}) {
  const result = await getWorkflow(config, name, { namespace });
  const nodes = result.workflow?.status?.nodes || {};
  return { status: result.status, nodes: Object.values(nodes).map(({ id, name, displayName, type, phase, message, startedAt, finishedAt, children }) => ({ id, name, displayName, type, phase, message, startedAt, finishedAt, children })) };
}

function namesFromLines(text) {
  return text.split('\n').map((s) => s.trim().replace(/^clusterworkflowtemplate\//, '').replace(/^cronworkflow\//, '')).filter(Boolean);
}

export async function listWorkflowTemplates(config, { namespace, cluster = true } = {}) {
  const args = cluster ? ['cluster-template', 'list', '-o', 'name'] : ['-n', namespace || config.namespace, 'template', 'list', '-o', 'name'];
  const result = await runCommand(config.argoBin, args);
  if (result.code !== 0) return { status: 'error', message: result.stderr, args: result.args };
  return { status: 'ok', templates: namesFromLines(result.stdout).map((name) => ({ name, cluster })), args: result.args };
}

export async function getWorkflowTemplate(config, { name, namespace, cluster = true } = {}) {
  safeName(name, 'template name');
  const args = cluster ? ['cluster-template', 'get', name, '--output', 'json'] : ['-n', namespace || config.namespace, 'template', 'get', name, '--output', 'json'];
  const result = await runCommand(config.argoBin, args);
  if (result.code !== 0) return { status: 'error', message: result.stderr, args: result.args };
  try { return { status: 'ok', template: JSON.parse(result.stdout), args: result.args }; }
  catch { return { status: 'ok', raw: result.stdout, args: result.args }; }
}

export async function listCronWorkflows(config, { namespace = config.namespace } = {}) {
  const result = await runCommand(config.argoBin, ['-n', namespace, 'cron', 'list', '-o', 'name']);
  if (result.code !== 0) return { status: 'error', message: result.stderr, args: result.args };
  return { status: 'ok', cronWorkflows: namesFromLines(result.stdout).map((name) => ({ name })), args: result.args };
}

export async function getCronWorkflow(config, { name, namespace = config.namespace } = {}) {
  safeName(name, 'cron workflow name');
  const result = await runCommand(config.argoBin, ['-n', namespace, 'cron', 'get', name, '--output', 'json']);
  if (result.code !== 0) return { status: 'error', message: result.stderr, args: result.args };
  try { return { status: 'ok', cronWorkflow: JSON.parse(result.stdout), args: result.args }; }
  catch { return { status: 'ok', raw: result.stdout, args: result.args }; }
}

export async function cronWorkflowOperation(config, { name, operation, namespace = config.namespace } = {}) {
  safeName(name, 'cron workflow name');
  const allowed = new Set(['suspend', 'resume', 'delete']);
  if (!allowed.has(operation)) throw new Error('Unsupported cron workflow operation');
  requireOperation(config, `cron:${operation}`);
  const result = await runCommand(config.argoBin, ['-n', namespace, 'cron', operation, name]);
  return { status: result.code === 0 ? 'ok' : 'error', operation, output: result.stdout || result.stderr, args: result.args };
}

export async function argoVersion(config) {
  const result = await runCommand(config.argoBin, ['version', '--short']);
  return { status: result.code === 0 ? 'ok' : 'error', version: result.stdout.trim(), error: result.stderr, args: result.args };
}

export async function lintArgoYaml(config, { yaml, kinds = ['workflows', 'workflowtemplates', 'cronworkflows', 'clusterworkflowtemplates'], offline = true } = {}) {
  const tmp = path.join(os.tmpdir(), `argo-webui-lint-${randomUUID()}.yaml`);
  await fs.writeFile(tmp, yaml || '');
  try {
    const args = ['lint', tmp, '--no-color', '--output', 'simple', '--kinds', Array.isArray(kinds) ? kinds.join(',') : kinds];
    if (offline) args.push('--offline');
    const result = await runCommand(config.argoBin, args);
    return { status: result.code === 0 ? 'ok' : 'error', output: result.stdout || result.stderr, args: result.args };
  } finally {
    fs.unlink(tmp).catch(() => {});
  }
}

export async function listArchivedWorkflows(config, { selector, output = 'json', chunkSize } = {}) {
  const args = ['archive', 'list', '-o', output];
  if (selector) args.push('-l', selector);
  if (chunkSize !== undefined) args.push('--chunk-size', String(chunkSize));
  const result = await runCommand(config.argoBin, args);
  if (result.code !== 0) return { status: 'error', message: result.stderr, args: result.args };
  if (output === 'json') {
    try { return { status: 'ok', archive: JSON.parse(result.stdout), args: result.args }; } catch {}
  }
  return { status: 'ok', output: result.stdout, args: result.args };
}

export async function getArchivedWorkflow(config, { uid, output = 'json' } = {}) {
  safeName(uid, 'archive workflow uid');
  const result = await runCommand(config.argoBin, ['archive', 'get', uid, '-o', output]);
  if (result.code !== 0) return { status: 'error', message: result.stderr, args: result.args };
  if (output === 'json') {
    try { return { status: 'ok', workflow: JSON.parse(result.stdout), args: result.args }; } catch {}
  }
  return { status: 'ok', output: result.stdout, args: result.args };
}

export async function listArchiveLabelKeys(config) {
  const result = await runCommand(config.argoBin, ['archive', 'list-label-keys']);
  return { status: result.code === 0 ? 'ok' : 'error', keys: namesFromLines(result.stdout), error: result.stderr, args: result.args };
}

export async function listArchiveLabelValues(config, { key } = {}) {
  if (!/^[A-Za-z0-9_.\-/]+$/.test(key || '')) throw new Error('Invalid label key');
  const result = await runCommand(config.argoBin, ['archive', 'list-label-values', key]);
  return { status: result.code === 0 ? 'ok' : 'error', values: namesFromLines(result.stdout), error: result.stderr, args: result.args };
}

export async function archiveWorkflowOperation(config, { uid, operation, parameters = {}, memoized = false } = {}) {
  safeName(uid, 'archive workflow uid');
  const allowed = new Set(['retry', 'resubmit', 'delete']);
  if (!allowed.has(operation)) throw new Error('Unsupported archive operation');
  requireOperation(config, `archive:${operation}`);
  const args = ['archive', operation, uid];
  for (const [key, value] of Object.entries(parameters)) args.push('-p', `${key}=${value}`);
  if (memoized && operation === 'resubmit') args.push('--memoized');
  const result = await runCommand(config.argoBin, args);
  return { status: result.code === 0 ? 'ok' : 'error', operation, output: result.stdout || result.stderr, args: result.args };
}

export function allowedOperations(config) {
  return { allowedOperations: config.mcpAllowedOperations || [] };
}

export function recentDeployments(limit = 20) {
  return [...new Set([...deployments.values()])].sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, limit);
}
