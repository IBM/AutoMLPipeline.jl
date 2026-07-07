import fs from 'node:fs/promises';
import path from 'node:path';
import { parseYamlDocuments, normalizeTemplate, validationHint } from './yaml.js';

const ROOT = path.resolve(new URL('../..', import.meta.url).pathname);
const NAME_ALIASES = new Map([
  ['automlai-dualsearch', 'automl-dualsearch'],
  ['automlai-unisearch', 'automl-unisearch']
]);

function withHints(t) {
  return { ...t, parameters: (t.parameters || []).map((p) => ({ ...p, validation: validationHint(p.name) })) };
}

async function localTemplates(root = ROOT) {
  const out = [];
  for (const dir of [root, path.join(root, 'argo-workflow')]) {
    const files = await fs.readdir(dir).catch(() => []);
    for (const file of files.filter((f) => f.endsWith('-template.yaml'))) {
      const fullPath = path.join(dir, file);
      const label = path.relative(root, fullPath) || file;
      try {
        const text = await fs.readFile(fullPath, 'utf8');
        for (const doc of parseYamlDocuments(text)) {
          if (doc?.kind?.includes('WorkflowTemplate')) out.push(withHints({ ...normalizeTemplate(doc, 'local', 'local'), file: label }));
        }
      } catch (error) {
        out.push({ name: file, source: 'local', metadataSource: 'error', parameters: [], warnings: [error.message] });
      }
    }
  }
  return out;
}

function remoteFromJson(json) {
  const items = Array.isArray(json) ? json : (json.items || json.templates || []);
  return items.map((item) => withHints(normalizeTemplate(item, 'remote', item.spec ? 'remote' : 'remote-name'))).filter((t) => t.name);
}

function remoteFromHtml(text) {
  const names = [...text.matchAll(/(?:clusterworkflowtemplate\/)?([A-Za-z0-9_.-]+?)(?:-template)?(?:\.ya?ml)?(?=["'<>\s]|$)/g)]
    .map((m) => m[1]).filter((n) => /automl|workflow|dualsearch|unisearch/i.test(n));
  return [...new Set(names)].map((name) => ({ name, source: 'remote', metadataSource: 'remote-name', parameters: [], warnings: ['Remote endpoint provided name only; local metadata may enrich this entry.'] }));
}

function localForRemote(localMap, name) {
  return localMap.get(name) || localMap.get(NAME_ALIASES.get(name)) || localMap.get([...NAME_ALIASES.entries()].find(([, v]) => v === name)?.[0]);
}

function argoApiUrl(url) {
  try {
    const u = new URL(url);
    if (u.pathname.replace(/\/$/, '') === '/cluster-workflow-templates') {
      u.pathname = '/api/v1/cluster-workflow-templates';
      u.search = '';
      return u.toString();
    }
  } catch {}
  return '';
}

async function fetchTemplates(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Remote templates returned HTTP ${res.status}`);
  const text = await res.text();
  const type = res.headers.get('content-type') || '';
  if (type.includes('json') || /^[\s\[]*[{\[]/.test(text)) return remoteFromJson(JSON.parse(text));
  try {
    const yamlTemplates = parseYamlDocuments(text).map((d) => withHints(normalizeTemplate(d, 'remote', 'remote'))).filter((t) => t.name);
    if (yamlTemplates.length) return yamlTemplates;
  } catch {}
  return remoteFromHtml(text);
}

async function remoteTemplates(url, timeoutMs = 5000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let lastError;
    for (const candidate of [...new Set([argoApiUrl(url), url].filter(Boolean))]) {
      try {
        const templates = await fetchTemplates(candidate, controller.signal);
        if (templates.length || candidate === url) return templates;
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error('Remote templates unavailable');
  } finally {
    clearTimeout(timer);
  }
}

export async function listTemplates({ templateUrl, root = ROOT, timeoutMs } = {}) {
  const local = await localTemplates(root);
  let remote = [];
  const warnings = [];
  try { remote = templateUrl ? await remoteTemplates(templateUrl, timeoutMs) : []; }
  catch (error) { warnings.push(`Remote templates unavailable: ${error.message}`); }

  const localMap = new Map(local.map((t) => [t.name, t]));
  const byName = new Map(local.map((t) => [t.name, t]));
  for (const r of remote) {
    const l = localForRemote(localMap, r.name);
    const needsLocal = (!r.parameters || r.parameters.length === 0) && l?.parameters?.length;
    byName.set(r.name, withHints({
      ...l,
      ...r,
      parameters: needsLocal ? l.parameters : r.parameters,
      metadataSource: needsLocal ? 'local-fallback' : r.metadataSource,
      localName: l?.name !== r.name ? l?.name : undefined,
      warnings: [...(r.warnings || []), ...(needsLocal ? [`Parameters enriched from local YAML ${l.file || l.name}.`] : [])]
    }));
  }
  return { templates: [...byName.values()].sort((a, b) => a.name.localeCompare(b.name)), warnings };
}
