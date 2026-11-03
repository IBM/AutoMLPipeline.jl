import YAML from 'yaml';

export function parseYamlDocuments(text) {
  try {
    return YAML.parseAllDocuments(text).map((doc) => {
      if (doc.errors?.length) throw doc.errors[0];
      return doc.toJSON();
    }).filter(Boolean);
  } catch (error) {
    const line = error.linePos?.[0]?.line;
    const col = error.linePos?.[0]?.col;
    const at = line ? ` at ${line}:${col}` : '';
    const e = new Error(`Invalid YAML${at}: ${error.message}`);
    e.cause = error;
    throw e;
  }
}

export function getName(doc) {
  return doc?.metadata?.name || doc?.name || '';
}

export function parametersFromDoc(doc) {
  const params = doc?.spec?.arguments?.parameters || [];
  return params.map((p) => ({
    name: String(p.name || ''),
    value: p.value ?? p.default ?? '',
    default: p.value ?? p.default ?? '',
    required: p.value == null && p.default == null
  })).filter((p) => p.name);
}

export function normalizeTemplate(doc, source = 'local', metadataSource = source) {
  return {
    name: getName(doc),
    kind: doc?.kind,
    apiVersion: doc?.apiVersion,
    source,
    metadataSource,
    parameters: parametersFromDoc(doc),
    warnings: []
  };
}

const RISKY_KEYS = new Set(['hostPath', 'hostNetwork', 'hostPID', 'hostIPC', 'serviceAccountName', 'imagePullSecrets', 'secretRef', 'secretKeyRef']);

function walk(value, path = [], findings = []) {
  if (!value || typeof value !== 'object') return findings;
  if (Array.isArray(value)) {
    value.forEach((v, i) => walk(v, path.concat(i), findings));
    return findings;
  }
  for (const [key, val] of Object.entries(value)) {
    const next = path.concat(key);
    if (RISKY_KEYS.has(key)) findings.push(next.join('.'));
    if (key === 'privileged' && val === true) findings.push(next.join('.'));
    if (key === 'runAsUser' && Number(val) === 0) findings.push(next.join('.'));
    walk(val, next, findings);
  }
  return findings;
}

export function validateUploadYaml(text, { maxBytes = 256_000 } = {}) {
  if (Buffer.byteLength(text, 'utf8') > maxBytes) throw new Error(`YAML exceeds ${maxBytes} byte limit`);
  const docs = parseYamlDocuments(text);
  if (docs.length !== 1) throw new Error('Upload must contain exactly one YAML document');
  const doc = docs[0];
  if (doc.apiVersion !== 'argoproj.io/v1alpha1') throw new Error('Only argoproj.io/v1alpha1 Argo YAML is supported');
  if (doc.kind === 'ClusterWorkflowTemplate') {
    return { mode: 'preview-template', doc, template: normalizeTemplate(doc, 'upload', 'upload') };
  }
  if (doc.kind !== 'Workflow') throw new Error(`Unsupported kind: ${doc.kind || 'unknown'}`);
  const risky = walk(doc);
  if (risky.length) throw new Error(`Unsafe workflow fields: ${risky.slice(0, 8).join(', ')}`);
  return { mode: 'submit-workflow', doc, template: normalizeTemplate(doc, 'upload', 'upload') };
}

export function validationHint(name) {
  if (['workers', 'folds'].includes(name)) return { type: 'integer', min: 1 };
  if (name === 'predictiontype') return { type: 'enum', values: ['classification', 'regression', 'anomalydetection'] };
  if (name === 'complexity') return { type: 'enum', values: ['low', 'high'] };
  if (name === 'input') return { type: 'csvName' };
  if (name === 'runid') return { type: 'runid' };
  return { type: 'text' };
}

export function validateParameters(params = {}) {
  const errors = {};
  for (const [key, value] of Object.entries(params)) {
    const hint = validationHint(key);
    const text = String(value ?? '');
    if (hint.type === 'integer' && (!/^\d+$/.test(text) || Number(text) < hint.min)) errors[key] = `Must be an integer >= ${hint.min}`;
    if (hint.type === 'enum' && text && !hint.values.includes(text)) errors[key] = `Must be one of ${hint.values.join(', ')}`;
    if (hint.type === 'csvName' && (!/^[A-Za-z0-9_.-]+\.csv$/.test(text) || text.includes('..'))) errors[key] = 'Must be a .csv filename without paths';
    if (hint.type === 'runid' && (text === '' || text === 'NONE')) errors[key] = 'Run ID is required for prediction templates';
  }
  return errors;
}
