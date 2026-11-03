import test from 'node:test';
import assert from 'node:assert/strict';
import { redact } from '../lib/redact.js';
import { summarizeRun, compareRuns, queryMlflowResults } from '../lib/mlflow.js';

test('redacts common secrets', () => {
  const out = redact('token=abc password=hunter2 https://user:pass@example.test');
  assert.doesNotMatch(out, /hunter2|pass@example|token=abc/);
});

test('summarizes run data into maps', async () => {
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ run: { info: { run_id: 'r1' }, data: { params: [{ key: 'workers', value: '5' }], metrics: [{ key: 'accuracy', value: 0.9 }], tags: [{ key: 'workflow', value: 'wf1' }] } } }));
  try {
    const out = await summarizeRun({ mlflowUrl: 'http://mlflow.test' }, { runId: 'r1' });
    assert.deepEqual(out.params, { workers: '5' });
    assert.deepEqual(out.metrics, { accuracy: 0.9 });
    assert.deepEqual(out.tags, { workflow: 'wf1' });
  } finally {
    globalThis.fetch = oldFetch;
  }
});

test('compares runs with selected metric keys', async () => {
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ runs: [{ info: { run_id: 'r1', status: 'FINISHED' }, data: { metrics: [{ key: 'accuracy', value: 0.9 }, { key: 'loss', value: 0.1 }], params: [], tags: [] } }] }));
  try {
    const out = await compareRuns({ mlflowUrl: 'http://mlflow.test' }, { metricKeys: ['accuracy'] });
    assert.deepEqual(out.runs[0].metrics, { accuracy: 0.9 });
  } finally {
    globalThis.fetch = oldFetch;
  }
});

test('queries MLflow runs for one deployed workflow tag', async () => {
  const oldFetch = globalThis.fetch;
  let body;
  globalThis.fetch = async (_url, options) => {
    body = JSON.parse(options.body);
    return new Response(JSON.stringify({ runs: [] }));
  };
  try {
    await queryMlflowResults({ mlflowUrl: 'http://mlflow.test' }, { workflowName: "wf'1" });
    assert.equal(body.filter, "tags.workflow = 'wf\\'1'");
    assert.deepEqual(body.order_by, ['attributes.start_time DESC']);
  } finally {
    globalThis.fetch = oldFetch;
  }
});
