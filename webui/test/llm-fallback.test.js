import test from 'node:test';
import assert from 'node:assert/strict';
import { answerPrompt, buildPromptContext } from '../lib/llm.js';

test('prompt falls back locally when LLM endpoint fails', async () => {
  const oldFetch = global.fetch;
  global.fetch = async () => ({ ok: false, status: 401, json: async () => ({ error: { message: 'bad token sk-secret' } }) });
  try {
    const out = await answerPrompt({ llm: { apiKey: 'bad', baseUrl: 'https://ete.example/v1', model: 'x' }, namespace: 'argo' }, {
      prompt: 'what template parameters?',
      context: { template: { name: 'automlai-dualsearch', parameters: [{ name: 'workers' }] } }
    });
    assert.equal(out.type, 'local_answer');
    assert.match(out.message, /workers/);
    assert.match(out.llmError, /bad token/);
  } finally {
    global.fetch = oldFetch;
  }
});

test('non-json MLflow context does not crash prompt context', () => {
  assert.equal(buildPromptContext({ mlflow: 'Metrics and runs will appear here.' }).mlflow, 'Metrics and runs will appear here.');
});

test('local fallback explains plot anomalies', async () => {
  const out = await answerPrompt({ llm: { disabled: true }, namespace: 'argo' }, {
    prompt: 'explain plot anomalies',
    context: { plot: { metric: 'cpu', label: 'CPU usage', unit: '%', hours: 1, stepMinutes: 10, query: 'q', points: [{ ts: 20, value: 9, anomaly: true }] } }
  });
  assert.equal(out.type, 'local_answer');
  assert.match(out.message, /1970-01-01T00:00:20.000Z/);
});
