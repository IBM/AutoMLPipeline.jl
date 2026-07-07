import test from 'node:test';
import assert from 'node:assert/strict';
import { answerPrompt, buildPromptContext, maybeConfirmation } from '../lib/llm.js';

test('redacts LLM context', () => {
  const ctx = buildPromptContext({ logs: 'Authorization: bearer abc123 password=secret' });
  assert.match(ctx.logs, /\[REDACTED\]/);
  assert.doesNotMatch(ctx.logs, /abc123|secret/);
});

test('deploy prompt returns confirmation payload', () => {
  const result = maybeConfirmation({ namespace: 'argo' }, 'deploy this', { template: { name: 'automl-classification' }, parameters: { input: 'iris.csv' } });
  assert.equal(result.type, 'confirmation_required');
  assert.equal(result.tool, 'submit_template_workflow');
});

test('ETE chat request omits unsupported temperature override', async () => {
  const oldFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (_url, init) => {
    request = { headers: init.headers, body: JSON.parse(init.body) };
    return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 });
  };
  try {
    const out = await answerPrompt({ namespace: 'argo', llm: { apiKey: 'sk-live', baseUrl: 'https://ete.example/v1', model: 'azure/gpt-5.5' } }, { prompt: 'summarize' });
    assert.equal(out.type, 'answer');
    assert.equal(request.headers.authorization, 'Bearer sk-live');
    assert.equal(request.body.model, 'azure/gpt-5.5');
    assert.equal('temperature' in request.body, false);
  } finally {
    globalThis.fetch = oldFetch;
  }
});
