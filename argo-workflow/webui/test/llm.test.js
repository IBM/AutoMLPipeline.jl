import test from 'node:test';
import assert from 'node:assert/strict';
import { answerPrompt, buildPromptContext, maybeConfirmation, readOnlyTools } from '../lib/llm.js';

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

test('ETE chat request omits unsupported temperature override and exposes read-only tools', async () => {
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
    assert.equal(request.body.tool_choice, 'auto');
    assert.ok(request.body.tools.some((t) => t.function.name === 'mlflow_search_runs'));
    assert.ok(request.body.tools.some((t) => t.function.name === 'list_workflows'));
  } finally {
    globalThis.fetch = oldFetch;
  }
});

test('LLM can call MLflow and Argo read-only tools without confirmation', async () => {
  const oldFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (_url, init) => {
    const body = JSON.parse(init.body);
    requests.push(body);
    if (requests.length === 1) {
      return new Response(JSON.stringify({ choices: [{ message: { tool_calls: [
        { id: 'argo1', type: 'function', function: { name: 'list_workflows', arguments: '{}' } },
        { id: 'ml1', type: 'function', function: { name: 'mlflow_search_runs', arguments: '{"max_results":1}' } }
      ] } }] }), { status: 200 });
    }
    return new Response(JSON.stringify({ choices: [{ message: { content: '## Results\n\n- Argo and MLflow queried.' } }] }), { status: 200 });
  };
  try {
    const out = await answerPrompt({
      namespace: 'argo',
      llm: { apiKey: 'sk-live', baseUrl: 'https://ete.example/v1', model: 'azure/gpt-5.5' },
      llmToolHandlers: {
        list_workflows: async () => ({ status: 'ok', workflows: [{ metadata: { name: 'wf' } }] }),
        mlflow_search_runs: async () => ({ runs: [{ info: { run_id: 'r1' } }] })
      }
    }, { prompt: 'show workflows and mlflow runs' });
    assert.equal(out.type, 'answer');
    assert.match(out.message, /Results/);
    assert.equal(requests.length, 2);
    assert.equal(requests[1].messages.filter((m) => m.role === 'tool').length, 2);
  } finally {
    globalThis.fetch = oldFetch;
  }
});

test('read-only tool list excludes mutating operations', () => {
  const names = readOnlyTools.map((t) => t.function.name);
  assert.ok(names.includes('mlflow_search_runs'));
  assert.ok(names.includes('get_workflow_logs'));
  assert.equal(names.includes('submit_template_workflow'), false);
  assert.equal(names.includes('operate_workflow'), false);
});
