import test from 'node:test';
import assert from 'node:assert/strict';
import { listTemplates } from '../lib/templates.js';

test('argo UI cluster template page uses API endpoint', async () => {
  const oldFetch = global.fetch;
  const calls = [];
  global.fetch = async (url) => {
    calls.push(String(url));
    if (String(url).endsWith('/api/v1/cluster-workflow-templates')) {
      return { ok: true, headers: new Map([['content-type', 'application/json']]), text: async () => JSON.stringify({ items: [{ metadata: { name: 'automlai-dualsearch' }, spec: { arguments: { parameters: [{ name: 'workers', value: '5' }] } } }] }) };
    }
    return { ok: true, headers: new Map([['content-type', 'text/html']]), text: async () => '<div id="app"></div>' };
  };
  try {
    const { templates } = await listTemplates({ templateUrl: 'http://argo.home:8080/cluster-workflow-templates' });
    assert.ok(calls.includes('http://argo.home:8080/api/v1/cluster-workflow-templates'));
    assert.equal(templates.find((t) => t.name === 'automlai-dualsearch')?.parameters[0]?.name, 'workers');
  } finally {
    global.fetch = oldFetch;
  }
});
