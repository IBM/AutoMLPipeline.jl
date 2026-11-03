import test from 'node:test';
import assert from 'node:assert/strict';
import { listTemplates } from '../lib/templates.js';

test('remote name can use local alias metadata', async () => {
  const oldFetch = global.fetch;
  global.fetch = async () => ({ ok: true, headers: new Map([['content-type', 'text/html']]), text: async () => '<a>automlai-dualsearch-template.yaml</a>' });
  try {
    const { templates } = await listTemplates({ templateUrl: 'http://example.test' });
    const t = templates.find((x) => x.name === 'automlai-dualsearch');
    assert.ok(t);
    assert.equal(t.metadataSource, 'local-fallback');
    assert.ok(t.parameters.some((p) => p.name === 'workers'));
  } finally {
    global.fetch = oldFetch;
  }
});
