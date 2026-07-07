import test from 'node:test';
import assert from 'node:assert/strict';
import { listTemplates } from '../lib/templates.js';

test('loads local example templates', async () => {
  const { templates } = await listTemplates({ timeoutMs: 1 });
  assert.ok(templates.some((t) => t.name === 'automlai-dualsearch'));
  const dual = templates.find((t) => t.name === 'automlai-dualsearch');
  assert.ok(dual.parameters.some((p) => p.name === 'workers'));
});
