import test from 'node:test';
import assert from 'node:assert/strict';
import { detectAnomalies } from '../lib/anomaly.js';

test('skips AutoAD for tiny series', async () => {
  const out = await detectAnomalies([{ value: 1 }, { value: 2 }]);
  assert.deepEqual(out.indexes, []);
  assert.equal(out.mode, 'quick');
  assert.match(out.warnings[0], /at least 25/);
});

test('maps AutoAD one-based indexes to point indexes', async () => {
  const points = Array.from({ length: 30 }, (_, value) => ({ value }));
  const out = await detectAnomalies(points, { runner: async () => ({ code: 0, stdout: '1,3,30', stderr: '' }) });
  assert.deepEqual(out.indexes, [0, 2, 29]);
  assert.equal(out.mode, 'quick');
});

test('passes bounded votepercent to AutoAD runner', async () => {
  const points = Array.from({ length: 30 }, (_, value) => ({ value }));
  let seen;
  await detectAnomalies(points, { votepercent: 2, mode: 'full', runner: async (_values, opts) => { seen = opts; return { code: 0, stdout: '', stderr: '' }; } });
  assert.equal(seen.votepercent, 1);
  assert.equal(seen.mode, 'full');
});
