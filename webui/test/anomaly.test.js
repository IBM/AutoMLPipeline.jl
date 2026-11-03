import test from 'node:test';
import assert from 'node:assert/strict';
import { detectAnomalies, QUICK_DETECTORS } from '../lib/anomaly.js';

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

test('allows quick detector choices', async () => {
  assert.equal(QUICK_DETECTORS.length, 4);
  assert.equal(QUICK_DETECTORS.some((d) => d.id === 'sk:OneClassSVM'), false);
  const points = Array.from({ length: 30 }, (_, value) => ({ value }));
  let seen;
  const out = await detectAnomalies(points, { detector: QUICK_DETECTORS[0].id, runner: async (_values, opts) => { seen = opts; return { code: 0, stdout: '2', stderr: '' }; } });
  assert.equal(seen.detector, QUICK_DETECTORS[0].id);
  assert.equal(out.detector, QUICK_DETECTORS[0].id);
  assert.deepEqual(out.indexes, [1]);
});
