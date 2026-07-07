import test from 'node:test';
import assert from 'node:assert/strict';
import { metricOptions, prometheusBase, queryMetricRange } from '../lib/prometheus.js';

test('publishes four metric choices', () => {
  const options = metricOptions();
  assert.deepEqual(options.map((m) => m.id), ['cpu', 'memory', 'network', 'latency']);
  assert.equal(options.find((m) => m.id === 'latency').label, 'Network probe p95 latency');
});

test('uses explicit Prometheus API URL before scrape metrics URL', () => {
  assert.equal(String(prometheusBase({ prometheusApiUrl: 'http://prom2.prometheus.home:8080', prometheusMetricsUrl: 'http://prom1.prometheus.home:8080/metrics' })), 'http://prom2.prometheus.home:8080/');
});

test('falls back by converting /metrics URL to Prometheus API base', () => {
  assert.equal(String(prometheusBase('http://prom1.prometheus.home:8080/metrics')), 'http://prom1.prometheus.home:8080/');
});

test('queries one day at 10 minute resolution', async () => {
  const oldFetch = global.fetch;
  let seen;
  global.fetch = async (url) => {
    seen = new URL(url);
    return { ok: true, json: async () => ({ status: 'success', data: { result: [{ values: [[1, '2.5'], [2, '3']] }] } }) };
  };
  try {
    const out = await queryMetricRange({ prometheusApiUrl: 'http://prom.test', prometheusMetricsUrl: 'http://wrong.test/metrics' }, { metric: 'cpu' });
    assert.equal(seen.pathname, '/api/v1/query_range');
    assert.equal(seen.searchParams.get('step'), '600');
    assert.equal(Number(seen.searchParams.get('end')) - Number(seen.searchParams.get('start')), 86400);
    assert.deepEqual(out.points, [{ ts: 1, value: 2.5 }, { ts: 2, value: 3 }]);
  } finally {
    global.fetch = oldFetch;
  }
});
