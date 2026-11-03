---
title: Prometheus Anomaly Stream Context for LLM-Assisted Web UI
date: 2026-07-07
category: design-patterns
module: webui
problem_type: design_pattern
component: assistant
severity: low
applies_when:
  - "A web UI plots Prometheus metrics and should preserve anomaly detector choices through backend queries, rendering, and assistant context"
  - "Operators need both an ad hoc plot and a fixed recent anomaly stream for prompt-panel explanations"
  - "Quick anomaly detectors should be selectable without exposing every slow or unsuitable detector"
tags:
  - argo-automl-command-deck
  - prompt-panel
  - prometheus
  - autoad
  - anomaly-detection
  - metric-stream
  - llm-context
  - webui
---

# Prometheus Anomaly Stream Context for LLM-Assisted Web UI

## Context

The Argo AutoML Web UI already had a configurable Prometheus plot, but the Prompt Panel only received that one plot. Operators needed a second fixed-window metric anomaly stream and wanted the assistant to compare anomalies across both plots instead of narrating one chart.

Session history showed this stream design was iterative: it started as CPU-only, expanded to CPU, memory, network throughput, and latency, moved below both plots, and had its detector set trimmed after speed checks. Slow detectors were removed from quick mode, then `OneClassSVM` was removed from the dropdown, leaving a short interactive list. (session history)

## Guidance

Publish quick detector choices as data from the backend. Do not hard-code the dropdown separately from detector validation, because detector ids cross the JavaScript-to-Julia boundary.

```js
export const QUICK_DETECTORS = [
  { id: 'sk:LocalOutlierFactor', label: 'LocalOutlierFactor' },
  { id: 'sk:EllipticEnvelope', label: 'EllipticEnvelope' },
  { id: 'caret:pca', label: 'PCA' },
  { id: 'caret:mcd', label: 'MCD' }
];

function detectorId(value) {
  return QUICK_DETECTORS.some((d) => d.id === value) ? value : '';
}
```

Thread the selected detector through the existing Prometheus range path. One query path keeps metric bounds, AutoAD warnings, response metadata, and chart rendering consistent.

```js
export async function queryMetricRange(config, {
  metric = 'cpu',
  hours = 24,
  stepMinutes = 10,
  votepercent = 0.5,
  anomalyMode = 'quick',
  detector = ''
} = {}) {
  const anomalies = await detectAnomalies(points, {
    votepercent: anomalyVotepercent,
    mode: detectorMode,
    detector
  });

  return {
    metric,
    anomalyMode: anomalies.mode,
    detector: anomalies.detector,
    warnings: anomalies.warnings,
    points
  };
}
```

Keep detector prefixes explicit in the Julia bridge and reject unsupported values before they reach Julia by using the JS allowlist.

```julia
function detector_result(name)
  if startswith(name, "sk:")
    return fit_transform!(SKAnomalyDetector(name[4:end]), X)
  end
  if startswith(name, "caret:")
    return fit_transform!(CaretAnomalyDetector(name[7:end]), X)
  end
  error("Unsupported detector: " * name)
end
```

Reuse chart drawing and summary helpers for both canvases. Difference belongs in parameters: original plot uses user-selected window/step/mode, while the stream uses a fixed `24h` / `5m` quick-detector flow.

```js
function drawMetric(data, canvasId = 'metricChart') {
  const canvas = $(canvasId);
  // shared canvas plot rendering
}

function plotSummary(data) {
  const detector = data.detector ? `, detector ${data.detector}` : '';
  return `${data.label}: ${data.points.length} points, ${data.hours}h window, ${data.stepMinutes}m step, ${data.anomalyMode || 'quick'} mode${detector}, votepercent ${data.votepercent}, ${data.anomalyCount || 0} anomalies.`;
}
```

Send both plot contexts to the Prompt Panel. Keep the original plot as `plot`; send the fixed-window stream as `streamPlot`.

```js
const data = await json('/api/prompt', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    prompt: $('prompt').value,
    context: {
      template: state.current,
      parameters,
      logs: $('logs').textContent,
      plot: state.plot,
      streamPlot: state.cpuStream
    }
  })
});
```

## Why This Matters

A single chart answers “what happened to this metric?” A second anomaly stream answers “does a quick detector flag related signals in the same recent window?” The assistant becomes useful for cross-signal triage only when it can see both contexts: metric label, range, step, detector, anomaly count, warning state, and anomalous timestamps.

Allowlisting quick detectors also keeps the operator surface honest. Some AutoAD detectors are fine offline but too slow or noisy for an interactive stream; session history recorded slow detectors being excluded and the quick list narrowed before the final UI shipped. (session history)

## When to Apply

- Use this pattern when a primary observability plot should stay configurable but the assistant also needs a stable recent stream for comparison.
- Keep a bounded detector list when user-facing detector ids are passed to a model bridge or native runtime.
- Reuse the existing metric-range endpoint when both plots share the same data source and response shape.
- Add a separate endpoint only when the secondary stream has a different data source, auth policy, or response contract.

## Examples

Expose metrics and quick detectors from the same discovery endpoint:

```js
app.get('/api/prometheus/metrics', (_req, res) => res.json({
  status: 'ok',
  metrics: metricOptions(),
  quickDetectors: quickDetectorOptions()
}));
```

Render the stream controls with PCA as the default quick detector:

```js
$('cpuStreamMetric').innerHTML = metricOptions;
$('cpuStreamDetector').innerHTML = state.quickDetectors
  .map((d) => `<option value="${d.id}">${d.label}</option>`)
  .join('');
$('cpuStreamDetector').value = 'caret:pca';
```

Query the fixed stream without disturbing the primary plot:

```js
const params = new URLSearchParams({
  metric: $('cpuStreamMetric').value || 'cpu',
  hours: '24',
  stepMinutes: '5',
  anomalyMode: 'quick',
  detector: $('cpuStreamDetector').value
});
const data = await json(`/api/prometheus/range?${params}`);
state.cpuStream = data;
drawMetric(data, 'cpuStreamChart');
```

Test detector passthrough and LLM context shape:

```js
assert.equal(QUICK_DETECTORS.some((d) => d.id === 'sk:OneClassSVM'), false);

const out = await detectAnomalies(points, {
  detector: QUICK_DETECTORS[0].id,
  runner: async (_values, opts) => {
    assert.equal(opts.detector, QUICK_DETECTORS[0].id);
    return { code: 0, stdout: '2', stderr: '' };
  }
});
assert.deepEqual(out.indexes, [1]);

const ctx = buildPromptContext({ plot, streamPlot: { ...plot, metric: 'memory' } });
assert.equal(ctx.plot.anomalyCount, 1);
assert.equal(ctx.streamPlot.anomalyCount, 1);
```

Verification for the shipped change:

```sh
cd webui && npm test      # 43 pass
cd webui && npm run check # pass
```

## Related

- `docs/solutions/workflow-issues/argo-workflow-webui-llm-readonly-tools.md` - complementary Prompt Panel safety pattern for read-only Argo/MLflow tools.
- `webui/lib/anomaly.js` - quick detector catalog, detector validation, Julia detector prefixes, anomaly metadata.
- `webui/lib/prometheus.js` - metric options, quick detector options, range query detector pass-through.
- `webui/server.js` - `/api/prometheus/metrics` publishes `quickDetectors`.
- `webui/public/index.html` - metric anomaly stream controls and second canvas.
- `webui/public/app.js` - shared chart rendering, stream query, prompt context with `streamPlot`.
- `webui/lib/llm.js` - compact dual-plot prompt context and comparison instructions.
- `webui/test/anomaly.test.js`, `webui/test/prometheus.test.js`, `webui/test/llm.test.js` - detector, query, and prompt-context coverage.
