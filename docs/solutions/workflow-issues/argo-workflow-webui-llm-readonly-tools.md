---
title: LLM Read-Only Tools for Argo and MLflow
date: 2026-07-07
category: workflow-issues
module: argo-workflow/webui
problem_type: workflow_issue
component: assistant
severity: medium
applies_when:
  - "A prompt-panel LLM needs operational context from Argo workflows, templates, logs, or MLflow results"
  - "Read-only tool calls can be safe without confirmation, but workflow mutations must stay gated"
tags:
  - argo-workflows
  - mlflow
  - llm-tools
  - read-only-tools
  - litellm
  - permission-gating
---

# LLM Read-Only Tools for Argo and MLflow

## Context

`argo-workflow/webui` has a prompt panel backed by an OpenAI-compatible ETE LiteLLM model. The panel needed first-class access to live operational context: Argo workflows, templates, logs, workflow nodes, and MLflow experiments, runs, metrics, and artifacts.

The important boundary: read-only inspection should not interrupt the user with confirmation prompts, but anything that submits, retries, deletes, suspends, resumes, or otherwise mutates workflow state must stay behind existing confirmation and permission gates.

Session history found no prior implementation attempt for this exact prompt-panel tool-access work. Older related sessions only refreshed graph metadata, so they did not change the solution direction. (session history)

## Guidance

Expose a small read-only tool registry to the LLM. Do not expose mutating helpers just because they exist in nearby backend modules.

```js
export const readOnlyTools = [
  tool('list_workflows', 'List recent Argo workflows', schema),
  tool('get_workflow_status', 'Get Argo workflow status', schema),
  tool('get_workflow_logs', 'Get Argo workflow logs', schema),
  tool('get_workflow_nodes', 'Get compact node status', schema),
  tool('mlflow_search_runs', 'Search MLflow runs', schema),
  tool('mlflow_get_run', 'Get one MLflow run', schema),
  tool('mlflow_get_metric_history', 'Get MLflow metric history', schema),
  tool('mlflow_list_artifacts', 'List MLflow artifacts', schema)
];
```

Route tool calls through an explicit handler map. This keeps policy visible and gives tests a cheap injection point.

```js
async function runReadOnlyTool(config, name, args) {
  if (config.llmToolHandlers?.[name]) return config.llmToolHandlers[name](args);

  const handlers = {
    list_workflows: (a) => listWorkflows(config, a),
    get_workflow_logs: (a) => getWorkflowLogs(config, a.name, a),
    mlflow_search_runs: (a) => searchRuns(config, a),
    mlflow_get_run: (a) => getRun(config, a.runId)
  };

  if (!handlers[name]) throw new Error(`Unsupported read-only tool ${name}`);
  return handlers[name](args);
}
```

Bound the tool loop. A model can ask for tools repeatedly; the server should not let that become an unbounded agent run.

```js
for (let i = 0; i < 3; i += 1) {
  const json = await chat(config, messages);
  const calls = json.choices?.[0]?.message?.tool_calls || [];
  if (!calls.length) return finalAnswer(json);
  await appendToolResults(calls);
}
```

Treat every tool result as untrusted prompt input. Redact and truncate before adding it back to the model context.

```js
messages.push({
  role: 'tool',
  tool_call_id: call.id,
  name,
  content: redact(JSON.stringify(result)).slice(0, 12000)
});
```

Keep deploy/submit intent outside the read-only flow. Detect mutation requests before calling the LLM and return a confirmation payload instead.

```js
const confirmation = maybeConfirmation(config, body.prompt, body.context || {});
if (confirmation) return confirmation;
```

## Why This Matters

LLM tool access is useful for operations questions:

- Which workflow failed?
- What do the latest logs say?
- Which MLflow run has the best metric?
- What artifacts did this run produce?

The same tool mechanism can also become dangerous if it includes mutating operations. A careless or prompt-injected answer could submit new workloads, retry expensive jobs, delete archived workflows, or leak secrets from logs. Minimal safe shape: default tools inspect only; state changes require a separate confirmation and the existing `ARGO_MCP_ALLOWED_OPERATIONS` gate.

## When to Apply

Use this pattern when an LLM assistant needs live context from operational systems:

- workflow engines such as Argo, Airflow, or Tekton
- experiment trackers such as MLflow
- CI/CD status and logs
- observability dashboards
- artifact browsers

No-confirmation tools should be idempotent reads. Any tool that creates, updates, deletes, retries, terminates, submits, deploys, or changes scheduling belongs outside the default tool registry.

## Examples

Test that read-only tools are exposed:

```js
const names = readOnlyTools.map((t) => t.function.name);
assert.ok(names.includes('mlflow_search_runs'));
assert.ok(names.includes('list_workflows'));
```

Test that the LLM can call Argo and MLflow read-only tools without confirmation:

```js
const out = await answerPrompt({
  llm: { apiKey: 'sk-live', baseUrl: 'https://ete.example/v1', model: 'azure/gpt-5.5' },
  llmToolHandlers: {
    list_workflows: async () => ({ status: 'ok', workflows: [] }),
    mlflow_search_runs: async () => ({ runs: [] })
  }
}, { prompt: 'show workflows and mlflow runs' });

assert.equal(out.type, 'answer');
```

Test that mutating tools stay excluded:

```js
const names = readOnlyTools.map((t) => t.function.name);
assert.equal(names.includes('submit_template_workflow'), false);
assert.equal(names.includes('operate_workflow'), false);
```

Verification for this change:

```sh
npm --prefix webui run check
npm --prefix webui test
```

Both passed from `argo-workflow` with 34 tests.

## Related

- `argo-workflow/webui/lib/llm.js` - read-only tool definitions, handler map, bounded tool loop, redaction, confirmation gate
- `argo-workflow/webui/test/llm.test.js` - coverage for tool exposure, no-confirmation read-only calls, mutating-tool exclusion
- `argo-workflow/webui/README.md` - user-facing prompt-panel tool access docs
- `docs/plans/2026-07-07-001-feat-argo-automl-webui-plan.md` - original safety architecture for UI, MCP, LLM, Argo, and MLflow surfaces
