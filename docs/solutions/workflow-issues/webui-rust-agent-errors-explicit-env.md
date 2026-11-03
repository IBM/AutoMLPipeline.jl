---
title: WebUI Rust Agent Errors Require Explicit Environment Configuration
date: 2026-10-18
category: workflow-issues
module: webui-rust
problem_type: integration_issue
component: assistant
symptoms:
  - "Prompt Panel returned local answers when the remote LLM endpoint was misconfigured or unreachable"
  - "LLM configuration could be silently sourced from fallback credentials instead of the intended runtime environment"
  - "Operators could not tell whether the agent or local diagnostic path answered a prompt"
root_cause: config_error
resolution_type: code_fix
severity: medium
tags:
  - webui-rust
  - prompt-panel
  - litellm
  - agent-error
  - zshenv
  - credentials
---

# WebUI Rust Agent Errors Require Explicit Environment Configuration

## Problem

`webui-rust` has a Prompt Panel that must call an OpenAI-compatible LiteLLM agent. When the agent endpoint or credentials were wrong, the server previously returned local diagnostic answers as if they were normal prompt responses. That hid whether the remote agent was reachable.

## Symptoms

- Prompt responses could have `type: local_answer` after a failed remote agent call.
- `/api/config` exposed only masked credential state, so the UI could show a key was present while the agent call was still unreachable.
- A malformed or incomplete base URL could build the wrong chat-completions endpoint.
- Same prompt could answer locally or remotely without the operator seeing which path ran.

## What Didn't Work

- Treating `OPENAI_*` and Pi `models.json` as fallback defaults made debugging ambiguous: the web UI could use credentials different from the operator's `BASE_URL` and `ETE_API_KEY` values.
- Returning local answers after `call_llm` failed converted an integration failure into a success response.
- Returning deterministic plot explanations locally avoided timeout but still violated the rule that the agent must answer or fail.
- Rendering only a generic thrown fetch error in the browser lost the structured error payload from `/api/prompt`.

## Solution

Make agent configuration strict and make every failed agent path explicit.

`load_llm_config` reads only `BASE_URL`, `ETE_API_KEY`, and optional `LLM_MODEL` from the process environment or simple assignments in `~/.zshenv`. It does not fall back to OpenAI variables, Pi `models.json`, or a default public OpenAI URL.

`/api/llm` probes the submitted endpoint/model/key through the same `/chat/completions` path used by `/api/prompt` before persisting UI credential changes. Bad credentials return non-2xx `agent_error` and leave the previous working config in place. This keeps save status synchronized with prompt/explanation reachability.

`/api/prompt` now has no fallback path. If credentials are missing, the remote call times out, the endpoint is unreachable, the response is non-success, JSON is invalid, or message content is empty, the route returns non-2xx `agent_error`. It does not include `localAnswer` and does not synthesize a plot explanation.

Prompt context is still compacted before remote LLM calls: raw plot `points` arrays are replaced with summary fields and capped anomaly points. Plot explanation prompts use the full-detector plot by default and are normalized to concise agent instructions before the same remote chat call. Quick-detector stream context is included only when the user asks for quick, stream, or uni-detector data. This reduces timeout-prone payload/wording without bypassing the agent.

`call_llm` uses a bounded timeout, retries transient transport failures against the same `/chat/completions` endpoint, preserves `/v1` when constructing `/chat/completions`, treats non-success HTTP status, invalid JSON, and empty content as explicit errors, and redacts the API key from surfaced error text.

The browser keeps the structured body from failed `fetch` calls, renders `agent_error` details in the answer panel and credential-save panel, and marks the LLM status badge as `agent error` or `missing BASE_URL` / `missing ETE_API_KEY`.

## Why This Works

Remote agent availability is part of the contract. Absence of the agent is materially different from a successful answer. By making failed calls non-2xx and removing fallback answers, the UI has only two prompt states:

- remote agent answered successfully (`type: answer`)
- remote agent did not answer (`type: agent_error`)

No local diagnostic can be confused for an agent answer.

## Prevention

- Keep agent defaults tied to the deployment contract: `BASE_URL`, `ETE_API_KEY`, and optional `LLM_MODEL`.
- Never convert failed remote dependency calls into successful prompt responses.
- Add route-level tests: missing config must return `agent_error` without `localAnswer`; fallback sources must stay ignored; save-time verification must reject unreachable chat-completions endpoints without replacing previous config.
- Validate live service state with both success and forced-unreachable smoke checks before handoff.
- Browser-test the Prompt Panel with Playwright so the real Plot → Ask agent path is covered, not only direct API calls.

## Related Issues

- `webui-rust/src/main.rs` - LLM config loading, prompt route, context compaction, chat URL construction, remote call error handling, and regression tests.
- `webui-rust/public/app.js` - browser fetch error preservation, LLM status badge, and rendered prompt output.
- `webui-rust/public/markdown.js` - `agent_error` answer-panel formatting.
- `webui-rust/README.md` - documented `BASE_URL`, `ETE_API_KEY`, and `LLM_MODEL` configuration.
- `docs/solutions/workflow-issues/argo-workflow-webui-llm-readonly-tools.md` - related Prompt Panel safety pattern for original Node web UI.
- `docs/solutions/design-patterns/argo-automl-webui-prometheus-anomaly-stream-context.md` - related plot-context pattern for Prometheus explanations.
