# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Argo AutoML Web UI

### Argo AutoML Command Deck
A local operator dashboard for submitting AutoML workflows, inspecting workflow state, and querying experiment results without exposing raw cluster command execution.

### Prompt Panel
An assistant surface in the command deck that can answer operational questions by reading workflow, experiment, and observability context, while workflow state changes remain separate confirmed actions.

### Metric Anomaly Stream
A fixed-window observability plot that samples one operational metric over recent time and flags anomalous points with a bounded quick detector so operators and the Prompt Panel can compare signals without changing the primary plot.

### Quick Detector
An allowlisted anomaly detector considered fast enough for interactive metric exploration, distinct from full ensemble detection intended for slower or broader analysis.

### Read-Only Tool Registry
The allowlist of assistant-callable tools that inspect workflow or experiment state without changing cluster, schedule, run, or artifact state.

### Mutating Workflow Operation
Any action that creates, retries, stops, resumes, suspends, terminates, deletes, or otherwise changes workflow execution state.

### Workflow Template
A reusable workflow definition that supplies the parameters and execution shape for an AutoML workflow submission.

### Experiment Run
A recorded MLflow execution with parameters, metrics, tags, and artifacts used to compare AutoML outcomes.
