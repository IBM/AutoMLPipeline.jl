import { z } from 'zod';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { listTemplates } from './templates.js';
import {
  allowedOperations,
  submitTemplateWorkflow,
  submitYamlWorkflow,
  listWorkflows,
  getWorkflow,
  getWorkflowLogs,
  watchWorkflow,
  workflowOperation,
  resubmitWorkflow,
  workflowNodes,
  listWorkflowTemplates,
  getWorkflowTemplate,
  listCronWorkflows,
  getCronWorkflow,
  cronWorkflowOperation,
  argoVersion,
  lintArgoYaml,
  listArchivedWorkflows,
  getArchivedWorkflow,
  listArchiveLabelKeys,
  listArchiveLabelValues,
  archiveWorkflowOperation
} from './argo.js';
import {
  queryMlflowResults,
  searchExperiments,
  getExperiment,
  getExperimentByName,
  searchRuns,
  getRun,
  summarizeRun,
  compareRuns,
  getMetricHistory,
  listArtifacts
} from './mlflow.js';

const content = (value) => ({ content: [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) }] });

export function createMcpServer(config) {
  const server = new McpServer({ name: 'argo-automl-command-deck', version: '0.1.0' });
  server.registerTool('list_workflow_templates', { description: 'List discovered ClusterWorkflowTemplates with UI parameter metadata.' }, async () => content(await listTemplates(config)));
  server.registerTool('list_argo_templates', {
    description: 'List Argo WorkflowTemplates or ClusterWorkflowTemplates via Argo CLI.',
    inputSchema: { cluster: z.boolean().optional(), namespace: z.string().optional() }
  }, async (args) => content(await listWorkflowTemplates(config, args)));
  server.registerTool('get_argo_template', {
    description: 'Get one Argo WorkflowTemplate or ClusterWorkflowTemplate as JSON.',
    inputSchema: { name: z.string(), cluster: z.boolean().optional(), namespace: z.string().optional() }
  }, async (args) => content(await getWorkflowTemplate(config, args)));
  server.registerTool('submit_template_workflow', {
    description: 'Submit an existing ClusterWorkflowTemplate with parameters.',
    inputSchema: { templateName: z.string(), parameters: z.record(z.string(), z.any()).optional(), requestId: z.string().optional() }
  }, async (args) => content(await submitTemplateWorkflow(config, args)));
  server.registerTool('submit_yaml_workflow', {
    description: 'Submit uploaded Argo Workflow YAML after validation. ClusterWorkflowTemplate YAML is preview-only.',
    inputSchema: { yaml: z.string(), dryRun: z.boolean().optional() }
  }, async (args) => content(await submitYamlWorkflow(config, args)));
  server.registerTool('list_workflows', { description: 'List recent Argo workflows.', inputSchema: { limit: z.number().optional(), namespace: z.string().optional() } }, async (args) => content(await listWorkflows(config, args)));
  server.registerTool('get_workflow_status', { description: 'Get workflow status.', inputSchema: { name: z.string(), namespace: z.string().optional() } }, async ({ name, namespace }) => content(await getWorkflow(config, name, { namespace })));
  server.registerTool('get_workflow_logs', { description: 'Get workflow logs.', inputSchema: { name: z.string(), tailLines: z.number().optional(), namespace: z.string().optional() } }, async ({ name, tailLines, namespace }) => content(await getWorkflowLogs(config, name, { tailLines, namespace })));
  server.registerTool('watch_workflow', { description: 'Watch one workflow briefly for status changes.', inputSchema: { name: z.string(), namespace: z.string().optional() } }, async (args) => content(await watchWorkflow(config, args)));
  server.registerTool('operate_workflow', {
    description: 'Run a gated workflow operation: suspend, resume, retry, terminate, stop, or delete. Allowed by ARGO_MCP_ALLOWED_OPERATIONS.',
    inputSchema: { name: z.string(), operation: z.enum(['suspend', 'resume', 'retry', 'terminate', 'stop', 'delete']), namespace: z.string().optional() }
  }, async (args) => content(await workflowOperation(config, args)));
  server.registerTool('resubmit_workflow', {
    description: 'Resubmit a completed workflow, optionally overriding parameters or using memoized steps. Gated by resubmit permission.',
    inputSchema: { name: z.string(), namespace: z.string().optional(), parameters: z.record(z.string(), z.any()).optional(), memoized: z.boolean().optional() }
  }, async (args) => content(await resubmitWorkflow(config, args)));
  server.registerTool('get_workflow_nodes', { description: 'Return compact node status list for one workflow.', inputSchema: { name: z.string(), namespace: z.string().optional() } }, async (args) => content(await workflowNodes(config, args)));
  server.registerTool('list_cron_workflows', { description: 'List Argo CronWorkflows.', inputSchema: { namespace: z.string().optional() } }, async (args) => content(await listCronWorkflows(config, args)));
  server.registerTool('get_cron_workflow', { description: 'Get one Argo CronWorkflow as JSON.', inputSchema: { name: z.string(), namespace: z.string().optional() } }, async (args) => content(await getCronWorkflow(config, args)));
  server.registerTool('operate_cron_workflow', {
    description: 'Run a gated CronWorkflow operation: suspend, resume, or delete. Allowed by ARGO_MCP_ALLOWED_OPERATIONS using cron:<operation>.',
    inputSchema: { name: z.string(), operation: z.enum(['suspend', 'resume', 'delete']), namespace: z.string().optional() }
  }, async (args) => content(await cronWorkflowOperation(config, args)));
  server.registerTool('argo_version', { description: 'Read Argo CLI/server version information.' }, async () => content(await argoVersion(config)));
  server.registerTool('lint_argo_yaml', {
    description: 'Lint Argo Workflow, WorkflowTemplate, CronWorkflow, or ClusterWorkflowTemplate YAML using argo lint.',
    inputSchema: { yaml: z.string(), kinds: z.array(z.enum(['workflows', 'workflowtemplates', 'cronworkflows', 'clusterworkflowtemplates'])).optional(), offline: z.boolean().optional() }
  }, async (args) => content(await lintArgoYaml(config, args)));
  server.registerTool('list_archived_workflows', { description: 'List archived workflows.', inputSchema: { selector: z.string().optional(), output: z.enum(['name', 'json', 'yaml', 'wide']).optional(), chunkSize: z.number().optional() } }, async (args) => content(await listArchivedWorkflows(config, args)));
  server.registerTool('get_archived_workflow', { description: 'Get one archived workflow by UID.', inputSchema: { uid: z.string(), output: z.enum(['json', 'yaml', 'wide']).optional() } }, async (args) => content(await getArchivedWorkflow(config, args)));
  server.registerTool('list_archive_label_keys', { description: 'List archived workflow label keys.' }, async () => content(await listArchiveLabelKeys(config)));
  server.registerTool('list_archive_label_values', { description: 'List values for one archived workflow label key.', inputSchema: { key: z.string() } }, async (args) => content(await listArchiveLabelValues(config, args)));
  server.registerTool('operate_archived_workflow', {
    description: 'Run a gated archive operation: retry, resubmit, or delete. Allowed by ARGO_MCP_ALLOWED_OPERATIONS using archive:<operation>.',
    inputSchema: { uid: z.string(), operation: z.enum(['retry', 'resubmit', 'delete']), parameters: z.record(z.string(), z.any()).optional(), memoized: z.boolean().optional() }
  }, async (args) => content(await archiveWorkflowOperation(config, args)));
  server.registerTool('get_mcp_permissions', { description: 'Show currently allowed mutating MCP operations.' }, async () => content(allowedOperations(config)));
  server.registerTool('query_mlflow_results', { description: 'Read-only MLflow result query.', inputSchema: { runId: z.string().optional(), experimentId: z.string().optional() } }, async (args) => content(await queryMlflowResults(config, args)));
  server.registerTool('mlflow_search_experiments', {
    description: 'Search MLflow experiments using MLflow v2 Tracking API.',
    inputSchema: { filter: z.string().optional(), maxResults: z.number().optional(), pageToken: z.string().optional(), orderBy: z.array(z.string()).optional(), viewType: z.enum(['ACTIVE_ONLY', 'DELETED_ONLY', 'ALL']).optional() }
  }, async ({ maxResults, pageToken, viewType, orderBy, ...args }) => content(await searchExperiments(config, { ...args, max_results: maxResults, page_token: pageToken, view_type: viewType, order_by: orderBy })));
  server.registerTool('mlflow_get_experiment', {
    description: 'Get one MLflow experiment by ID or name.',
    inputSchema: { experimentId: z.string().optional(), name: z.string().optional() }
  }, async ({ experimentId, name }) => content(name ? await getExperimentByName(config, name) : await getExperiment(config, experimentId)));
  server.registerTool('mlflow_search_runs', {
    description: 'Search MLflow runs by experiment IDs, filter, ordering, and page size.',
    inputSchema: { experimentIds: z.array(z.string()).optional(), filter: z.string().optional(), maxResults: z.number().optional(), orderBy: z.array(z.string()).optional(), pageToken: z.string().optional() }
  }, async ({ experimentIds, maxResults, orderBy, pageToken, ...args }) => content(await searchRuns(config, { ...args, experiment_ids: experimentIds, max_results: maxResults, order_by: orderBy, page_token: pageToken })));
  server.registerTool('mlflow_get_run', { description: 'Get one MLflow run with latest metrics, params, tags, inputs, and info.', inputSchema: { runId: z.string() } }, async ({ runId }) => content(await getRun(config, runId)));
  server.registerTool('mlflow_summarize_run', { description: 'Summarize one MLflow run into params, metrics, tags, and info maps.', inputSchema: { runId: z.string() } }, async (args) => content(await summarizeRun(config, args)));
  server.registerTool('mlflow_compare_runs', {
    description: 'Compare MLflow runs, optionally selecting metric keys.',
    inputSchema: { experimentIds: z.array(z.string()).optional(), filter: z.string().optional(), metricKeys: z.array(z.string()).optional(), maxResults: z.number().optional(), orderBy: z.array(z.string()).optional() }
  }, async (args) => content(await compareRuns(config, args)));
  server.registerTool('mlflow_get_metric_history', { description: 'Get full value history for one MLflow metric key in one run.', inputSchema: { runId: z.string(), metricKey: z.string() } }, async (args) => content(await getMetricHistory(config, args)));
  server.registerTool('mlflow_list_artifacts', { description: 'List MLflow artifacts for a run and optional artifact path.', inputSchema: { runId: z.string(), path: z.string().optional() } }, async (args) => content(await listArtifacts(config, args)));
  return server;
}

export async function handleMcpRequest(config, req, res) {
  const server = createMcpServer(config);
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on('close', () => server.close().catch(() => {}));
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
}
