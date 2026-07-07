import test from 'node:test';
import assert from 'node:assert/strict';
import { createMcpServer } from '../lib/mcp.js';

test('creates MCP server', () => {
  const server = createMcpServer({ namespace: 'argo', argoBin: 'argo', mlflowUrl: 'http://mlflow.home:8080', mcpAllowedOperations: ['submit'] });
  assert.equal(typeof server.connect, 'function');
});
