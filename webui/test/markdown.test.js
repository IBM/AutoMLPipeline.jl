import test from 'node:test';
import assert from 'node:assert/strict';
import { markdownToHtml, promptResultMarkdown } from '../public/markdown.js';

test('renders basic markdown and escapes html', () => {
  const html = markdownToHtml('## Hello\n\n**safe** `<script>`\n\n- one');
  assert.match(html, /<h2>Hello<\/h2>/);
  assert.match(html, /<strong>safe<\/strong>/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /<li>one<\/li>/);
});

test('formats prompt result as markdown', () => {
  const md = promptResultMarkdown({ type: 'confirmation_required', tool: 'submit_template_workflow', namespace: 'argo', templateName: 'automl-classification', parameters: { input: 'iris.csv' } });
  assert.match(md, /## Confirm deployment/);
  assert.match(md, /```json/);
  assert.match(md, /automl-classification/);
});
