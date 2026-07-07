import test from 'node:test';
import assert from 'node:assert/strict';
import { parseYamlDocuments, parametersFromDoc, validateUploadYaml, validateParameters } from '../lib/yaml.js';

const cwt = `apiVersion: argoproj.io/v1alpha1
kind: ClusterWorkflowTemplate
metadata:
  name: automlai-dualsearch
spec:
  arguments:
    parameters:
      - name: workers
        value: 5
      - name: input
        value: iris.csv
`;

test('extracts parameters from template YAML', () => {
  const doc = parseYamlDocuments(cwt)[0];
  assert.deepEqual(parametersFromDoc(doc).map((p) => p.name), ['workers', 'input']);
});

test('uploaded ClusterWorkflowTemplate is preview-only', () => {
  assert.equal(validateUploadYaml(cwt).mode, 'preview-template');
});

test('rejects unsafe workflow fields', () => {
  const wf = `apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata: {name: bad}
spec:
  templates:
  - name: main
    container:
      image: alpine
      securityContext: {privileged: true}
`;
  assert.throws(() => validateUploadYaml(wf), /Unsafe workflow fields/);
});

test('validates known AutoML params', () => {
  assert.deepEqual(validateParameters({ workers: '-1', input: '../x.csv', predictiontype: 'nope' }), {
    workers: 'Must be an integer >= 1',
    input: 'Must be a .csv filename without paths',
    predictiontype: 'Must be one of classification, regression, anomalydetection'
  });
});
