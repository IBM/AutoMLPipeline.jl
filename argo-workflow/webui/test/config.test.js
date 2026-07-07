import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { buildConfig, publicConfig } from '../lib/config.js';

function modelsFile(apiKey = 'ete-key') {
  const file = path.join(os.tmpdir(), `models-${Date.now()}-${Math.random()}.json`);
  fs.writeFileSync(file, JSON.stringify({ providers: { external: { baseUrl: 'https://ete.example/v1', apiKey, models: [{ id: 'azure/gpt-5.5' }] } } }));
  return file;
}

test('Pi external ETE key wins over stale OPENAI_API_KEY by default', () => {
  const cfg = buildConfig({ PI_MODELS_FILE: modelsFile(), OPENAI_API_KEY: 'sk-stale' });
  assert.equal(cfg.llm.apiKey, 'ete-key');
  assert.equal(cfg.llm.baseUrl, 'https://ete.example/v1');
  assert.equal(cfg.llm.model, 'azure/gpt-5.5');
  assert.equal(cfg.llm.source, 'pi-external');
});

test('explicit OPENAI_BASE_URL override opts into env LLM config', () => {
  const cfg = buildConfig({ PI_MODELS_FILE: modelsFile(), OPENAI_API_KEY: 'sk-env', OPENAI_BASE_URL: 'https://other.example/v1', OPENAI_MODEL: 'x' });
  assert.equal(cfg.llm.apiKey, 'sk-env');
  assert.equal(cfg.llm.baseUrl, 'https://other.example/v1');
  assert.equal(cfg.llm.model, 'x');
  assert.equal(cfg.llm.source, 'env');
});

test('public config reports source without key', () => {
  const pub = publicConfig(buildConfig({ PI_MODELS_FILE: modelsFile(), OPENAI_API_KEY: 'sk-stale' }));
  assert.equal(pub.llm.source, 'pi-external');
  assert.equal(pub.llm.hasKey, true);
  assert.equal(JSON.stringify(pub).includes('ete-key'), false);
});

test('expands Pi external env var API key references', () => {
  const cfg = buildConfig({ PI_MODELS_FILE: modelsFile('$ETE_API_KEY'), ETE_API_KEY: 'sk-live' });
  assert.equal(cfg.llm.apiKey, 'sk-live');
  assert.equal(cfg.llm.source, 'pi-external');
});

test('does not send unresolved Pi external env var references', () => {
  const cfg = buildConfig({ PI_MODELS_FILE: modelsFile('$MISSING_ETE_KEY') });
  assert.equal(cfg.llm.apiKey, '');
  assert.equal(cfg.llm.source, 'env-fallback');
});
