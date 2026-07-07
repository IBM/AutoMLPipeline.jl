import test from 'node:test';
import assert from 'node:assert/strict';
import { requireMutationAuth, config } from '../server.js';

function req({ cookie = '', origin = 'http://127.0.0.1:8090', host = '127.0.0.1:8090', authorization = '' } = {}) {
  return { get(name) { return { cookie, origin, host, authorization }[name.toLowerCase()] || ''; } };
}

function res() {
  return { code: 200, body: null, status(code) { this.code = code; return this; }, json(body) { this.body = body; return this; } };
}

test('accepts same-origin httpOnly session cookie', () => {
  const out = res();
  let next = false;
  requireMutationAuth(req({ cookie: `argo_webui_token=${config.sessionToken}` }), out, () => { next = true; });
  assert.equal(next, true);
});

test('rejects cross-origin mutation with valid cookie', () => {
  const out = res();
  requireMutationAuth(req({ cookie: `argo_webui_token=${config.sessionToken}`, origin: 'http://evil.test' }), out, () => {});
  assert.equal(out.code, 403);
});
