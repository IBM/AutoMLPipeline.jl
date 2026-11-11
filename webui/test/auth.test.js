import test from 'node:test';
import assert from 'node:assert/strict';
import { app, requireMutationAuth, config } from '../server.js';

function req({ cookie = '', origin = `http://${config.host}:${config.port}`, host = 'rebind.example:8090', authorization = '', token = '' } = {}) {
  const headers = { cookie, origin, host, authorization, 'x-argo-webui-token': token };
  return { get(name) { return headers[name.toLowerCase()] || ''; } };
}

function res() {
  return { code: 200, body: null, status(code) { this.code = code; return this; }, json(body) { this.body = body; return this; } };
}

test('accepts same-origin header session token', () => {
  const out = res();
  let next = false;
  requireMutationAuth(req({ token: config.sessionToken }), out, () => { next = true; });
  assert.equal(next, true);
});

test('rejects self-issued cookie token even from same-origin request', () => {
  const out = res();
  requireMutationAuth(req({ cookie: `argo_webui_token=${config.sessionToken}` }), out, () => {});
  assert.equal(out.code, 403);
});

test('rejects mutation when Origin is missing', () => {
  const out = res();
  requireMutationAuth(req({ origin: '', token: config.sessionToken }), out, () => {});
  assert.equal(out.code, 403);
});

test('rejects cross-origin mutation with valid header token', () => {
  const out = res();
  requireMutationAuth(req({ token: config.sessionToken, origin: 'http://evil.test' }), out, () => {});
  assert.equal(out.code, 403);
});

test('rejects Host-matched DNS-rebinding-shaped Origin', () => {
  const out = res();
  requireMutationAuth(req({ token: config.sessionToken, origin: 'http://rebind.example:8090', host: 'rebind.example:8090' }), out, () => {});
  assert.equal(out.code, 403);
});

test('GET /api/config does not issue mutation token cookie', async () => {
  const server = app.listen(0, '127.0.0.1');
  await new Promise((resolve) => server.once('listening', resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/config`);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('set-cookie'), null);
  } finally {
    await new Promise((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
  }
});
