import { spawn } from 'node:child_process';
import path from 'node:path';
import { redact } from './redact.js';

const ROOT = path.resolve(new URL('../..', import.meta.url).pathname);
const JULIA_SCRIPT = `
using AutoAD, DataFrames
vals = parse.(Float64, split(read(stdin, String)))
if length(vals) < 25
  exit()
end
votepercent = parse(Float64, ARGS[1])
model = AutoAnomalyDetection(Dict(:votepercent => votepercent))
out = fit_transform!(model, DataFrame(auto = vals))
flags = Bool.(out[!, names(out)[end]])
print(join(findall(flags), ","))
`;

function boundedVotepercent(value) {
  const n = Number(value ?? 0.5);
  return Number.isFinite(n) ? Math.min(1, Math.max(0.1, n)) : 0.5;
}

function runJulia(values, { timeoutMs = 120000, votepercent = 0.5 } = {}) {
  return new Promise((resolve) => {
    const child = spawn('julia', [`--project=${path.join(ROOT, 'AutoAD')}`, '-e', JULIA_SCRIPT, String(boundedVotepercent(votepercent))], { stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    const timer = setTimeout(() => child.kill('SIGTERM'), timeoutMs);
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr: redact(stderr) });
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      resolve({ code: 127, stdout: '', stderr: redact(error.message) });
    });
    child.stdin.end(values.join('\n'));
  });
}

export async function detectAnomalies(points, { runner = runJulia, votepercent = 0.5 } = {}) {
  if (points.length < 25) return { indexes: [], warnings: ['Need at least 25 points for AutoAD anomaly detection.'] };
  const result = await runner(points.map((p) => p.value), { votepercent: boundedVotepercent(votepercent) });
  if (result.code !== 0) return { indexes: [], warnings: [`AutoAD unavailable: ${result.stderr || 'julia failed'}`] };
  const indexes = String(result.stdout || '').split(',').map((n) => Number(n) - 1).filter((n) => Number.isInteger(n) && n >= 0 && n < points.length);
  return { indexes, warnings: [] };
}
