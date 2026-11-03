import { spawn } from 'node:child_process';
import path from 'node:path';
import { redact } from './redact.js';

const ROOT = path.resolve(new URL('../..', import.meta.url).pathname);
const JULIA_SCRIPT = `
using AutoAD, DataFrames, Statistics
using AutoAD.CaretAnomalyDetectors: caretadlearner_dict

vals = parse.(Float64, split(read(stdin, String)))
if length(vals) < 25
  exit()
end
votepercent = parse(Float64, ARGS[1])
mode = ARGS[2]
detector = ARGS[3]
X = DataFrame(auto = vals)

function detector_result(name)
  if startswith(name, "sk:")
    return fit_transform!(SKAnomalyDetector(name[4:end]), X)
  end
  if startswith(name, "caret:")
    return fit_transform!(CaretAnomalyDetector(name[7:end]), X)
  end
  error("Unsupported detector: " * name)
end

if detector != ""
  flags = Bool.(detector_result(detector))
elseif mode == "full"
  out = fit_transform!(AutoAnomalyDetection(Dict(:votepercent => votepercent)), X)
  flags = Bool.(out[!, names(out)[end]])
else
  cols = Vector{Vector{Int}}()
  for learner in ["EllipticEnvelope", "OneClassSVM", "LocalOutlierFactor"]
    push!(cols, fit_transform!(SKAnomalyDetector(learner), X))
  end
  for learner in ["cof", "iforest", "knn", "lof", "mcd", "pca", "sos", "svm"]
    push!(cols, fit_transform!(CaretAnomalyDetector(learner), X))
  end
  flags = [mean(col[i] for col in cols) >= votepercent for i in eachindex(vals)]
end
print(join(findall(flags), ","))
`;

function boundedVotepercent(value) {
  const n = Number(value ?? 0.5);
  return Number.isFinite(n) ? Math.min(1, Math.max(0.1, n)) : 0.5;
}

function detectorMode(value) {
  return value === 'full' ? 'full' : 'quick';
}

export const QUICK_DETECTORS = [
  { id: 'sk:LocalOutlierFactor', label: 'LocalOutlierFactor' },
  { id: 'sk:EllipticEnvelope', label: 'EllipticEnvelope' },
  { id: 'caret:pca', label: 'PCA' },
  { id: 'caret:mcd', label: 'MCD' }
];

function detectorId(value) {
  return QUICK_DETECTORS.some((d) => d.id === value) ? value : '';
}

function runJulia(values, { timeoutMs = 120000, votepercent = 0.5, mode = 'quick', detector = '' } = {}) {
  return new Promise((resolve) => {
    const child = spawn('julia', [`--project=${path.join(ROOT, 'AutoAD')}`, '-e', JULIA_SCRIPT, String(boundedVotepercent(votepercent)), detectorMode(mode), detectorId(detector)], { stdio: ['pipe', 'pipe', 'pipe'] });
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

export async function detectAnomalies(points, { runner = runJulia, votepercent = 0.5, mode = 'quick', detector = '' } = {}) {
  const selectedMode = detectorMode(mode);
  const selectedDetector = detectorId(detector);
  if (points.length < 25) return { indexes: [], mode: selectedMode, detector: selectedDetector, warnings: ['Need at least 25 points for AutoAD anomaly detection.'] };
  const result = await runner(points.map((p) => p.value), { votepercent: boundedVotepercent(votepercent), mode: selectedMode, detector: selectedDetector });
  if (result.code !== 0) return { indexes: [], mode: selectedMode, detector: selectedDetector, warnings: [`AutoAD unavailable: ${result.stderr || 'julia failed'}`] };
  const indexes = String(result.stdout || '').split(',').map((n) => Number(n) - 1).filter((n) => Number.isInteger(n) && n >= 0 && n < points.length);
  return { indexes, mode: selectedMode, detector: selectedDetector, warnings: [] };
}
