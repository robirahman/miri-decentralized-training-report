// ── Hardware-failure & straggler-mitigation model ───────────────────────────
// Direct port of reliability_model() / straggler_factor() in evasion_calculator.py
// (kept in lock-step; the cross-language parity harness in parity/parity.ts +
// verify_reliability_parity.py asserts identical knob values). See
// Simulator_Documentation §5.

export const MITIGATION_STRATEGIES = ['none', 'synchronous', 'threshold', 'relay',
                                      'async', 'backup_workers', 'checkpoint_elastic'] as const
export const TAIL_COEF_BASE = 0.05 / (0.10 * (1 / 0.60 - 1))   // ~0.75
export const RELAY_TAIL_FACTOR = 0.40
export const BACKUP_TAIL_FACTOR = 0.30
export const LOCAL_CHECKPOINT_BW_BPS = 3e9 * 8
export const ASYNC_GOODPUT_FLOOR = 0.88
export const THRESHOLD_QUALITY_PENALTY = 0.85
export const ASYNC_STALENESS_PENALTY = 0.97

// Slowest-worker sync-tail multiplier for a synchronous all-reduce over n workers.
// Mirrors straggler_factor() / reliability_model()'s f_tail exactly, so the flat
// and hierarchical/PP paths agree across the Python and web simulators.
export function stragglerFactor(n: number, mode: string,
                                slowFraction: number, slowSeverity: number): number {
  if (n <= 1) return 1.0
  const base = TAIL_COEF_BASE * slowFraction * (1 / Math.max(1e-6, slowSeverity) - 1) * Math.log2(n)
  if (mode === 'threshold' || mode === 'async') return 1.0
  if (mode === 'relay') return 1 + RELAY_TAIL_FACTOR * base
  if (mode === 'backup_workers') return 1 + BACKUP_TAIL_FACTOR * base
  return 1 + base   // none / synchronous / checkpoint_elastic
}

export interface RelInput {
  nNodes: number; gpuCount: number; timeSeconds: number; params: number;
  bytesPerParam: number; vramGb: number; mode: string; tailN?: number;
  failureRate: number; recoveryTimeS: number; checkpointMode: string;
  slowFraction: number; slowSeverity: number; backupFraction: number;
  ckptReplicas?: number;
}

export function reliabilityModel(i: RelInput) {
  const g = Math.max(1, i.nNodes * i.gpuCount)
  const tH = i.timeSeconds / 3600
  const lam = Math.max(0, i.failureRate)
  const clusterRate = g * lam
  const recoveryH = i.recoveryTimeS / 3600
  const expectedFailures = clusterRate * tH
  const mtbfH = clusterRate > 0 ? 1 / clusterRate : Infinity
  const pDown = 1 - Math.exp(-lam * recoveryH)
  const ckptReplicas = i.ckptReplicas ?? 4

  // f_tail: slowest-worker sync tail
  const nt = i.tailN ?? i.nNodes
  let base = 0
  if (nt > 1) {
    const dcoef = TAIL_COEF_BASE * i.slowFraction * (1 / Math.max(1e-6, i.slowSeverity) - 1)
    base = dcoef * Math.log2(nt)
  }
  let fTail: number
  if (i.mode === 'threshold' || i.mode === 'async') fTail = 1.0
  else if (i.mode === 'relay') fTail = 1 + RELAY_TAIL_FACTOR * base
  else if (i.mode === 'backup_workers') fTail = 1 + BACKUP_TAIL_FACTOR * base
  else fTail = 1 + base   // none / synchronous / checkpoint_elastic

  // checkpoint cost (hours)
  const stateBytes = i.params * i.bytesPerParam
  const cRawH = (stateBytes * 8 / LOCAL_CHECKPOINT_BW_BPS) / 3600
  const freeVram = i.vramGb * 1e9 - stateBytes
  const gpuMemFeasible = freeVram >= ckptReplicas * (stateBytes / Math.max(1, i.nNodes))
  let cEffH: number
  if (i.checkpointMode === 'async') cEffH = cRawH * 0.02
  else if (i.checkpointMode === 'gpu_memory' && gpuMemFeasible) cEffH = cRawH * 0.005
  else cEffH = cRawH

  const youngDaly = (cH: number, fullRecovery: boolean): [number, number] => {
    if (clusterRate <= 0) return [0, Infinity]
    const tStar = cH > 0 ? Math.sqrt(2 * cH / clusterRate) : Infinity
    if (!isFinite(tStar)) return [clusterRate * recoveryH, tStar]
    const rec = fullRecovery ? recoveryH : 0
    return [cH / tStar + clusterRate * (tStar / 2 + rec), tStar]
  }

  // u: goodput
  let u: number, tStarH = Infinity
  if (i.mode === 'none') {
    u = 1 / (1 + clusterRate * tH / 2)
  } else if (i.mode === 'synchronous') {
    const [L, t] = youngDaly(cEffH, true); tStarH = t; u = Math.max(0, 1 - L)
  } else if (i.mode === 'checkpoint_elastic') {
    const [L, t] = youngDaly(cEffH, false); tStarH = t; u = Math.max(0, 1 - L - pDown)
  } else if (i.mode === 'backup_workers') {
    u = 1 - Math.max(0, pDown - i.backupFraction)
  } else if (i.mode === 'async') {
    const churn = 1 - Math.exp(-clusterRate * recoveryH)
    u = 1 - pDown - (1 - ASYNC_GOODPUT_FLOOR) * churn
  } else {   // relay / threshold
    u = 1 - pDown
  }

  const etaMit = i.mode === 'threshold' ? THRESHOLD_QUALITY_PENALTY
    : i.mode === 'async' ? ASYNC_STALENESS_PENALTY : 1.0
  const costMult = i.mode === 'backup_workers' ? 1 + i.backupFraction : 1.0

  return {
    mitigation: i.mode, fTail, u, etaMit, costMult, totalGpus: g,
    clusterFailuresPerDay: clusterRate * 24, mtbfHours: mtbfH,
    expectedFailures, pNodeDown: pDown, checkpointIntervalH: tStarH,
    gpuMemCheckpointFeasible: gpuMemFeasible,
    gpuHoursWasted: g * tH * (1 - u), timeInflation: u > 0 ? 1 / u : Infinity,
  }
}
