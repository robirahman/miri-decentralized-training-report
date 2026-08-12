// Web side of the cross-language reliability-model parity harness.
// Driven by ../../verify_reliability_parity.py, which passes a cases file and
// diffs this output against the Python reliability_model() / straggler_factor().
//
// Run standalone (Node >= 22.6):
//   node --experimental-strip-types simulator-web/parity/parity.ts <cases.json>
//
// It imports the SAME reliabilityModel.ts the web app uses, so any drift
// between the Python and TypeScript implementations fails the harness.
import { readFileSync } from 'node:fs'
import { reliabilityModel, stragglerFactor } from '../src/reliabilityModel.ts'

// JSON has no Infinity/NaN; map non-finite numbers to sentinel strings so the
// Python side can compare them exactly (JSON.stringify would emit `null`).
const canon = (v: unknown): unknown => {
  if (typeof v === 'number' && !isFinite(v)) {
    return v > 0 ? 'inf' : v < 0 ? '-inf' : 'nan'
  }
  return v
}

const casesPath = process.argv[2]
if (!casesPath) {
  console.error('usage: parity.ts <cases.json>')
  process.exit(2)
}
const data = JSON.parse(readFileSync(casesPath, 'utf8'))

const reliability = data.reliability.map((c: Parameters<typeof reliabilityModel>[0]) => {
  const r = reliabilityModel(c) as Record<string, unknown>
  const out: Record<string, unknown> = {}
  for (const k of Object.keys(r)) out[k] = canon(r[k])
  return out
})

const straggler = data.straggler.map(
  (c: { n: number; mode: string; slowFraction: number; slowSeverity: number }) =>
    ({ f: canon(stragglerFactor(c.n, c.mode, c.slowFraction, c.slowSeverity)) })
)

process.stdout.write(JSON.stringify({ reliability, straggler }))
