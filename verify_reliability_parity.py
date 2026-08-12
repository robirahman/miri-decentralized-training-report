"""Cross-language parity harness for the hardware-failure & straggler-mitigation
model.

The reliability model is implemented twice — once in evasion_calculator.py
(reliability_model / straggler_factor) and once in the web app
(simulator-web/src/reliabilityModel.ts). They must stay in lock-step. This
script builds an input grid, evaluates it with the Python implementation, runs
the *actual* TypeScript implementation via Node, and asserts the two agree to
within a tight relative tolerance.

Usage:
    python verify_reliability_parity.py

Requires Node >= 22.6 (for `--experimental-strip-types`). Exits 0 on parity,
1 on mismatch, 2 if Node is unavailable.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

from evasion_calculator import reliability_model, straggler_factor

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY_TS = os.path.join(HERE, "simulator-web", "parity", "parity.ts")

TOL = 1e-9                      # max allowed relative difference
TIME_S = 1.5 * 365.25 * 24 * 3600

# Python reliability_model key -> TypeScript reliabilityModel key. Diagnostics
# that only one side returns (checkpoint_cost_s, failure_rate) are not compared.
KEYMAP = {
    "f_tail": "fTail",
    "u": "u",
    "eta_mit": "etaMit",
    "cost_mult": "costMult",
    "total_gpus": "totalGpus",
    "cluster_failures_per_day": "clusterFailuresPerDay",
    "mtbf_hours": "mtbfHours",
    "expected_failures": "expectedFailures",
    "p_node_down": "pNodeDown",
    "checkpoint_interval_h": "checkpointIntervalH",
    "gpu_mem_checkpoint_feasible": "gpuMemCheckpointFeasible",
    "gpu_hours_wasted": "gpuHoursWasted",
    "time_inflation": "timeInflation",
}

STRATEGIES = ["none", "synchronous", "threshold", "relay",
              "async", "backup_workers", "checkpoint_elastic"]


def build_cases():
    """7 strategies x 3 node counts x 2 failure rates x 3 checkpoint modes
    x 2 model sizes = 252 reliability cases, plus a straggler-tail sweep."""
    reliability = []
    for mode in STRATEGIES:
        for n in (72, 500, 4000):
            for fr in (2e-5, 1e-4):
                for cm in ("sync", "async", "gpu_memory"):
                    for p in (144e9, 1000e9):   # 1000B: model exceeds VRAM -> gpu_mem infeasible
                        reliability.append({
                            "nNodes": n, "gpuCount": 50, "timeSeconds": TIME_S,
                            "params": p, "bytesPerParam": 16, "vramGb": 4000,
                            "mode": mode, "failureRate": fr, "recoveryTimeS": 600,
                            "checkpointMode": cm, "slowFraction": 0.10,
                            "slowSeverity": 0.60, "backupFraction": 0.05,
                        })
    # straggler_factor() reads SLOW_NODE_FRACTION/SEVERITY globals (defaults
    # 0.10/0.60); pass the same to the TS pure function.
    straggler = [{"n": n, "mode": mode, "slowFraction": 0.10, "slowSeverity": 0.60}
                 for mode in STRATEGIES
                 for n in (1, 2, 10, 50, 500, 4000)]
    return {"reliability": reliability, "straggler": straggler}


def canon(v):
    """Match parity.ts: non-finite numbers -> sentinel strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if math.isnan(v):
            return "nan"
    return v


def py_reliability(c):
    r = reliability_model(
        c["nNodes"], c["gpuCount"], c["timeSeconds"], c["params"],
        c["bytesPerParam"], c["vramGb"], mode=c["mode"],
        failure_rate=c["failureRate"], recovery_time_s=c["recoveryTimeS"],
        checkpoint_mode=c["checkpointMode"], slow_fraction=c["slowFraction"],
        slow_severity=c["slowSeverity"], backup_fraction=c["backupFraction"],
    )
    return r


def compare(label, py_val, ts_val, mismatches):
    """Compare one scalar (sentinel string or number) and record failures."""
    pc, tc = canon(py_val), canon(ts_val)
    if isinstance(pc, str) or isinstance(tc, str) or isinstance(pc, bool) or isinstance(tc, bool):
        if pc != tc:
            mismatches.append((label, pc, tc, None))
        return 0.0
    rel = abs(pc - tc) / max(1.0, abs(pc))
    if rel > TOL:
        mismatches.append((label, pc, tc, rel))
    return rel


def main():
    cases = build_cases()

    # Python side.
    py_reliability_out = [py_reliability(c) for c in cases["reliability"]]
    py_straggler_out = [straggler_factor(c["n"], c["mode"]) for c in cases["straggler"]]

    # TypeScript side (runs the actual web reliabilityModel.ts via Node).
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(cases, f)
        cases_path = f.name
    try:
        proc = subprocess.run(
            ["node", "--experimental-strip-types", PARITY_TS, cases_path],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        print("Node not found on PATH — cannot run the TypeScript side.", file=sys.stderr)
        return 2
    finally:
        os.unlink(cases_path)

    if proc.returncode != 0:
        print("Node failed to run parity.ts:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return 2
    ts = json.loads(proc.stdout)

    # Compare.
    mismatches = []
    max_rel = 0.0
    for idx, (c, pr, tr) in enumerate(zip(cases["reliability"], py_reliability_out, ts["reliability"])):
        ctx = f"reliability[{idx}] mode={c['mode']} n={c['nNodes']} fr={c['failureRate']} ckpt={c['checkpointMode']} p={c['params']:.0f}"
        if pr["mitigation"] != tr.get("mitigation"):
            mismatches.append((f"{ctx}.mitigation", pr["mitigation"], tr.get("mitigation"), None))
        for pk, tk in KEYMAP.items():
            max_rel = max(max_rel, compare(f"{ctx}.{pk}", pr[pk], tr.get(tk), mismatches))
    for idx, (c, pf, tf) in enumerate(zip(cases["straggler"], py_straggler_out, ts["straggler"])):
        ctx = f"straggler[{idx}] mode={c['mode']} n={c['n']}"
        max_rel = max(max_rel, compare(f"{ctx}.f", pf, tf["f"], mismatches))

    n_cases = len(cases["reliability"]) + len(cases["straggler"])
    if mismatches:
        print(f"PARITY FAILED: {len(mismatches)} mismatch(es) across {n_cases} cases "
              f"({len(cases['reliability'])} reliability + {len(cases['straggler'])} straggler).")
        for label, pv, tv, rel in mismatches[:20]:
            extra = f"  (rel={rel:.2e})" if rel is not None else ""
            print(f"  {label}: py={pv!r} ts={tv!r}{extra}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        return 1

    print(f"PARITY OK: Python and TypeScript agree on all {n_cases} cases "
          f"({len(cases['reliability'])} reliability + {len(cases['straggler'])} straggler); "
          f"max relative difference {max_rel:.2e} (tol {TOL:.0e}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
