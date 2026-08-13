#!/usr/bin/env python3
"""Generate tables for the final report, using the MIRI Decentralized Training Simulator.

Produces three markdown tables:
  1. clocal_results.md         — min-cost hardware per C_local target at 100 Mbps
  2. bw_sensitivity_results.md — min-cost config at C_local = 10^25 across bandwidths
  3. latency_sensitivity_results.md — min-cost config at C_local = 10^25 across latencies

For each target (and each bandwidth/latency scenario), the script searches over:
  - All hardware configs in CONFIGS and CONFIGS_FP8
  - Three training modes: flat DiLoCo, hierarchical DiLoCo (groups of 4/8/16), PP-DiLoCo
  - Model sizes from 0.5B to 10 000B

and returns the minimum-cost configuration that meets the C_local target, subject to
the overtraining-ratio constraint (1x <= OT <= 100x).

Run from repo root:  python generate_paper_tables.py
"""
import math
from evasion_calculator import (
    CONFIGS, CONFIGS_FP8,
    compute_generic_scenario, compute_hierarchical_scenario, compute_pp_diloco_scenario,
    MFU, COMPRESSION, CHIP_TO_SERVER, SERVER_TO_CLUSTER, BYTES_PER_PARAM,
)

# ── Fixed assumptions ─────────────────────────────────────────────────────────
TIME_S = 740 * 86400
MODEL_SIZES = [
    0.5, 1, 2, 5, 10,
    25, 28, 30, 32, 35, 40, 45, 50, 60, 70, 78, 85, 91, 92, 100, 110, 120, 130, 140, 150,
    155, 160, 165, 170, 172, 175, 178, 180, 185, 195, 200, 210, 225, 235, 240, 245, 248,
    250, 252, 255, 258, 260, 265, 270, 275, 280, 290, 300, 310, 315, 320, 325, 330, 335,
    340, 350, 375, 400, 450, 500, 625, 750, 1000, 1500, 2500, 5000, 10000,
]
OT_MIN, OT_MAX = 1.0, 100.0
ALL_CFGS = list(CONFIGS.items()) + list(CONFIGS_FP8.items())

# ── Table 1: C_local targets at fixed network ────────────────────────────────
CLOCAL_TARGETS = [1e24, 3.3e24, 1e25, 2.1e25, 3.8e25, 6.6e25, 1e26]
BW_DEFAULT = 100e6
LATENCY_DEFAULT = 0.1

# ── Table 2: bandwidth sweep at fixed C_local ─────────────────────────────────
BW_SWEEP = [
    (10e6,   "10 Mbps"),
    (30e6,   "30 Mbps"),
    ((57e6, 310e6),  "US avg (310 / 57 Mbps)"),
    ((47e6, 207e6),  "China avg (207 / 47 Mbps)"),
    (100e6,  "100 Mbps"),
    (300e6,  "300 Mbps"),
    (1e9,    "1 Gbps"),
    (1000e9, "1000 Gbps"),
]
BW_SWEEP_LATENCY = {1000e9: 0.0001}  # datacenter latency for Tbps row
BW_SWEEP_TARGET = 1e25

# ── Table 3: latency sweep at fixed bandwidth & C_local ───────────────────────
LATENCY_SWEEP = [0.01, 0.03, 0.1, 0.3]
LATENCY_SWEEP_BW = 100e6
LATENCY_SWEEP_TARGET = 1e25


# ── Helpers ───────────────────────────────────────────────────────────────────
def max_single_node_b(cfg):
    return cfg['vram_gb'] / cfg.get('bytes_per_param', BYTES_PER_PARAM)

def get_h(r):
    """Different modes store the effective H under different keys."""
    return r.get('h_used', r.get('h_eff', r.get('h_min', 0)))

def fmt_mode(mode_label, r):
    """Flat → 'Flat'; hierarchical → 'Hier (nodes_per_group × n_groups)';
    PP → 'PP (pp_stages × n_groups)'."""
    if mode_label.startswith("Hier"):
        return f"Hier ({r.get('nodes_per_group',0)}x{r.get('n_groups',0)})"
    if mode_label.startswith("PP"):
        return f"PP ({r.get('pp_stages',0)}x{r.get('n_groups',0)})"
    return mode_label

def unpack_bw(bw):
    """bw is either a scalar (symmetric) or a (up, down) tuple (asymmetric)."""
    if isinstance(bw, tuple):
        return bw[0], bw[1]
    return bw, bw

def try_all_modes(cfg, n_nodes, model_b, bw, latency):
    """Enumerate flat, hierarchical (3 group sizes), and PP-DiLoCo candidates."""
    bw_up, bw_down = unpack_bw(bw)
    bw_peer = min(bw_up, bw_down)  # conservative: peer-to-peer rate limited by upload
    out = []
    if model_b <= max_single_node_b(cfg):
        r = compute_generic_scenario(
            cfg, n_nodes, time_seconds=TIME_S, target_params_b=model_b,
            bw_up_bps=bw_up, bw_down_bps=bw_down, latency_s=latency)
        if r:
            out.append(("Flat", r))
        for npg in [4, 8, 10, 12, 14, 16, 18, 20, 24, 32]:
            if n_nodes >= 2 * npg:
                r = compute_hierarchical_scenario(
                    None, n_nodes, nodes_per_group=npg, cfg=cfg,
                    time_seconds=TIME_S, target_params_b=model_b,
                    bw_up_bps=bw_up, bw_down_bps=bw_down, latency_s=latency,
                    regional_bw_bps=bw_peer, regional_latency_s=latency)
                if r and 'n_groups' in r:
                    out.append((f"Hier({npg})", r))
    else:
        r = compute_pp_diloco_scenario(
            None, n_nodes, target_params_b=model_b, cfg=cfg,
            time_seconds=TIME_S,
            bw_up_bps=bw_up, bw_down_bps=bw_down, latency_s=latency,
            pp_bw_bps=bw_peer, pp_latency_s=latency)
        if r:
            out.append((f"PP({r['pp_stages']}stg)", r))
    return out

def _cq(r):
    return r.get('c_quality', r['c_local'] * r['chi'])

def best_at_nodes(cfg, n_nodes, bw, latency, target=None):
    """At fixed (cfg, n_nodes), pick (mode, model).
    If target is given: among candidates with C_local >= target, maximize C_quality;
    if none meet target, fall back to max C_local (so binary search sees the feasibility
    frontier). If target is None: max C_local (used for binary-search mid-point probes)."""
    best_mode, best_r = None, None
    for pb in MODEL_SIZES:
        for mode, r in try_all_modes(cfg, n_nodes, pb, bw, latency):
            ot = r.get('overtraining_ratio', 0)
            if ot < OT_MIN or ot > OT_MAX:
                continue
            if best_r is None:
                best_r, best_mode = r, mode
                continue
            if target is None:
                if r['c_local'] > best_r['c_local']:
                    best_r, best_mode = r, mode
            else:
                curr_meets = r['c_local'] >= target
                best_meets = best_r['c_local'] >= target
                if curr_meets and not best_meets:
                    best_r, best_mode = r, mode
                elif curr_meets and best_meets:
                    if _cq(r) > _cq(best_r):
                        best_r, best_mode = r, mode
                elif (not curr_meets) and (not best_meets):
                    if r['c_local'] > best_r['c_local']:
                        best_r, best_mode = r, mode
    return best_mode, best_r

def find_min_nodes(cfg, target, bw, latency, max_n=100000):
    """Binary search for the smallest n_nodes meeting target C_local."""
    m1, r1 = best_at_nodes(cfg, 1, bw, latency, target=target)
    if r1 and r1['c_local'] >= target:
        return m1, r1
    _, r_hi = best_at_nodes(cfg, max_n, bw, latency)
    if not r_hi or r_hi['c_local'] < target:
        return None, None
    lo, hi = 2, max_n
    while lo < hi:
        mid = (lo + hi) // 2
        _, r = best_at_nodes(cfg, mid, bw, latency)
        if r and r['c_local'] >= target:
            hi = mid
        else:
            lo = mid + 1
    return best_at_nodes(cfg, lo, bw, latency, target=target)

def cheapest_config(target, bw, latency):
    """Over all hardware configs, return the one with lowest cost that meets target.
    On exact cost ties, prefer the config with higher C_quality."""
    best_name, best_mode, best_r = None, None, None
    for name, cfg in ALL_CFGS:
        m, r = find_min_nodes(cfg, target, bw, latency)
        if not r:
            continue
        if best_r is None or r['cost_usd'] < best_r['cost_usd']:
            best_r, best_mode, best_name = r, m, name
        elif r['cost_usd'] == best_r['cost_usd'] and _cq(r) > _cq(best_r):
            best_r, best_mode, best_name = r, m, name
    return best_name, best_mode, best_r

# ── Formatters ────────────────────────────────────────────────────────────────
def fmt_flop(f):
    exp = int(math.floor(math.log10(f))) if f > 0 else 0
    return f"{f/10**exp:.1f}e{exp}"

def fmt_chi(c):
    if c >= 0.001: return f"{c:.4f}"
    if c > 0: return f"{c:.1e}"
    return "~0"

def fmt_model(pb):
    return f"{pb:.1f}B" if pb < 1 else f"{pb:.0f}B"

def fmt_ot(ot):
    return f"{ot:,.0f}x" if ot >= 10 else f"{ot:.1f}x"

def fmt_cost(c):
    return f"${c/1e9:.2f}B" if c >= 1e9 else f"${c/1e6:.1f}M"

def fmt_target(t):
    exp = int(math.floor(math.log10(t)))
    coeff = t / 10**exp
    return f"10^{exp}" if abs(coeff - 1.0) < 0.01 else f"{coeff:.1f} x 10^{exp}"

def fmt_bw(bw_bps):
    if bw_bps >= 1e9:
        return f"{bw_bps/1e9:.0f} Gbps"
    return f"{bw_bps/1e6:.0f} Mbps"

def row_fields(r, mode):
    """Return the 10 per-result fields that appear in every markdown row."""
    cq = r.get('c_quality', r['c_local'] * r['chi'])
    return {
        'nodes':     f"{r['n_nodes']:,}",
        'mode':      fmt_mode(mode, r),
        'model':     fmt_model(r['params_b']),
        'h':         f"{get_h(r):.0f}",
        'eta':       f"{r['eta']:.4f}",
        'c_local':   fmt_flop(r['c_local']),
        'chi':       fmt_chi(r['chi']),
        'c_quality': fmt_flop(cq),
        'ot':        fmt_ot(r['overtraining_ratio']),
        'cost':      fmt_cost(r['cost_usd']),
    }

BASE_COLS = ("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |")
BASE_HDR = "|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|"


# ── Table 1: C_local targets ──────────────────────────────────────────────────
def table_clocal_targets():
    lines = [
        f"Minimum-cost hardware to reach C_local targets in 740 days, using actual "
        f"per-node VRAM. All network connections at {BW_DEFAULT/1e6:.0f} Mbps with "
        f"{LATENCY_DEFAULT*1000:.0f} ms latency. MFU = {MFU*100:.0f}%, "
        f"compression = {COMPRESSION}x, overtraining constrained to "
        f"{OT_MIN:.0f}x--{OT_MAX:.0f}x. Hardware cost applies Cottier et al. (2024) "
        f"multipliers: {CHIP_TO_SERVER}x chip-to-server, "
        f"{SERVER_TO_CLUSTER}x server-to-cluster.",
        "",
        "| Target | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |",
        BASE_HDR,
    ]
    for t in CLOCAL_TARGETS:
        name, mode, r = cheapest_config(t, BW_DEFAULT, LATENCY_DEFAULT)
        if r:
            f = row_fields(r, mode)
            lines.append(
                f"| {fmt_target(t)} | {name} | {f['nodes']} | {f['mode']} | {f['model']} "
                f"| {f['h']} | {f['eta']} | {f['c_local']} | {f['chi']} | {f['c_quality']} "
                f"| {f['ot']} | {f['cost']} |"
            )
        else:
            lines.append(f"| {fmt_target(t)} | --- infeasible --- | | | | | | | | | | |")
    return "\n".join(lines) + "\n"


# ── Table 2: bandwidth sweep ──────────────────────────────────────────────────
def table_bandwidth_sweep():
    header = [
        f"Bandwidth sensitivity: minimum-cost hardware to reach "
        f"{fmt_target(BW_SWEEP_TARGET)} FLOP C_local in 740 days, using actual "
        f"per-node VRAM. Symmetric rows set WAN, regional, and PP channels to the "
        f"listed bandwidth with 100 ms latency (0.1 ms at 1000 Gbps to model "
        f"intra-datacenter InfiniBand/NVLink conditions). Asymmetric rows (US avg, "
        f"China avg) use the listed down/up values for WAN; regional and PP channels "
        f"use the upload rate (peer-to-peer bottleneck). MFU = {MFU*100:.0f}%, "
        f"compression = {COMPRESSION}x, overtraining constrained to "
        f"{OT_MIN:.0f}x--{OT_MAX:.0f}x. Hardware cost applies Cottier et al. (2024) "
        f"multipliers: {CHIP_TO_SERVER}x chip-to-server, "
        f"{SERVER_TO_CLUSTER}x server-to-cluster. Rows sorted by cost, descending.",
        "",
        "| BW | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |",
        BASE_HDR,
    ]
    rows = []
    for bw, label in BW_SWEEP:
        latency = BW_SWEEP_LATENCY.get(bw, 0.1)
        name, mode, r = cheapest_config(BW_SWEEP_TARGET, bw, latency)
        if r:
            f = row_fields(r, mode)
            row_md = (
                f"| {label} | {name} | {f['nodes']} | {f['mode']} | {f['model']} "
                f"| {f['h']} | {f['eta']} | {f['c_local']} | {f['chi']} | {f['c_quality']} "
                f"| {f['ot']} | {f['cost']} |"
            )
            rows.append((r['cost_usd'], row_md))
    rows.sort(key=lambda x: -x[0])
    return "\n".join(header + [r for _, r in rows]) + "\n"


# ── Table 3: latency sweep ────────────────────────────────────────────────────
def table_latency_sweep():
    lines = [
        f"Latency sensitivity: minimum-cost hardware to reach "
        f"{fmt_target(LATENCY_SWEEP_TARGET)} FLOP C_local in 740 days, using actual "
        f"per-node VRAM. All connections at {LATENCY_SWEEP_BW/1e6:.0f} Mbps with "
        f"the listed one-way latency. MFU = {MFU*100:.0f}%, "
        f"compression = {COMPRESSION}x, overtraining constrained to "
        f"{OT_MIN:.0f}x--{OT_MAX:.0f}x. Hardware cost applies Cottier et al. (2024) "
        f"multipliers: {CHIP_TO_SERVER}x chip-to-server, "
        f"{SERVER_TO_CLUSTER}x server-to-cluster.",
        "",
        "| Latency | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |",
        BASE_HDR,
    ]
    for lat in LATENCY_SWEEP:
        name, mode, r = cheapest_config(LATENCY_SWEEP_TARGET, LATENCY_SWEEP_BW, lat)
        if r:
            f = row_fields(r, mode)
            lines.append(
                f"| {lat*1000:.0f} ms | {name} | {f['nodes']} | {f['mode']} | {f['model']} "
                f"| {f['h']} | {f['eta']} | {f['c_local']} | {f['chi']} | {f['c_quality']} "
                f"| {f['ot']} | {f['cost']} |"
            )
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────
OUTPUTS = [
    ("clocal_results.md",            table_clocal_targets),
    ("bw_sensitivity_results.md",    table_bandwidth_sweep),
    ("latency_sensitivity_results.md", table_latency_sweep),
]

if __name__ == "__main__":
    for filename, builder in OUTPUTS:
        md = builder()
        with open(filename, "w") as f:
            f.write(md)
        print(f"=== {filename} ===")
        print(md)
