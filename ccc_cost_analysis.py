"""
CCC Rule Cost Analysis: Lowest cost to reach C_local targets under current
and proposed (memory-amended) CCC rules.

Uses the evasion_calculator simulator formulas. Computes η as throughput
efficiency (η_H × η_compression), matching the documented simulator outputs.
The replica divergence penalty is reported separately via η_chinchilla.

Costs are full system acquisition costs: chip price × Cottier et al. (2024)
chip-to-server and server-to-cluster multipliers (2.02× combined), matching
evasion_calculator.py. See Simulator_Documentation.md §7.
"""
import math
import sys

# Ensure Unicode (box-drawing, ×, λ, …) prints even when stdout is redirected
# on a non-UTF-8 default console, e.g. Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, "/mnt/chromeos/MyFiles/Documents/MIRI decentralized training report")
from evasion_calculator import (
    straggler_factor, reliability_model, efficiency, alpha, compression_quality,
    replica_loss_multiplier, chinchilla_efficiency, chinchilla_loss,
    COMPRESSION, BYTES_PER_PARAM, BITS_PER_PSEUDO_GRAD,
    BW_UP_BPS, BW_DOWN_BPS, LATENCY_S, MFU, LOCAL_BATCH, TIME_SECONDS,
    CHINCHILLA_TOKENS_PER_PARAM, CHIP_TO_SERVER, SERVER_TO_CLUSTER,
)

# ── Hardware configurations ───────────────────────────────────────────────────
# All configs must be under 16 H100-equivalents (15,840 TFLOPS FP16)
# GPU prices: March 2026 market rates (web search)

HARDWARE = {
    # ── Pricing rationale (March 2026) ──────────────────────────────────────
    # A100 80GB: Abundant on secondary market as enterprises upgrade to
    #   Blackwell. Used SXM units: $5-9K (ALTA Technologies, eBay, Fluence).
    #   Mid-range used: $7K. New PCIe still $10-15K.
    # H100 SXM: Starting at $27K/GPU (IntuitionLabs, Jarvislabs). 8-GPU
    #   board ~$215K = $27K/GPU. Bulk discounts 10-15% for large orders.
    # H200 SXM: 8-GPU board $308-315K = $39K/GPU (IntuitionLabs, TRG).
    #   46% premium over H100 driven by HBM3e (141GB vs 80GB, 4.8 TB/s).
    # GH200: $35-45K per superchip (TheRegister, Vipera). Integrated Grace
    #   CPU + 144GB HBM3 + NVLink-C2C. Mid-range: $40K.
    # ─────────────────────────────────────────────────────────────────────
    "48x A100 80GB": {
        "pflops": 48 * 312e12 / 1e15,     # 14.976 PFLOPS FP16
        "vram_gb": 48 * 80,                # 3,840 GB
        "gpu_count": 48,
        "gpu_cost_usd": 7_000,             # Used SXM, mid-range secondary market
        "h100_equiv": 48 * 312 / 990,      # 15.13
        "bytes_per_param": BYTES_PER_PARAM, # 16 (FP16 mixed precision)
        "bits_per_pseudo_grad": BITS_PER_PSEUDO_GRAD,  # 16
    },
    "16x GH200": {
        "pflops": 15.84,                   # 16 × 990 TFLOPS
        "vram_gb": 16 * 144,               # 2,304 GB
        "gpu_count": 16,
        "gpu_cost_usd": 40_000,            # Superchip w/ Grace CPU + HBM3
        "h100_equiv": 16.0,
        "bytes_per_param": BYTES_PER_PARAM,
        "bits_per_pseudo_grad": BITS_PER_PSEUDO_GRAD,
    },
    "16x H200 SXM": {
        "pflops": 16 * 990e12 / 1e15,      # 15.84 PFLOPS FP16
        "vram_gb": 16 * 141,               # 2,256 GB
        "gpu_count": 16,
        "gpu_cost_usd": 39_000,            # 8-GPU board $315K ÷ 8; HBM3e premium
        "h100_equiv": 16.0,
        "bytes_per_param": BYTES_PER_PARAM,
        "bits_per_pseudo_grad": BITS_PER_PSEUDO_GRAD,
    },
    "16x H100 SXM (FP16)": {
        "pflops": 15.84,                   # 16 × 990 TFLOPS FP16
        "vram_gb": 16 * 80,                # 1,280 GB
        "gpu_count": 16,
        "gpu_cost_usd": 27_000,            # Starting SXM price ($27K, IntuitionLabs)
        "h100_equiv": 16.0,
        "bytes_per_param": BYTES_PER_PARAM,
        "bits_per_pseudo_grad": BITS_PER_PSEUDO_GRAD,
    },
    "16x H100 SXM (FP8)": {
        "pflops": 16 * 1980e12 / 1e15,     # 31.68 PFLOPS FP8
        "vram_gb": 16 * 80,                # 1,280 GB
        "gpu_count": 16,
        "gpu_cost_usd": 27_000,            # Same hardware as FP16, software mode
        "h100_equiv": 16.0,                # CCC threshold uses FP16
        "bytes_per_param": 14,             # FP8: 1+1+4+4+4 bytes
        "bits_per_pseudo_grad": 8,         # FP8 pseudo-gradients
    },
    "9x H200 SXM": {
        "pflops": 9 * 990e12 / 1e15,       # 8.91 PFLOPS FP16
        "vram_gb": 9 * 141,                # 1,269 GB
        "gpu_count": 9,
        "gpu_cost_usd": 39_000,            # Same per-GPU as 16x H200
        "h100_equiv": 9.0,
        "bytes_per_param": BYTES_PER_PARAM,
        "bits_per_pseudo_grad": BITS_PER_PSEUDO_GRAD,
    },
    "16x A100 80GB": {
        "pflops": 16 * 312e12 / 1e15,      # 4.992 PFLOPS FP16
        "vram_gb": 16 * 80,                # 1,280 GB
        "gpu_count": 16,
        "gpu_cost_usd": 7_000,             # Used SXM, same per-GPU as 48x config
        "h100_equiv": 16 * 312 / 990,      # 5.04
        "bytes_per_param": BYTES_PER_PARAM,
        "bits_per_pseudo_grad": BITS_PER_PSEUDO_GRAD,
    },
}

# ── Rule definitions ──────────────────────────────────────────────────────────

def passes_current_rule(cfg):
    """Current CCC: ≤16 H100-eq in FLOP/s only."""
    return cfg["h100_equiv"] <= 16.01

def passes_amended_rule(cfg):
    """Amended CCC: ≤16 H100-eq in FLOP/s AND ≤1,280 GB accelerator memory."""
    return cfg["h100_equiv"] <= 16.01 and cfg["vram_gb"] <= 1280

CURRENT_CONFIGS = {k: v for k, v in HARDWARE.items() if passes_current_rule(v)}
AMENDED_CONFIGS = {k: v for k, v in HARDWARE.items() if passes_amended_rule(v)}

# ── Simulator core ────────────────────────────────────────────────────────────

def compute(cfg, n_nodes, compression=COMPRESSION, scenario="expected"):
    """Compute C_local for a config and node count.
    η = η_H × η_compression (throughput efficiency, matching documented results).
    Replica penalty reported separately via η_chinchilla."""
    bpp = cfg["bytes_per_param"]
    bpg = cfg["bits_per_pseudo_grad"]
    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    max_params_b = vram_gb / bpp
    params_b = max_params_b
    params = params_b * 1e9

    effective_flops = pflops * 1e15 * MFU
    t_comp = (6 * params * LOCAL_BATCH) / effective_flops

    v_bits = params * bpg / compression
    t_sync_base = v_bits / BW_UP_BPS + v_bits / BW_DOWN_BPS + LATENCY_S
    rel = reliability_model(n_nodes, cfg["gpu_count"], TIME_SECONDS, params,
                            bpp, vram_gb)
    f_n = rel["f_tail"]
    t_sync = t_sync_base * f_n

    if n_nodes == 1:
        h_min = 1
        eta = 1.0
    else:
        h_min = max(1, math.ceil(t_sync / t_comp))
        eta = efficiency(h_min, params_b, compression_ratio=compression,
                         scenario=scenario)
    eta *= rel["eta_mit"]

    c_actual = n_nodes * effective_flops * TIME_SECONDS * rel["u"]
    c_local = c_actual * eta

    # Training details
    total_tokens = c_actual / (6 * params)
    chinchilla_tokens = CHINCHILLA_TOKENS_PER_PARAM * params
    overtraining_ratio = total_tokens / chinchilla_tokens

    # Chinchilla efficiency (overtraining penalty only, no replica)
    if n_nodes > 1:
        eta_chin = chinchilla_efficiency(params, total_tokens, c_actual,
                                         loss_multiplier=1.0)
    else:
        eta_chin = 1.0
    c_quality = c_local * eta_chin

    # System acquisition cost: chips × Cottier et al. (2024) server/cluster
    # multipliers × any mitigation redundancy (matches evasion_calculator.py)
    cost_usd = (n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]
                * CHIP_TO_SERVER * SERVER_TO_CLUSTER * rel["cost_mult"])

    return {
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "params_b": params_b,
        "h_min": h_min,
        "f_straggler": f_n,
        "mitigation": rel["mitigation"],
        "goodput": rel["u"],
        "cost_mult": rel["cost_mult"],
        "gpu_hours_wasted": rel["gpu_hours_wasted"],
        "mtbf_hours": rel["mtbf_hours"],
        "eta": eta,
        "c_actual": c_actual,
        "c_local": c_local,
        "eta_chinchilla": eta_chin,
        "c_quality": c_quality,
        "overtraining_ratio": overtraining_ratio,
        "total_tokens_T": total_tokens / 1e12,
        "cost_usd": cost_usd,
    }


def find_min_nodes(cfg, target_c_local):
    """Binary search for minimum nodes to reach target C_local."""
    r = compute(cfg, 1)
    if r["c_local"] >= target_c_local:
        return r

    lo, hi = 2, 200_000
    r_hi = compute(cfg, hi)
    if r_hi["c_local"] < target_c_local:
        return None

    while lo < hi:
        mid = (lo + hi) // 2
        r = compute(cfg, mid)
        if r["c_local"] >= target_c_local:
            hi = mid
        else:
            lo = mid + 1

    return compute(cfg, lo)


# ── Validation against current simulator results ──────────────────────────────
def validate():
    """Self-consistency check against current evasion_calculator outputs.
    Recalibrated to the asymmetric-bandwidth / unified-efficiency / reliability-model
    engine (default COMPRESSION=150, error feedback, default 'relay' mitigation with
    goodput u=1.0 at the default failure rate). The previous reference constants
    (eta=0.858, h_min=200) came from a pre-refactor simulator_output.txt and are obsolete."""
    # 48x A100 80GB, N=72: eta=0.9214, C_local=1.88e25, h_min=19, goodput=1.0
    r = compute(HARDWARE["48x A100 80GB"], 72)
    assert abs(r["eta"] - 0.9214) < 0.002, f"48x A100 N=72: eta={r['eta']:.4f}, expected ~0.9214"
    assert abs(r["c_local"] / 1.881e25 - 1.0) < 0.02, f"C_local={r['c_local']:.3e}, expected ~1.88e25"
    assert r["h_min"] == 19, f"h_min={r['h_min']}, expected 19"
    assert abs(r["goodput"] - 1.0) < 0.001, f"goodput={r['goodput']:.4f}, expected ~1.0 (relay, default lambda)"

    # 16x H100 SXM (FP16), N=72: eta=0.9154, C_local=1.98e25, h_min=20
    r2 = compute(HARDWARE["16x H100 SXM (FP16)"], 72)
    assert abs(r2["eta"] - 0.9154) < 0.006, f"16x H100 N=72: eta={r2['eta']:.4f}, expected ~0.9154"
    assert r2["h_min"] == 20, f"h_min={r2['h_min']}, expected 20"
    print("  Validation passed: results match current simulator output.")

validate()


# ── Analysis ──────────────────────────────────────────────────────────────────

TARGETS = [1e24, 1e25, 1e26]
TARGET_LABELS = ["10^24", "10^25", "10^26"]

def format_cost(usd):
    if usd >= 1e9:
        return f"${usd/1e9:.2f}B"
    elif usd >= 1e6:
        return f"${usd/1e6:.1f}M"
    else:
        return f"${usd/1e3:.0f}K"

def print_rule_analysis(rule_name, configs):
    print(f"\n{'='*130}")
    print(f"  {rule_name}")
    print(f"  Training window: 1.5 years | WAN: 100 Mbps, 100 ms RTT | MFU: 40% | Compression: 16x | Streaming DiLoCo")
    print(f"{'='*130}")

    # Config summary
    for name, cfg in configs.items():
        bpp = cfg["bytes_per_param"]
        max_model = cfg["vram_gb"] / bpp
        precision = "FP8" if bpp < 16 else "FP16"
        print(f"  {name:>25}: {cfg['pflops']:>6.2f} PFLOPS ({precision}), {cfg['vram_gb']:>5,} GB, "
              f"{cfg['h100_equiv']:>5.1f} H100-eq, ${cfg['gpu_cost_usd']:>6,}/GPU, max model {max_model:>5.0f}B")

    print(f"\n  Costs below are full system acquisition cost: chip price x {CHIP_TO_SERVER} "
          f"(server) x {SERVER_TO_CLUSTER} (cluster) = {CHIP_TO_SERVER * SERVER_TO_CLUSTER:.2f}x "
          f"chip cost, per Cottier et al. (2024). The $/GPU column is system cost per GPU.")

    all_results = {}  # target -> [(name, result)]

    for target, label in zip(TARGETS, TARGET_LABELS):
        print(f"\n  {'─'*125}")
        print(f"  Target: C_local ≥ {label} FLOP")
        print(f"  {'─'*125}")
        print(f"  {'Config':>25} | {'Nodes':>5} | {'GPUs':>7} | {'Cost':>10} | "
              f"{'$/GPU':>7} | {'Model':>6} | {'H':>5} | {'η':>5} | "
              f"{'C_local':>10} | {'OT':>6} | {'η_chin':>6} | {'C_quality':>10}")
        print(f"  {'-'*125}")

        results = []
        for name, cfg in configs.items():
            r = find_min_nodes(cfg, target)
            if r is None:
                print(f"  {name:>25} | {'N/A — exceeds 200K nodes':>80}")
                continue
            results.append((name, r))
            cost_str = format_cost(r['cost_usd'])
            gpu_cost = r['cost_usd'] / r['total_gpus'] if r['total_gpus'] > 0 else 0
            print(f"  {name:>25} | {r['n_nodes']:>5} | {r['total_gpus']:>7,} | {cost_str:>10} | "
                  f"${gpu_cost:>5,.0f} | {r['params_b']:>5.0f}B | {r['h_min']:>5} | "
                  f"{r['eta']:>5.3f} | {r['c_local']:>10.2e} | {r['overtraining_ratio']:>5.1f}x | "
                  f"{r['eta_chinchilla']:>6.3f} | {r['c_quality']:>10.2e}")

        if results:
            best_name, best_r = min(results, key=lambda x: x[1]["cost_usd"])
            print(f"\n  >>> LOWEST COST: {best_name} — {best_r['n_nodes']} nodes, "
                  f"{best_r['total_gpus']:,} GPUs, {format_cost(best_r['cost_usd'])}")
            all_results[label] = (best_name, best_r)

    return all_results


# ── Header ────────────────────────────────────────────────────────────────────
print("=" * 130)
print("  MINIMUM COST TO REACH C_local COMPUTE TARGETS BELOW CCC THRESHOLD")
print("  Current vs. Proposed Memory-Amended CCC Definition")
print("=" * 130)
print()
print("  GPU Prices (March 2026 market rates):")
print("  ┌─────────────────┬──────────┬────────────────────────────────────────────────────────────┐")
print("  │ Accelerator      │ $/unit   │ Source / rationale                                        │")
print("  ├─────────────────┼──────────┼────────────────────────────────────────────────────────────┤")
print("  │ A100 80GB SXM   │  $7,000  │ Used, mid-range secondary market (ALTA, Fluence, eBay)    │")
print("  │ H100 SXM 80GB   │ $27,000  │ Starting SXM price; 8-GPU board ~$215K (IntuitionLabs)    │")
print("  │ H200 SXM 141GB  │ $39,000  │ 8-GPU board $315K ÷ 8; HBM3e premium (IntuitionLabs,TRG) │")
print("  │ GH200 144GB     │ $40,000  │ $35-45K superchip (TheRegister, Vipera); Grace CPU+HBM3   │")
print("  └─────────────────┴──────────┴────────────────────────────────────────────────────────────┘")
print()
print("  CCC Threshold: 16 H100-equivalents = 15,840 TFLOPS FP16")
print("  Proposed amendment adds: ≤1,280 GB aggregate accelerator memory (= 16 × H100 80GB)")

current_results = print_rule_analysis(
    "SCENARIO A — CURRENT CCC RULE: Compute only (≤16 H100-eq in FLOP/s)",
    CURRENT_CONFIGS,
)

amended_results = print_rule_analysis(
    "SCENARIO B — PROPOSED AMENDED RULE: Compute (≤16 H100-eq) AND Memory (≤1,280 GB)",
    AMENDED_CONFIGS,
)

# ── Side-by-side summary ─────────────────────────────────────────────────────
print(f"\n\n{'='*130}")
print("  SUMMARY: Lowest-cost path to each C_local target")
print(f"{'='*130}")
print(f"\n  {'Target':>8} │ {'Current Rule (compute only)':>50} │ {'Amended Rule (+memory cap)':>50} │ {'Δ Cost':>8}")
print(f"  {'─'*8}─┼─{'─'*50}─┼─{'─'*50}─┼─{'─'*8}")

for label in TARGET_LABELS:
    cr = current_results.get(label)
    ar = amended_results.get(label)
    if cr and ar:
        c_name, c_r = cr
        a_name, a_r = ar
        c_str = f"{c_name}: {c_r['n_nodes']}N × {c_r['total_gpus']//c_r['n_nodes']}GPU = {format_cost(c_r['cost_usd'])}"
        a_str = f"{a_name}: {a_r['n_nodes']}N × {a_r['total_gpus']//a_r['n_nodes']}GPU = {format_cost(a_r['cost_usd'])}"
        ratio = a_r['cost_usd'] / c_r['cost_usd'] if c_r['cost_usd'] > 0 else float('inf')
        print(f"  {label:>8} │ {c_str:>50} │ {a_str:>50} │ {ratio:>7.2f}x")


# ── Key hardware details ─────────────────────────────────────────────────────
print(f"\n\n{'='*130}")
print("  KEY: Why certain configs dominate")
print(f"{'='*130}")
print("""
  Under the CURRENT rule (compute-only threshold):
    - 48× A100 80GB node: 15.1 H100-eq, 3,840 GB VRAM → fits 240B model (FP16)
      Cheapest GPUs ($7K each used) + huge VRAM = lowest cost for large models
    - 16× H100 FP8 node: 16.0 H100-eq, 1,280 GB VRAM → fits 91B model (FP8)
      2× compute throughput via FP8, but 3.9× more expensive per GPU ($27K)

  Under the AMENDED rule (compute + 1,280 GB memory cap):
    - 48× A100 node is EXCLUDED (3,840 GB >> 1,280 GB limit)
    - 16× GH200 EXCLUDED (2,304 GB), 16× H200 EXCLUDED (2,256 GB)
    - Best remaining: 16× H100 SXM (exactly at both limits: 16.0 H100-eq, 1,280 GB)
    - 16× H100 FP8 uses same hardware, with 2× compute throughput

  The memory amendment forces evaders from cheap, high-memory A100 clusters
  to expensive H100 nodes, increasing the per-GPU cost from $7K to $27K (3.9×).
  It also reduces maximum model size from 240B to 80-91B parameters.
""")
