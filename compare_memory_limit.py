"""
Minimum-cost evasion analysis: before vs. after the 1,280 GB memory limit.

Finds the cheapest hardware configuration to achieve 10^24, 10^25, and 10^26
quality-adjusted FLOP (C_quality) using sub-CCC nodes over WAN, comparing:
  - Before: CCC defined by compute threshold only (>16 H100-equivalents)
  - After:  CCC defined by compute OR memory threshold (>1,280 GB accelerator memory)

Uses early-2026 GPU prices and the evasion_calculator infrastructure.
"""

import math
import sys
import io

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from evasion_calculator import (
    find_optimal_config_for_target,
    COMPRESSION, BW_BPS, LATENCY_S,
)

# ── GPU hardware definitions (early 2026 prices) ────────────────────────────

GPUS = {
    "A100 80GB": {
        "hbm_gb": 80,
        "fp16_tflops": 312,
        "fp8_tflops": None,       # No FP8 support (Ampere)
        "price_usd": 7_000,       # Used/secondary market
    },
    "H100 SXM": {
        "hbm_gb": 80,
        "fp16_tflops": 989,
        "fp8_tflops": 1_979,
        "price_usd": 25_000,
    },
    "GH200": {
        "hbm_gb": 144,
        "fp16_tflops": 989,
        "fp8_tflops": 1_979,
        "price_usd": 28_000,
    },
    "B100": {
        "hbm_gb": 192,
        "fp16_tflops": 1_750,
        "fp8_tflops": 3_500,
        "price_usd": 32_000,
    },
    "B200": {
        "hbm_gb": 192,
        "fp16_tflops": 2_250,
        "fp8_tflops": 4_500,
        "price_usd": 47_000,
    },
    # Chinese chips (only available domestically in China; not export-available)
    "Ascend 910B": {
        "hbm_gb": 64,          # 64 GB HBM2e (published)
        "fp16_tflops": 320,    # published
        "fp8_tflops": 640,     # published
        "price_usd": 16_000,   # ¥110,000 (~$16k)
    },
    "Ascend 910C": {
        "hbm_gb": 128,         # est. from benchmarks
        "fp16_tflops": 600,    # est. from benchmarks
        "fp8_tflops": 1_200,   # est.
        "price_usd": 26_000,   # ¥180,000 (~$26k)
    },
    # Google TPUs (BF16 TFLOPS treated as FP16-equivalent;
    # prices are capital equivalents derived from cloud rental rates)
    "TPU v4": {
        "hbm_gb": 32,          # 32 GB HBM2e (published)
        "fp16_tflops": 275,    # BF16 peak (published)
        "fp8_tflops": None,    # no FP8 support
        "price_usd": 12_000,   # est. from cloud rates
    },
    "TPU v5e": {
        "hbm_gb": 16,          # 16 GB HBM (published)
        "fp16_tflops": 197,    # BF16 peak (published)
        "fp8_tflops": None,    # INT8 only, no true FP8
        "price_usd": 6_000,    # est. from cloud rates
    },
    "TPU v5p": {
        "hbm_gb": 95,          # 95 GB HBM2e (published)
        "fp16_tflops": 459,    # BF16 peak (published)
        "fp8_tflops": None,    # INT8 only, no true FP8
        "price_usd": 20_000,   # est. from cloud rates
    },
    "TPU v6e": {
        "hbm_gb": 32,          # 32 GB HBM (published)
        "fp16_tflops": 918,    # BF16 peak (published)
        "fp8_tflops": 1_836,   # FP8 support (published)
        "price_usd": 25_000,   # est. from cloud rates
    },
}

# ── Thresholds ───────────────────────────────────────────────────────────────

CCC_COMPUTE_TFLOPS = 15_840   # >16 H100-equivalents (16 × 990 TFLOPS FP16)
H100_FP16_TFLOPS = 990        # Reference for H100-equivalent calculation
MEMORY_LIMIT_GB = 1_280       # Proposed amendment: >1,280 GB

TARGETS = [1e24, 1e25, 1e26]
TARGET_LABELS = ["10^24", "10^25", "10^26"]


# ── Node configuration builder ───────────────────────────────────────────────

def max_gpus_compute(gpu):
    """Max GPUs under CCC compute threshold (strict inequality: must be ≤ 15,840 TFLOPS)."""
    return math.floor(CCC_COMPUTE_TFLOPS / gpu["fp16_tflops"])


def max_gpus_memory(gpu):
    """Max GPUs with aggregate memory ≤ 1,280 GB (strict inequality: must be ≤ 1,280)."""
    return math.floor(MEMORY_LIMIT_GB / gpu["hbm_gb"])


def build_node_cfg(gpu_name, gpu, n_gpus):
    """Build a cfg dict compatible with evasion_calculator functions.
    Uses FP8 training if supported, FP16 otherwise."""
    fp8 = gpu["fp8_tflops"] is not None
    tflops_per_gpu = gpu["fp8_tflops"] if fp8 else gpu["fp16_tflops"]
    cfg = {
        "gpu_count": n_gpus,
        "pflops": n_gpus * tflops_per_gpu * 1e12 / 1e15,
        "pflops_fp16": n_gpus * gpu["fp16_tflops"] * 1e12 / 1e15,
        "vram_gb": n_gpus * gpu["hbm_gb"],
        "gpu_cost_usd": gpu["price_usd"],
        "h100_equiv": n_gpus * gpu["fp16_tflops"] / H100_FP16_TFLOPS,
    }
    if fp8:
        cfg["bytes_per_param"] = 14
        cfg["bits_per_pseudo_grad"] = 8
    return cfg


def build_configs_for_regime(with_memory_limit):
    """Build node configs for all GPUs under the given regime.
    Returns list of (gpu_name, cfg, max_model_b) tuples."""
    configs = []
    for name, gpu in GPUS.items():
        n_compute = max_gpus_compute(gpu)
        if with_memory_limit:
            n_memory = max_gpus_memory(gpu)
            n_gpus = min(n_compute, n_memory)
        else:
            n_gpus = n_compute

        if n_gpus < 1:
            continue

        cfg = build_node_cfg(name, gpu, n_gpus)
        bytes_per_param = cfg.get("bytes_per_param", 16)
        max_model_b = cfg["vram_gb"] / bytes_per_param
        configs.append((name, cfg, n_gpus, max_model_b))

    return configs


# ── Main analysis ────────────────────────────────────────────────────────────

def find_cheapest(configs, target):
    """Find the cheapest config to reach target C_quality.
    Returns (gpu_name, result_dict) or (None, None) if infeasible."""
    best_name = None
    best_result = None

    for gpu_name, cfg, n_gpus, max_model_b in configs:
        result = find_optimal_config_for_target(cfg, target)
        if result is not None:
            if best_result is None or result["cost_usd"] < best_result["cost_usd"]:
                best_name = gpu_name
                best_result = result

    return best_name, best_result


def format_cost(cost_usd):
    if cost_usd >= 1e9:
        return f"${cost_usd/1e9:.2f}B"
    elif cost_usd >= 1e6:
        return f"${cost_usd/1e6:.1f}M"
    elif cost_usd >= 1e3:
        return f"${cost_usd/1e3:.0f}K"
    else:
        return f"${cost_usd:.0f}"


def format_mode(r):
    if "mode" in r and "PP" in r.get("mode", ""):
        return r["mode"]
    elif "mode" in r:
        return r["mode"]
    elif "n_groups" in r and "nodes_per_group" in r:
        return f"Hier {r['nodes_per_group']}x{r['n_groups']}"
    else:
        return "Flat DiLoCo"


def main():
    print("=" * 110)
    print("MINIMUM-COST EVASION ANALYSIS: BEFORE vs. AFTER 1,280 GB MEMORY LIMIT")
    print("=" * 110)

    # ── Table 0: Node configurations ──────────────────────────────────────
    print("\n── Sub-CCC Node Configurations ─────────────────────────────────────")
    for regime_label, with_mem in [("BEFORE memory limit (compute threshold only)",  False),
                                    ("AFTER memory limit (compute + memory ≤ 1,280 GB)", True)]:
        print(f"\n  {regime_label}:")
        print(f"  {'GPU':>12} | {'GPUs':>4} | {'VRAM':>8} | {'H100-eq':>7} | {'PFLOPS':>8} | "
              f"{'Precision':>9} | {'Max Model':>9} | {'Node Cost':>10}")
        print("  " + "-" * 90)

        configs = build_configs_for_regime(with_mem)
        for gpu_name, cfg, n_gpus, max_model_b in configs:
            prec = "FP8" if "bytes_per_param" in cfg and cfg["bytes_per_param"] == 14 else "FP16"
            node_cost = n_gpus * GPUS[gpu_name]["price_usd"]
            print(f"  {gpu_name:>12} | {n_gpus:>4} | {cfg['vram_gb']:>5} GB | "
                  f"{cfg['h100_equiv']:>7.1f} | {cfg['pflops']:>6.2f}  | "
                  f"{prec:>9} | {max_model_b:>6.0f}B  | {format_cost(node_cost):>10}")

    # ── Table 1: Cheapest config per (target × regime) ────────────────────
    print("\n\n── Minimum Cost to Achieve C_quality Target ────────────────────────")
    print(f"\n  {'Target':>7} | {'Regime':>8} | {'GPU':>12} | {'Mode':>20} | {'Nodes':>6} | "
          f"{'Cost':>9} | {'Model':>6} | {'OT':>5} | {'eta':>5} | {'C_quality':>10}")
    print("  " + "-" * 115)

    results = {}  # (target_idx, regime) -> (gpu_name, result)

    for i, target in enumerate(TARGETS):
        for regime_label, with_mem, regime_key in [
            ("Before", False, "before"),
            ("After",  True,  "after"),
        ]:
            configs = build_configs_for_regime(with_mem)
            gpu_name, r = find_cheapest(configs, target)

            results[(i, regime_key)] = (gpu_name, r)

            if r is None:
                print(f"  {TARGET_LABELS[i]:>7} | {regime_label:>8} | {'N/A':>12} | "
                      f"{'Infeasible':>20} | {'':>6} | {'':>9} | {'':>6} | {'':>5} | {'':>5} | {'':>10}")
            else:
                mode = format_mode(r)
                params_b = r.get("params_b", r.get("max_params_b", 0))
                ot = r.get("overtraining_ratio", 0)
                ot_str = f"{ot:.1f}x" if ot < 100 else f"{ot:.0f}x"
                eta = r.get("eta", 0)
                c_q = r.get("c_quality", 0)
                print(f"  {TARGET_LABELS[i]:>7} | {regime_label:>8} | {gpu_name:>12} | "
                      f"{mode:>20} | {r['n_nodes']:>6,} | {format_cost(r['cost_usd']):>9} | "
                      f"{params_b:>4.0f}B | {ot_str:>5} | {eta:>5.3f} | {c_q:>10.2e}")

        print("  " + "-" * 115)

    # ── Table 2: All GPU options per (target × regime) ────────────────────
    print("\n\n── All GPU Options (sorted by cost) ────────────────────────────────")

    for i, target in enumerate(TARGETS):
        for regime_label, with_mem in [("Before memory limit", False),
                                        ("After memory limit",  True)]:
            print(f"\n  Target: {TARGET_LABELS[i]} C_quality — {regime_label}")
            print(f"  {'GPU':>12} | {'Mode':>20} | {'Nodes':>6} | {'GPUs':>7} | "
                  f"{'Cost':>9} | {'Model':>6} | {'OT':>5} | {'eta':>5} | {'C_quality':>10}")
            print("  " + "-" * 105)

            configs = build_configs_for_regime(with_mem)
            all_results = []
            for gpu_name, cfg, n_gpus, max_model_b in configs:
                r = find_optimal_config_for_target(cfg, target)
                if r is not None:
                    all_results.append((gpu_name, r))

            all_results.sort(key=lambda x: x[1]["cost_usd"])

            for gpu_name, r in all_results:
                mode = format_mode(r)
                params_b = r.get("params_b", r.get("max_params_b", 0))
                ot = r.get("overtraining_ratio", 0)
                ot_str = f"{ot:.1f}x" if ot < 100 else f"{ot:.0f}x"
                eta = r.get("eta", 0)
                c_q = r.get("c_quality", 0)
                total_gpus = r.get("total_gpus", r["n_nodes"] * cfg.get("gpu_count", 0))
                print(f"  {gpu_name:>12} | {mode:>20} | {r['n_nodes']:>6,} | "
                      f"{total_gpus:>7,} | {format_cost(r['cost_usd']):>9} | "
                      f"{params_b:>4.0f}B | {ot_str:>5} | {eta:>5.3f} | {c_q:>10.2e}")

            if not all_results:
                print(f"  {'':>12} | {'No feasible config':>20}")

    # ── Table 3: Impact summary ───────────────────────────────────────────
    print("\n\n── Impact Summary ──────────────────────────────────────────────────")
    print(f"\n  {'Target':>7} | {'Cost Before':>11} | {'Cost After':>11} | "
          f"{'Ratio':>6} | {'GPU Before':>12} | {'GPU After':>12} | "
          f"{'Model Before':>12} | {'Model After':>12}")
    print("  " + "-" * 105)

    for i, target in enumerate(TARGETS):
        gn_b, r_b = results[(i, "before")]
        gn_a, r_a = results[(i, "after")]

        if r_b is not None and r_a is not None:
            ratio = r_a["cost_usd"] / r_b["cost_usd"]
            pb_b = r_b.get("params_b", r_b.get("max_params_b", 0))
            pb_a = r_a.get("params_b", r_a.get("max_params_b", 0))
            print(f"  {TARGET_LABELS[i]:>7} | {format_cost(r_b['cost_usd']):>11} | "
                  f"{format_cost(r_a['cost_usd']):>11} | {ratio:>5.2f}x | "
                  f"{gn_b:>12} | {gn_a:>12} | {pb_b:>9.0f}B   | {pb_a:>9.0f}B")
        else:
            print(f"  {TARGET_LABELS[i]:>7} | {'N/A' if r_b is None else format_cost(r_b['cost_usd']):>11} | "
                  f"{'N/A' if r_a is None else format_cost(r_a['cost_usd']):>11} | "
                  f"{'N/A':>6} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")

    # ── Note on newly caught configurations ───────────────────────────────
    print("\n\n── Configurations Newly Caught by 1,280 GB Memory Threshold ────────")
    for name, gpu in GPUS.items():
        n_before = max_gpus_compute(gpu)
        n_after = min(n_before, max_gpus_memory(gpu))
        if n_after < n_before:
            vram_before = n_before * gpu["hbm_gb"]
            vram_after = n_after * gpu["hbm_gb"]
            bpp = 14 if gpu["fp8_tflops"] else 16
            model_before = vram_before / bpp
            model_after = vram_after / bpp
            print(f"  {name:>12}: {n_before} GPUs ({vram_before} GB, {model_before:.0f}B) "
                  f"→ {n_after} GPUs ({vram_after} GB, {model_after:.0f}B)  "
                  f"[{(1 - model_after/model_before)*100:.0f}% model size reduction]")

    for name, gpu in GPUS.items():
        n_before = max_gpus_compute(gpu)
        n_after = min(n_before, max_gpus_memory(gpu))
        if n_after == n_before:
            print(f"  {name:>12}: unchanged ({n_before} GPUs, {n_before * gpu['hbm_gb']} GB)")

    print()


if __name__ == "__main__":
    main()
