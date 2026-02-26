"""
Treaty Evasion Scenario Calculator
Implements the MIRI Decentralized Training Simulator formulas to compute
the maximum local-equivalent FLOPs achievable using sub-CCC nodes over WAN.

Based on formulas from Simulator_Documentation.md.
"""

import math

# ── Fixed parameters ──────────────────────────────────────────────────────────

# Node configurations (all under 16 H100-equivalents = 15,840 TFLOPS FP16)
CONFIGS = {
    "48x A100 80GB": {
        "pflops": 48 * 312e12 / 1e15,   # 48 x 312 TFLOPS = 14.976 PFLOPS
        "vram_gb": 48 * 80,              # 3,840 GB
        "gpu_count": 48,
        "gpu_cost_usd": 15_000,          # ~$15k per A100 80GB (2025)
        "h100_equiv": 48 * 312 / 990,    # ~15.1
    },
    "16x GH200": {
        "pflops": 15.84,                 # 16 x 990 TFLOPS
        "vram_gb": 16 * 144,             # 2,304 GB
        "gpu_count": 16,
        "gpu_cost_usd": 30_000,          # ~$30k per GH200 (2025 est.)
        "h100_equiv": 16.0,
    },
    "16x H100 SXM": {
        "pflops": 15.84,                 # 16 x 990 TFLOPS
        "vram_gb": 16 * 80,              # 1,280 GB
        "gpu_count": 16,
        "gpu_cost_usd": 30_000,          # ~$30k per H100 (2025)
        "h100_equiv": 16.0,
    },
}

# Network
BW_MBPS = 100           # Symmetric WAN bandwidth (Mbps)
BW_BPS = BW_MBPS * 1e6  # bits/s
LATENCY_S = 0.1         # 100 ms RTT

# Training
MFU = 0.40
COMPRESSION = 16        # 4-bit quantization + 25% sparsification
LOCAL_BATCH = 131_072   # tokens per local step
BYTES_PER_PARAM = 16    # FP16 mixed-precision training
BITS_PER_PSEUDO_GRAD = 16  # FP16 pseudo-gradients before compression

# Time
TIME_YEARS = 1.5
TIME_SECONDS = TIME_YEARS * 365.25 * 86400  # 47,335,400 seconds

# Node sweep
NODE_COUNTS = [1, 2, 4, 8, 16, 32, 72, 144, 500, 1000]
NODE_COUNTS_LARGE = [500, 1000, 2000, 3000, 4000, 5000]

# Hierarchical DiLoCo regional parameters
REGIONAL_BW_MBPS = 1000    # 1 Gbps regional interconnect
REGIONAL_BW_BPS = REGIONAL_BW_MBPS * 1e6
REGIONAL_LATENCY_S = 0.02  # 20 ms regional RTT
NODES_PER_GROUP = 8        # Nodes per regional cluster

# Time sensitivity analysis
TIME_VARIANTS = {
    "6 months": 0.5 * 365.25 * 86400,
    "1 year": 1.0 * 365.25 * 86400,
    "1.5 years": 1.5 * 365.25 * 86400,
}

# FP8 node configuration (same CCC threshold, 2x compute throughput)
CONFIGS_FP8 = {
    "16x H100 FP8": {
        "pflops": 16 * 1980e12 / 1e15,  # 16 x 1980 TFLOPS FP8 = 31.68 PFLOPS
        "pflops_fp16": 15.84,             # For CCC threshold calculation
        "vram_gb": 16 * 80,              # 1,280 GB
        "gpu_count": 16,
        "gpu_cost_usd": 30_000,
        "h100_equiv": 16.0,              # CCC threshold uses FP16 capacity
        "bytes_per_param": 14,           # FP8: 1+1+4+4+4 = 14 bytes
        "bits_per_pseudo_grad": 8,       # FP8 pseudo-gradients (8 bits)
    },
}


# ── Simulator formulas ────────────────────────────────────────────────────────

def straggler_factor(n):
    """f(n) = 1 + 0.05 * log2(n)"""
    if n <= 1:
        return 1.0
    return 1.0 + 0.05 * math.log2(n)


def alpha(params_billion):
    """alpha = 0.08 / (1 + log10(P/1e9) / 5), where P is in parameters."""
    log_p = math.log10(params_billion)  # already in billions
    return 0.08 / (1.0 + log_p / 5.0)


def efficiency(h, params_billion):
    """eta = max(0.4, 1 - alpha * log10(H))"""
    a = alpha(params_billion)
    eta = 1.0 - a * math.log10(h)
    return max(0.4, eta)


def compute_scenario(config_name, n_nodes, compression=COMPRESSION,
                     time_seconds=None, bytes_per_param=BYTES_PER_PARAM,
                     bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                     bw_bps=None, latency_s=None):
    """Compute all metrics for a given node configuration and node count."""
    if config_name in CONFIGS:
        cfg = CONFIGS[config_name]
    else:
        cfg = CONFIGS_FP8[config_name]
        bytes_per_param = cfg["bytes_per_param"]
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_bps is None:
        bw_bps = BW_BPS
    if latency_s is None:
        latency_s = LATENCY_S

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    # Max dense model size
    max_params_b = vram_gb / bytes_per_param  # billions of params
    params_b = max_params_b  # Train the largest model that fits
    params = params_b * 1e9

    # Effective FLOPS
    effective_flops = pflops * 1e15 * MFU

    # Per-step compute time
    t_comp = (6 * params * LOCAL_BATCH) / effective_flops

    # Communication volume per sync (bits, after compression)
    v_bits = params * bits_per_pseudo_grad / compression

    # Sync time (base, before straggler)
    t_sync_base = 2 * v_bits / bw_bps + latency_s

    # Straggler factor
    f_n = straggler_factor(n_nodes)

    # Sync time with straggler
    t_sync = t_sync_base * f_n

    # Minimum H for compute-bound regime (streaming DiLoCo)
    if n_nodes == 1:
        h_min = 1
        eta = 1.0
    else:
        h_min = math.ceil(t_sync / t_comp)
        eta = efficiency(h_min, params_b)

    # Total local-equivalent FLOPs (compute-bound regime)
    c_actual = n_nodes * effective_flops * time_seconds
    c_local = c_actual * eta

    # Training details
    total_tokens = c_actual / (6 * params)
    chinchilla_tokens = 20 * params
    overtraining_ratio = total_tokens / chinchilla_tokens

    # Cost
    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    # Verify under CCC threshold
    h100_eq = cfg.get("h100_equiv", cfg.get("pflops_fp16", pflops) * 1000 / 990)
    assert h100_eq <= 16.01, f"{config_name} exceeds CCC threshold!"

    return {
        "config": config_name,
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "h100_equiv_per_node": cfg["h100_equiv"],
        "pflops_per_node": pflops,
        "vram_gb": vram_gb,
        "max_params_b": max_params_b,
        "t_comp": t_comp,
        "t_sync_base": t_sync_base,
        "f_straggler": f_n,
        "t_sync": t_sync,
        "h_min": h_min,
        "alpha": alpha(params_b) if n_nodes > 1 else 0,
        "eta": eta,
        "c_actual": c_actual,
        "c_local": c_local,
        "total_tokens_T": total_tokens / 1e12,
        "chinchilla_tokens_T": chinchilla_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
        "compression": compression,
        "time_seconds": time_seconds,
        "bw_mbps": bw_bps / 1e6,
        "latency_ms": latency_s * 1000,
    }


def compute_hierarchical_scenario(config_name, n_nodes, nodes_per_group=NODES_PER_GROUP,
                                  compression=COMPRESSION, time_seconds=None,
                                  bytes_per_param=BYTES_PER_PARAM,
                                  bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                                  bw_bps=None, latency_s=None,
                                  regional_bw_bps=None, regional_latency_s=None):
    """Compute metrics for hierarchical DiLoCo (two-tier topology)."""
    if config_name in CONFIGS:
        cfg = CONFIGS[config_name]
    else:
        cfg = CONFIGS_FP8[config_name]
        bytes_per_param = cfg["bytes_per_param"]
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_bps is None:
        bw_bps = BW_BPS
    if latency_s is None:
        latency_s = LATENCY_S
    if regional_bw_bps is None:
        regional_bw_bps = REGIONAL_BW_BPS
    if regional_latency_s is None:
        regional_latency_s = REGIONAL_LATENCY_S

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    max_params_b = vram_gb / bytes_per_param
    params_b = max_params_b
    params = params_b * 1e9

    effective_flops = pflops * 1e15 * MFU

    # Per-step compute time
    t_comp = (6 * params * LOCAL_BATCH) / effective_flops

    # Communication volume (bits, after compression)
    v_bits = params * bits_per_pseudo_grad / compression

    # Number of groups
    n_groups = n_nodes // nodes_per_group
    if n_groups < 2:
        # Fall back to flat DiLoCo
        return compute_scenario(config_name, n_nodes, compression, time_seconds,
                                bytes_per_param, bits_per_pseudo_grad,
                                bw_bps, latency_s)

    # Regional sync (fast LAN)
    f_regional = straggler_factor(nodes_per_group)
    t_regional_sync = (2 * v_bits / regional_bw_bps + regional_latency_s) * f_regional
    h_inner_min = max(1, math.ceil(t_regional_sync / t_comp))

    # Global sync (slow WAN)
    f_global = straggler_factor(n_groups)
    t_global_sync = (2 * v_bits / bw_bps + latency_s) * f_global

    # Regional cycle time (streaming)
    t_regional_cycle = max(h_inner_min * t_comp, t_regional_sync)

    # Minimum H_regional for compute-bound global sync
    h_regional_min = max(1, math.ceil(t_global_sync / t_regional_cycle))

    # Effective H (hierarchical formula)
    h_eff = h_inner_min * math.sqrt(h_regional_min)

    eta = efficiency(h_eff, params_b)

    # Total local-equivalent FLOPs
    c_actual = n_nodes * effective_flops * time_seconds
    c_local = c_actual * eta

    # Training details
    total_tokens = c_actual / (6 * params)
    chinchilla_tokens = 20 * params
    overtraining_ratio = total_tokens / chinchilla_tokens

    # Cost
    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "config": config_name + " (hierarchical)",
        "n_nodes": n_nodes,
        "n_groups": n_groups,
        "nodes_per_group": nodes_per_group,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "h100_equiv_per_node": cfg["h100_equiv"],
        "pflops_per_node": pflops,
        "vram_gb": vram_gb,
        "max_params_b": max_params_b,
        "t_comp": t_comp,
        "t_regional_sync": t_regional_sync,
        "t_global_sync": t_global_sync,
        "f_regional": f_regional,
        "f_global": f_global,
        "h_inner": h_inner_min,
        "h_regional": h_regional_min,
        "h_eff": h_eff,
        "alpha": alpha(params_b),
        "eta": eta,
        "c_actual": c_actual,
        "c_local": c_local,
        "total_tokens_T": total_tokens / 1e12,
        "chinchilla_tokens_T": chinchilla_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
        "compression": compression,
    }


def compute_moe_ep_scenario(config_name, n_nodes, total_params_b, active_params_b,
                            n_moe_layers=32, compression=COMPRESSION,
                            time_seconds=None):
    """Compute metrics for MoE + Expert Parallelism scenario."""
    if config_name in CONFIGS:
        cfg = CONFIGS[config_name]
        bpp = BYTES_PER_PARAM
        bpg = BITS_PER_PSEUDO_GRAD
    else:
        cfg = CONFIGS_FP8[config_name]
        bpp = cfg["bytes_per_param"]
        bpg = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    # EP memory reduction
    p_shared = active_params_b * 1e9
    p_experts = (total_params_b - active_params_b) * 1e9
    mem_node_gb = (p_shared + p_experts / n_nodes) * bpp / 1e9
    fits = mem_node_gb <= vram_gb

    # Effective FLOPS (only active params contribute to compute)
    effective_flops = pflops * 1e15 * MFU
    params_active = active_params_b * 1e9

    # Per-step compute time (uses active params)
    t_comp = (6 * params_active * LOCAL_BATCH) / effective_flops

    # EP All-to-All latency per step
    t_ep = 2 * LATENCY_S * n_moe_layers

    # Communication volume for DiLoCo sync (total params, after compression)
    v_bits = total_params_b * 1e9 * bpg / compression

    # Sync time
    f_n = straggler_factor(n_nodes)
    t_sync_base = 2 * v_bits / BW_BPS + LATENCY_S
    t_sync = t_sync_base * f_n

    # Minimum H
    t_step = t_comp + t_ep
    if n_nodes <= 1:
        h_min = 1
        eta = 1.0
    else:
        h_min = math.ceil(t_sync / t_step)
        eta = efficiency(h_min, total_params_b)

    # Total local-equivalent FLOPs (based on active params compute rate)
    # Adjust for EP latency overhead
    ep_overhead = t_ep / (t_comp + t_ep)
    c_actual = n_nodes * effective_flops * time_seconds * (1 - ep_overhead)
    c_local = c_actual * eta

    # Training details (based on active params)
    total_tokens = c_actual / (6 * params_active)
    chinchilla_tokens_active = 20 * params_active
    overtraining_ratio = total_tokens / chinchilla_tokens_active

    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "config": config_name + f" (MoE {total_params_b:.0f}B/{active_params_b:.0f}B)",
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "total_params_b": total_params_b,
        "active_params_b": active_params_b,
        "mem_node_gb": mem_node_gb,
        "vram_gb": vram_gb,
        "fits_on_node": fits,
        "t_comp": t_comp,
        "t_ep": t_ep,
        "ep_overhead_pct": ep_overhead * 100,
        "t_sync": t_sync,
        "f_straggler": f_n,
        "h_min": h_min,
        "eta": eta,
        "c_actual": c_actual,
        "c_local": c_local,
        "total_tokens_T": total_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def print_config_summary(config_name):
    cfg = CONFIGS[config_name]
    max_b = cfg["vram_gb"] / BYTES_PER_PARAM
    print(f"\n{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"  Peak FP16:    {cfg['pflops']:.2f} PFLOPS ({cfg['h100_equiv']:.1f} H100-equiv)")
    print(f"  VRAM:         {cfg['vram_gb']:,} GB")
    print(f"  Max model:    {max_b:.0f}B params (FP16 training)")
    print(f"  GPU cost:     ${cfg['gpu_cost_usd']:,}/GPU x {cfg['gpu_count']} = ${cfg['gpu_count'] * cfg['gpu_cost_usd']:,.0f}/node")
    print(f"{'='*80}")


def print_results_table(config_name):
    print(f"\n{'N':>5} | {'GPUs':>7} | {'Cost':>8} | {'f(N)':>5} | {'H_min':>5} | "
          f"{'eta':>5} | {'C_local':>10} | {'x10^24':>7} | {'Model':>6} | "
          f"{'Tokens':>7} | {'OT ratio':>8}")
    print("-" * 105)

    for n in NODE_COUNTS:
        r = compute_scenario(config_name, n)
        cost_str = f"${r['cost_usd']/1e6:.1f}M"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.1f}x"
        model_str = f"{r['max_params_b']:.0f}B"
        tokens_str = f"{r['total_tokens_T']:.1f}T"
        ot_str = f"{r['overtraining_ratio']:.1f}x"

        print(f"{n:>5} | {r['total_gpus']:>7,} | {cost_str:>8} | {r['f_straggler']:>5.3f} | "
              f"{r['h_min']:>5} | {r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7} | "
              f"{model_str:>6} | {tokens_str:>7} | {ot_str:>8}")


def print_detailed(config_name, n_nodes):
    r = compute_scenario(config_name, n_nodes)
    print(f"\n--- Detailed: {config_name}, N={n_nodes} ---")
    print(f"  Per-step compute:    {r['t_comp']:.2f} seconds")
    print(f"  Sync base time:     {r['t_sync_base']:.1f} seconds")
    print(f"  Straggler factor:   {r['f_straggler']:.3f}")
    print(f"  Sync time (total):  {r['t_sync']:.1f} seconds")
    print(f"  Min inner steps H:  {r['h_min']}")
    print(f"  Alpha:              {r['alpha']:.5f}")
    print(f"  Efficiency eta:       {r['eta']:.4f}")
    print(f"  C_actual:           {r['c_actual']:.3e} FLOP")
    print(f"  C_local:            {r['c_local']:.3e} FLOP (local-equivalent)")
    print(f"  x Strict Threshold: {r['strict_threshold_multiple']:.1f}x")
    print(f"  Model:              {r['max_params_b']:.0f}B params")
    print(f"  Tokens:             {r['total_tokens_T']:.1f}T")
    print(f"  Chinchilla tokens:  {r['chinchilla_tokens_T']:.1f}T")
    print(f"  Overtraining:       {r['overtraining_ratio']:.1f}x")
    print(f"  Cost:               ${r['cost_usd']:,.0f}")


def print_large_scale_table(config_name, compression=COMPRESSION):
    """Print results for large node counts targeting 10^27."""
    print(f"\n{'N':>5} | {'GPUs':>7} | {'Cost':>8} | {'f(N)':>5} | {'H_min':>5} | "
          f"{'eta':>5} | {'C_local':>10} | {'x10^24':>7}")
    print("-" * 75)

    for n in NODE_COUNTS_LARGE:
        r = compute_scenario(config_name, n, compression=compression)
        cost_str = f"${r['cost_usd']/1e9:.1f}B" if r['cost_usd'] >= 1e9 else f"${r['cost_usd']/1e6:.0f}M"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.0f}x"

        print(f"{n:>5} | {r['total_gpus']:>7,} | {cost_str:>8} | {r['f_straggler']:>5.3f} | "
              f"{r['h_min']:>5} | {r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7}")


def print_hierarchical_table(config_name, compression=COMPRESSION):
    """Print hierarchical DiLoCo results for large node counts."""
    print(f"\n{'N':>5} | {'Groups':>6} | {'H_in':>5} | {'H_reg':>5} | {'H_eff':>5} | "
          f"{'eta':>5} | {'C_local':>10} | {'x10^24':>7}")
    print("-" * 75)

    for n in NODE_COUNTS_LARGE:
        r = compute_hierarchical_scenario(config_name, n, compression=compression)
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.0f}x"

        print(f"{n:>5} | {r.get('n_groups', 'N/A'):>6} | "
              f"{r.get('h_inner', r.get('h_min', 'N/A')):>5} | "
              f"{r.get('h_regional', '-'):>5} | "
              f"{r.get('h_eff', r.get('h_min', '-')):>5.0f} | "
              f"{r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7}")


def print_time_sensitivity(config_name, n_nodes):
    """Print C_local at different training durations."""
    print(f"\n  Time sensitivity for {config_name}, N={n_nodes}:")
    for label, t_sec in TIME_VARIANTS.items():
        r = compute_scenario(config_name, n_nodes, time_seconds=t_sec)
        print(f"    {label:>10}: C_local = {r['c_local']:.2e} FLOP ({r['strict_threshold_multiple']:.1f}x threshold)")


def print_10e27_comparison():
    """Print the 10^27 FLOP scenario comparison table."""
    print("\n" + "=" * 100)
    print("10^27 FLOP CONFIGURATION COMPARISON")
    print("=" * 100)

    scenarios = []

    # A: Flat DiLoCo, 48x A100 FP16, 16x compression
    rA = compute_scenario("48x A100 80GB", 4000)
    scenarios.append(("A: Flat, A100 FP16, 16x comp", rA))

    # B: Hierarchical, 48x A100 FP16, 16x compression
    rB = compute_hierarchical_scenario("48x A100 80GB", 4000)
    scenarios.append(("B: Hierarchical, A100 FP16, 16x", rB))

    # C: Flat DiLoCo, H100 FP8, 16x compression
    rC = compute_scenario("16x H100 FP8", 2000)
    scenarios.append(("C: Flat, H100 FP8, 16x comp", rC))

    # D: Hierarchical, H100 FP8, 16x compression
    rD = compute_hierarchical_scenario("16x H100 FP8", 2000)
    scenarios.append(("D: Hier, H100 FP8, 16x comp", rD))

    # E: Flat, A100 FP16, 100x compression
    rE = compute_scenario("48x A100 80GB", 4000, compression=100)
    scenarios.append(("E: Flat, A100 FP16, 100x comp", rE))

    # F: Hierarchical + 100x compression + FP8 H100
    rF = compute_hierarchical_scenario("16x H100 FP8", 2000, compression=100)
    scenarios.append(("F: Hier+100x, H100 FP8", rF))

    # G: MoE + EP (600B total, 100B active)
    rG = compute_moe_ep_scenario("48x A100 80GB", 4000, total_params_b=600,
                                  active_params_b=100)
    scenarios.append(("G: MoE+EP 600B/100B, A100", rG))

    print(f"\n{'Config':>35} | {'Nodes':>5} | {'GPUs':>7} | {'Cost':>8} | "
          f"{'Model':>18} | {'eta':>5} | {'C_local':>10} | {'x10^24':>7}")
    print("-" * 115)

    for label, r in scenarios:
        cost = r['cost_usd']
        cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.0f}x"
        if 'total_params_b' in r:
            model_str = f"{r['total_params_b']:.0f}B MoE ({r['active_params_b']:.0f}B act)"
        else:
            model_str = f"{r['max_params_b']:.0f}B dense"

        print(f"{label:>35} | {r['n_nodes']:>5} | {r['total_gpus']:>7,} | "
              f"{cost_str:>8} | {model_str:>18} | {r['eta']:>5.3f} | "
              f"{c_str:>10} | {mult_str:>7}")


# ── Network sensitivity parameters ───────────────────────────────────────────

# Bandwidth sweep values (Mbps)
BANDWIDTH_SWEEP_MBPS = [10, 25, 50, 100, 250, 500, 1000]

# Latency scenarios based on real-world measurements
# Sources: Azure Network Latency Stats (June 2025), Verizon IP Latency,
#          Epsilon Telecom, AWS inter-region (CloudPing.co)
LATENCY_SCENARIOS = {
    "Same cloud region":     0.002,    #   2 ms — AWS cross-AZ, Azure intra-region
    "Same continent (EU)":   0.020,    #  20 ms — London–Frankfurt (Epsilon)
    "Continental US":        0.065,    #  65 ms — NYC–LA (Epsilon 64ms, Azure 71ms avg)
    "Transatlantic":         0.075,    #  75 ms — NYC–London (Verizon 70ms, Azure 79ms)
    "Transpacific":          0.105,    # 105 ms — LA–Tokyo (AWS Oregon-Tokyo, Epsilon)
    "US East–Asia":          0.230,    # 230 ms — Virginia–Singapore (AWS 224ms)
    "Global worst-case":     0.340,    # 340 ms — Brazil–SE Asia (Azure 332-343ms)
}

# Combined scenarios: realistic network profiles for different deployment models
DEPLOYMENT_PROFILES = {
    "Colocated (same metro)":  {"bw_mbps": 1000,  "latency_ms":   5, "description": "Nodes in nearby DCs, business fiber"},
    "Same country (US)":       {"bw_mbps":  500,  "latency_ms":  35, "description": "Nodes across US, enterprise connections"},
    "Continental (US coasts)": {"bw_mbps":  100,  "latency_ms":  65, "description": "NYC to LA, commodity broadband"},
    "Continental (EU)":        {"bw_mbps":  100,  "latency_ms":  20, "description": "London to Frankfurt, commodity broadband"},
    "Transatlantic":           {"bw_mbps":  100,  "latency_ms":  75, "description": "US to Europe, commodity broadband"},
    "Transpacific":            {"bw_mbps":   50,  "latency_ms": 105, "description": "US to Japan, limited bandwidth"},
    "Global (adversarial)":    {"bw_mbps":   25,  "latency_ms": 230, "description": "Maximally distributed, avoid detection"},
    "Global worst-case":       {"bw_mbps":   10,  "latency_ms": 340, "description": "Brazil to SE Asia, consumer broadband"},
}


def bandwidth_sensitivity(config_name, n_nodes, compression=COMPRESSION,
                          latency_s=None, use_hierarchical=False):
    """Sweep bandwidth values and return results for each."""
    if latency_s is None:
        latency_s = LATENCY_S
    results = []
    for bw_mbps in BANDWIDTH_SWEEP_MBPS:
        bw_bps = bw_mbps * 1e6
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_bps=bw_bps, latency_s=latency_s)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_bps=bw_bps, latency_s=latency_s)
        r["bw_mbps"] = bw_mbps
        results.append(r)
    return results


def latency_sensitivity(config_name, n_nodes, compression=COMPRESSION,
                        bw_bps=None, use_hierarchical=False):
    """Sweep latency values and return results for each."""
    if bw_bps is None:
        bw_bps = BW_BPS
    results = []
    for name, lat_s in LATENCY_SCENARIOS.items():
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_bps=bw_bps, latency_s=lat_s)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_bps=bw_bps, latency_s=lat_s)
        r["latency_scenario"] = name
        r["latency_ms"] = lat_s * 1000
        results.append(r)
    return results


def deployment_profile_sweep(config_name, n_nodes, compression=COMPRESSION,
                             use_hierarchical=False):
    """Test all deployment profiles and return results."""
    results = []
    for name, profile in DEPLOYMENT_PROFILES.items():
        bw_bps = profile["bw_mbps"] * 1e6
        lat_s = profile["latency_ms"] / 1000.0
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_bps=bw_bps, latency_s=lat_s)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_bps=bw_bps, latency_s=lat_s)
        r["profile"] = name
        r["profile_description"] = profile["description"]
        r["bw_mbps"] = profile["bw_mbps"]
        r["latency_ms"] = profile["latency_ms"]
        results.append(r)
    return results


def print_bandwidth_sensitivity(config_name, n_nodes, compression=COMPRESSION,
                                use_hierarchical=False):
    """Print bandwidth sensitivity table."""
    label = "hierarchical" if use_hierarchical else "flat"
    print(f"\n  Bandwidth sensitivity: {config_name}, N={n_nodes}, {label}, {compression}x comp")
    print(f"  {'BW (Mbps)':>10} | {'H_min/eff':>9} | {'eta':>5} | {'C_local':>10} | {'x10^24':>7} | {'Regime':>12}")
    print("  " + "-" * 72)
    results = bandwidth_sensitivity(config_name, n_nodes, compression,
                                    use_hierarchical=use_hierarchical)
    for r in results:
        h = r.get("h_eff", r.get("h_min", 1))
        h_str = f"{h:.0f}" if isinstance(h, float) else f"{h}"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.1f}x"
        # Determine if comm-bound: if not streaming, eta would be lower
        regime = "compute" if r['eta'] > 0.5 else "comm-bound"
        print(f"  {r['bw_mbps']:>10} | {h_str:>9} | {r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7} | {regime:>12}")
    return results


def print_latency_sensitivity(config_name, n_nodes, compression=COMPRESSION,
                              use_hierarchical=False):
    """Print latency sensitivity table."""
    label = "hierarchical" if use_hierarchical else "flat"
    print(f"\n  Latency sensitivity: {config_name}, N={n_nodes}, {label}, {compression}x comp")
    print(f"  {'Scenario':>25} | {'RTT (ms)':>8} | {'H_min/eff':>9} | {'eta':>5} | {'C_local':>10} | {'x10^24':>7}")
    print("  " + "-" * 82)
    results = latency_sensitivity(config_name, n_nodes, compression,
                                  use_hierarchical=use_hierarchical)
    for r in results:
        h = r.get("h_eff", r.get("h_min", 1))
        h_str = f"{h:.0f}" if isinstance(h, float) else f"{h}"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.1f}x"
        print(f"  {r['latency_scenario']:>25} | {r['latency_ms']:>8.0f} | {h_str:>9} | "
              f"{r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7}")
    return results


def print_deployment_profiles(config_name, n_nodes, compression=COMPRESSION,
                              use_hierarchical=False):
    """Print deployment profile sweep table."""
    label = "hierarchical" if use_hierarchical else "flat"
    print(f"\n  Deployment profiles: {config_name}, N={n_nodes}, {label}, {compression}x comp")
    print(f"  {'Profile':>25} | {'BW':>6} | {'RTT':>5} | {'H':>5} | {'eta':>5} | {'C_local':>10} | {'x10^24':>7}")
    print("  " + "-" * 82)
    results = deployment_profile_sweep(config_name, n_nodes, compression,
                                       use_hierarchical=use_hierarchical)
    for r in results:
        h = r.get("h_eff", r.get("h_min", 1))
        h_str = f"{h:.0f}" if isinstance(h, float) else f"{h}"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.1f}x"
        print(f"  {r['profile']:>25} | {r['bw_mbps']:>4}M | {r['latency_ms']:>3.0f}ms | "
              f"{h_str:>5} | {r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7}")
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("TREATY EVASION SCENARIO: Maximum Distributed Training Below CCC Threshold")
    print(f"Time limit: {TIME_YEARS} years ({TIME_SECONDS/86400:.0f} days)")
    print(f"WAN: {BW_MBPS} Mbps, {LATENCY_S*1000:.0f} ms latency")
    print(f"MFU: {MFU*100:.0f}%  |  Compression: {COMPRESSION}x  |  Streaming: Yes")
    print(f"CCC threshold: 16 H100-equivalents = 15,840 TFLOPS FP16")
    print("=" * 80)

    # ── PART 1: Existing analysis (sub-threshold evasion) ────────────────────

    # Primary analysis: 48x A100
    print_config_summary("48x A100 80GB")
    print_results_table("48x A100 80GB")
    print_detailed("48x A100 80GB", 4)
    print_detailed("48x A100 80GB", 72)

    # Secondary: 16x GH200
    print_config_summary("16x GH200")
    print_results_table("16x GH200")
    print_detailed("16x GH200", 72)

    # Tertiary: 16x H100
    print_config_summary("16x H100 SXM")
    print_results_table("16x H100 SXM")

    # ── PART 2: 10^27 FLOP scenarios ────────────────────────────────────────

    print("\n" + "=" * 80)
    print("PART 2: SCALING TO 10^27 FLOP")
    print("=" * 80)

    # Large-scale flat DiLoCo
    print("\n--- A: Flat DiLoCo, 48x A100 FP16, 16x compression ---")
    print_large_scale_table("48x A100 80GB")

    # Hierarchical DiLoCo
    print(f"\n--- B: Hierarchical DiLoCo (groups of {NODES_PER_GROUP}), 48x A100 FP16, 16x ---")
    print_hierarchical_table("48x A100 80GB")

    # FP8 H100
    print("\n--- C: Flat DiLoCo, 16x H100 FP8, 16x compression ---")
    print_large_scale_table("16x H100 FP8")

    # Hierarchical FP8 H100
    print(f"\n--- D: Hierarchical DiLoCo, 16x H100 FP8, 16x compression ---")
    print_hierarchical_table("16x H100 FP8")

    # 100x compression flat
    print("\n--- E: Flat DiLoCo, 48x A100 FP16, 100x compression ---")
    print_large_scale_table("48x A100 80GB", compression=100)

    # Hierarchical + 100x + FP8
    print(f"\n--- F: Hierarchical + 100x compression, 16x H100 FP8 ---")
    print_hierarchical_table("16x H100 FP8", compression=100)

    # MoE + EP
    print("\n--- G: MoE + Expert Parallelism (600B total / 100B active) ---")
    for n in [72, 500, 2000, 4000]:
        r = compute_moe_ep_scenario("48x A100 80GB", n, total_params_b=600,
                                     active_params_b=100)
        cost = r['cost_usd']
        cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
        print(f"  N={n:>5}: mem/node={r['mem_node_gb']:.0f}GB ({'FITS' if r['fits_on_node'] else 'NO FIT'}), "
              f"EP overhead={r['ep_overhead_pct']:.1f}%, eta={r['eta']:.3f}, "
              f"C_local={r['c_local']:.2e}, cost={cost_str}")

    # Comparison table
    print_10e27_comparison()

    # ── PART 3: Enforcement time sensitivity ────────────────────────────────

    print("\n" + "=" * 80)
    print("PART 3: ENFORCEMENT TIME SENSITIVITY")
    print("=" * 80)

    print_time_sensitivity("48x A100 80GB", 4)
    print_time_sensitivity("48x A100 80GB", 72)
    print_time_sensitivity("48x A100 80GB", 4000)
    print_time_sensitivity("16x H100 FP8", 2000)

    # ── PART 4: Network sensitivity analysis ─────────────────────────────────

    print("\n" + "=" * 80)
    print("PART 4: NETWORK SENSITIVITY ANALYSIS")
    print("=" * 80)

    # --- 4A: Bandwidth sensitivity ---
    print("\n" + "-" * 80)
    print("4A: BANDWIDTH SENSITIVITY (varying BW, fixed 100ms latency)")
    print("-" * 80)

    # Small-scale evasion (72 nodes, 48x A100, reaching for 10^25)
    print_bandwidth_sensitivity("48x A100 80GB", 72)

    # Medium-scale (500 nodes, targeting 10^26)
    print_bandwidth_sensitivity("48x A100 80GB", 500)

    # Large-scale flat (4000 nodes A100, targeting 10^27)
    print_bandwidth_sensitivity("48x A100 80GB", 4000)

    # Large-scale FP8 flat (2000 nodes H100 FP8, targeting 10^27)
    print_bandwidth_sensitivity("16x H100 FP8", 2000)

    # Large-scale hierarchical (best config: H100 FP8, hier, 100x)
    print_bandwidth_sensitivity("16x H100 FP8", 2000, compression=100,
                                use_hierarchical=True)

    # --- 4B: Latency sensitivity ---
    print("\n" + "-" * 80)
    print("4B: LATENCY SENSITIVITY (varying latency, fixed 100 Mbps BW)")
    print("-" * 80)

    # Reference: 72 nodes (10^25 target)
    print_latency_sensitivity("48x A100 80GB", 72)

    # 500 nodes (10^26 target)
    print_latency_sensitivity("48x A100 80GB", 500)

    # 2000 nodes H100 FP8 flat (10^27 target)
    print_latency_sensitivity("16x H100 FP8", 2000)

    # 2000 nodes H100 FP8 hierarchical + 100x (best config)
    print_latency_sensitivity("16x H100 FP8", 2000, compression=100,
                              use_hierarchical=True)

    # --- 4C: Deployment profiles (combined BW + latency) ---
    print("\n" + "-" * 80)
    print("4C: DEPLOYMENT PROFILES (realistic BW + latency combinations)")
    print("-" * 80)

    # 72 nodes (sub-$100M evasion)
    print_deployment_profiles("48x A100 80GB", 72)

    # 500 nodes (10^26 target)
    print_deployment_profiles("48x A100 80GB", 500)

    # 2000 nodes FP8 (10^27 target) — flat
    print_deployment_profiles("16x H100 FP8", 2000)

    # 2000 nodes FP8 (10^27 target) — hierarchical + 100x
    print_deployment_profiles("16x H100 FP8", 2000, compression=100,
                              use_hierarchical=True)

    # ── PART 5: Cross-validation ────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("PART 5: CROSS-VALIDATION")
    print("=" * 80)

    # Check 1: Single 16xH100 node vs treaty claim
    r1 = compute_scenario("16x H100 SXM", 1)
    print(f"\n1. Single 16xH100 node in 1.5 years (BF16, 40% MFU): {r1['c_local']:.2e} FLOP")
    print(f"   Treaty says 10^24 in 2 years at FP8/50% -> our BF16/40% = {r1['c_local']/1e24:.2f}x threshold")
    print(f"   Expected ~0.3x (2.5x slower than treaty's FP8/50%): {'PASS' if 0.2 < r1['c_local']/1e24 < 0.4 else 'CHECK'}")

    # Check 2: 72xGH200 vs existing baseline
    r2 = compute_scenario("16x GH200", 72)
    print(f"\n2. 72xGH200 C_local: {r2['c_local']:.2e} FLOP")
    print(f"   Existing baseline: 1.02e25 FLOP (raw C=6PD for 144Bx12T tokens)")
    print(f"   Our C_actual for 72 nodes: {r2['c_actual']:.2e}")
    ratio = r2['c_actual'] / 1.02e25
    print(f"   Ratio: {ratio:.2f}x")
    print(f"   Note: Baseline used 32 PFLOPS/node; we use 15.84 -> ratio should be ~{15.84/32:.2f}x")

    # Check 3: CCC threshold verification
    print(f"\n3. CCC threshold verification:")
    for name, cfg in {**CONFIGS, **CONFIGS_FP8}.items():
        status = "UNDER" if cfg["h100_equiv"] <= 16.0 else "OVER"
        print(f"   {name}: {cfg['h100_equiv']:.1f} H100-equiv -> {status} threshold")

    # Check 4: FP8 gives ~2x throughput per node
    r_fp16 = compute_scenario("16x H100 SXM", 1)
    r_fp8 = compute_scenario("16x H100 FP8", 1)
    ratio_fp8 = r_fp8['c_local'] / r_fp16['c_local']
    print(f"\n4. FP8 vs FP16 throughput (single H100 node):")
    print(f"   FP16: {r_fp16['c_local']:.2e}, FP8: {r_fp8['c_local']:.2e}")
    print(f"   Ratio: {ratio_fp8:.2f}x (expected ~2x): {'PASS' if 1.8 < ratio_fp8 < 2.5 else 'CHECK'}")

    # Check 5: Hierarchical eta > flat eta at large N
    r_flat = compute_scenario("48x A100 80GB", 4000)
    r_hier = compute_hierarchical_scenario("48x A100 80GB", 4000)
    print(f"\n5. Hierarchical vs flat eta at N=4000:")
    print(f"   Flat: eta={r_flat['eta']:.4f} (H={r_flat['h_min']})")
    print(f"   Hier: eta={r_hier['eta']:.4f} (H_eff={r_hier['h_eff']:.1f})")
    print(f"   Improvement: {(r_hier['eta']-r_flat['eta'])*100:.1f}pp: {'PASS' if r_hier['eta'] > r_flat['eta'] else 'CHECK'}")

    # Check 6: MoE+EP memory fits
    r_moe = compute_moe_ep_scenario("48x A100 80GB", 72, total_params_b=600,
                                     active_params_b=100)
    print(f"\n6. MoE+EP memory check (600B MoE, N=72, 48xA100):")
    print(f"   Per-node memory: {r_moe['mem_node_gb']:.0f} GB (VRAM: {r_moe['vram_gb']} GB)")
    print(f"   Fits: {r_moe['fits_on_node']}: {'PASS' if r_moe['fits_on_node'] else 'FAIL'}")

    # Check 7: Time scaling (C_local at 6mo = C_local at 18mo * 6/18)
    r_full = compute_scenario("48x A100 80GB", 72)
    r_half = compute_scenario("48x A100 80GB", 72, time_seconds=TIME_VARIANTS["6 months"])
    ratio_time = r_half['c_local'] / r_full['c_local']
    print(f"\n7. Time scaling check (6mo vs 1.5yr):")
    print(f"   6mo: {r_half['c_local']:.2e}, 1.5yr: {r_full['c_local']:.2e}")
    print(f"   Ratio: {ratio_time:.3f} (expected 0.333): {'PASS' if abs(ratio_time - 1/3) < 0.01 else 'CHECK'}")
