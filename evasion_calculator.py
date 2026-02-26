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


# ── Countermeasure analysis configs ───────────────────────────────────────────

# Lowered CCC threshold: max A100-80GB node under each threshold
# A100 80GB: 312 TFLOPS FP16 each, $15k each, 80 GB VRAM each
# Max GPUs = floor(threshold_h100_equiv * 990 / 312)
LOWERED_CCC_A100 = {
    "16 H100-eq (current)": {"gpu_count": 48, "pflops": 48 * 312e12 / 1e15,
                              "vram_gb": 48 * 80, "gpu_cost_usd": 15_000,
                              "h100_equiv": 48 * 312 / 990},
    "8 H100-eq":  {"gpu_count": 25, "pflops": 25 * 312e12 / 1e15,
                    "vram_gb": 25 * 80, "gpu_cost_usd": 15_000,
                    "h100_equiv": 25 * 312 / 990},
    "4 H100-eq":  {"gpu_count": 12, "pflops": 12 * 312e12 / 1e15,
                    "vram_gb": 12 * 80, "gpu_cost_usd": 15_000,
                    "h100_equiv": 12 * 312 / 990},
    "2 H100-eq":  {"gpu_count": 6,  "pflops": 6 * 312e12 / 1e15,
                    "vram_gb": 6 * 80,  "gpu_cost_usd": 15_000,
                    "h100_equiv": 6 * 312 / 990},
    "1 H100-eq":  {"gpu_count": 3,  "pflops": 3 * 312e12 / 1e15,
                    "vram_gb": 3 * 80,  "gpu_cost_usd": 15_000,
                    "h100_equiv": 3 * 312 / 990},
}

# Lowered CCC threshold: max H100 SXM node under each threshold (FP8 compute)
# H100 SXM: 990 TFLOPS FP16 each (1980 FP8), $30k each, 80 GB VRAM each
LOWERED_CCC_H100_FP8 = {
    "16 H100-eq (current)": {"gpu_count": 16, "pflops": 16 * 1980e12 / 1e15,
                              "pflops_fp16": 15.84, "vram_gb": 16 * 80,
                              "gpu_cost_usd": 30_000, "h100_equiv": 16.0,
                              "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "8 H100-eq":  {"gpu_count": 8,  "pflops": 8 * 1980e12 / 1e15,
                    "pflops_fp16": 7.92,  "vram_gb": 8 * 80,
                    "gpu_cost_usd": 30_000, "h100_equiv": 8.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "4 H100-eq":  {"gpu_count": 4,  "pflops": 4 * 1980e12 / 1e15,
                    "pflops_fp16": 3.96,  "vram_gb": 4 * 80,
                    "gpu_cost_usd": 30_000, "h100_equiv": 4.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "2 H100-eq":  {"gpu_count": 2,  "pflops": 2 * 1980e12 / 1e15,
                    "pflops_fp16": 1.98,  "vram_gb": 2 * 80,
                    "gpu_cost_usd": 30_000, "h100_equiv": 2.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "1 H100-eq":  {"gpu_count": 1,  "pflops": 1 * 1980e12 / 1e15,
                    "pflops_fp16": 0.99,  "vram_gb": 1 * 80,
                    "gpu_cost_usd": 30_000, "h100_equiv": 1.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
}

# Memory thresholds: VRAM limits that trigger CCC registration
MEMORY_THRESHOLDS_GB = [256, 512, 1024, 2048]

# Collateral damage: representative legitimate computing systems
# ── Compression quality model ────────────────────────────────────────────────
# Multiplicative penalty on eta from gradient compression quality loss.
# Based on literature review:
#   - FP4 (4x): lossless at 4B (Streaming DiLoCo 2501.18512), 15B (MuLoCo 2505.23725)
#   - 16x: FP4 + 4x sparsification; FP4 validated, sparsification limited evidence
#   - 100x: 2-bit + TopK 3% or FP4 + 25x sparse; validated only at 512M (SparseLoCo 2508.15706)
# "optimistic" = best-case (literature supports lossless at small scale)
# "expected"   = accounts for extrapolation uncertainty to 100B+ scale
# "conservative" = pessimistic bound covering compounding unknowns

COMPRESSION_QUALITY = {
    1:   {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    4:   {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    16:  {"optimistic": 1.00, "expected": 0.98, "conservative": 0.95},
    100: {"optimistic": 0.99, "expected": 0.95, "conservative": 0.90},
}

DEFAULT_SCENARIO = "expected"

LEGITIMATE_SYSTEMS = [
    {"name": "Consumer gaming PC (1x RTX 5090)",    "gpus": 1,   "vram_gb": 32,    "tflops_fp16": 104,  "h100_equiv": 0.11, "category": "Consumer"},
    {"name": "Enthusiast workstation (2x RTX 4090)", "gpus": 2,   "vram_gb": 48,    "tflops_fp16": 330,  "h100_equiv": 0.33, "category": "Consumer"},
    {"name": "Research workstation (4x A100 40GB)",  "gpus": 4,   "vram_gb": 160,   "tflops_fp16": 1248, "h100_equiv": 1.26, "category": "Research"},
    {"name": "Research workstation (4x A100 80GB)",  "gpus": 4,   "vram_gb": 320,   "tflops_fp16": 1248, "h100_equiv": 1.26, "category": "Research"},
    {"name": "AI lab server (8x A100 80GB)",         "gpus": 8,   "vram_gb": 640,   "tflops_fp16": 2496, "h100_equiv": 2.52, "category": "Research"},
    {"name": "AWS p4d.24xlarge (8x A100 40GB)",      "gpus": 8,   "vram_gb": 320,   "tflops_fp16": 2496, "h100_equiv": 2.52, "category": "Cloud"},
    {"name": "AWS p5.48xlarge (8x H100 SXM)",        "gpus": 8,   "vram_gb": 640,   "tflops_fp16": 7920, "h100_equiv": 8.0,  "category": "Cloud"},
    {"name": "HPC simulation node (4x A100 80GB)",   "gpus": 4,   "vram_gb": 320,   "tflops_fp16": 1248, "h100_equiv": 1.26, "category": "Scientific"},
    {"name": "Molecular dynamics cluster (8x H100)",  "gpus": 8,   "vram_gb": 640,   "tflops_fp16": 7920, "h100_equiv": 8.0,  "category": "Scientific"},
    {"name": "Inference server (8x L40S)",           "gpus": 8,   "vram_gb": 384,   "tflops_fp16": 1472, "h100_equiv": 1.49, "category": "Commercial"},
    {"name": "Rendering farm node (4x RTX 6000 Ada)","gpus": 4,   "vram_gb": 192,   "tflops_fp16": 597,  "h100_equiv": 0.60, "category": "Commercial"},
    {"name": "Princeton AI cluster node (8x H100)",  "gpus": 8,   "vram_gb": 640,   "tflops_fp16": 7920, "h100_equiv": 8.0,  "category": "Research"},
    {"name": "DGX A100 (8x A100 80GB)",             "gpus": 8,   "vram_gb": 640,   "tflops_fp16": 2496, "h100_equiv": 2.52, "category": "Research"},
    {"name": "DGX H100 (8x H100 SXM)",              "gpus": 8,   "vram_gb": 640,   "tflops_fp16": 7920, "h100_equiv": 8.0,  "category": "Research"},
]


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


def compression_quality(compression_ratio, scenario=None):
    """Multiplicative quality factor for gradient compression.
    Interpolates log-linearly between known compression thresholds."""
    if scenario is None:
        scenario = DEFAULT_SCENARIO
    if compression_ratio <= 1:
        return 1.0
    # Known thresholds
    thresholds = sorted(COMPRESSION_QUALITY.keys())
    # Exact match
    if compression_ratio in COMPRESSION_QUALITY:
        return COMPRESSION_QUALITY[compression_ratio][scenario]
    # Log-linear interpolation
    log_cr = math.log10(compression_ratio)
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if lo <= compression_ratio <= hi:
            if lo <= 1:
                lo_log = 0
            else:
                lo_log = math.log10(lo)
            hi_log = math.log10(hi)
            t = (log_cr - lo_log) / (hi_log - lo_log) if hi_log > lo_log else 0
            q_lo = COMPRESSION_QUALITY[lo][scenario]
            q_hi = COMPRESSION_QUALITY[hi][scenario]
            return q_lo + t * (q_hi - q_lo)
    # Above max threshold: extrapolate from last two
    lo, hi = thresholds[-2], thresholds[-1]
    return COMPRESSION_QUALITY[hi][scenario]


def replica_penalty(n_replicas, params_billion):
    """Multiplicative penalty from averaging n_replicas' pseudo-gradients.
    Based on DiLoCo Scaling Laws (2503.09799) Table 5:
    M=8 at 2.4B: ~1.2% penalty. Penalty decreases with model size."""
    if n_replicas <= 1:
        return 1.0
    # ~0.5% per doubling at 2.4B, scales inversely with model size
    base_per_doubling = 0.005
    scale_adj = min(2.4, params_billion) / max(params_billion, 0.1)
    penalty = base_per_doubling * scale_adj * math.log2(n_replicas)
    return max(0.85, 1.0 - penalty)


def efficiency(h, params_billion, compression_ratio=1, scenario=None, n_replicas=1):
    """Combined efficiency: eta_H * eta_compression * eta_replicas.
    eta_H = max(0.4, 1 - alpha * log10(H))  [sync interval penalty]
    eta_compression = compression_quality()   [gradient compression quality]
    eta_replicas = replica_penalty()           [multi-replica averaging]"""
    a = alpha(params_billion)
    eta_h = 1.0 - a * math.log10(h)
    eta_h = max(0.4, eta_h)
    eta_c = compression_quality(compression_ratio, scenario)
    eta_r = replica_penalty(n_replicas, params_billion)
    return eta_h * eta_c * eta_r


def compute_scenario(config_name, n_nodes, compression=COMPRESSION,
                     time_seconds=None, bytes_per_param=BYTES_PER_PARAM,
                     bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                     bw_bps=None, latency_s=None, scenario=None):
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
        eta = efficiency(h_min, params_b, compression_ratio=compression,
                         scenario=scenario, n_replicas=n_nodes)

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
                                  regional_bw_bps=None, regional_latency_s=None,
                                  scenario=None):
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
                                bw_bps, latency_s, scenario=scenario)

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

    eta = efficiency(h_eff, params_b, compression_ratio=compression,
                     scenario=scenario, n_replicas=n_nodes)

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
                            time_seconds=None, scenario=None):
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
        eta = efficiency(h_min, total_params_b, compression_ratio=compression,
                         scenario=scenario, n_replicas=n_nodes)

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
                          latency_s=None, use_hierarchical=False, scenario=None):
    """Sweep bandwidth values and return results for each."""
    if latency_s is None:
        latency_s = LATENCY_S
    results = []
    for bw_mbps in BANDWIDTH_SWEEP_MBPS:
        bw_bps = bw_mbps * 1e6
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_bps=bw_bps, latency_s=latency_s,
                                              scenario=scenario)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_bps=bw_bps, latency_s=latency_s,
                                 scenario=scenario)
        r["bw_mbps"] = bw_mbps
        results.append(r)
    return results


def latency_sensitivity(config_name, n_nodes, compression=COMPRESSION,
                        bw_bps=None, use_hierarchical=False, scenario=None):
    """Sweep latency values and return results for each."""
    if bw_bps is None:
        bw_bps = BW_BPS
    results = []
    for name, lat_s in LATENCY_SCENARIOS.items():
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_bps=bw_bps, latency_s=lat_s,
                                              scenario=scenario)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_bps=bw_bps, latency_s=lat_s,
                                 scenario=scenario)
        r["latency_scenario"] = name
        r["latency_ms"] = lat_s * 1000
        results.append(r)
    return results


def deployment_profile_sweep(config_name, n_nodes, compression=COMPRESSION,
                             use_hierarchical=False, scenario=None):
    """Test all deployment profiles and return results."""
    results = []
    for name, profile in DEPLOYMENT_PROFILES.items():
        bw_bps = profile["bw_mbps"] * 1e6
        lat_s = profile["latency_ms"] / 1000.0
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_bps=bw_bps, latency_s=lat_s,
                                              scenario=scenario)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_bps=bw_bps, latency_s=lat_s,
                                 scenario=scenario)
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


def compute_generic_scenario(cfg, n_nodes, compression=COMPRESSION,
                             time_seconds=None, bytes_per_param=BYTES_PER_PARAM,
                             bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                             bw_bps=None, latency_s=None, scenario=None):
    """Compute scenario for an arbitrary config dict (not from named configs)."""
    if "bytes_per_param" in cfg:
        bytes_per_param = cfg["bytes_per_param"]
    if "bits_per_pseudo_grad" in cfg:
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_bps is None:
        bw_bps = BW_BPS
    if latency_s is None:
        latency_s = LATENCY_S

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    max_params_b = vram_gb / bytes_per_param
    params_b = max_params_b
    params = params_b * 1e9

    effective_flops = pflops * 1e15 * MFU
    t_comp = (6 * params * LOCAL_BATCH) / effective_flops
    v_bits = params * bits_per_pseudo_grad / compression
    t_sync_base = 2 * v_bits / bw_bps + latency_s
    f_n = straggler_factor(n_nodes)
    t_sync = t_sync_base * f_n

    if n_nodes == 1:
        h_min = 1
        eta = 1.0
    else:
        h_min = math.ceil(t_sync / t_comp)
        eta = efficiency(h_min, params_b, compression_ratio=compression,
                         scenario=scenario, n_replicas=n_nodes)

    c_actual = n_nodes * effective_flops * time_seconds
    c_local = c_actual * eta
    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "pflops_per_node": pflops,
        "vram_gb": vram_gb,
        "max_params_b": max_params_b,
        "h_min": h_min,
        "eta": eta,
        "c_local": c_local,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
    }


def find_nodes_for_target(cfg, target_flop, compression=COMPRESSION,
                          bytes_per_param=BYTES_PER_PARAM,
                          bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD):
    """Binary search for minimum nodes to reach target_flop."""
    if "bytes_per_param" in cfg:
        bytes_per_param = cfg["bytes_per_param"]
    if "bits_per_pseudo_grad" in cfg:
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    # Check if 1 node suffices
    r = compute_generic_scenario(cfg, 1, compression=compression,
                                 bytes_per_param=bytes_per_param,
                                 bits_per_pseudo_grad=bits_per_pseudo_grad)
    if r["c_local"] >= target_flop:
        return r

    # Binary search between 2 and 100000
    lo, hi = 2, 100000
    while lo < hi:
        mid = (lo + hi) // 2
        r = compute_generic_scenario(cfg, mid, compression=compression,
                                     bytes_per_param=bytes_per_param,
                                     bits_per_pseudo_grad=bits_per_pseudo_grad)
        if r["c_local"] >= target_flop:
            hi = mid
        else:
            lo = mid + 1

    return compute_generic_scenario(cfg, lo, compression=compression,
                                    bytes_per_param=bytes_per_param,
                                    bits_per_pseudo_grad=bits_per_pseudo_grad)


def print_countermeasure_ccc_threshold():
    """Analyze effect of lowering CCC compute threshold."""
    print("\n" + "=" * 100)
    print("COUNTERMEASURE: LOWERING CCC COMPUTE THRESHOLD")
    print("=" * 100)

    targets = [1e24, 1e25, 1e26]
    target_labels = ["10^24", "10^25", "10^26"]

    # A100-based (FP16) nodes
    print("\n--- A100 80GB nodes (FP16 training, max VRAM per compute) ---")
    print(f"\n  {'CCC Threshold':>20} | {'GPUs/node':>9} | {'PFLOPS':>7} | {'VRAM':>7} | "
          f"{'Max Model':>10} | {'H100-eq':>7}")
    print("  " + "-" * 75)
    for label, cfg in LOWERED_CCC_A100.items():
        model_b = cfg["vram_gb"] / BYTES_PER_PARAM
        print(f"  {label:>20} | {cfg['gpu_count']:>9} | {cfg['pflops']:>7.2f} | "
              f"{cfg['vram_gb']:>5} GB | {model_b:>7.0f}B | {cfg['h100_equiv']:>7.1f}")

    print(f"\n  {'CCC Threshold':>20} | ", end="")
    for tl in target_labels:
        print(f"{'N->' + tl:>18} | {'Cost':>8} | ", end="")
    print()
    print("  " + "-" * 105)

    for label, cfg in LOWERED_CCC_A100.items():
        print(f"  {label:>20} | ", end="")
        for target in targets:
            r = find_nodes_for_target(cfg, target)
            cost = r["cost_usd"]
            cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
            print(f"{r['n_nodes']:>18,} | {cost_str:>8} | ", end="")
        print()

    # H100-based (FP8) nodes
    print("\n--- H100 SXM nodes (FP8 training, max compute per CCC threshold) ---")
    print(f"\n  {'CCC Threshold':>20} | {'GPUs/node':>9} | {'FP8 PFLOPS':>10} | {'VRAM':>7} | "
          f"{'Max Model':>10} | {'H100-eq':>7}")
    print("  " + "-" * 80)
    for label, cfg in LOWERED_CCC_H100_FP8.items():
        model_b = cfg["vram_gb"] / cfg["bytes_per_param"]
        print(f"  {label:>20} | {cfg['gpu_count']:>9} | {cfg['pflops']:>10.2f} | "
              f"{cfg['vram_gb']:>5} GB | {model_b:>7.0f}B | {cfg['h100_equiv']:>7.1f}")

    print(f"\n  {'CCC Threshold':>20} | ", end="")
    for tl in target_labels:
        print(f"{'N->' + tl:>18} | {'Cost':>8} | ", end="")
    print()
    print("  " + "-" * 105)

    for label, cfg in LOWERED_CCC_H100_FP8.items():
        print(f"  {label:>20} | ", end="")
        for target in targets:
            r = find_nodes_for_target(cfg, target)
            cost = r["cost_usd"]
            cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
            print(f"{r['n_nodes']:>18,} | {cost_str:>8} | ", end="")
        print()


def print_countermeasure_memory_threshold():
    """Analyze effect of adding memory (VRAM) to CCC definition."""
    print("\n" + "=" * 100)
    print("COUNTERMEASURE: ADDING MEMORY (VRAM) THRESHOLD TO CCC DEFINITION")
    print("=" * 100)

    print("\n  Current exploit: 48x A100 80GB = 3,840 GB VRAM at 15.1 H100-equiv (under 16)")
    print("  Adding a VRAM threshold constrains the max node to min(compute_limit, memory_limit)")

    targets = [1e24, 1e25, 1e26]
    target_labels = ["10^24", "10^25", "10^26"]

    print(f"\n  {'VRAM Limit':>12} | {'Max A100s':>9} | {'Actual VRAM':>11} | {'PFLOPS':>7} | "
          f"{'Max Model':>10} | {'H100-eq':>7} | ", end="")
    for tl in target_labels:
        print(f"{'N->' + tl:>10} | {'Cost':>8} | ", end="")
    print()
    print("  " + "-" * 140)

    for mem_limit in MEMORY_THRESHOLDS_GB:
        # Max A100 80GB GPUs under both compute (16 H100-eq) and memory limits
        max_by_compute = 48  # floor(16 * 990 / 312) = 50, but we use 48 to stay under
        max_by_memory = mem_limit // 80  # 80 GB per A100
        n_gpus = min(max_by_compute, max_by_memory)
        if n_gpus < 1:
            n_gpus = 1

        cfg = {
            "gpu_count": n_gpus,
            "pflops": n_gpus * 312e12 / 1e15,
            "vram_gb": n_gpus * 80,
            "gpu_cost_usd": 15_000,
            "h100_equiv": n_gpus * 312 / 990,
        }
        model_b = cfg["vram_gb"] / BYTES_PER_PARAM

        print(f"  {mem_limit:>8} GB | {n_gpus:>9} | {cfg['vram_gb']:>8} GB | "
              f"{cfg['pflops']:>7.2f} | {model_b:>7.0f}B | {cfg['h100_equiv']:>7.1f} | ", end="")

        for target in targets:
            r = find_nodes_for_target(cfg, target)
            cost = r["cost_usd"]
            cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
            print(f"{r['n_nodes']:>10,} | {cost_str:>8} | ", end="")
        print()

    # Also show H100 FP8 under memory limits
    print(f"\n  H100 SXM (FP8) under memory limits:")
    print(f"  {'VRAM Limit':>12} | {'Max H100s':>9} | {'Actual VRAM':>11} | {'FP8 PFLOPS':>10} | "
          f"{'Max Model':>10} | {'H100-eq':>7} | ", end="")
    for tl in target_labels:
        print(f"{'N->' + tl:>10} | {'Cost':>8} | ", end="")
    print()
    print("  " + "-" * 145)

    for mem_limit in MEMORY_THRESHOLDS_GB:
        max_by_compute = 16
        max_by_memory = mem_limit // 80
        n_gpus = min(max_by_compute, max_by_memory)
        if n_gpus < 1:
            n_gpus = 1

        cfg = {
            "gpu_count": n_gpus,
            "pflops": n_gpus * 1980e12 / 1e15,
            "pflops_fp16": n_gpus * 990e12 / 1e15,
            "vram_gb": n_gpus * 80,
            "gpu_cost_usd": 30_000,
            "h100_equiv": n_gpus * 1.0,
            "bytes_per_param": 14,
            "bits_per_pseudo_grad": 8,
        }
        model_b = cfg["vram_gb"] / cfg["bytes_per_param"]

        print(f"  {mem_limit:>8} GB | {n_gpus:>9} | {cfg['vram_gb']:>8} GB | "
              f"{cfg['pflops']:>10.2f} | {model_b:>7.0f}B | {cfg['h100_equiv']:>7.1f} | ", end="")

        for target in targets:
            r = find_nodes_for_target(cfg, target)
            cost = r["cost_usd"]
            cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
            print(f"{r['n_nodes']:>10,} | {cost_str:>8} | ", end="")
        print()


def print_collateral_damage():
    """Show which legitimate computing systems would be caught by each threshold."""
    print("\n" + "=" * 100)
    print("COLLATERAL DAMAGE: LEGITIMATE SYSTEMS CAUGHT BY EACH THRESHOLD")
    print("=" * 100)

    compute_thresholds = [16, 8, 4, 2, 1]
    memory_thresholds = [2048, 1024, 512, 256]

    print(f"\n  {'System':>45} | {'GPUs':>4} | {'VRAM':>7} | {'H100-eq':>7} | ", end="")
    for ct in compute_thresholds:
        print(f"{'<'+str(ct):>4} | ", end="")
    print("| ", end="")
    for mt in memory_thresholds:
        print(f"{'<'+str(mt)+'G':>6} | ", end="")
    print()
    print("  " + "-" * 140)

    for sys in LEGITIMATE_SYSTEMS:
        print(f"  {sys['name']:>45} | {sys['gpus']:>4} | {sys['vram_gb']:>5}GB | "
              f"{sys['h100_equiv']:>7.2f} | ", end="")
        for ct in compute_thresholds:
            caught = "X" if sys["h100_equiv"] > ct else "."
            print(f"  {caught:>2} | ", end="")
        print("| ", end="")
        for mt in memory_thresholds:
            caught = "X" if sys["vram_gb"] > mt else "."
            print(f"    {caught:>2} | ", end="")
        print(f"  [{sys['category']}]")

    # Count caught at each threshold
    print(f"\n  {'Systems caught':>45} |      |         |         | ", end="")
    for ct in compute_thresholds:
        count = sum(1 for s in LEGITIMATE_SYSTEMS if s["h100_equiv"] > ct)
        print(f"{count:>4} | ", end="")
    print("| ", end="")
    for mt in memory_thresholds:
        count = sum(1 for s in LEGITIMATE_SYSTEMS if s["vram_gb"] > mt)
        print(f"  {count:>4} | ", end="")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("TREATY EVASION SCENARIO: Maximum Distributed Training Below CCC Threshold")
    print(f"Time limit: {TIME_YEARS} years ({TIME_SECONDS/86400:.0f} days)")
    print(f"WAN: {BW_MBPS} Mbps, {LATENCY_S*1000:.0f} ms latency")
    print(f"MFU: {MFU*100:.0f}%  |  Compression: {COMPRESSION}x  |  Streaming: Yes")
    print(f"Compression quality scenario: {DEFAULT_SCENARIO}")
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

    # ── PART 6: Treaty modification analysis ──────────────────────────────────

    print("\n" + "=" * 80)
    print("PART 6: TREATY MODIFICATION ANALYSIS")
    print("=" * 80)

    print_countermeasure_ccc_threshold()
    print_countermeasure_memory_threshold()
    print_collateral_damage()

    # -- PART 7: Compression quality sensitivity ---------------------------------

    print("\n" + "=" * 80)
    print("PART 7: COMPRESSION QUALITY SENSITIVITY")
    print("=" * 80)
    print(f"\nDefault scenario: {DEFAULT_SCENARIO}")
    print("Compression quality factors (multiplicative on eta):")
    for cr, factors in sorted(COMPRESSION_QUALITY.items()):
        print(f"  {cr:>4}x: optimistic={factors['optimistic']:.2f}, "
              f"expected={factors['expected']:.2f}, "
              f"conservative={factors['conservative']:.2f}")

    # Key scenarios under all three quality assumptions
    key_scenarios = [
        ("72 nodes, 48xA100, 16x comp", "48x A100 80GB", 72, 16, False),
        ("500 nodes, 48xA100, 16x comp", "48x A100 80GB", 500, 16, False),
        ("4000 nodes, 48xA100, 16x comp", "48x A100 80GB", 4000, 16, False),
        ("2000 nodes, H100 FP8, 16x comp", "16x H100 FP8", 2000, 16, False),
        ("2000 nodes, H100 FP8, 100x hier", "16x H100 FP8", 2000, 100, True),
        ("4000 nodes, 48xA100, 100x comp", "48x A100 80GB", 4000, 100, False),
    ]

    print(f"\n  {'Scenario':>40} | {'Opt eta':>7} | {'Opt C_local':>11} | "
          f"{'Exp eta':>7} | {'Exp C_local':>11} | "
          f"{'Con eta':>7} | {'Con C_local':>11}")
    print("  " + "-" * 115)

    for label, cfg_name, n, comp, hier in key_scenarios:
        results = {}
        for sc in ["optimistic", "expected", "conservative"]:
            if hier:
                r = compute_hierarchical_scenario(cfg_name, n, compression=comp,
                                                   scenario=sc)
            else:
                r = compute_scenario(cfg_name, n, compression=comp, scenario=sc)
            results[sc] = r

        opt, exp, con = results["optimistic"], results["expected"], results["conservative"]
        print(f"  {label:>40} | {opt['eta']:>7.3f} | {opt['c_local']:>11.2e} | "
              f"{exp['eta']:>7.3f} | {exp['c_local']:>11.2e} | "
              f"{con['eta']:>7.3f} | {con['c_local']:>11.2e}")

    # Bandwidth sensitivity under expected scenario (revised "3-6%" analysis)
    print("\n\n--- Bandwidth sensitivity WITH compression quality (expected scenario) ---")
    print("  Compare to PART 4 results (which used optimistic/no compression quality)")

    for label, cfg_name, n, comp, hier in [
        ("72 nodes, 48xA100, 16x", "48x A100 80GB", 72, 16, False),
        ("2000 nodes, H100 FP8, hier+100x", "16x H100 FP8", 2000, 100, True),
    ]:
        print(f"\n  {label}:")
        print(f"  {'BW (Mbps)':>10} | {'H':>5} | {'eta':>7} | {'C_local':>11} | {'x10^24':>7} | {'% of 1Gbps':>10}")

        results_bw = bandwidth_sensitivity(cfg_name, n, compression=comp,
                                           use_hierarchical=hier, scenario="expected")
        best_c = max(r['c_local'] for r in results_bw)
        for r in results_bw:
            h = r.get("h_eff", r.get("h_min", 1))
            h_str = f"{h:.0f}" if isinstance(h, float) else f"{h}"
            pct = r['c_local'] / best_c * 100
            print(f"  {r['bw_mbps']:>10} | {h_str:>5} | {r['eta']:>7.3f} | "
                  f"{r['c_local']:>11.2e} | {r['strict_threshold_multiple']:>7.1f} | {pct:>9.0f}%")
