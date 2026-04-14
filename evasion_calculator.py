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
    "50x A100 80GB": {
        "pflops": 50 * 312e12 / 1e15,   # 50 x 312 TFLOPS = 15.600 PFLOPS
        "vram_gb": 50 * 80,              # 4,000 GB
        "gpu_count": 50,
        "gpu_cost_usd": 7_000,           # ~$7k per A100 80GB (early 2026, secondary market)
        "h100_equiv": 50 * 312 / 990,    # ~15.8
    },
    "16x GH200": {
        "pflops": 15.84,                 # 16 x 990 TFLOPS
        "vram_gb": 16 * 144,             # 2,304 GB
        "gpu_count": 16,
        "gpu_cost_usd": 28_000,          # ~$28k per GH200 (early 2026 est.)
        "h100_equiv": 16.0,
    },
    "16x H100 SXM": {
        "pflops": 15.84,                 # 16 x 990 TFLOPS
        "vram_gb": 16 * 80,              # 1,280 GB
        "gpu_count": 16,
        "gpu_cost_usd": 25_000,          # ~$25k per H100 (early 2026)
        "h100_equiv": 16.0,
    },
    # Chinese chips (only available domestically in China; not export-available)
    "49x Ascend 910B": {
        "pflops": 49 * 320e12 / 1e15,   # 49 x 320 TFLOPS = 15.68 PFLOPS
        "vram_gb": 49 * 64,             # 3,136 GB HBM2e
        "gpu_count": 49,
        "gpu_cost_usd": 16_000,         # ¥110,000 (~$16k)
        "h100_equiv": 49 * 320 / 990,   # ~15.8
    },
    "26x Ascend 910C": {
        "pflops": 26 * 600e12 / 1e15,   # 26 x 600 TFLOPS = 15.60 PFLOPS (est.)
        "vram_gb": 26 * 128,            # 3,328 GB HBM (est.)
        "gpu_count": 26,
        "gpu_cost_usd": 26_000,         # ¥180,000 (~$26k)
        "h100_equiv": 26 * 600 / 990,   # ~15.8
    },
    # Google TPUs (BF16 TFLOPS treated as FP16-equivalent)
    "57x TPU v4": {
        "pflops": 57 * 275e12 / 1e15,   # 57 x 275 TFLOPS = 15.675 PFLOPS
        "vram_gb": 57 * 32,             # 1,824 GB HBM2e
        "gpu_count": 57,
        "gpu_cost_usd": 12_000,         # est. capital equiv. from cloud rental rates
        "h100_equiv": 57 * 275 / 990,   # ~15.8
    },
    "80x TPU v5e": {
        "pflops": 80 * 197e12 / 1e15,   # 80 x 197 TFLOPS = 15.76 PFLOPS
        "vram_gb": 80 * 16,             # 1,280 GB HBM
        "gpu_count": 80,
        "gpu_cost_usd": 6_000,          # est. capital equiv. from cloud rental rates
        "h100_equiv": 80 * 197 / 990,   # ~15.9
    },
    "34x TPU v5p": {
        "pflops": 34 * 459e12 / 1e15,   # 34 x 459 TFLOPS = 15.606 PFLOPS
        "vram_gb": 34 * 95,             # 3,230 GB HBM2e
        "gpu_count": 34,
        "gpu_cost_usd": 20_000,         # est. capital equiv. from cloud rental rates
        "h100_equiv": 34 * 459 / 990,   # ~15.8
    },
    "17x TPU v6e": {
        "pflops": 17 * 918e12 / 1e15,   # 17 x 918 TFLOPS = 15.606 PFLOPS
        "vram_gb": 17 * 32,             # 544 GB HBM
        "gpu_count": 17,
        "gpu_cost_usd": 25_000,         # est. capital equiv. from cloud rental rates
        "h100_equiv": 17 * 918 / 990,   # ~15.8
    },
}

# Network (asymmetric WAN — upload is typically the bottleneck)
BW_UP_MBPS = 100          # WAN upload bandwidth (Mbps)
BW_DOWN_MBPS = 500        # WAN download bandwidth (Mbps)
BW_UP_BPS = BW_UP_MBPS * 1e6    # bits/s
BW_DOWN_BPS = BW_DOWN_MBPS * 1e6  # bits/s
LATENCY_S = 0.1           # 100 ms RTT

# Training
MFU = 0.40
COMPRESSION = 150       # 4-bit quantization + 25% sparsification
LOCAL_BATCH = 131_072   # tokens per local step
BYTES_PER_PARAM = 16    # FP16 mixed-precision training
BITS_PER_PSEUDO_GRAD = 16  # FP16 pseudo-gradients before compression
STRAGGLER_MODE = "relay"   # "synchronous", "threshold", or "relay"

# Time
TIME_YEARS = 1.5
TIME_SECONDS = TIME_YEARS * 365.25 * 86400  # 47,335,400 seconds

# Node sweep
NODE_COUNTS = [1, 2, 4, 8, 16, 32, 72, 144, 500, 1000]
NODE_COUNTS_LARGE = [500, 1000, 2000, 3000, 4000, 5000]

# Replica penalty H-dependence
REPLICA_H_REF = 30   # Charles et al. (2025) experimental inner step count

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
        "gpu_cost_usd": 25_000,
        "h100_equiv": 16.0,              # CCC threshold uses FP16 capacity
        "bytes_per_param": 14,           # FP8: 1+1+4+4+4 = 14 bytes
        "bits_per_pseudo_grad": 8,       # FP8 pseudo-gradients (8 bits)
    },
    "16x GH200 FP8": {
        "pflops": 16 * 1980e12 / 1e15,  # 16 x 1980 TFLOPS FP8 = 31.68 PFLOPS
        "pflops_fp16": 15.84,             # For CCC threshold calculation
        "vram_gb": 16 * 144,             # 2,304 GB (144 GB per GH200)
        "gpu_count": 16,
        "gpu_cost_usd": 28_000,          # ~$28k per GH200 (early 2026 est.)
        "h100_equiv": 16.0,              # CCC threshold uses FP16 capacity
        "bytes_per_param": 14,           # FP8: 1+1+4+4+4 = 14 bytes
        "bits_per_pseudo_grad": 8,       # FP8 pseudo-gradients (8 bits)
    },
    # Chinese chips with FP8 support (only available domestically in China)
    "49x Ascend 910B FP8": {
        "pflops": 49 * 640e12 / 1e15,   # 49 x 640 TFLOPS FP8 = 31.36 PFLOPS
        "pflops_fp16": 49 * 320e12 / 1e15,
        "vram_gb": 49 * 64,             # 3,136 GB HBM2e
        "gpu_count": 49,
        "gpu_cost_usd": 16_000,         # ¥110,000 (~$16k)
        "h100_equiv": 49 * 320 / 990,
        "bytes_per_param": 14,
        "bits_per_pseudo_grad": 8,
    },
    "26x Ascend 910C FP8": {
        "pflops": 26 * 1200e12 / 1e15,  # 26 x 1200 TFLOPS FP8 = 31.20 PFLOPS (est.)
        "vram_gb": 26 * 128,            # 3,328 GB HBM (est.)
        "pflops_fp16": 26 * 600e12 / 1e15,
        "gpu_count": 26,
        "gpu_cost_usd": 26_000,         # ¥180,000 (~$26k)
        "h100_equiv": 26 * 600 / 990,
        "bytes_per_param": 14,
        "bits_per_pseudo_grad": 8,
    },
    # Google TPU v6e with FP8 support
    "17x TPU v6e FP8": {
        "pflops": 17 * 1836e12 / 1e15,  # 17 x 1836 TFLOPS FP8 = 31.21 PFLOPS
        "pflops_fp16": 17 * 918e12 / 1e15,
        "vram_gb": 17 * 32,             # 544 GB HBM
        "gpu_count": 17,
        "gpu_cost_usd": 25_000,         # est. capital equiv. from cloud rental rates
        "h100_equiv": 17 * 918 / 990,
        "bytes_per_param": 14,
        "bits_per_pseudo_grad": 8,
    },
}


# ── Countermeasure analysis configs ───────────────────────────────────────────

# Lowered CCC threshold: max A100-80GB node under each threshold
# A100 80GB: 312 TFLOPS FP16 each, $7k each (early 2026), 80 GB HBM each
# Max GPUs = floor(threshold_h100_equiv * 990 / 312)
LOWERED_CCC_A100 = {
    "16 H100-eq (current)": {"gpu_count": 50, "pflops": 50 * 312e12 / 1e15,
                              "vram_gb": 50 * 80, "gpu_cost_usd": 7_000,
                              "h100_equiv": 50 * 312 / 990},
    "8 H100-eq":  {"gpu_count": 25, "pflops": 25 * 312e12 / 1e15,
                    "vram_gb": 25 * 80, "gpu_cost_usd": 7_000,
                    "h100_equiv": 25 * 312 / 990},
    "4 H100-eq":  {"gpu_count": 12, "pflops": 12 * 312e12 / 1e15,
                    "vram_gb": 12 * 80, "gpu_cost_usd": 7_000,
                    "h100_equiv": 12 * 312 / 990},
    "2 H100-eq":  {"gpu_count": 6,  "pflops": 6 * 312e12 / 1e15,
                    "vram_gb": 6 * 80,  "gpu_cost_usd": 7_000,
                    "h100_equiv": 6 * 312 / 990},
    "1 H100-eq":  {"gpu_count": 3,  "pflops": 3 * 312e12 / 1e15,
                    "vram_gb": 3 * 80,  "gpu_cost_usd": 7_000,
                    "h100_equiv": 3 * 312 / 990},
}

# Lowered CCC threshold: max H100 SXM node under each threshold (FP8 compute)
# H100 SXM: 990 TFLOPS FP16 each (1980 FP8), $25k each (early 2026), 80 GB HBM each
LOWERED_CCC_H100_FP8 = {
    "16 H100-eq (current)": {"gpu_count": 16, "pflops": 16 * 1980e12 / 1e15,
                              "pflops_fp16": 15.84, "vram_gb": 16 * 80,
                              "gpu_cost_usd": 25_000, "h100_equiv": 16.0,
                              "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "8 H100-eq":  {"gpu_count": 8,  "pflops": 8 * 1980e12 / 1e15,
                    "pflops_fp16": 7.92,  "vram_gb": 8 * 80,
                    "gpu_cost_usd": 25_000, "h100_equiv": 8.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "4 H100-eq":  {"gpu_count": 4,  "pflops": 4 * 1980e12 / 1e15,
                    "pflops_fp16": 3.96,  "vram_gb": 4 * 80,
                    "gpu_cost_usd": 25_000, "h100_equiv": 4.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "2 H100-eq":  {"gpu_count": 2,  "pflops": 2 * 1980e12 / 1e15,
                    "pflops_fp16": 1.98,  "vram_gb": 2 * 80,
                    "gpu_cost_usd": 25_000, "h100_equiv": 2.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
    "1 H100-eq":  {"gpu_count": 1,  "pflops": 1 * 1980e12 / 1e15,
                    "pflops_fp16": 0.99,  "vram_gb": 1 * 80,
                    "gpu_cost_usd": 25_000, "h100_equiv": 1.0,
                    "bytes_per_param": 14, "bits_per_pseudo_grad": 8},
}

# Memory thresholds: VRAM limits that trigger CCC registration
MEMORY_THRESHOLDS_GB = [256, 512, 1024, 2048]

# Collateral damage: representative legitimate computing systems
# ── Compression quality model ────────────────────────────────────────────────
# Multiplicative penalty on eta from gradient compression quality loss.
# Based on literature review:
#   - FP4 (4x): lossless at 4B (Streaming DiLoCo 2501.18512), 15B (MuLoCo 2505.23725)
#   - 16x: FP4 + 4x sparsification; implicitly validated at 72B by Covenant
#   - 100x: 2-bit + TopK 3%; validated at 512M (SparseLoCo 2508.15706),
#           implicitly validated at 72B (Covenant uses 146x, arXiv:2603.08163)
#   - 500x: speculative; not demonstrated, similar extrapolation risk as
#           100x had pre-Covenant (from validated scale to ~3.4x beyond)
# "optimistic" = best-case (literature supports lossless at validated scale)
# "expected"   = central estimate accounting for remaining extrapolation to 100B+
# "conservative" = genuinely pessimistic given remaining uncertainty

COMPRESSION_QUALITY_NO_EF = {
    1:   {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    4:   {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    16:  {"optimistic": 1.00, "expected": 0.99, "conservative": 0.95},
    100: {"optimistic": 1.00, "expected": 0.98, "conservative": 0.90},
    150: {"optimistic": 0.99, "expected": 0.96, "conservative": 0.85},
    500: {"optimistic": 0.99, "expected": 0.95, "conservative": 0.75},
}

COMPRESSION_QUALITY_EF = {
    1:   {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    4:   {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    16:  {"optimistic": 1.00, "expected": 1.00, "conservative": 0.97},
    100: {"optimistic": 1.00, "expected": 0.99, "conservative": 0.93},
    150: {"optimistic": 1.00, "expected": 0.99, "conservative": 0.91},
    500: {"optimistic": 0.99, "expected": 0.96, "conservative": 0.80},
}

ERROR_FEEDBACK = True
COMPRESSION_QUALITY = COMPRESSION_QUALITY_EF  # backward-compat alias

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
    {"name": "Huawei Atlas 900 node (8x Ascend 910B)", "gpus": 8, "vram_gb": 512,  "tflops_fp16": 2560, "h100_equiv": 2.59, "category": "Research"},
    {"name": "Google Cloud TPU v4 pod slice (8 chips)", "gpus": 8, "vram_gb": 256,  "tflops_fp16": 2200, "h100_equiv": 2.22, "category": "Cloud"},
    {"name": "Google Cloud TPU v5e pod slice (8 chips)","gpus": 8, "vram_gb": 128,  "tflops_fp16": 1576, "h100_equiv": 1.59, "category": "Cloud"},
]


# ── Simulator formulas ────────────────────────────────────────────────────────

def straggler_factor(n, mode=None):
    """Straggler penalty factor for synchronous aggregation.
    Modes:
      "synchronous" — full penalty: f(n) = 1 + 0.05 * log2(n)
      "threshold"   — stragglers dropped: f(n) = 1.0
      "relay"       — async relay (e.g. R2): f(n) = 1 + 0.02 * log2(n)
    """
    if mode is None:
        mode = STRAGGLER_MODE
    if n <= 1:
        return 1.0
    if mode == "threshold":
        return 1.0
    if mode == "relay":
        return 1.0 + 0.02 * math.log2(n)
    # "synchronous" (default/legacy)
    return 1.0 + 0.05 * math.log2(n)


def alpha(params_billion):
    """alpha = 0.08 / (1 + log10(P/1e9) / 5), where P is in parameters."""
    log_p = math.log10(params_billion)  # already in billions
    return 0.08 / (1.0 + log_p / 5.0)


def _interpolate_quality(table, ratio, scenario=None):
    """Log-linear interpolation in a quality table keyed by compression ratio.
    Shared helper for pseudo-gradient and activation compression quality."""
    if scenario is None:
        scenario = DEFAULT_SCENARIO
    if ratio <= 1:
        return 1.0
    thresholds = sorted(table.keys())
    if ratio in table:
        return table[ratio][scenario]
    log_r = math.log10(ratio)
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if lo <= ratio <= hi:
            lo_log = 0 if lo <= 1 else math.log10(lo)
            hi_log = math.log10(hi)
            t = (log_r - lo_log) / (hi_log - lo_log) if hi_log > lo_log else 0
            return table[lo][scenario] + t * (table[hi][scenario] - table[lo][scenario])
    return table[thresholds[-1]][scenario]


def compression_quality(compression_ratio, scenario=None, error_feedback=None):
    """Multiplicative quality factor for pseudo-gradient compression."""
    if error_feedback is None:
        error_feedback = ERROR_FEEDBACK
    table = COMPRESSION_QUALITY_EF if error_feedback else COMPRESSION_QUALITY_NO_EF
    return _interpolate_quality(table, compression_ratio, scenario)


# ── Activation compression quality model ─────────────────────────────────────
# Per-stage-boundary quality factor for activation compression in PP mode.
# Unlike pseudo-gradient errors (which average across replicas), activation
# errors accumulate through the pipeline (forward + backward passes).
# Literature:
#   - FP8 (2x): universally near-lossless (COAT ICLR 2025; SWARM ICML 2023)
#   - 4-bit adaptive (4x): near-lossless (TAH-Quant 2025; GACT ICML 2022)
#   - 10-16x: structural methods needed; Protocol Models 2025: 100x lossless
#     at 8B via subspace decomposition, but not validated at 100B+
ACTIVATION_COMPRESSION_QUALITY = {
    1:   {"optimistic": 1.00, "expected": 1.00, "conservative": 1.00},
    2:   {"optimistic": 1.00, "expected": 1.00, "conservative": 0.99},
    4:   {"optimistic": 1.00, "expected": 0.995, "conservative": 0.96},
    10:  {"optimistic": 0.995, "expected": 0.98, "conservative": 0.90},
}

PP_COMPRESSION = 4      # Default: 4-bit activation quantization (well-validated)
MICRO_BATCHES = 8       # GPipe micro-batch count


def activation_compression_quality(pp_compression, pp_stages, scenario=None):
    """Quality factor for activation compression in PP mode.
    Errors compound at each stage boundary: 2*(S-1) boundaries (fwd + bwd).
    Returns per_boundary_quality ** (2*(S-1))."""
    if pp_compression <= 1 or pp_stages <= 1:
        return 1.0
    per_boundary = _interpolate_quality(ACTIVATION_COMPRESSION_QUALITY,
                                         pp_compression, scenario)
    n_boundaries = 2 * (pp_stages - 1)
    return per_boundary ** n_boundaries


def replica_loss_multiplier(n_replicas, params_billion, h=REPLICA_H_REF):
    """Loss multiplier from averaging n_replicas' pseudo-gradients.
    Uses power law fit to Charles et al. (2025) Table 4 (35M-2.4B, H=30):
      L(N,M)/L(N,1) = M^beta(N)  where  beta(N) = 1.0923 * N^(-0.2342)
    H-dependent scaling: beta(N,H) = beta_0(N) * ln(H) / ln(H_ref).
    At H=1 (DDP), beta=0 (no replica penalty). At H=H_ref=30, matches Charles.
    Validated on 4B and 10B at H=30. Extrapolated beyond 10B and H=30.
    The compute_*_scenario() functions convert this to a FLOP penalty
    (eta_replica) via Chinchilla decomposition and fold it into eta."""
    if n_replicas <= 1:
        return 1.0
    if h <= 1:
        return 1.0  # H=1 is DDP: no replica divergence
    params = params_billion * 1e9
    beta_base = 1.0923 * params ** (-0.2342)
    h_scale = math.log(h) / math.log(REPLICA_H_REF)
    beta = beta_base * h_scale
    return n_replicas ** beta


# ── Chinchilla scaling law (corrected) ────────────────────────────────────────
# Besiroglu et al. 2024, "Chinchilla Scaling: A Replication Attempt"
# (arXiv:2404.10102). Corrects Hoffmann et al. 2022 Approach 3 parameters.
# L(N,D) = E + A/N^alpha + B/D^beta
# Optimal ratio D* = 25.6*N (including outliers).

CHINCHILLA_TOKENS_PER_PARAM = 25.6

_CHIN_E = 1.8172
_CHIN_A = 482.01
_CHIN_ALPHA = 0.3478
_CHIN_B = 2085.43
_CHIN_BETA = 0.3658


def chinchilla_loss(params, tokens):
    """Predicted loss for N parameters trained on D tokens.
    params and tokens in raw counts (not billions/trillions)."""
    if params <= 0 or tokens <= 0:
        return float('inf')
    return _CHIN_E + _CHIN_A * params**(-_CHIN_ALPHA) + _CHIN_B * tokens**(-_CHIN_BETA)


def chinchilla_optimal_allocation(c_flop, overtrain_target=1.0):
    """Given total compute C = 6*N*D, find (N_opt, D_opt) minimizing loss.
    With D* = overtrain_target * 25.6 * N:
      C = 6 * N * (overtrain_target * 25.6 * N) = 153.6 * overtrain_target * N^2.
    At overtrain_target=1.0 (default), this is Chinchilla-optimal.
    At overtrain_target=3.0, models are 3x overtrained (industry-standard)."""
    tokens_per_param = CHINCHILLA_TOKENS_PER_PARAM * overtrain_target
    n_opt = math.sqrt(c_flop / (6 * tokens_per_param))
    d_opt = tokens_per_param * n_opt
    return n_opt, d_opt


def chinchilla_efficiency(params, tokens, c_flop, loss_multiplier=1.0,
                          overtrain_target=1.0):
    """Fraction of compute that is 'effective' vs reference-optimal allocation.
    Returns eta_chinchilla in [0, 1]. At reference-optimal returns ~1.0.
    overtrain_target: the overtraining ratio considered "optimal" (default 1.0).
      At 3.0, a 3x-overtrained model scores eta=1.0.
    loss_multiplier > 1.0 models quality degradation (e.g., from replica averaging).
    Method: binary search for C' such that L_ref(C') = L(params, tokens) * loss_multiplier."""
    l_actual = chinchilla_loss(params, tokens) * loss_multiplier
    # Find C_effective: compute needed at reference-optimal to reach same loss
    lo, hi = 1e10, c_flop * 10  # search up to 10x current compute
    for _ in range(200):
        mid = math.sqrt(lo * hi)  # geometric bisection
        n_mid, d_mid = chinchilla_optimal_allocation(mid, overtrain_target)
        l_mid = chinchilla_loss(n_mid, d_mid)
        if l_mid > l_actual:
            lo = mid
        else:
            hi = mid
    c_effective = math.sqrt(lo * hi)
    return min(1.0, c_effective / c_flop)


def efficiency(h, params_billion, compression_ratio=1, scenario=None, error_feedback=None):
    """Throughput efficiency: eta_H * eta_compression.
    eta_H = max(0.4, 1 - alpha * log10(H))  [sync interval penalty]
    eta_compression = compression_quality()   [gradient compression quality]
    error_feedback: if True, use error-feedback compression table (higher quality).
    Note: replica penalty (eta_replica) is computed separately in each
    compute_*_scenario() function via Chinchilla decomposition and folded
    into the returned eta."""
    a = alpha(params_billion)
    eta_h = 1.0 - a * math.log10(h)
    eta_h = max(0.4, eta_h)
    eta_c = compression_quality(compression_ratio, scenario, error_feedback=error_feedback)
    return eta_h * eta_c


def compute_scenario(config_name, n_nodes, compression=COMPRESSION,
                     time_seconds=None, bytes_per_param=BYTES_PER_PARAM,
                     bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                     bw_up_bps=None, bw_down_bps=None, latency_s=None,
                     scenario=None, target_params_b=None, h_override=None,
                     straggler_mode=None, error_feedback=None):
    """Compute all metrics for a given node configuration and node count.
    If target_params_b is specified, train that model size instead of max-VRAM.
    If h_override is specified, use that H instead of h_min (may be comm-bound).
    The model must fit on a single node (no PP).
    Returns eta including replica penalty, eta_chinchilla for overtraining only."""
    if config_name in CONFIGS:
        cfg = CONFIGS[config_name]
    else:
        cfg = CONFIGS_FP8[config_name]
        bytes_per_param = cfg["bytes_per_param"]
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_up_bps is None:
        bw_up_bps = BW_UP_BPS
    if bw_down_bps is None:
        bw_down_bps = BW_DOWN_BPS
    if latency_s is None:
        latency_s = LATENCY_S
    if straggler_mode is None:
        straggler_mode = STRAGGLER_MODE
    if error_feedback is None:
        error_feedback = ERROR_FEEDBACK

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    # Max dense model size
    max_params_b = vram_gb / bytes_per_param  # billions of params
    if target_params_b is not None:
        assert target_params_b <= max_params_b, \
            f"target_params_b={target_params_b}B exceeds max {max_params_b:.0f}B for {config_name}"
        params_b = target_params_b
    else:
        params_b = max_params_b  # Train the largest model that fits
    params = params_b * 1e9

    # Effective FLOPS
    effective_flops = pflops * 1e15 * MFU

    # Per-step compute time
    t_comp = (6 * params * LOCAL_BATCH) / effective_flops

    # Communication volume per sync (bits, after compression)
    v_bits = params * bits_per_pseudo_grad / compression

    # Sync time (base, before straggler): upload + download + latency
    t_sync_base = v_bits / bw_up_bps + v_bits / bw_down_bps + latency_s

    # Straggler factor
    f_n = straggler_factor(n_nodes, mode=straggler_mode)

    # Sync time with straggler
    t_sync = t_sync_base * f_n

    # Minimum H for compute-bound regime (streaming DiLoCo)
    if n_nodes == 1:
        h_min = 1
        h_used = h_override if h_override is not None else 1
        eta = 1.0
    else:
        h_min = math.ceil(t_sync / t_comp)
        h_used = h_override if h_override is not None else h_min
        eta = efficiency(h_used, params_b, compression_ratio=compression,
                         scenario=scenario, error_feedback=error_feedback)

    # Throughput: accounts for comm-bound at low H
    outer_step_time = max(h_used * t_comp, t_sync) if n_nodes > 1 else h_used * t_comp
    n_outer_steps = time_seconds / outer_step_time if outer_step_time > 0 else 0
    total_tokens = n_outer_steps * h_used * LOCAL_BATCH * n_nodes
    c_actual = 6 * params * total_tokens
    c_local = c_actual * eta

    # Training details
    chinchilla_tokens = CHINCHILLA_TOKENS_PER_PARAM * params
    overtraining_ratio = total_tokens / chinchilla_tokens if chinchilla_tokens > 0 else 0

    # Bandwidth / network metrics
    bw_duty_cycle = t_sync / outer_step_time if outer_step_time > 0 and n_nodes > 1 else 0
    comm_gb_per_sync = 2 * params * bits_per_pseudo_grad / compression / 8 / 1e9

    # Chinchilla-optimality + replica quality penalty
    loss_mult = replica_loss_multiplier(n_nodes, params_b, h=h_used) if n_nodes > 1 else 1.0
    eta_chin_full = chinchilla_efficiency(params, total_tokens, c_actual,
                                          loss_multiplier=loss_mult) if n_nodes > 1 else 1.0
    # Decompose: overtraining-only vs replica-only
    eta_chin_ot = chinchilla_efficiency(params, total_tokens, c_actual,
                                        loss_multiplier=1.0) if n_nodes > 1 else 1.0
    eta_replica = eta_chin_full / eta_chin_ot if eta_chin_ot > 0 else 0

    # Fold replica penalty into eta; C_local is now true local-equivalent
    eta = eta * eta_replica
    c_local = c_actual * eta
    c_quality = c_local * eta_chin_ot

    # Cost
    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    # Verify under CCC threshold
    h100_eq = cfg.get("h100_equiv", cfg.get("pflops_fp16", pflops) * 1000 / 990)
    assert h100_eq <= 16.01, f"{config_name} exceeds CCC threshold!"

    return {
        "config": config_name,
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "h100_equiv_per_node": cfg.get("h100_equiv", 0),
        "pflops_per_node": pflops,
        "vram_gb": vram_gb,
        "max_params_b": max_params_b,
        "params_b": params_b,
        "t_comp": t_comp,
        "t_sync_base": t_sync_base,
        "f_straggler": f_n,
        "t_sync": t_sync,
        "h_min": h_min,
        "h_used": h_used,
        "alpha": alpha(params_b) if n_nodes > 1 else 0,
        "eta": eta,
        "eta_replica": eta_replica,
        "c_actual": c_actual,
        "c_local": c_local,
        "eta_chinchilla": eta_chin_ot,
        "c_quality": c_quality,
        "total_tokens_T": total_tokens / 1e12,
        "chinchilla_tokens_T": chinchilla_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
        "compression": compression,
        "time_seconds": time_seconds,
        "bw_up_mbps": bw_up_bps / 1e6,
        "bw_down_mbps": bw_down_bps / 1e6,
        "latency_ms": latency_s * 1000,
        "bw_duty_cycle": bw_duty_cycle,
        "comm_gb_per_sync": comm_gb_per_sync,
    }


def compute_hierarchical_scenario(config_name, n_nodes, nodes_per_group=NODES_PER_GROUP,
                                  compression=COMPRESSION, time_seconds=None,
                                  bytes_per_param=BYTES_PER_PARAM,
                                  bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                                  bw_up_bps=None, bw_down_bps=None, latency_s=None,
                                  regional_bw_bps=None, regional_latency_s=None,
                                  scenario=None, cfg=None, target_params_b=None,
                                  straggler_mode=None, error_feedback=None):
    """Compute metrics for hierarchical DiLoCo (two-tier topology).
    If cfg is provided, use it directly instead of looking up config_name.
    If target_params_b is specified, train that model size instead of max-VRAM.
    Returns None if target_params_b exceeds single-node VRAM capacity.
    Returns eta including replica penalty, eta_chinchilla for overtraining only."""
    if cfg is None:
        if config_name in CONFIGS:
            cfg = CONFIGS[config_name]
        else:
            cfg = CONFIGS_FP8[config_name]
            bytes_per_param = cfg["bytes_per_param"]
            bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]
    else:
        if "bytes_per_param" in cfg:
            bytes_per_param = cfg["bytes_per_param"]
        if "bits_per_pseudo_grad" in cfg:
            bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_up_bps is None:
        bw_up_bps = BW_UP_BPS
    if bw_down_bps is None:
        bw_down_bps = BW_DOWN_BPS
    if latency_s is None:
        latency_s = LATENCY_S
    if regional_bw_bps is None:
        regional_bw_bps = REGIONAL_BW_BPS
    if regional_latency_s is None:
        regional_latency_s = REGIONAL_LATENCY_S
    if straggler_mode is None:
        straggler_mode = STRAGGLER_MODE
    if error_feedback is None:
        error_feedback = ERROR_FEEDBACK

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    max_params_b = vram_gb / bytes_per_param
    params_b = target_params_b if target_params_b is not None else max_params_b
    if params_b > max_params_b:
        return None
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
        return compute_generic_scenario(cfg, n_nodes, compression, time_seconds,
                                        bytes_per_param, bits_per_pseudo_grad,
                                        bw_up_bps=bw_up_bps, bw_down_bps=bw_down_bps,
                                        latency_s=latency_s, scenario=scenario,
                                        target_params_b=target_params_b,
                                        straggler_mode=straggler_mode,
                                        error_feedback=error_feedback)

    # Regional sync (fast LAN)
    f_regional = straggler_factor(nodes_per_group, mode=straggler_mode)
    t_regional_sync = (2 * v_bits / regional_bw_bps + regional_latency_s) * f_regional
    h_inner_min = max(1, math.ceil(t_regional_sync / t_comp))

    # Global sync (slow WAN)
    f_global = straggler_factor(n_groups, mode=straggler_mode)
    t_global_sync = (v_bits / bw_up_bps + v_bits / bw_down_bps + latency_s) * f_global

    # Regional cycle time (streaming)
    t_regional_cycle = max(h_inner_min * t_comp, t_regional_sync)

    # Minimum H_regional for compute-bound global sync
    h_regional_min = max(1, math.ceil(t_global_sync / t_regional_cycle))

    # Effective H (hierarchical formula)
    h_eff = h_inner_min * math.sqrt(h_regional_min)

    eta = efficiency(h_eff, params_b, compression_ratio=compression,
                     scenario=scenario, error_feedback=error_feedback)

    # Total local-equivalent FLOPs
    c_actual = n_nodes * effective_flops * time_seconds
    c_local = c_actual * eta

    # Training details
    total_tokens = c_actual / (6 * params)
    chinchilla_tokens = CHINCHILLA_TOKENS_PER_PARAM * params
    overtraining_ratio = total_tokens / chinchilla_tokens

    # Chinchilla-optimality + replica quality penalty
    loss_mult = replica_loss_multiplier(n_nodes, params_b, h=h_eff)
    eta_chin_full = chinchilla_efficiency(params, total_tokens, c_actual,
                                          loss_multiplier=loss_mult)
    # Decompose: overtraining-only vs replica-only
    eta_chin_ot = chinchilla_efficiency(params, total_tokens, c_actual,
                                        loss_multiplier=1.0)
    eta_replica = eta_chin_full / eta_chin_ot if eta_chin_ot > 0 else 0

    # Fold replica penalty into eta; C_local is now true local-equivalent
    eta = eta * eta_replica
    c_local = c_actual * eta
    c_quality = c_local * eta_chin_ot

    # Cost
    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "config": (config_name or "custom") + " (hierarchical)",
        "mode": f"Hier {nodes_per_group}x{n_groups}",
        "n_nodes": n_nodes,
        "n_groups": n_groups,
        "nodes_per_group": nodes_per_group,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "h100_equiv_per_node": cfg.get("h100_equiv", 0),
        "pflops_per_node": pflops,
        "vram_gb": vram_gb,
        "max_params_b": max_params_b,
        "params_b": params_b,
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
        "eta_replica": eta_replica,
        "c_actual": c_actual,
        "c_local": c_local,
        "eta_chinchilla": eta_chin_ot,
        "c_quality": c_quality,
        "total_tokens_T": total_tokens / 1e12,
        "chinchilla_tokens_T": chinchilla_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
        "compression": compression,
    }


def compute_moe_ep_scenario(config_name, n_nodes, total_params_b, active_params_b,
                            n_moe_layers=32, compression=COMPRESSION,
                            time_seconds=None, scenario=None,
                            straggler_mode=None, error_feedback=None):
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
    if straggler_mode is None:
        straggler_mode = STRAGGLER_MODE
    if error_feedback is None:
        error_feedback = ERROR_FEEDBACK

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
    f_n = straggler_factor(n_nodes, mode=straggler_mode)
    t_sync_base = v_bits / BW_UP_BPS + v_bits / BW_DOWN_BPS + LATENCY_S
    t_sync = t_sync_base * f_n

    # Minimum H
    t_step = t_comp + t_ep
    if n_nodes <= 1:
        h_min = 1
        eta = 1.0
    else:
        h_min = math.ceil(t_sync / t_step)
        eta = efficiency(h_min, total_params_b, compression_ratio=compression,
                         scenario=scenario, error_feedback=error_feedback)

    # Total local-equivalent FLOPs (based on active params compute rate)
    # Adjust for EP latency overhead
    ep_overhead = t_ep / (t_comp + t_ep)
    c_actual = n_nodes * effective_flops * time_seconds * (1 - ep_overhead)
    c_local = c_actual * eta

    # Training details (based on active params)
    total_tokens = c_actual / (6 * params_active)
    chinchilla_tokens_active = CHINCHILLA_TOKENS_PER_PARAM * params_active
    overtraining_ratio = total_tokens / chinchilla_tokens_active

    # Chinchilla-optimality + replica quality penalty
    loss_mult = replica_loss_multiplier(n_nodes, total_params_b, h=h_min) if n_nodes > 1 else 1.0
    eta_chin_full = chinchilla_efficiency(params_active, total_tokens, c_actual,
                                          loss_multiplier=loss_mult) if n_nodes > 1 else 1.0
    # Decompose: overtraining-only vs replica-only
    eta_chin_ot = chinchilla_efficiency(params_active, total_tokens, c_actual,
                                        loss_multiplier=1.0) if n_nodes > 1 else 1.0
    eta_replica = eta_chin_full / eta_chin_ot if eta_chin_ot > 0 else 0

    # Fold replica penalty into eta; C_local is now true local-equivalent
    eta = eta * eta_replica
    c_local = c_actual * eta
    c_quality = c_local * eta_chin_ot

    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "config": config_name + f" (MoE {total_params_b:.0f}B/{active_params_b:.0f}B)",
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "total_params_b": total_params_b,
        "active_params_b": active_params_b,
        "params_b": active_params_b,
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
        "eta_replica": eta_replica,
        "c_actual": c_actual,
        "c_local": c_local,
        "eta_chinchilla": eta_chin_ot,
        "c_quality": c_quality,
        "total_tokens_T": total_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
    }


# ── PP-Group DiLoCo ───────────────────────────────────────────────────────────

# PP groups co-locate within a region — better bandwidth/latency than WAN
PP_BW_BPS = 1e9         # 1 Gbps (regional interconnect)
PP_LATENCY_S = 0.020    # 20 ms (same-continent latency)


def compute_pp_diloco_scenario(config_name, n_nodes, target_params_b,
                                pp_compression=PP_COMPRESSION,
                                micro_batches=MICRO_BATCHES,
                                compression=COMPRESSION,
                                bytes_per_param=BYTES_PER_PARAM,
                                bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                                time_seconds=None, bw_up_bps=None, bw_down_bps=None,
                                latency_s=None, pp_bw_bps=None, pp_latency_s=None,
                                scenario=None, cfg=None,
                                straggler_mode=None, error_feedback=None):
    """PP-Group DiLoCo: pipeline parallelism within co-located groups,
    DiLoCo synchronization across groups over WAN.

    Allows training models larger than single-node VRAM by sharding across
    pipeline stages. Each group of S co-located nodes holds the full model;
    G = N//S groups run DiLoCo outer loop.

    If cfg is provided, use it directly instead of looking up config_name.
    Returns None if insufficient nodes to form at least 2 groups.
    Returns eta including replica penalty, eta_chinchilla for overtraining only."""
    if cfg is None:
        if config_name in CONFIGS:
            cfg = CONFIGS[config_name]
        else:
            cfg = CONFIGS_FP8[config_name]
            bytes_per_param = cfg["bytes_per_param"]
            bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]
    else:
        if "bytes_per_param" in cfg:
            bytes_per_param = cfg["bytes_per_param"]
        if "bits_per_pseudo_grad" in cfg:
            bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_up_bps is None:
        bw_up_bps = BW_UP_BPS
    if bw_down_bps is None:
        bw_down_bps = BW_DOWN_BPS
    if latency_s is None:
        latency_s = LATENCY_S
    if pp_bw_bps is None:
        pp_bw_bps = PP_BW_BPS
    if pp_latency_s is None:
        pp_latency_s = PP_LATENCY_S
    if straggler_mode is None:
        straggler_mode = STRAGGLER_MODE
    if error_feedback is None:
        error_feedback = ERROR_FEEDBACK

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    # Pipeline stages needed
    mem_bytes = target_params_b * 1e9 * bytes_per_param
    pp_stages = math.ceil(mem_bytes / (vram_gb * 1e9))
    if pp_stages <= 1:
        # Model fits on one node — use regular DiLoCo instead
        return None

    # Groups
    n_groups = n_nodes // pp_stages
    if n_groups < 2:
        return None  # Not enough nodes for DiLoCo across groups

    params = target_params_b * 1e9
    bytes_per_value = bytes_per_param / 8  # approximate: FP16=2, FP8=1.75, FP4=1.625

    # Compute time: each stage processes 1/S of the model for all micro-batches
    effective_flops = pflops * 1e15 * MFU
    flops_per_step = 6 * params * LOCAL_BATCH
    # Wall-clock time for one node to compute the full model (no PP)
    compute_per_step_full = flops_per_step / effective_flops
    # In GPipe, each pipeline "tick" = one stage processing one micro-batch
    # Per-stage compute = full_model / S; per-micro = that / M
    compute_per_micro = compute_per_step_full / (pp_stages * micro_batches)

    # Hidden dimension approximation (for activation size)
    hidden_dim = 0.03 * math.sqrt(params)

    # Activation bits per step (forward pass activation tensor at stage boundary)
    # Shape: [local_batch, hidden_dim] in the given precision, compressed by pp_compression
    precision_bytes = 2 if bytes_per_param >= 16 else (1 if bytes_per_param >= 14 else 0.5)
    activation_bits = (LOCAL_BATCH * hidden_dim * precision_bytes * 8) / pp_compression

    # PP communication time per micro-batch (activation transfer between stages)
    comm_per_micro = (2 * activation_bits / micro_batches) / pp_bw_bps

    # GPipe bubble formula: (M + S - 1) micro-steps
    pp_straggler = straggler_factor(pp_stages, mode=straggler_mode)
    pp_step_time = (micro_batches + pp_stages - 1) * (
        compute_per_micro + (comm_per_micro + pp_latency_s) * pp_straggler
    )

    # DiLoCo sync across groups (pseudo-gradients over WAN)
    v_bits = params * bits_per_pseudo_grad / compression
    f_n = straggler_factor(n_groups, mode=straggler_mode)
    t_sync_base = v_bits / bw_up_bps + v_bits / bw_down_bps + latency_s
    t_sync = t_sync_base * f_n

    # Minimum H (inner steps before DiLoCo sync)
    h_min = max(1, math.ceil(t_sync / pp_step_time))

    # Efficiency components (throughput only; replica penalty via Chinchilla)
    eta_h_c_r = efficiency(h_min, target_params_b, compression_ratio=compression,
                           scenario=scenario, error_feedback=error_feedback)
    eta_act = activation_compression_quality(pp_compression, pp_stages, scenario)
    eta = eta_h_c_r * eta_act

    # PP bubble overhead (fraction of time in bubble)
    bubble_frac = (pp_stages - 1) / (micro_batches + pp_stages - 1)

    # Tokens and compute (wall-clock based)
    # One outer step = max(H * pp_step_time, t_sync) with streaming
    outer_step_time = max(h_min * pp_step_time, t_sync)
    n_outer_steps = time_seconds / outer_step_time
    total_tokens = n_outer_steps * h_min * LOCAL_BATCH * n_groups

    c_actual = 6 * params * total_tokens
    c_local = c_actual * eta

    # Chinchilla-optimality + replica quality penalty
    loss_mult = replica_loss_multiplier(n_groups, target_params_b, h=h_min) if n_groups > 1 else 1.0
    eta_chin_full = chinchilla_efficiency(params, total_tokens, c_actual,
                                          loss_multiplier=loss_mult) if n_groups > 1 else 1.0
    # Decompose: overtraining-only vs replica-only
    eta_chin_ot = chinchilla_efficiency(params, total_tokens, c_actual,
                                        loss_multiplier=1.0) if n_groups > 1 else 1.0
    eta_replica = eta_chin_full / eta_chin_ot if eta_chin_ot > 0 else 0

    # Fold replica penalty into eta; C_local is now true local-equivalent
    eta = eta * eta_replica
    c_local = c_actual * eta
    c_quality = c_local * eta_chin_ot

    # Overtraining
    chinchilla_tokens = CHINCHILLA_TOKENS_PER_PARAM * params
    overtraining_ratio = total_tokens / chinchilla_tokens

    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "config": (config_name or "custom") + f" (PP-DiLoCo {pp_stages}x{n_groups})",
        "mode": f"PP-Group DiLoCo ({pp_stages} stages x {n_groups} groups)",
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "params_b": target_params_b,
        "pp_stages": pp_stages,
        "n_groups": n_groups,
        "pp_step_time": pp_step_time,
        "bubble_frac": bubble_frac,
        "t_sync": t_sync,
        "f_straggler": f_n,
        "h_min": h_min,
        "eta_h_c_r": eta_h_c_r,
        "eta_activation": eta_act,
        "eta": eta,
        "eta_replica": eta_replica,
        "eta_chinchilla": eta_chin_ot,
        "c_actual": c_actual,
        "c_local": c_local,
        "c_quality": c_quality,
        "total_tokens_T": total_tokens / 1e12,
        "chinchilla_tokens_T": chinchilla_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
        "pp_compression": pp_compression,
        "compression": compression,
    }


# ── Model size sweep ──────────────────────────────────────────────────────────

def sweep_model_sizes(config_name, n_nodes, compression=COMPRESSION,
                      scenario=None, time_seconds=None,
                      straggler_mode=None, error_feedback=None):
    """Evaluate multiple model sizes and return all results.
    Includes DiLoCo candidates (fit on one node) and PP-DiLoCo candidates
    (require pipeline parallelism). Returns list of result dicts sorted by
    C_quality descending."""
    if config_name in CONFIGS:
        cfg = CONFIGS[config_name]
        bytes_per_param = BYTES_PER_PARAM
    else:
        cfg = CONFIGS_FP8[config_name]
        bytes_per_param = cfg["bytes_per_param"]

    max_single_b = cfg["vram_gb"] / bytes_per_param
    results = []

    # DiLoCo candidates: various fractions of max single-node model
    for frac in [0.25, 0.50, 0.75, 1.0]:
        target = frac * max_single_b
        try:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 scenario=scenario, time_seconds=time_seconds,
                                 target_params_b=target,
                                 straggler_mode=straggler_mode,
                                 error_feedback=error_feedback)
            r["mode_type"] = "DiLoCo"
            r["pp_stages"] = 1
            r["n_groups"] = n_nodes
            r["bubble_frac"] = 0.0
            r["eta_activation"] = 1.0
            results.append(r)
        except Exception:
            pass

    # PP-DiLoCo candidates: models larger than single-node VRAM
    for mult in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        target = mult * max_single_b
        r = compute_pp_diloco_scenario(config_name, n_nodes,
                                        target_params_b=target,
                                        compression=compression,
                                        scenario=scenario,
                                        time_seconds=time_seconds,
                                        straggler_mode=straggler_mode,
                                        error_feedback=error_feedback)
        if r:
            r["mode_type"] = "PP-DiLoCo"
            results.append(r)

    # Sort by C_quality descending
    results.sort(key=lambda x: x.get("c_quality", 0), reverse=True)
    return results


def print_model_size_sweep(config_name, n_nodes, compression=COMPRESSION,
                           scenario=None):
    """Print a model size optimization table for given config and node count."""
    if config_name in CONFIGS:
        cfg = CONFIGS[config_name]
        bytes_per_param = BYTES_PER_PARAM
    else:
        cfg = CONFIGS_FP8[config_name]
        bytes_per_param = cfg["bytes_per_param"]

    max_b = cfg["vram_gb"] / bytes_per_param
    c_budget = n_nodes * cfg["pflops"] * 1e15 * MFU * TIME_SECONDS
    n_opt, d_opt = chinchilla_optimal_allocation(c_budget)

    print(f"\n--- {n_nodes} nodes, {config_name} (Chinchilla-optimal: ~{n_opt/1e9:.0f}B, max single-node: {max_b:.0f}B) ---")
    print(f"\n  {'Mode':>12} | {'Model':>7} | {'PP':>2} | {'Groups':>6} | {'OT':>6} | "
          f"{'eta':>5} | {'eta_act':>7} | {'eta_chin':>8} | {'C_local':>10} | {'C_quality':>10}")
    print("  " + "-" * 105)

    results = sweep_model_sizes(config_name, n_nodes, compression=compression,
                                scenario=scenario)
    best = results[0] if results else None

    for r in results:
        mode = r.get("mode_type", "DiLoCo")
        model_str = f"{r['params_b']:.0f}B"
        pp = r.get("pp_stages", 1)
        groups = r.get("n_groups", n_nodes)
        ot = r.get("overtraining_ratio", 0)
        eta = r.get("eta", 0)
        eta_act = r.get("eta_activation", 1.0)
        eta_chin = r.get("eta_chinchilla", 1.0)
        c_local = r.get("c_local", 0)
        c_quality = r.get("c_quality", 0)
        marker = " *" if r is best else ""

        print(f"  {mode:>12} | {model_str:>7} | {pp:>2} | {groups:>6} | {ot:>5.1f}x | "
              f"{eta:>5.3f} | {eta_act:>7.3f} | {eta_chin:>8.3f} | {c_local:>10.2e} | "
              f"{c_quality:>10.2e}{marker}")

    if best:
        print(f"\n  * Best quality-adjusted compute: {best['params_b']:.0f}B "
              f"({best.get('mode_type', 'DiLoCo')}), C_quality = {best['c_quality']:.2e}")


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
          f"{'Tokens':>7} | {'OT ratio':>8} | {'eta_chin':>8} | {'C_quality':>10}")
    print("-" * 135)

    for n in NODE_COUNTS:
        r = compute_scenario(config_name, n)
        cost_str = f"${r['cost_usd']/1e6:.1f}M"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.1f}x"
        model_str = f"{r['max_params_b']:.0f}B"
        tokens_str = f"{r['total_tokens_T']:.1f}T"
        ot_str = f"{r['overtraining_ratio']:.1f}x"
        echin_str = f"{r['eta_chinchilla']:.3f}"
        cq_str = f"{r['c_quality']:.2e}"

        print(f"{n:>5} | {r['total_gpus']:>7,} | {cost_str:>8} | {r['f_straggler']:>5.3f} | "
              f"{r['h_min']:>5} | {r['eta']:>5.3f} | {c_str:>10} | {mult_str:>7} | "
              f"{model_str:>6} | {tokens_str:>7} | {ot_str:>8} | {echin_str:>8} | {cq_str:>10}")


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
    rA = compute_scenario("50x A100 80GB", 4000)
    scenarios.append(("A: Flat, A100 FP16, 16x comp", rA))

    # B: Hierarchical, 48x A100 FP16, 16x compression
    rB = compute_hierarchical_scenario("50x A100 80GB", 4000)
    scenarios.append(("B: Hierarchical, A100 FP16, 16x", rB))

    # C: Flat DiLoCo, H100 FP8, 16x compression
    rC = compute_scenario("16x H100 FP8", 2000)
    scenarios.append(("C: Flat, H100 FP8, 16x comp", rC))

    # D: Hierarchical, H100 FP8, 16x compression
    rD = compute_hierarchical_scenario("16x H100 FP8", 2000)
    scenarios.append(("D: Hier, H100 FP8, 16x comp", rD))

    # E: Flat, A100 FP16, 100x compression
    rE = compute_scenario("50x A100 80GB", 4000, compression=100)
    scenarios.append(("E: Flat, A100 FP16, 100x comp", rE))

    # F: Hierarchical + 100x compression + FP8 H100
    rF = compute_hierarchical_scenario("16x H100 FP8", 2000, compression=100)
    scenarios.append(("F: Hier+100x, H100 FP8", rF))

    # G: MoE + EP (600B total, 100B active)
    rG = compute_moe_ep_scenario("50x A100 80GB", 4000, total_params_b=600,
                                  active_params_b=100)
    scenarios.append(("G: MoE+EP 600B/100B, A100", rG))

    # H: PP-Group DiLoCo, 48x A100, 960B (optimal from sweep)
    rH = compute_pp_diloco_scenario("50x A100 80GB", 4000, target_params_b=960)
    if rH:
        scenarios.append(("H: PP-DiLoCo 960B, A100", rH))

    # I: PP-Group DiLoCo, H100 FP8, 480B
    rI = compute_pp_diloco_scenario("16x H100 FP8", 2000, target_params_b=480)
    if rI:
        scenarios.append(("I: PP-DiLoCo 480B, H100 FP8", rI))

    print(f"\n{'Config':>35} | {'Nodes':>5} | {'GPUs':>7} | {'Cost':>8} | "
          f"{'Model':>18} | {'eta':>5} | {'C_local':>10} | {'x10^24':>7} | "
          f"{'eta_chin':>8} | {'C_quality':>10}")
    print("-" * 145)

    for label, r in scenarios:
        cost = r['cost_usd']
        cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
        c_str = f"{r['c_local']:.2e}"
        mult_str = f"{r['strict_threshold_multiple']:.0f}x"
        if 'total_params_b' in r:
            model_str = f"{r['total_params_b']:.0f}B MoE ({r['active_params_b']:.0f}B act)"
        elif r.get('pp_stages', 1) > 1:
            model_str = f"{r['params_b']:.0f}B PP-{r['pp_stages']}x{r['n_groups']}"
        else:
            model_str = f"{r.get('max_params_b', r.get('params_b', 0)):.0f}B dense"
        echin = r.get('eta_chinchilla', 1.0)
        cq = r.get('c_quality', r['c_local'])

        print(f"{label:>35} | {r['n_nodes']:>5} | {r['total_gpus']:>7,} | "
              f"{cost_str:>8} | {model_str:>18} | {r['eta']:>5.3f} | "
              f"{c_str:>10} | {mult_str:>7} | {echin:>8.3f} | {cq:>10.2e}")


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
                          latency_s=None, use_hierarchical=False, scenario=None,
                          straggler_mode=None, error_feedback=None):
    """Sweep bandwidth values and return results for each."""
    if latency_s is None:
        latency_s = LATENCY_S
    results = []
    for bw_mbps in BANDWIDTH_SWEEP_MBPS:
        bw_bps = bw_mbps * 1e6  # symmetric for sweep
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_up_bps=bw_bps, bw_down_bps=bw_bps,
                                              latency_s=latency_s,
                                              scenario=scenario,
                                              straggler_mode=straggler_mode,
                                              error_feedback=error_feedback)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_up_bps=bw_bps, bw_down_bps=bw_bps,
                                 latency_s=latency_s,
                                 scenario=scenario,
                                 straggler_mode=straggler_mode,
                                 error_feedback=error_feedback)
        r["bw_mbps"] = bw_mbps
        results.append(r)
    return results


def latency_sensitivity(config_name, n_nodes, compression=COMPRESSION,
                        bw_up_bps=None, bw_down_bps=None,
                        use_hierarchical=False, scenario=None,
                        straggler_mode=None, error_feedback=None):
    """Sweep latency values and return results for each."""
    if bw_up_bps is None:
        bw_up_bps = BW_UP_BPS
    if bw_down_bps is None:
        bw_down_bps = BW_DOWN_BPS
    results = []
    for name, lat_s in LATENCY_SCENARIOS.items():
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_up_bps=bw_up_bps,
                                              bw_down_bps=bw_down_bps,
                                              latency_s=lat_s,
                                              scenario=scenario,
                                              straggler_mode=straggler_mode,
                                              error_feedback=error_feedback)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_up_bps=bw_up_bps, bw_down_bps=bw_down_bps,
                                 latency_s=lat_s,
                                 scenario=scenario,
                                 straggler_mode=straggler_mode,
                                 error_feedback=error_feedback)
        r["latency_scenario"] = name
        r["latency_ms"] = lat_s * 1000
        results.append(r)
    return results


def deployment_profile_sweep(config_name, n_nodes, compression=COMPRESSION,
                             use_hierarchical=False, scenario=None,
                             straggler_mode=None, error_feedback=None):
    """Test all deployment profiles and return results."""
    results = []
    for name, profile in DEPLOYMENT_PROFILES.items():
        bw_bps = profile["bw_mbps"] * 1e6  # symmetric for profiles
        lat_s = profile["latency_ms"] / 1000.0
        if use_hierarchical:
            r = compute_hierarchical_scenario(config_name, n_nodes,
                                              compression=compression,
                                              bw_up_bps=bw_bps, bw_down_bps=bw_bps,
                                              latency_s=lat_s,
                                              scenario=scenario,
                                              straggler_mode=straggler_mode,
                                              error_feedback=error_feedback)
        else:
            r = compute_scenario(config_name, n_nodes, compression=compression,
                                 bw_up_bps=bw_bps, bw_down_bps=bw_bps,
                                 latency_s=lat_s, scenario=scenario,
                                 straggler_mode=straggler_mode,
                                 error_feedback=error_feedback)
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
                             bw_up_bps=None, bw_down_bps=None, latency_s=None,
                             scenario=None, target_params_b=None, h_override=None,
                             straggler_mode=None, error_feedback=None):
    """Compute scenario for an arbitrary config dict (not from named configs).
    If target_params_b is specified, train that model size instead of max-VRAM.
    If h_override is specified, use that H instead of h_min (may be comm-bound).
    Returns None if target_params_b exceeds single-node VRAM capacity."""
    if "bytes_per_param" in cfg:
        bytes_per_param = cfg["bytes_per_param"]
    if "bits_per_pseudo_grad" in cfg:
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    if time_seconds is None:
        time_seconds = TIME_SECONDS
    if bw_up_bps is None:
        bw_up_bps = BW_UP_BPS
    if bw_down_bps is None:
        bw_down_bps = BW_DOWN_BPS
    if latency_s is None:
        latency_s = LATENCY_S
    if straggler_mode is None:
        straggler_mode = STRAGGLER_MODE
    if error_feedback is None:
        error_feedback = ERROR_FEEDBACK

    pflops = cfg["pflops"]
    vram_gb = cfg["vram_gb"]

    max_params_b = vram_gb / bytes_per_param
    params_b = target_params_b if target_params_b is not None else max_params_b
    if params_b > max_params_b:
        return None
    params = params_b * 1e9

    effective_flops = pflops * 1e15 * MFU
    t_comp = (6 * params * LOCAL_BATCH) / effective_flops
    v_bits = params * bits_per_pseudo_grad / compression
    t_sync_base = v_bits / bw_up_bps + v_bits / bw_down_bps + latency_s
    f_n = straggler_factor(n_nodes, mode=straggler_mode)
    t_sync = t_sync_base * f_n

    if n_nodes == 1:
        h_min = 1
        h_used = h_override if h_override is not None else 1
        eta = 1.0
    else:
        h_min = math.ceil(t_sync / t_comp)
        h_used = h_override if h_override is not None else h_min
        eta = efficiency(h_used, params_b, compression_ratio=compression,
                         scenario=scenario, error_feedback=error_feedback)

    # Throughput: accounts for comm-bound at low H
    outer_step_time = max(h_used * t_comp, t_sync) if n_nodes > 1 else h_used * t_comp
    n_outer_steps = time_seconds / outer_step_time if outer_step_time > 0 else 0
    total_tokens = n_outer_steps * h_used * LOCAL_BATCH * n_nodes
    c_actual = 6 * params * total_tokens
    c_local = c_actual * eta

    # Training details
    chinchilla_tokens = CHINCHILLA_TOKENS_PER_PARAM * params
    overtraining_ratio = total_tokens / chinchilla_tokens if chinchilla_tokens > 0 else 0

    # Bandwidth / network metrics
    bw_duty_cycle = t_sync / outer_step_time if outer_step_time > 0 and n_nodes > 1 else 0
    comm_gb_per_sync = 2 * params * bits_per_pseudo_grad / compression / 8 / 1e9

    # Chinchilla-optimality + replica quality penalty
    loss_mult = replica_loss_multiplier(n_nodes, params_b, h=h_used) if n_nodes > 1 else 1.0
    eta_chin_full = chinchilla_efficiency(params, total_tokens, c_actual,
                                          loss_multiplier=loss_mult) if n_nodes > 1 else 1.0
    # Decompose: overtraining-only vs replica-only
    eta_chin_ot = chinchilla_efficiency(params, total_tokens, c_actual,
                                        loss_multiplier=1.0) if n_nodes > 1 else 1.0
    eta_replica = eta_chin_full / eta_chin_ot if eta_chin_ot > 0 else 0

    # Fold replica penalty into eta; C_local is now true local-equivalent
    eta = eta * eta_replica
    c_local = c_actual * eta
    c_quality = c_local * eta_chin_ot

    cost_usd = n_nodes * cfg["gpu_count"] * cfg["gpu_cost_usd"]

    return {
        "n_nodes": n_nodes,
        "total_gpus": n_nodes * cfg["gpu_count"],
        "pflops_per_node": pflops,
        "vram_gb": vram_gb,
        "max_params_b": max_params_b,
        "params_b": params_b,
        "h_min": h_min,
        "h_used": h_used,
        "eta": eta,
        "eta_replica": eta_replica,
        "c_local": c_local,
        "eta_chinchilla": eta_chin_ot,
        "c_quality": c_quality,
        "total_tokens_T": total_tokens / 1e12,
        "overtraining_ratio": overtraining_ratio,
        "cost_usd": cost_usd,
        "strict_threshold_multiple": c_local / 1e24,
        "bw_duty_cycle": bw_duty_cycle,
        "comm_gb_per_sync": comm_gb_per_sync,
    }


def find_nodes_for_target(cfg, target_flop, compression=COMPRESSION,
                          bytes_per_param=BYTES_PER_PARAM,
                          bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                          straggler_mode=None, error_feedback=None):
    """Binary search for minimum nodes to reach target_flop."""
    if "bytes_per_param" in cfg:
        bytes_per_param = cfg["bytes_per_param"]
    if "bits_per_pseudo_grad" in cfg:
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]

    # Check if 1 node suffices
    r = compute_generic_scenario(cfg, 1, compression=compression,
                                 bytes_per_param=bytes_per_param,
                                 bits_per_pseudo_grad=bits_per_pseudo_grad,
                                 straggler_mode=straggler_mode,
                                 error_feedback=error_feedback)
    if r["c_local"] >= target_flop:
        return r

    # Binary search between 2 and 100000
    lo, hi = 2, 100000
    while lo < hi:
        mid = (lo + hi) // 2
        r = compute_generic_scenario(cfg, mid, compression=compression,
                                     bytes_per_param=bytes_per_param,
                                     bits_per_pseudo_grad=bits_per_pseudo_grad,
                                     straggler_mode=straggler_mode,
                                     error_feedback=error_feedback)
        if r["c_local"] >= target_flop:
            hi = mid
        else:
            lo = mid + 1

    return compute_generic_scenario(cfg, lo, compression=compression,
                                    bytes_per_param=bytes_per_param,
                                    bits_per_pseudo_grad=bits_per_pseudo_grad,
                                    straggler_mode=straggler_mode,
                                    error_feedback=error_feedback)


def _h_candidates(h_min):
    """Generate H values to try: h_min (compute-bound) plus lower values that
    trade throughput for reduced replica penalty."""
    candidates = {max(1, h_min)}
    for h in [10, 30, 50, 100, 200]:
        if 1 < h < h_min:
            candidates.add(h)
    if h_min > 60:
        candidates.add(max(2, h_min // 2))
    return sorted(candidates)


def _best_over_h(compute_fn, n, h_override_supported=True):
    """Try multiple H values at node count n, return best C_quality result.
    compute_fn(n) uses h_min by default; compute_fn(n, h_override=h) tries
    specific H. Falls back to default-only if h_override not supported."""
    result_default = compute_fn(n)
    if result_default is None:
        return None
    if not h_override_supported:
        return result_default

    h_min = result_default.get("h_min", result_default.get("h_used", 1))
    best = result_default

    for h in _h_candidates(h_min):
        if h == h_min:
            continue  # already tried as default
        try:
            r = compute_fn(n, h_override=h)
        except TypeError:
            # compute_fn doesn't support h_override
            return best
        if r is not None and r["c_quality"] > best["c_quality"]:
            best = r
    return best


def _binary_search_nodes_for_c_quality(compute_fn, target_c_quality, max_nodes=100000,
                                        search_h=False):
    """Binary search for minimum nodes where compute_fn(n)['c_quality'] >= target.
    compute_fn(n) should return a result dict or None.
    If search_h=True, tries multiple H values at each node count (Option B).
    Returns the result dict at the minimum node count, or None if not achievable."""
    def eval_fn(n):
        if search_h:
            return _best_over_h(compute_fn, n)
        return compute_fn(n)

    # Check if 2 nodes suffices (minimum for DiLoCo)
    r = eval_fn(2)
    if r is None:
        # Try larger node counts — PP modes need more nodes
        lo, hi = 3, max_nodes
    elif r["c_quality"] >= target_c_quality:
        return r
    else:
        lo, hi = 3, max_nodes

    # Find upper bound first
    r_hi = eval_fn(hi)
    if r_hi is None or r_hi["c_quality"] < target_c_quality:
        return None  # Not achievable even at max_nodes

    while lo < hi:
        mid = (lo + hi) // 2
        r = eval_fn(mid)
        if r is not None and r["c_quality"] >= target_c_quality:
            hi = mid
        else:
            lo = mid + 1

    result = eval_fn(lo)
    if result is not None and result["c_quality"] >= target_c_quality:
        return result
    return None


def find_optimal_config_for_target(cfg, target_c_quality, compression=COMPRESSION,
                                   bytes_per_param=BYTES_PER_PARAM,
                                   bits_per_pseudo_grad=BITS_PER_PSEUDO_GRAD,
                                   bw_up_bps=None, bw_down_bps=None,
                                   latency_s=None, scenario=None,
                                   straggler_mode=None, error_feedback=None):
    """Search over flat DiLoCo, hierarchical DiLoCo, and PP-Group DiLoCo at
    various model sizes to find the minimum-cost configuration reaching
    target_c_quality. All connections use the same bandwidth (WAN scenario).

    Returns the best result dict, or None if no configuration reaches the target."""
    if "bytes_per_param" in cfg:
        bytes_per_param = cfg["bytes_per_param"]
    if "bits_per_pseudo_grad" in cfg:
        bits_per_pseudo_grad = cfg["bits_per_pseudo_grad"]
    if bw_up_bps is None:
        bw_up_bps = BW_UP_BPS
    if bw_down_bps is None:
        bw_down_bps = BW_DOWN_BPS
    if latency_s is None:
        latency_s = LATENCY_S

    max_single_b = cfg["vram_gb"] / bytes_per_param
    best = None

    def update_best(result):
        nonlocal best
        if result is not None and (best is None or result["cost_usd"] < best["cost_usd"]):
            best = result

    # --- Flat DiLoCo at various model sizes ---
    for frac in [0.25, 0.5, 0.75, 1.0]:
        params_b = frac * max_single_b
        if params_b < 1:
            continue

        def flat_fn(n, pb=params_b, h_override=None):
            return compute_generic_scenario(
                cfg, n, compression=compression, bytes_per_param=bytes_per_param,
                bits_per_pseudo_grad=bits_per_pseudo_grad,
                bw_up_bps=bw_up_bps, bw_down_bps=bw_down_bps,
                latency_s=latency_s, scenario=scenario, target_params_b=pb,
                h_override=h_override,
                straggler_mode=straggler_mode, error_feedback=error_feedback)

        update_best(_binary_search_nodes_for_c_quality(
            flat_fn, target_c_quality, search_h=True))

    # --- Hierarchical DiLoCo at various model sizes (WAN bandwidth for regional) ---
    # Note: hierarchical doesn't support h_override yet; search_h=False
    for frac in [0.25, 0.5, 0.75, 1.0]:
        params_b = frac * max_single_b
        if params_b < 1:
            continue
        for nodes_per_group in [4, 8, 16]:

            def hier_fn(n, pb=params_b, npg=nodes_per_group):
                return compute_hierarchical_scenario(
                    None, n, nodes_per_group=npg, compression=compression,
                    bytes_per_param=bytes_per_param,
                    bits_per_pseudo_grad=bits_per_pseudo_grad,
                    bw_up_bps=bw_up_bps, bw_down_bps=bw_down_bps,
                    latency_s=latency_s,
                    regional_bw_bps=bw_up_bps, regional_latency_s=latency_s,
                    scenario=scenario, cfg=cfg, target_params_b=pb,
                    straggler_mode=straggler_mode, error_feedback=error_feedback)

            update_best(_binary_search_nodes_for_c_quality(hier_fn, target_c_quality))

    # --- PP-Group DiLoCo at larger model sizes (WAN bandwidth for PP) ---
    # Note: PP doesn't support h_override yet; search_h=False
    for pp_comp in [PP_COMPRESSION, 10]:
        for mult in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
            params_b = mult * max_single_b
            if params_b < 1:
                continue

            def pp_fn(n, pb=params_b, ppc=pp_comp):
                return compute_pp_diloco_scenario(
                    None, n, target_params_b=pb, pp_compression=ppc,
                    compression=compression,
                    bytes_per_param=bytes_per_param,
                    bits_per_pseudo_grad=bits_per_pseudo_grad,
                    bw_up_bps=bw_up_bps, bw_down_bps=bw_down_bps,
                    latency_s=latency_s,
                    pp_bw_bps=bw_up_bps, pp_latency_s=latency_s,
                    scenario=scenario, cfg=cfg,
                    straggler_mode=straggler_mode, error_feedback=error_feedback)

            update_best(_binary_search_nodes_for_c_quality(pp_fn, target_c_quality))

    return best


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

    # Optimized table: search over flat, hierarchical, and PP-DiLoCo
    print(f"\n  Optimal evasion config for each C_quality target (all links 100 Mbps / 100 ms):")
    print(f"\n  {'CCC Threshold':>20} | {'Target':>7} | {'Nodes':>7} | {'Cost':>8} | "
          f"{'Mode':>22} | {'Model':>7} | {'OT':>5} | {'eta':>5} | {'C_quality':>10}")
    print("  " + "-" * 115)

    for label, cfg in LOWERED_CCC_A100.items():
        for i, target in enumerate(targets):
            r = find_optimal_config_for_target(cfg, target)
            if r is None:
                print(f"  {label:>20} | {target_labels[i]:>7} | {'N/A':>7} | {'N/A':>8} | "
                      f"{'N/A':>22} | {'N/A':>7} | {'N/A':>5} | {'N/A':>5} | {'N/A':>10}")
            else:
                cost = r["cost_usd"]
                cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
                # Determine mode description
                if "mode" in r and "PP" in r.get("mode", ""):
                    mode = r["mode"]
                    pp_comp = r.get("pp_compression", PP_COMPRESSION)
                    if pp_comp != PP_COMPRESSION:
                        mode += f" act{pp_comp}x"
                elif "mode" in r:
                    mode = r["mode"]
                elif "n_groups" in r and "nodes_per_group" in r:
                    mode = f"Hier {r['nodes_per_group']}x{r['n_groups']}"
                else:
                    mode = "Flat DiLoCo"
                params_b = r.get("params_b", r.get("max_params_b", 0))
                ot = r.get("overtraining_ratio", 0)
                eta = r.get("eta", 0)
                c_q = r.get("c_quality", 0)
                ot_str = f"{ot:.1f}x" if ot < 100 else f"{ot:.0f}x"
                print(f"  {label:>20} | {target_labels[i]:>7} | {r['n_nodes']:>7,} | {cost_str:>8} | "
                      f"{mode:>22} | {params_b:>5.0f}B | {ot_str:>5} | {eta:>5.3f} | {c_q:>10.2e}")
        print("  " + "-" * 115)

    # Also print the old flat-only table for comparison
    print(f"\n  [Old table for comparison — flat DiLoCo only, targeting C_local not C_quality:]")
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

    print("\n  Current exploit: 50x A100 80GB = 4,000 GB VRAM at 15.8 H100-equiv (under 16)")
    print("  Adding a VRAM threshold constrains the max node to min(compute_limit, memory_limit)")

    targets = [1e24, 1e25, 1e26]
    target_labels = ["10^24", "10^25", "10^26"]

    print(f"\n  {'VRAM Limit':>12} | {'Max A100s':>9} | {'Actual VRAM':>11} | {'PFLOPS':>7} | "
          f"{'Max Model':>10} | {'H100-eq':>7} | ", end="")
    for tl in target_labels:
        print(f"{'N->' + tl:>10} | {'Cost':>8} | ", end="")
    print()
    print("  " + "-" * 140)

    # Build configs for each memory threshold (A100)
    # Include "no limit" (effective unlimited) as baseline
    a100_mem_cfgs = []
    for mem_limit in [99999] + sorted(MEMORY_THRESHOLDS_GB, reverse=True):
        max_by_compute = 50  # floor(16 * 990 / 312) = 50
        max_by_memory = mem_limit // 80  # 80 GB per A100
        n_gpus = min(max_by_compute, max_by_memory)
        if n_gpus < 1:
            n_gpus = 1
        cfg = {
            "gpu_count": n_gpus,
            "pflops": n_gpus * 312e12 / 1e15,
            "vram_gb": n_gpus * 80,
            "gpu_cost_usd": 7_000,
            "h100_equiv": n_gpus * 312 / 990,
        }
        model_b = cfg["vram_gb"] / BYTES_PER_PARAM
        a100_mem_cfgs.append((mem_limit, n_gpus, cfg, model_b))

    for mem_limit, n_gpus, cfg, model_b in a100_mem_cfgs:
        lbl = "No limit" if mem_limit >= 99999 else f"{mem_limit} GB"
        print(f"  {lbl:>12} | {n_gpus:>9} | {cfg['vram_gb']:>8} GB | "
              f"{cfg['pflops']:>7.2f} | {model_b:>7.0f}B | {cfg['h100_equiv']:>7.1f} | ", end="")
        for target in targets:
            r = find_nodes_for_target(cfg, target)
            cost = r["cost_usd"]
            cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
            print(f"{r['n_nodes']:>10,} | {cost_str:>8} | ", end="")
        print()

    # Optimized table: search over flat, hierarchical, and PP-DiLoCo (A100)
    print(f"\n  Optimal evasion config for each C_quality target (all links 100 Mbps / 100 ms):")
    print(f"\n  {'VRAM Limit':>12} | {'Target':>7} | {'Nodes':>7} | {'Cost':>8} | "
          f"{'Mode':>22} | {'Model':>7} | {'OT':>5} | {'eta':>5} | {'C_quality':>10}")
    print("  " + "-" * 115)

    for mem_limit, n_gpus, cfg, model_b in a100_mem_cfgs:
        label = f"{mem_limit} GB" if mem_limit < 99999 else "No limit"
        for i, target in enumerate(targets):
            r = find_optimal_config_for_target(cfg, target)
            if r is None:
                print(f"  {label:>12} | {target_labels[i]:>7} | {'N/A':>7} | {'N/A':>8} | "
                      f"{'N/A':>22} | {'N/A':>7} | {'N/A':>5} | {'N/A':>5} | {'N/A':>10}")
            else:
                cost = r["cost_usd"]
                cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
                if "mode" in r and "PP" in r.get("mode", ""):
                    mode = r["mode"]
                    pp_comp = r.get("pp_compression", PP_COMPRESSION)
                    if pp_comp != PP_COMPRESSION:
                        mode += f" act{pp_comp}x"
                elif "mode" in r:
                    mode = r["mode"]
                elif "n_groups" in r and "nodes_per_group" in r:
                    mode = f"Hier {r['nodes_per_group']}x{r['n_groups']}"
                else:
                    mode = "Flat DiLoCo"
                params_b = r.get("params_b", r.get("max_params_b", 0))
                ot = r.get("overtraining_ratio", 0)
                eta = r.get("eta", 0)
                c_q = r.get("c_quality", 0)
                ot_str = f"{ot:.1f}x" if ot < 100 else f"{ot:.0f}x"
                print(f"  {label:>12} | {target_labels[i]:>7} | {r['n_nodes']:>7,} | {cost_str:>8} | "
                      f"{mode:>22} | {params_b:>5.0f}B | {ot_str:>5} | {eta:>5.3f} | {c_q:>10.2e}")
        print("  " + "-" * 115)

    # Also show H100 FP8 under memory limits
    print(f"\n  H100 SXM (FP8) under memory limits:")
    print(f"  {'VRAM Limit':>12} | {'Max H100s':>9} | {'Actual VRAM':>11} | {'FP8 PFLOPS':>10} | "
          f"{'Max Model':>10} | {'H100-eq':>7} | ", end="")
    for tl in target_labels:
        print(f"{'N->' + tl:>10} | {'Cost':>8} | ", end="")
    print()
    print("  " + "-" * 145)

    h100_mem_cfgs = []
    for mem_limit in [99999] + sorted(MEMORY_THRESHOLDS_GB, reverse=True):
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
            "gpu_cost_usd": 25_000,
            "h100_equiv": n_gpus * 1.0,
            "bytes_per_param": 14,
            "bits_per_pseudo_grad": 8,
        }
        model_b = cfg["vram_gb"] / cfg["bytes_per_param"]
        h100_mem_cfgs.append((mem_limit, n_gpus, cfg, model_b))

    for mem_limit, n_gpus, cfg, model_b in h100_mem_cfgs:
        lbl = "No limit" if mem_limit >= 99999 else f"{mem_limit} GB"
        print(f"  {lbl:>12} | {n_gpus:>9} | {cfg['vram_gb']:>8} GB | "
              f"{cfg['pflops']:>10.2f} | {model_b:>7.0f}B | {cfg['h100_equiv']:>7.1f} | ", end="")
        for target in targets:
            r = find_nodes_for_target(cfg, target)
            cost = r["cost_usd"]
            cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
            print(f"{r['n_nodes']:>10,} | {cost_str:>8} | ", end="")
        print()

    # Optimized table for H100 FP8
    print(f"\n  Optimal H100 FP8 evasion config for each C_quality target:")
    print(f"\n  {'VRAM Limit':>12} | {'Target':>7} | {'Nodes':>7} | {'Cost':>8} | "
          f"{'Mode':>22} | {'Model':>7} | {'OT':>5} | {'eta':>5} | {'C_quality':>10}")
    print("  " + "-" * 115)

    for mem_limit, n_gpus, cfg, model_b in h100_mem_cfgs:
        label = f"{mem_limit} GB" if mem_limit < 99999 else "No limit"
        for i, target in enumerate(targets):
            r = find_optimal_config_for_target(cfg, target)
            if r is None:
                print(f"  {label:>12} | {target_labels[i]:>7} | {'N/A':>7} | {'N/A':>8} | "
                      f"{'N/A':>22} | {'N/A':>7} | {'N/A':>5} | {'N/A':>5} | {'N/A':>10}")
            else:
                cost = r["cost_usd"]
                cost_str = f"${cost/1e9:.1f}B" if cost >= 1e9 else f"${cost/1e6:.0f}M"
                if "mode" in r and "PP" in r.get("mode", ""):
                    mode = r["mode"]
                    pp_comp = r.get("pp_compression", PP_COMPRESSION)
                    if pp_comp != PP_COMPRESSION:
                        mode += f" act{pp_comp}x"
                elif "mode" in r:
                    mode = r["mode"]
                elif "n_groups" in r and "nodes_per_group" in r:
                    mode = f"Hier {r['nodes_per_group']}x{r['n_groups']}"
                else:
                    mode = "Flat DiLoCo"
                params_b = r.get("params_b", r.get("max_params_b", 0))
                ot = r.get("overtraining_ratio", 0)
                eta = r.get("eta", 0)
                c_q = r.get("c_quality", 0)
                ot_str = f"{ot:.1f}x" if ot < 100 else f"{ot:.0f}x"
                print(f"  {label:>12} | {target_labels[i]:>7} | {r['n_nodes']:>7,} | {cost_str:>8} | "
                      f"{mode:>22} | {params_b:>5.0f}B | {ot_str:>5} | {eta:>5.3f} | {c_q:>10.2e}")
        print("  " + "-" * 115)


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
    print(f"WAN: {BW_UP_MBPS} Mbps up / {BW_DOWN_MBPS} Mbps down, {LATENCY_S*1000:.0f} ms latency")
    print(f"MFU: {MFU*100:.0f}%  |  Compression: {COMPRESSION}x  |  Error feedback: {'Yes' if ERROR_FEEDBACK else 'No'}  |  Streaming: Yes  |  Straggler: {STRAGGLER_MODE}")
    print(f"Compression quality scenario: {DEFAULT_SCENARIO}")
    print(f"CCC threshold: 16 H100-equivalents = 15,840 TFLOPS FP16")
    print("=" * 80)

    # ── PART 1: Existing analysis (sub-threshold evasion) ────────────────────

    # Primary analysis: 48x A100
    print_config_summary("50x A100 80GB")
    print_results_table("50x A100 80GB")
    print_detailed("50x A100 80GB", 4)
    print_detailed("50x A100 80GB", 72)

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
    print_large_scale_table("50x A100 80GB")

    # Hierarchical DiLoCo
    print(f"\n--- B: Hierarchical DiLoCo (groups of {NODES_PER_GROUP}), 48x A100 FP16, 16x ---")
    print_hierarchical_table("50x A100 80GB")

    # FP8 H100
    print("\n--- C: Flat DiLoCo, 16x H100 FP8, 16x compression ---")
    print_large_scale_table("16x H100 FP8")

    # Hierarchical FP8 H100
    print(f"\n--- D: Hierarchical DiLoCo, 16x H100 FP8, 16x compression ---")
    print_hierarchical_table("16x H100 FP8")

    # 100x compression flat
    print("\n--- E: Flat DiLoCo, 48x A100 FP16, 100x compression ---")
    print_large_scale_table("50x A100 80GB", compression=100)

    # Hierarchical + 100x + FP8
    print(f"\n--- F: Hierarchical + 100x compression, 16x H100 FP8 ---")
    print_hierarchical_table("16x H100 FP8", compression=100)

    # MoE + EP
    print("\n--- G: MoE + Expert Parallelism (600B total / 100B active) ---")
    for n in [72, 500, 2000, 4000]:
        r = compute_moe_ep_scenario("50x A100 80GB", n, total_params_b=600,
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

    print_time_sensitivity("50x A100 80GB", 4)
    print_time_sensitivity("50x A100 80GB", 72)
    print_time_sensitivity("50x A100 80GB", 4000)
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
    print_bandwidth_sensitivity("50x A100 80GB", 72)

    # Medium-scale (500 nodes, targeting 10^26)
    print_bandwidth_sensitivity("50x A100 80GB", 500)

    # Large-scale flat (4000 nodes A100, targeting 10^27)
    print_bandwidth_sensitivity("50x A100 80GB", 4000)

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
    print_latency_sensitivity("50x A100 80GB", 72)

    # 500 nodes (10^26 target)
    print_latency_sensitivity("50x A100 80GB", 500)

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
    print_deployment_profiles("50x A100 80GB", 72)

    # 500 nodes (10^26 target)
    print_deployment_profiles("50x A100 80GB", 500)

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
    r_flat = compute_scenario("50x A100 80GB", 4000)
    r_hier = compute_hierarchical_scenario("50x A100 80GB", 4000)
    print(f"\n5. Hierarchical vs flat eta at N=4000:")
    print(f"   Flat: eta={r_flat['eta']:.4f} (H={r_flat['h_min']})")
    print(f"   Hier: eta={r_hier['eta']:.4f} (H_eff={r_hier['h_eff']:.1f})")
    print(f"   Improvement: {(r_hier['eta']-r_flat['eta'])*100:.1f}pp: {'PASS' if r_hier['eta'] > r_flat['eta'] else 'CHECK'}")

    # Check 6: MoE+EP memory fits
    r_moe = compute_moe_ep_scenario("50x A100 80GB", 72, total_params_b=600,
                                     active_params_b=100)
    print(f"\n6. MoE+EP memory check (600B MoE, N=72, 48xA100):")
    print(f"   Per-node memory: {r_moe['mem_node_gb']:.0f} GB (VRAM: {r_moe['vram_gb']} GB)")
    print(f"   Fits: {r_moe['fits_on_node']}: {'PASS' if r_moe['fits_on_node'] else 'FAIL'}")

    # Check 7: Time scaling (C_local at 6mo = C_local at 18mo * 6/18)
    r_full = compute_scenario("50x A100 80GB", 72)
    r_half = compute_scenario("50x A100 80GB", 72, time_seconds=TIME_VARIANTS["6 months"])
    ratio_time = r_half['c_local'] / r_full['c_local']
    print(f"\n7. Time scaling check (6mo vs 1.5yr):")
    print(f"   6mo: {r_half['c_local']:.2e}, 1.5yr: {r_full['c_local']:.2e}")
    print(f"   Ratio: {ratio_time:.3f} (expected 0.333): {'PASS' if abs(ratio_time - 1/3) < 0.01 else 'CHECK'}")

    # Check 8: Chinchilla loss at known scale matches intuition
    # GPT-3 175B trained on 300B tokens → loss ~1.73 (Table D.1 Brown et al.)
    l_gpt3 = chinchilla_loss(175e9, 300e12)
    print(f"\n8. Chinchilla loss sanity check:")
    print(f"   L(175B, 300B tokens) = {l_gpt3:.4f}")
    print(f"   Expected ~1.8-1.9 (GPT-3 actual ~1.73, scaling law approximate)")
    print(f"   {'PASS' if 1.7 < l_gpt3 < 2.0 else 'CHECK'}")

    # Check 9: Chinchilla-optimal allocation round-trip
    c_test = 6 * 100e9 * 25.6 * 100e9  # Chinchilla-optimal compute for 100B
    n_opt, d_opt = chinchilla_optimal_allocation(c_test)
    print(f"\n9. Chinchilla allocation round-trip:")
    print(f"   C for 100B optimal = {c_test:.2e}")
    print(f"   N_opt = {n_opt/1e9:.1f}B, D_opt = {d_opt/1e12:.2f}T")
    print(f"   D/N ratio = {d_opt/n_opt:.1f} (expected 25.6)")
    print(f"   {'PASS' if abs(n_opt/1e9 - 100) < 1 and abs(d_opt/n_opt - 25.6) < 0.1 else 'CHECK'}")

    # Check 10: eta_chinchilla = 1.0 at optimal allocation
    eta_at_opt = chinchilla_efficiency(n_opt, d_opt, c_test)
    print(f"\n10. eta_chinchilla at optimal = {eta_at_opt:.4f}")
    print(f"    {'PASS' if eta_at_opt > 0.99 else 'CHECK'}")

    # Check 11: C_quality <= C_local always
    r_ot = compute_scenario("50x A100 80GB", 500)
    print(f"\n11. C_quality <= C_local check (500 nodes, 48xA100):")
    print(f"    C_local = {r_ot['c_local']:.2e}, C_quality = {r_ot['c_quality']:.2e}")
    print(f"    {'PASS' if r_ot['c_quality'] <= r_ot['c_local'] else 'FAIL'}")

    # Check 12: Activation compression quality = 1.0 when no compression or 1 stage
    act_no_comp = activation_compression_quality(1, 4)
    act_1_stage = activation_compression_quality(4, 1)
    print(f"\n12. Activation compression edge cases:")
    print(f"    No compression (ratio=1, S=4): {act_no_comp:.4f} {'PASS' if act_no_comp == 1.0 else 'FAIL'}")
    print(f"    One stage (ratio=4, S=1): {act_1_stage:.4f} {'PASS' if act_1_stage == 1.0 else 'FAIL'}")

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
    print(f"Error feedback: {'Yes' if ERROR_FEEDBACK else 'No'}")
    print("\nCompression quality (with error feedback):")
    for cr, factors in sorted(COMPRESSION_QUALITY_EF.items()):
        print(f"  {cr:>4}x: optimistic={factors['optimistic']:.2f}, "
              f"expected={factors['expected']:.2f}, "
              f"conservative={factors['conservative']:.2f}")
    print("\nCompression quality (without error feedback):")
    for cr, factors in sorted(COMPRESSION_QUALITY_NO_EF.items()):
        print(f"  {cr:>4}x: optimistic={factors['optimistic']:.2f}, "
              f"expected={factors['expected']:.2f}, "
              f"conservative={factors['conservative']:.2f}")

    # Key scenarios under all three quality assumptions
    key_scenarios = [
        ("72 nodes, 48xA100, 16x comp", "50x A100 80GB", 72, 16, False),
        ("500 nodes, 48xA100, 16x comp", "50x A100 80GB", 500, 16, False),
        ("4000 nodes, 48xA100, 16x comp", "50x A100 80GB", 4000, 16, False),
        ("2000 nodes, H100 FP8, 16x comp", "16x H100 FP8", 2000, 16, False),
        ("2000 nodes, H100 FP8, 100x hier", "16x H100 FP8", 2000, 100, True),
        ("4000 nodes, 48xA100, 100x comp", "50x A100 80GB", 4000, 100, False),
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
        ("72 nodes, 48xA100, 16x", "50x A100 80GB", 72, 16, False),
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

    # ── PART 8: Model size optimization & PP-Group DiLoCo ─────────────────────

    print("\n" + "=" * 80)
    print("PART 8: MODEL SIZE OPTIMIZATION & PP-GROUP DiLoCo")
    print("=" * 80)
    print(f"\nActivation compression: {PP_COMPRESSION}x (4-bit quantization)")
    print(f"PP interconnect: {PP_BW_BPS/1e9:.0f} Gbps, {PP_LATENCY_S*1000:.0f} ms (regional co-location)")
    print(f"Chinchilla optimal D/N ratio: {CHINCHILLA_TOKENS_PER_PARAM}")
    print(f"Micro-batches: {MICRO_BATCHES}")

    print("\nActivation compression quality factors (per stage boundary):")
    for cr, factors in sorted(ACTIVATION_COMPRESSION_QUALITY.items()):
        print(f"  {cr:>4}x: optimistic={factors['optimistic']:.3f}, "
              f"expected={factors['expected']:.3f}, "
              f"conservative={factors['conservative']:.3f}")

    # Key node counts for model size sweep
    sweep_configs = [
        ("50x A100 80GB", 72),
        ("50x A100 80GB", 500),
        ("50x A100 80GB", 4000),
        ("16x H100 FP8", 2000),
    ]

    for cfg_name, n in sweep_configs:
        print_model_size_sweep(cfg_name, n)

    # Summary: best C_quality at each scale
    print("\n\n--- Summary: Optimal model size at each scale ---")
    print(f"  {'Config':>20} | {'Nodes':>5} | {'Best Mode':>12} | {'Model':>7} | "
          f"{'PP':>2} | {'C_quality':>10} | {'vs DiLoCo max':>13}")
    print("  " + "-" * 90)

    for cfg_name, n in sweep_configs:
        results = sweep_model_sizes(cfg_name, n)
        best = results[0] if results else None
        # Find the max-VRAM DiLoCo result for comparison
        diloco_max = [r for r in results if r.get("mode_type") == "DiLoCo"
                      and r.get("pp_stages", 1) == 1]
        diloco_max_c = max((r["c_quality"] for r in diloco_max), default=0)
        improvement = best["c_quality"] / diloco_max_c if diloco_max_c > 0 else float('inf')

        if best:
            model_str = f"{best['params_b']:.0f}B"
            print(f"  {cfg_name:>20} | {n:>5} | {best.get('mode_type','?'):>12} | "
                  f"{model_str:>7} | "
                  f"{best.get('pp_stages',1):>2} | {best['c_quality']:>10.2e} | "
                  f"{improvement:>12.2f}x")
