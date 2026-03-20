# Registration Impact of a 1,280 GB HBM Threshold

## Document by Claude | 2026-03-20

## Background

The main report recommends adding a memory-based component to the Covered Compute Cluster (CCC) definition: any cluster with aggregate HBM (High Bandwidth Memory) **greater than** 1,280 GB would require registration, in addition to the existing compute threshold of **greater than** 15,840 TFLOPS FP16 (>16 H100-equivalents). Both thresholds use strict inequality — a cluster at exactly 1,280 GB or exactly 16.0 H100-equivalents is *not* a CCC.

This document examines which hardware configurations would be **newly caught** by the HBM threshold — i.e., configurations with >1,280 GB HBM but ≤16 H100-equivalents of compute — and assesses the resulting registration burden on non-AI-training use cases.

## Why HBM rather than all VRAM

The threshold specifically targets HBM rather than all accelerator memory (VRAM). There are two families of GPU memory in current hardware:

- **HBM (High Bandwidth Memory):** Used in all datacenter AI training accelerators — NVIDIA A100/H100/H200/GH200/B100/B200, AMD MI210/MI250X/MI300X, Intel Gaudi 2/3, Google TPU v4/v5. HBM provides very high bandwidth (1,500-3,350 GB/s per chip), which is essential for efficient large-model training.
- **GDDR (Graphics DDR):** Used in consumer GPUs (RTX 4090, RTX 5090), professional/workstation GPUs (RTX A6000, RTX 6000 Ada), and some datacenter GPUs marketed for inference or rendering (A40, L40, L40S). GDDR provides lower bandwidth (~864-1,008 GB/s per chip) and is primarily optimized for graphics rendering and inference workloads.

By targeting HBM specifically, the threshold catches all hardware suitable for large-scale AI training while excluding VFX rendering farms, workstation clusters, and consumer GPU pools. This significantly narrows the registration burden to AI-relevant hardware.

**No practical evasion gap:** The only GDDR-based datacenter GPU with meaningful compute density is the L40S (48 GB GDDR6, 366 TFLOPS FP16). However, its memory bandwidth (864 GB/s) is less than half the A100's (2,039 GB/s). Since training requires reading all model weights on every forward and backward pass, this bandwidth bottleneck limits training MFU to approximately 15-20% on GDDR cards, compared to ~40% on HBM cards. A cluster of 43 L40S cards (2,064 GB GDDR, 15.9 H100-eq) would achieve roughly 2.3x less effective training throughput than 50 A100 80GB cards despite similar raw TFLOPS. GDDR-based clusters are not a practical evasion vector for large-scale training.

## Accelerator specifications

The tables below list current accelerators with their per-card VRAM, memory type, and FP16 compute, along with the minimum number of cards needed to **exceed** 1,280 GB of HBM (i.e., the smallest N where N x HBM > 1,280 GB) and the resulting compute in H100-equivalents. Cards using GDDR memory are listed for comparison but are **not covered** by the HBM threshold.

### NVIDIA datacenter GPUs

| Accelerator | VRAM | Memory Type | FP16 Dense (TFLOPS) | Cards to exceed 1,280 GB HBM | Aggregate HBM | Aggregate compute | H100-equiv | Covered? |
| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |
| A100 80GB | 80 GB | HBM2e | 312 | 17 | 1,360 GB | 5,304 TFLOPS | 5.4 | **Yes** |
| A100 40GB | 40 GB | HBM2e | 312 | 33 | 1,320 GB | 10,296 TFLOPS | 10.4 | **Yes** |
| V100 32GB | 32 GB | HBM2 | 125 | 41 | 1,312 GB | 5,125 TFLOPS | 5.2 | **Yes** |
| V100 16GB | 16 GB | HBM2 | 125 | 81 | 1,296 GB | 10,125 TFLOPS | 10.2 | **Yes** |
| H200 | 141 GB | HBM3e | 990 | 10 | 1,410 GB | 9,900 TFLOPS | 10.0 | **Yes** |
| H100 SXM | 80 GB | HBM3 | 990 | 17 | 1,360 GB | 16,830 TFLOPS | 17.0 | Compute threshold triggers first |
| GH200 | 144 GB | HBM3e | 990 | 9 | 1,296 GB | 8,910 TFLOPS | 9.0 | **Yes** |
| B100 | 192 GB | HBM3e | 1,800 | 7 | 1,344 GB | 12,600 TFLOPS | 12.7 | **Yes** |
| B200 | 192 GB | HBM3e | 2,250 | 7 | 1,344 GB | 15,750 TFLOPS | 15.9 | **Yes** |
| A40 | 48 GB | GDDR6 | 150 | — | 0 GB HBM | 4,050 TFLOPS | 4.1 | No (GDDR) |
| L40 | 48 GB | GDDR6 | 181 | — | 0 GB HBM | 4,887 TFLOPS | 4.9 | No (GDDR) |
| L40S | 48 GB | GDDR6 | 366 | — | 0 GB HBM | 9,882 TFLOPS | 10.0 | No (GDDR) |

Note: 16x H100 SXM = exactly 1,280 GB HBM and exactly 16.0 H100-eq, which does **not** exceed either threshold. A 17th H100 triggers both simultaneously.

### AMD datacenter GPUs

| Accelerator | VRAM | Memory Type | FP16 Dense (TFLOPS) | Cards to exceed 1,280 GB HBM | Aggregate HBM | Aggregate compute | H100-equiv | Covered? |
| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |
| MI210 | 64 GB | HBM2e | 181 | 21 | 1,344 GB | 3,801 TFLOPS | 3.8 | **Yes** |
| MI250X | 128 GB | HBM2e | 383 | 11 | 1,408 GB | 4,213 TFLOPS | 4.3 | **Yes** |
| MI300X | 192 GB | HBM3 | 1,307 | 7 | 1,344 GB | 9,149 TFLOPS | 9.2 | **Yes** |

### Intel accelerators

| Accelerator | VRAM | Memory Type | BF16 (TFLOPS) | Cards to exceed 1,280 GB HBM | Aggregate HBM | Aggregate compute | H100-equiv | Covered? |
| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |
| Gaudi 2 | 96 GB | HBM2e | 432 | 14 | 1,344 GB | 6,048 TFLOPS | 6.1 | **Yes** |
| Gaudi 3 | 128 GB | HBM2 | 1,835 | 11 | 1,408 GB | 20,185 TFLOPS | 20.4 | Compute threshold triggers first |

### NVIDIA professional / workstation GPUs

| Accelerator | VRAM | Memory Type | FP16 Dense (TFLOPS) | Covered by HBM threshold? |
| :-- | --: | :-- | --: | :-- |
| RTX A6000 | 48 GB | GDDR6 | 155 | No (GDDR) |
| RTX 6000 Ada | 48 GB | GDDR6 | 728 | No (GDDR) |
| RTX 5880 Ada | 48 GB | GDDR6 | 554 | No (GDDR) |

All professional/workstation GPUs use GDDR memory and are **entirely excluded** from the HBM threshold, regardless of cluster size. A render farm with hundreds of RTX A6000 cards would have 0 GB of HBM.

Note: Consumer GPUs (RTX 4090 at 24 GB GDDR6X, RTX 5090 at 32 GB GDDR6X) also use GDDR memory and are excluded from the HBM threshold entirely.

## Notable thresholds

- **2x DGX A100** (16x A100 80GB): exactly 1,280 GB HBM, 5.0 H100-eq. This is **not caught** — both thresholds use strict inequality (>1,280 GB), and 1,280 GB is not greater than 1,280 GB. Adding a single additional A100 80GB (17 total, 1,360 GB) would trigger the HBM threshold.
- **2x DGX H100** (16x H100 SXM): exactly 1,280 GB HBM, exactly 16.0 H100-eq. Also **not caught** by either threshold. A 17th H100 (1,360 GB, 17.0 H100-eq) would trigger both.
- **1x DGX A100 or 1x DGX H100** (8 GPUs, 640 GB): well under both thresholds.
- **3x DGX A100** (24x A100 80GB): 1,920 GB HBM, 7.6 H100-eq. This is the smallest standard multi-DGX configuration that is clearly caught by the HBM threshold alone.

## Legitimate use cases affected

### 1. LLM inference serving

**Impact: Moderate.** This is the most significant category of newly burdened users.

Serving large language models requires enough VRAM to hold model weights and KV cache. Representative VRAM requirements:

| Model | FP16 weights | FP8 weights | With KV cache (128K context) |
| :-- | --: | --: | --: |
| 70B (Llama 3) | ~140 GB | ~70 GB | ~180-250 GB |
| 405B (Llama 3.1) | ~810 GB | ~405 GB | ~900-1,200 GB |
| 671B MoE (DeepSeek-V3) | ~1,340 GB | ~670 GB | ~1,500-2,000 GB |

A **single** DGX A100 or DGX H100 (640 GB) can serve a 70B model comfortably and a 405B model in FP8 with limited context. This setup would remain unregistered.

However, production inference deployments typically run **multiple server nodes** for throughput and redundancy. Two DGX A100 nodes (16x A100 80GB, exactly 1,280 GB) would remain just under the threshold, but a 405B model in FP16 with long-context support would require more — three DGX A100 nodes (1,920 GB, 7.6 H100-eq) would clearly exceed the HBM threshold. Large inference providers (cloud API services, enterprise deployments) routinely operate clusters of 3+ DGX nodes.

Note: inference providers using GDDR-based cards (L40S, which is marketed for inference) would not be caught regardless of cluster size.

### 2. Shared university and research clusters

**Impact: Moderate to high, but mitigated by existing visibility.**

University GPU clusters typically have 50-300+ GPUs shared across many independent researchers via a job scheduler (e.g., SLURM). Examples:

- A modest 20x A100 80GB cluster: 1,600 GB HBM, 6.3 H100-eq
- A mid-size 80x A100 80GB cluster: 6,400 GB HBM, 25.3 H100-eq (already caught by compute)
- Princeton's 300x H100 cluster: 24,000 GB HBM, 300 H100-eq (already caught by compute)

Small-to-medium university clusters (17-50 A100 80GB cards) would be newly caught by the HBM threshold despite falling well under the compute threshold. A cluster with exactly 16x A100 80GB (1,280 GB) would remain just under the HBM threshold.

However, university clusters are **already highly visible** — they are operated by known institutions with public procurement records, published research, and no incentive to hide. The marginal registration burden (paperwork and potential monitoring) is real but does not create a surveillance gap, since these systems are not covert.

An important definitional question: if a university cluster runs independent 1-8 GPU jobs for different researchers, does it constitute hardware "networked to perform workloads together"? If the CCC definition is interpreted per-workload rather than per-facility, most university clusters would not be caught. But this interpretation also weakens the threshold's enforcement value, since an evader could claim their nodes serve independent workloads.

### 3. Scientific HPC clusters

**Impact: Low to moderate.**

GPU-accelerated scientific computing (climate modeling, molecular dynamics, computational fluid dynamics, genomics) uses clusters that can exceed 1,280 GB HBM:

- Climate modeling: 32-256 GPUs per simulation (2,560-20,480 GB with A100 80GB)
- Molecular dynamics (GROMACS, NAMD): typically 4-48 GPUs per job
- Computational fluid dynamics: 16-64 GPUs for large simulations

Large HPC clusters are often already above the compute threshold. Smaller ones (17-50 HBM-equipped GPUs) would be newly caught by the HBM threshold. As with university clusters, these systems are typically operated by known institutions (national labs, research centers) with high existing visibility.

### 4. VFX and GPU rendering farms

**Impact: None.** Movie studios and VFX houses use GDDR-based professional GPUs (A40, L40, RTX A6000) for GPU rendering. Since these cards use GDDR memory rather than HBM, rendering farms are entirely excluded from the HBM threshold regardless of size. This is a key advantage of the HBM-specific approach over a general VRAM threshold.

### 5. Autonomous vehicle development

**Impact: Negligible (already caught).** AV training clusters (Tesla, Waymo, Cruise) operate thousands of HBM-equipped GPUs that already exceed the compute threshold by orders of magnitude. The HBM threshold adds no new burden.

### 6. Cryptocurrency mining operations

**Impact: None.** Cryptocurrency mining hardware uses GDDR-based consumer GPUs (RTX 4090, RTX 3090, etc.) and is entirely excluded from the HBM threshold. Even mining farms repurposed for AI inference with hundreds of consumer cards would have 0 GB of HBM.

## Summary of registration impact

| Use case | Typical setup | Currently registered? | Newly registered under HBM threshold? | Burden |
| :-- | :-- | :-- | :-- | :-- |
| Single DGX server | 8 GPUs, 640 GB HBM | No | No | None |
| LLM inference (70B) | 1 DGX node, 640 GB HBM | No | No | None |
| LLM inference (405B, production) | 3-4 DGX A100 nodes, 1,920-2,560 GB HBM | No | **Yes** | Moderate |
| University cluster (small) | 17-50 A100 80GBs, 1,360-4,000 GB HBM | No | **Yes** | Low-moderate |
| University cluster (large) | 80+ A100s or H100s | Already registered (compute) | N/A | None (already registered) |
| Scientific HPC (small) | 17-50 HBM GPUs | No | **Yes** | Low |
| Scientific HPC (large) | 100+ GPUs | Already registered (compute) | N/A | None |
| VFX render farm | Any size (A40/L40/RTX A6000) | No | **No (GDDR)** | None |
| AV development | 1,000+ GPUs | Already registered (compute) | N/A | None |
| Crypto mining | Any size (consumer GPUs) | No | **No (GDDR)** | None |
| Cloud provider (per-customer) | Varies | Depends on allocation | Depends on allocation | See below |

### Cloud provider considerations

Cloud providers (AWS, GCP, Azure) operate clusters that vastly exceed both thresholds and are already subject to CCC registration for their aggregate infrastructure. The HBM threshold raises a question about **per-customer allocations**: if a single cloud customer provisions 17+ A100 80GB instances (1,360 GB aggregate HBM), does this constitute a newly formed CCC? The compute-only threshold already creates this question at 51+ A100 instances; the HBM threshold lowers this to 17 A100 80GB instances. Cloud providers would likely need clearer guidance on per-tenant vs. per-facility registration.

## Assessment

The claim that the HBM threshold is a "low-impact addition" that would "not substantially increase compliance costs and registration burden of users who are not doing AI training" is **largely accurate**, with some caveats:

**What's accurate:**
- Single-server configurations (1 DGX box) remain unregistered
- The threshold doesn't create a fundamentally new category of surveillance — the largest newly-caught systems (universities, national labs) are already visible institutions
- The primary enforcement benefit is real: it forces evasion hardware from A100-class (4,000 GB HBM, 250B model) down to H100-class (1,280 GB, 91B model), greatly reducing evasion capability
- The HBM-specific threshold entirely excludes VFX rendering farms, workstation clusters, consumer GPU pools, and cryptocurrency mining operations — all significant categories of non-AI-training GPU use
- All newly-caught hardware uses datacenter AI accelerators (HBM-equipped), making it inherently AI-adjacent

**Caveats:**
- Production LLM inference deployments with 3+ DGX A100 nodes would be newly registered — this is a growing category of legitimate non-training use, though it uses the same datacenter AI hardware the threshold is designed to monitor
- Small-to-medium university and HPC clusters (17-50 HBM-equipped GPUs) would face new registration requirements, though these institutions are already highly visible

**Recommendation:** The report's characterization of the HBM threshold as "low-impact" is defensible. The remaining newly-caught use cases (inference clusters, university research clusters) all use datacenter AI hardware and are operated by known entities. The report should note that medium-scale inference deployments would face new registration requirements, while emphasizing that the HBM-specific framing avoids burdening non-AI-training industries entirely.
