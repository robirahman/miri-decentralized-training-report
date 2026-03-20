# Registration Impact of a 1,280 GB VRAM Threshold

## Document by Claude | 2026-03-20

## Background

The main report recommends adding a memory-based component to the Covered Compute Cluster (CCC) definition: any cluster with aggregate accelerator VRAM **greater than** 1,280 GB would require registration, in addition to the existing compute threshold of **greater than** 15,840 TFLOPS FP16 (>16 H100-equivalents). Both thresholds use strict inequality — a cluster at exactly 1,280 GB or exactly 16.0 H100-equivalents is *not* a CCC.

This document examines which hardware configurations would be **newly caught** by the VRAM threshold — i.e., configurations with >1,280 GB VRAM but ≤16 H100-equivalents of compute — and assesses the resulting registration burden on non-AI-training use cases.

## Accelerator specifications

The table below lists current datacenter and professional accelerators with their per-card VRAM and FP16 compute, along with the minimum number of cards needed to **exceed** 1,280 GB VRAM (i.e., the smallest N where N × VRAM > 1,280 GB) and the resulting compute in H100-equivalents.

### NVIDIA datacenter GPUs

| Accelerator | VRAM | FP16 Dense (TFLOPS) | Cards to exceed 1,280 GB | Aggregate VRAM | Aggregate compute | H100-equiv | Already caught by compute? |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| A100 80GB | 80 GB | 312 | 17 | 1,360 GB | 5,304 TFLOPS | 5.4 | No |
| A100 40GB | 40 GB | 312 | 33 | 1,320 GB | 10,296 TFLOPS | 10.4 | No |
| V100 32GB | 32 GB | 125 | 41 | 1,312 GB | 5,125 TFLOPS | 5.2 | No |
| V100 16GB | 16 GB | 125 | 81 | 1,296 GB | 10,125 TFLOPS | 10.2 | No |
| A40 | 48 GB | 150 | 27 | 1,296 GB | 4,050 TFLOPS | 4.1 | No |
| L40 | 48 GB | 181 | 27 | 1,296 GB | 4,887 TFLOPS | 4.9 | No |
| L40S | 48 GB | 366 | 27 | 1,296 GB | 9,882 TFLOPS | 10.0 | No |
| H200 | 141 GB | 990 | 10 | 1,410 GB | 9,900 TFLOPS | 10.0 | No |
| H100 SXM | 80 GB | 990 | 17 | 1,360 GB | 16,830 TFLOPS | 17.0 | Yes |
| GH200 (HBM3e) | 144 GB | 990 | 9 | 1,296 GB | 8,910 TFLOPS | 9.0 | No |
| B100 | 192 GB | 1,800 | 7 | 1,344 GB | 12,600 TFLOPS | 12.7 | No |
| B200 | 192 GB | 2,250 | 7 | 1,344 GB | 15,750 TFLOPS | 15.9 | No |

Note: 16x H100 SXM = exactly 1,280 GB VRAM and exactly 16.0 H100-eq, which does **not** exceed either threshold. A 17th H100 triggers both simultaneously.

### AMD datacenter GPUs

| Accelerator | VRAM | FP16 Dense (TFLOPS) | Cards to exceed 1,280 GB | Aggregate VRAM | Aggregate compute | H100-equiv | Already caught by compute? |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| MI210 | 64 GB | 181 | 21 | 1,344 GB | 3,801 TFLOPS | 3.8 | No |
| MI250X | 128 GB | 383 | 11 | 1,408 GB | 4,213 TFLOPS | 4.3 | No |
| MI300X | 192 GB | 1,307 | 7 | 1,344 GB | 9,149 TFLOPS | 9.2 | No |

### Intel accelerators

| Accelerator | VRAM | BF16 (TFLOPS) | Cards to exceed 1,280 GB | Aggregate VRAM | Aggregate compute | H100-equiv | Already caught by compute? |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| Gaudi 2 | 96 GB | 432 | 14 | 1,344 GB | 6,048 TFLOPS | 6.1 | No |
| Gaudi 3 | 128 GB | 1,835 | 11 | 1,408 GB | 20,185 TFLOPS | 20.4 | Yes |

### NVIDIA professional / workstation GPUs

| Accelerator | VRAM | FP16 Dense (TFLOPS) | Cards to exceed 1,280 GB | Aggregate VRAM | Aggregate compute | H100-equiv | Already caught by compute? |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| RTX A6000 | 48 GB | 155 | 27 | 1,296 GB | 4,185 TFLOPS | 4.2 | No |
| RTX 6000 Ada | 48 GB | 728 | 27 | 1,296 GB | 19,656 TFLOPS | 19.9 | Yes |
| RTX 5880 Ada | 48 GB | 554 | 27 | 1,296 GB | 14,958 TFLOPS | 15.1 | No |

Note: Consumer GPUs (RTX 4090 at 24 GB, RTX 5090 at 32 GB) would require 54 or 41 cards respectively to exceed 1,280 GB. Clusters of this size are uncommon outside of cryptocurrency mining operations and would have 4-5 H100-equivalents of compute.

## Notable thresholds

- **2x DGX A100** (16x A100 80GB): exactly 1,280 GB VRAM, 5.0 H100-eq. This is **not caught** — both thresholds use strict inequality (>1,280 GB), and 1,280 GB is not greater than 1,280 GB. Adding a single additional A100 80GB (17 total, 1,360 GB) would trigger the VRAM threshold.
- **2x DGX H100** (16x H100 SXM): exactly 1,280 GB VRAM, exactly 16.0 H100-eq. Also **not caught** by either threshold. A 17th H100 (1,360 GB, 17.0 H100-eq) would trigger both.
- **1x DGX A100 or 1x DGX H100** (8 GPUs, 640 GB): well under both thresholds.
- **3x DGX A100** (24x A100 80GB): 1,920 GB VRAM, 7.6 H100-eq. This is the smallest standard multi-DGX configuration that is clearly caught by the VRAM threshold alone.

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

However, production inference deployments typically run **multiple server nodes** for throughput and redundancy. Two DGX A100 nodes (16x A100 80GB, exactly 1,280 GB) would remain just under the threshold, but a 405B model in FP16 with long-context support would require more — three DGX A100 nodes (1,920 GB, 7.6 H100-eq) would clearly exceed the VRAM threshold. Large inference providers (cloud API services, enterprise deployments) routinely operate clusters of 3+ DGX nodes.

### 2. Shared university and research clusters

**Impact: Moderate to high, but mitigated by existing visibility.**

University GPU clusters typically have 50-300+ GPUs shared across many independent researchers via a job scheduler (e.g., SLURM). Examples:

- A modest 20x A100 80GB cluster: 1,600 GB VRAM, 6.3 H100-eq
- A mid-size 80x A100 80GB cluster: 6,400 GB VRAM, 25.3 H100-eq (already caught by compute)
- Princeton's 300x H100 cluster: 24,000 GB VRAM, 300 H100-eq (already caught by compute)

Small-to-medium university clusters (17-50 A100 80GB cards) would be newly caught by the VRAM threshold despite falling well under the compute threshold. A cluster with exactly 16x A100 80GB (1,280 GB) would remain just under the VRAM threshold.

However, university clusters are **already highly visible** — they are operated by known institutions with public procurement records, published research, and no incentive to hide. The marginal registration burden (paperwork and potential monitoring) is real but does not create a surveillance gap, since these systems are not covert.

An important definitional question: if a university cluster runs independent 1-8 GPU jobs for different researchers, does it constitute hardware "networked to perform workloads together"? If the CCC definition is interpreted per-workload rather than per-facility, most university clusters would not be caught. But this interpretation also weakens the threshold's enforcement value, since an evader could claim their nodes serve independent workloads.

### 3. Scientific HPC clusters

**Impact: Low to moderate.**

GPU-accelerated scientific computing (climate modeling, molecular dynamics, computational fluid dynamics, genomics) uses clusters that can exceed 1,280 GB VRAM:

- Climate modeling: 32-256 GPUs per simulation (2,560-20,480 GB with A100 80GB)
- Molecular dynamics (GROMACS, NAMD): typically 4-48 GPUs per job
- Computational fluid dynamics: 16-64 GPUs for large simulations

Large HPC clusters are often already above the compute threshold. Smaller ones (16-50 GPUs) would be newly caught by the VRAM threshold. As with university clusters, these systems are typically operated by known institutions (national labs, research centers) with high existing visibility.

### 4. VFX and GPU rendering farms

**Impact: Low.**

Movie studios and VFX houses increasingly use GPU rendering (NVIDIA RTX A6000, A40, L40 cards). A large render farm with 27+ A40 GPUs (1,296 GB VRAM, 4.1 H100-eq) would cross the VRAM threshold.

However, most GPU render farms use consumer or professional cards with lower memory density — 27 cards is a moderately large installation. Many studios still use CPU rendering or cloud burst rendering. The VFX industry would see some registration requirements for its largest on-premises GPU farms, but this is a relatively small number of facilities.

### 5. Autonomous vehicle development

**Impact: Negligible (already caught).** AV training clusters (Tesla, Waymo, Cruise) operate thousands of GPUs that already exceed the compute threshold by orders of magnitude. The VRAM threshold adds no new burden.

### 6. Cryptocurrency mining operations repurposed for AI

**Impact: Low.** Crypto mining GPUs have relatively low VRAM (consumer cards). A mining farm repurposed for AI with 40+ RTX 4090s (24 GB each) would cross 960 GB, but reaching 1,280 GB requires 54 cards at only ~4.5 H100-eq. These operations would be newly caught, though they are also potential evasion vectors and arguably should be monitored.

## Summary of registration impact

| Use case | Typical setup | Currently registered? | Newly registered under VRAM threshold? | Burden |
| :-- | :-- | :-- | :-- | :-- |
| Single DGX server | 8 GPUs, 640 GB | No | No | None |
| LLM inference (70B) | 1 DGX node, 640 GB | No | No | None |
| LLM inference (405B, production) | 3-4 DGX A100 nodes, 1,920-2,560 GB | No | **Yes** | Moderate |
| University cluster (small) | 17-50 A100 80GBs, 1,360-4,000 GB | No | **Yes** | Low-moderate |
| University cluster (large) | 80+ A100s or H100s | Already registered (compute) | N/A | None (already registered) |
| Scientific HPC (small) | 17-50 GPUs | No | **Yes** | Low |
| Scientific HPC (large) | 100+ GPUs | Already registered (compute) | N/A | None |
| VFX render farm | 27+ A40/L40 | No | **Yes (if large)** | Low |
| AV development | 1,000+ GPUs | Already registered (compute) | N/A | None |
| Cloud provider (per-customer) | Varies | Depends on allocation | Depends on allocation | See below |

### Cloud provider considerations

Cloud providers (AWS, GCP, Azure) operate clusters that vastly exceed both thresholds and are already subject to CCC registration for their aggregate infrastructure. The VRAM threshold raises a question about **per-customer allocations**: if a single cloud customer provisions 16+ A100 80GB instances (1,280 GB aggregate VRAM), does this constitute a newly formed CCC? The compute-only threshold already creates this question at 51+ A100 instances; the VRAM threshold lowers this to 17 A100 80GB instances. Cloud providers would likely need clearer guidance on per-tenant vs. per-facility registration.

## Assessment

The claim that the VRAM threshold is a "low-impact addition" that would "not substantially increase compliance costs and registration burden of users who are not doing AI training" is **partially accurate but overstated**:

**What's accurate:**
- Single-server configurations (1 DGX box) remain unregistered
- The threshold doesn't create a fundamentally new category of surveillance — the largest newly-caught systems (universities, national labs) are already visible institutions
- The primary enforcement benefit is real: it forces evasion hardware from A100-class (4,000 GB VRAM, 250B model) down to H100-class (1,280 GB, 91B model), greatly reducing evasion capability

**What's overstated:**
- Production LLM inference deployments with 3+ DGX A100 nodes would be newly registered — this is a large and growing category of legitimate non-training use
- Small-to-medium university and HPC clusters (17-50 high-memory GPUs) would face new registration requirements
- The threshold is low enough that it catches modest-scale legitimate deployments (3 DGX A100 boxes = 1,920 GB, just 7.6 H100-eq of compute)

**Recommendation:** The report should acknowledge that the VRAM threshold would require registration of medium-scale inference deployments and small institutional GPU clusters, while noting that these systems are typically operated by known entities with existing regulatory visibility. The net compliance burden is real but bounded, and significantly smaller than the enforcement benefit against distributed training evasion.
