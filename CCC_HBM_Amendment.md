# Proposed Amendment: Adding Accelerator Memory to the CCC Definition

## 1. Current Treaty Text (Article II, Definition 13)

> **Covered chip cluster (CCC)** means any set of AI chips or networked cluster with aggregate effective computing capacity greater than 16 H100-equivalents. A networked cluster refers to chips that either are physically co-located, have inter-node aggregate bandwidth — defined as the sum of bandwidth between distinct hosts/chassis — greater than 25 Gbit/s, or are networked to perform workloads together.
>
> The aggregate effective computing capacity of 16 H100 chips is 15,840 TFLOP/s, or 253,440 TPP, and is based on the sum of per-chip TPP. Examples of CCCs would include: the GB200 NVL72 server, three eight-way H100 HGX servers residing in the same building, CloudMatrix 384, a pod with 32 TPUv6e chips, every supercomputer.

Current Notes on Article II state:

> Other chip metrics are important in AI training (such as high bandwidth memory), but overall, these matter less than the number of operations per second.

## 2. Proposed Amended Text

### Definition 13 (amended)

> **Covered chip cluster (CCC)** means any set of AI chips or networked cluster with (a) aggregate effective computing capacity greater than 16 H100-equivalents, **or (b) aggregate accelerator memory greater than 1,280 GB**. A networked cluster refers to chips that either are physically co-located, have inter-node aggregate bandwidth — defined as the sum of bandwidth between distinct hosts/chassis — greater than 25 Gbit/s, or are networked to perform workloads together.
>
> The aggregate effective computing capacity of 16 H100 chips is 15,840 TFLOP/s, or 253,440 TPP, and is based on the sum of per-chip TPP. **The aggregate accelerator memory of 16 H100 SXM chips is 1,280 GB (16 x 80 GB HBM3).** Examples of CCCs would include: the GB200 NVL72 server, three eight-way H100 HGX servers residing in the same building, CloudMatrix 384, a pod with 32 TPUv6e chips, every supercomputer, **a cluster of 25 or more A100 80 GB accelerators, or a cluster of 21 or more Huawei Ascend 910B accelerators**.

### New Definition 13a

> **Accelerator memory** means the aggregate high-bandwidth memory (HBM), or equivalent on-chip or on-package memory directly accessible by the AI chip's processing cores, measured in gigabytes (GB). For chips with unified memory architectures, accelerator memory is the total unified memory capacity. Host system memory (DDR, LPDDR) not directly accessible by the AI chip's processing cores is excluded.

### Amended Notes on Article II

Replace:

> Other chip metrics are important in AI training (such as high bandwidth memory), but overall, these matter less than the number of operations per second.

With:

> Both computing capacity and accelerator memory are critical metrics for AI training. Computing capacity determines the rate at which training operations execute. Accelerator memory determines the maximum model size that can be trained on a single node without cross-node model parallelism. A cluster's aggregate accelerator memory is therefore a direct constraint on the complexity of models it can produce. The CCC definition thresholds both metrics at the level of 16 H100 SXM accelerators (15,840 TFLOP/s computing capacity and 1,280 GB accelerator memory), so that any cluster exceeding a 16x H100 node in either dimension is subject to registration and monitoring.
>
> The memory threshold closes an evasion path identified in distributed training analysis: without it, an actor could select high-memory, low-compute accelerators (e.g., 50 A100 80 GB GPUs providing 4,000 GB of memory at only 15.8 H100-equivalents of compute) to train models of 250 billion parameters on unregistered nodes. With the memory threshold, the most capable unregistered node is limited to 16 accelerators of H100 class or below (1,280 GB, approximately 80 billion parameters).

## 3. Impact on Hardware Configurations

| Configuration | Compute | Memory | CCC (current)? | CCC (amended)? | Change |
|:--|:--|:--|:--|:--|:--|
| **NVIDIA** | | | | | |
| 50x A100 80GB | 15.8 H100-eq | 4,000 GB | No | **Yes** (memory) | **Newly caught** |
| 25x A100 80GB | 7.9 H100-eq | 2,000 GB | No | **Yes** (memory) | **Newly caught** |
| 17x A100 80GB | 5.3 H100-eq | 1,360 GB | No | **Yes** (memory) | **Newly caught** |
| 16x GH200 NVL2 | 16.0 H100-eq | 2,304 GB | Borderline | **Yes** (memory) | Unambiguously caught |
| 16x H200 | 16.0 H100-eq | 2,256 GB | Borderline | **Yes** (memory) | Unambiguously caught |
| 9x B100 | 15.9 H100-eq | 1,728 GB | No | **Yes** (memory) | **Newly caught** |
| 16x H100 SXM | 16.0 H100-eq | 1,280 GB | Borderline | Borderline | No change |
| 16x A100 80GB | 5.0 H100-eq | 1,280 GB | No | No | No change |
| 12x H100 SXM | 12.0 H100-eq | 960 GB | No | No | No change |
| 8x H100 SXM (DGX) | 8.0 H100-eq | 640 GB | No | No | No change |
| 8x A100 80GB (DGX) | 2.5 H100-eq | 640 GB | No | No | No change |
| 4x H100 SXM | 4.0 H100-eq | 320 GB | No | No | No change |
| 3x 8-way H100 HGX | 24.0 H100-eq | 1,920 GB | Yes (compute) | Yes (both) | No change |
| GB200 NVL72 | >>16 H100-eq | >>1,280 GB | Yes (compute) | Yes (both) | No change |
| **Chinese chips** | | | | | |
| 49x Ascend 910B | 15.8 H100-eq | 3,136 GB | No | **Yes** (memory) | **Newly caught** |
| 26x Ascend 910C | 15.8 H100-eq | 3,328 GB | No | **Yes** (memory) | **Newly caught** |
| 8x Ascend 910B (Atlas 900) | 2.6 H100-eq | 512 GB | No | No | No change |
| **Google TPUs** | | | | | |
| 57x TPU v4 | 15.8 H100-eq | 1,824 GB | No | **Yes** (memory) | **Newly caught** |
| 34x TPU v5p | 15.8 H100-eq | 3,230 GB | No | **Yes** (memory) | **Newly caught** |
| 80x TPU v5e | 15.9 H100-eq | 1,280 GB | No | Borderline | No change |
| 17x TPU v6e | 15.8 H100-eq | 544 GB | No | No | No change |
| 8x TPU v5e (pod slice) | 1.6 H100-eq | 128 GB | No | No | No change |

**Key observations:**

- **No new collateral on standard legitimate systems.** All newly caught configurations exceed 16 accelerators and are already unusual outside of dedicated ML clusters. Standard research workstations (4-8 GPUs), cloud instances (up to 8 GPUs), and small TPU pod slices (8 chips) are unaffected.
- **The 16x H100 SXM node remains at the boundary** of both thresholds, exactly as under the current treaty. It is the most performant non-CCC configuration under the amended definition.
- **The high-HBM exploit is closed across all chip families.** Maximum model capacity on a single unregistered node drops from 250B (50x A100) to ~91B (16x H100 FP8 or 20x Ascend 910B) — roughly a 3x reduction.
- **Chinese chip evasion paths are constrained.** Huawei Ascend chips are only available domestically in China due to export controls, but a Chinese actor could use 49x Ascend 910B nodes (3,136 GB, 224B model) as a cost-effective evasion path. The memory threshold forces these down to 20 chips (1,280 GB, 91B model) — a 59% reduction in model capacity.
- **Chips with large per-unit memory but low compute** (e.g., hypothetical future chips with 256 GB HBM but low FLOP/s) cannot be accumulated beyond 5 units without triggering registration.

## 4. Rationale

### 4.1 Why a memory threshold?

The current CCC definition thresholds only on compute capacity (FLOP/s). This creates an exploitable asymmetry: an evader can select hardware that maximizes memory while staying under the compute threshold, enabling training of much larger AI models than the compute threshold alone would suggest.

The [Governance Analysis](Governance%20Analysis%20by%20Claude.md) (Section 10.2) identifies this as the single most impactful countermeasure against distributed training evasion. A 1,280 GB threshold:

- **Increases minimum evasion cost** for reaching 10^25 quality-adjusted FLOP from ~$831M to ~$1.22B (1.5x). Without the memory threshold, the cheapest path uses Ascend 910B nodes with 3,136 GB HBM (available only to a Chinese actor); the cheapest non-Chinese path uses B100 nodes at $906M. With the threshold, the cheapest path for any actor is B100 nodes at $1.22B. (Costs computed using early-2026 GPU prices; Ascend prices based on ¥110,000/910B and ¥180,000/910C.)
- **Reduces maximum model size** on a single unregistered node from 250B (50x A100 80GB) to ~91B parameters (16x H100 FP8), forcing severe overtraining at moderate node counts or complex pipeline parallelism at large node counts.
- **Has zero collateral** on representative legitimate computing systems — no standard research workstation, cloud instance, TPU pod slice, or academic cluster is affected.

### 4.2 Why 1,280 GB specifically?

The threshold is set at the aggregate accelerator memory of 16 H100 SXM accelerators (16 x 80 GB = 1,280 GB). This choice has three advantages:

1. **Consistency with the compute threshold.** Both thresholds reference the same hardware baseline (16x H100 SXM), making the definition conceptually clean: a CCC is any cluster that exceeds a 16x H100 node in either processing power or memory.

2. **No additional collateral.** Any legitimate system with more than 1,280 GB of accelerator memory almost certainly also exceeds 16 H100-equivalents of compute (since high-memory systems are generally also high-compute). The memory threshold catches configurations that the compute threshold misses only when an actor deliberately selects low-compute, high-memory hardware — which is precisely the evasion behavior the threshold is designed to prevent.

3. **Future-proofing.** As accelerator memory per chip increases (e.g., H200 at 141 GB, GH200 at 144 GB, future chips at 192-256 GB), the 1,280 GB threshold automatically limits the number of next-generation chips that can be aggregated without registration — roughly 9 H200s or 8 GH200s. This prevents the threshold from becoming permissive as memory density increases, without requiring frequent numerical updates.

### 4.3 Why a flat threshold, not an HBM:FLOPS ratio?

A ratio-based threshold (e.g., "HBM per H100-equivalent must not exceed X") would more surgically target the exploit but has a critical flaw: **it is gameable at the cluster level**. An evader could add a small number of high-FLOPS, low-memory chips to an A100 cluster, diluting the aggregate ratio below the threshold while retaining the A100 HBM. A per-chip ratio would amount to banning certain chip types from clusters, which is better handled by the flat memory threshold that directly constrains what matters (total memory available for model training).

### 4.4 Effect on distributed training evasion

Under the amended definition, the optimal evasion node for NVIDIA hardware becomes 16x H100 SXM (80 GB each, 1,280 GB total, 16.0 H100-equivalents), limiting single-node model capacity to approximately **91 billion parameters** in FP8 — down from 250B (50x A100 FP16) under the current definition. For Chinese hardware, the optimal node drops from 49x Ascend 910B (3,136 GB, 224B) to 20x Ascend 910B (1,280 GB, 91B).

The minimum-cost evasion configurations at key compute scales (all links 100 Mbps / 100 ms WAN, early-2026 GPU prices):

| Target C_quality | Current definition (cheapest) | Amended definition (cheapest) | Cost ratio |
|:--|:--|:--|:--|
| 10^24 | 3 nodes B100, 93B model, Flat DiLoCo, $864K | 3 nodes H100, 91B model, Flat DiLoCo, $1.2M | 1.4x |
| 10^25 (Chinese actor) | 1,060 nodes Ascend 910B, 896B PP-DiLoCo, $831M | 6,352 nodes B100, 658B PP-DiLoCo, $1.22B | 1.5x |
| 10^25 (non-Chinese actor) | 3,144 nodes B100, 741B PP-DiLoCo, $906M | 6,352 nodes B100, 658B PP-DiLoCo, $1.22B | 1.3x |

At the 10^25 scale — roughly the compute budget of a GPT-4-class training run — the memory threshold forces minimum evasion cost from **$831M to $1.22B** (1.5x increase). Without the memory threshold, a Chinese actor could exploit Ascend 910B nodes (49 chips, 3,136 GB HBM, $784K/node) using only 4-stage PP-Group DiLoCo. Huawei Ascend chips are only available domestically in China due to export controls; a non-Chinese actor's cheapest path is B100 nodes at $906M. With the threshold, the cheapest path for any actor converges to B100 nodes at $1.22B, requiring 8 pipeline stages instead of 4-6.

The evader's options under the amended definition are:

1. **Accept the overtraining penalty** — with model size capped at ~91B on a single node, flat DiLoCo becomes severely overtrained at moderate node counts, producing a less capable model per FLOP spent.
2. **Use PP-Group DiLoCo** to shard a larger model across multiple nodes — but this requires co-locating pipeline stages, partially re-creating the physical clustering that the treaty's monitoring mechanisms detect, and incurring pipeline bubble overhead.
3. **Use fewer nodes** to reduce overtraining — but this reduces total compute proportionally.

All three options represent a meaningful degradation of the evader's position relative to the current definition.
