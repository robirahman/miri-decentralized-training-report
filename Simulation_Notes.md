# Simulation Notes

## Scenario 1: 10^25 FLOP Baseline

### Configuration
*   **Target:** 1.02e25 FLOPs (12T tokens, 144B parameter model)
*   **Hardware:** 72 nodes, each with 16x GH200 (estimated 32 PFLOPS/node at 40% MFU)
*   **Network:** 100 Mbps symmetric WAN bandwidth per node
*   **Algorithm:** Streaming DiLoCo
    *   Inner steps: 128
    *   Compression: 16x (e.g., 4-bit quantization + sparsification)
    *   Local Batch: 131k tokens

### Results
*   **Estimated Training Time:** 331.14 days
*   **Bottleneck:** Communication (Network Bandwidth)
*   **Duty Cycle Analysis:**
    *   Compute Block (128 steps): ~1,132 seconds (~19 mins)
    *   Sync Block (Upload/Download): ~2,880 seconds (~48 mins)
    *   Even with infrequent synchronization, the 100 Mbps link keeps the GPUs idle for significant portions of the run if synchronization is blocking.

### Conclusions
The $10^{25}$ FLOP run is technically feasible within a one-year window, but it is highly sensitive to inter-node bandwidth. Any reduction in WAN speed or increase in model size (requiring more frequent sync) quickly pushes the project into "Impractical" territory (>1 year).

---

## Scenario 2: Treaty Evasion — Maximum Compute Below CCC Threshold

Under the [MIRI treaty proposal](https://arxiv.org/abs/2511.10783), clusters exceeding 16 H100-equivalents (15,840 TFLOPS FP16) must be registered. This scenario examines the maximum local-equivalent compute achievable with unregistered sub-threshold nodes over WAN. Full analysis in [Governance_Analysis.md](Governance_Analysis.md).

### Configuration
*   **Hardware:** Nodes of 48x A100 80GB (14.98 PFLOPS = 15.1 H100-equiv, **under threshold**)
*   **VRAM:** 3,840 GB per node (enables 240B dense model at FP16)
*   **Network:** 100 Mbps symmetric WAN, 100 ms latency
*   **Algorithm:** Streaming DiLoCo, 16x compression, optimized inner steps
*   **Time Limit:** 1.5 years (548 days), based on slower growth rates under treaty restrictions

### Results

| Nodes | GPUs | Est. Cost | Inner Steps H | $\eta$ | C_local (FLOP) | x Strict Threshold |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | 48 | $0.7M | — | 1.000 | 2.84e23 | 0.3x |
| 4 | 192 | $2.9M | 168 | 0.879 | **9.97e23** | **1.0x** |
| 8 | 384 | $5.8M | 176 | 0.878 | 1.99e24 | 2.0x |
| 16 | 768 | $12M | 183 | 0.877 | 3.98e24 | 4.0x |
| 72 | 3,456 | $52M | 200 | 0.875 | **1.79e25** | **17.9x** |
| 144 | 6,912 | $104M | 207 | 0.874 | 3.57e25 | 35.7x |
| 500 | 24,000 | $360M | 221 | 0.873 | 1.24e26 | 123.8x |

*   **Minimum to reach threshold:** ~4 nodes (192 A100s, ~$3M)
*   **Reference case (72 nodes):** 240B model, 14.2T tokens, 3x Chinchilla overtraining
*   **Bottleneck:** Compute-bound (H optimized to balance compute and communication)
*   **Algorithmic efficiency:** Stable at ~87% across all node counts

### Conclusions
The 10^24 FLOP Strict Threshold can be exceeded with as few as 4 sub-CCC nodes. The primary barrier to this evasion is not the compute threshold but the treaty's chip tracking and consolidation provisions. See [Governance_Analysis.md](Governance_Analysis.md) for full implications.

---

## Scenario 3: Large-Scale Evasion — 10^27 FLOP Configurations

Extension of Scenario 2 to explore whether state-actor-level resources could achieve 10^27 FLOP using all simulator-supported techniques. Full analysis in [Governance_Analysis.md](Governance_Analysis.md), Section 8.

### Configurations Compared

| Config | Hardware | Technique | Nodes | GPUs | Cost | $\eta$ | C_local (FLOP) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| A | 48x A100 FP16 | Flat DiLoCo, 16x comp | 4,000 | 192,000 | $2.9B | 0.871 | 9.87e26 |
| B | 48x A100 FP16 | Hierarchical, 16x comp | 4,000 | 192,000 | $2.9B | 0.902 | 1.02e27 |
| C | 16x H100 FP8 | Flat DiLoCo, 16x comp | 2,000 | 32,000 | $960M | 0.862 | 1.03e27 |
| D | 16x H100 FP8 | Hierarchical, 16x comp | 2,000 | 32,000 | $960M | 0.896 | 1.07e27 |
| E | 48x A100 FP16 | Flat DiLoCo, 100x comp | 4,000 | 192,000 | $2.9B | 0.914 | 1.04e27 |
| F | 16x H100 FP8 | Hier + 100x comp | 2,000 | 32,000 | $960M | 0.941 | **1.13e27** |

*   Hierarchical: groups of 8 nodes, 1 Gbps regional bandwidth, 20 ms latency
*   100x compression: FP4 pseudo-gradient quantization + aggressive sparsification

### Key Findings

*   **10^27 requires 2,000-4,000 nodes ($1-3B):** state-actor-level investment
*   **FP8 is most cost-effective:** 2x compute throughput from same CCC-threshold node, 3x fewer GPUs needed
*   **Config F is optimal:** hierarchical + 100x compression + FP8 achieves $\eta = 0.941$ and 1.13e27 FLOP at $960M
*   **Hierarchical DiLoCo:** +3pp efficiency over flat (0.902 vs 0.871) by reducing effective H from 244 to 65
*   **MoE+EP:** doesn't increase C_local but enables 600B-1T MoE models within same compute budget

### Conclusions

10^27 FLOP is technically achievable with DiLoCo over WAN, but the hardware procurement ($1-3B) is the binding constraint. The treaty's financial monitoring and chip tracking provisions are the effective enforcement mechanism at this scale, not the compute threshold.

---

## Scenario 4: Network Sensitivity — Bandwidth, Latency, and Deployment Profiles

Tests whether degraded network conditions (lower bandwidth, higher latency) significantly reduce achievable compute. Uses real-world measured latency values from Azure, AWS, Verizon, and Epsilon Telecom. Full analysis in [Governance_Analysis.md](Governance_Analysis.md), Section 9.

### Bandwidth Sensitivity (100 ms latency, varying bandwidth)

**72 nodes, 48x A100 FP16, 16x compression:**

| BW (Mbps) | H_min | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 1,994 | 0.821 | 1.68e25 | 88% |
| 100 (baseline) | 200 | 0.875 | 1.79e25 | 94% |
| 1,000 | 20 | 0.929 | 1.90e25 | 100% |

**2,000 nodes, 16x H100 FP8, hier+100x (Config F):**

| BW (Mbps) | H_eff | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 33 | 0.913 | 1.10e27 | 95% |
| 100 (baseline) | 11 | 0.941 | 1.13e27 | 98% |
| 1,000 | 4 | 0.964 | 1.16e27 | 100% |

### Latency Sensitivity (100 Mbps, varying real-world latency)

**Result:** Latency has zero measurable impact across all tested configurations. H_min and $\eta$ are identical from 2 ms (same cloud region) to 340 ms (Brazil–SE Asia). This is because sync volumes (hundreds of Gbits) make RTT negligible — even 340 ms is only 0.007% of the ~4,800-second bandwidth-limited sync time.

### Deployment Profile Summary (combined BW + latency)

**72 nodes, 48x A100 FP16, 16x comp:**

| Deployment | BW | RTT | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 0.929 | 1.90e25 | 100% |
| Continental US | 100 Mbps | 65 ms | 0.875 | 1.79e25 | 94% |
| Global worst-case | 10 Mbps | 340 ms | 0.821 | 1.68e25 | 88% |

**2,000 nodes, 16x H100 FP8, hier+100x:**

| Deployment | BW | RTT | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 0.964 | 1.16e27 | 100% |
| Continental US | 100 Mbps | 65 ms | 0.941 | 1.13e27 | 97% |
| Global worst-case | 10 Mbps | 340 ms | 0.913 | 1.10e27 | 95% |

### Conclusions

*   **Bandwidth is the only network parameter that matters**, and even a 10x bandwidth reduction (100→10 Mbps) costs only 6-12% of C_local. DiLoCo compensates by increasing H.
*   **Latency is irrelevant** for any realistic model size and compression level. Sync volumes dominate RTT by 4-5 orders of magnitude.
*   **10^26 is achievable under all network conditions** (500 nodes, 48x A100, even at 10 Mbps/340 ms).
*   **10^27 requires hierarchical+100x at very low bandwidths** (flat DiLoCo falls short below ~50 Mbps), but with Config F it is achievable even at 10 Mbps globally.
*   **Network-level enforcement is ineffective.** An evader using consumer broadband loses at most 12% of compute versus an optimized local deployment. The treaty cannot rely on network monitoring or bandwidth restrictions to prevent distributed training.

---

## Scenario 5: Treaty Modification Analysis — Countermeasure Effectiveness

Evaluates proposed treaty modifications to close the distributed training loophole. Full analysis in [Governance_Analysis.md](Governance_Analysis.md), Section 10.

### Lowering the CCC Compute Threshold

For each candidate threshold, shows the maximum sub-threshold node (A100 80GB) and nodes/cost to reach 10^24 FLOP:

| CCC Threshold | Max GPUs/node | VRAM | Max Model | Nodes for 10^24 | Cost |
|:--|:--|:--|:--|:--|:--|
| 16 (current) | 48 | 3,840 GB | 240B | 5 | $4M |
| 8 | 25 | 2,000 GB | 125B | 8 | $3M |
| 4 | 12 | 960 GB | 60B | 16 | $3M |
| 2 | 6 | 480 GB | 30B | 31 | $3M |
| 1 | 3 | 240 GB | 15B | 61 | $3M |

*   **Cost is nearly constant (~$3M) regardless of threshold.** DiLoCo redistributes the same total GPUs into more, smaller nodes.
*   **Model quality degrades significantly:** 240B at current threshold vs 15B at threshold=1.
*   **Registration burden escalates:** 0 legitimate systems caught at 16; 4 at threshold 4; 11 at threshold 1. (Systems are subject to registration and inspection, not banned.)
*   **Indirect benefit:** More nodes = more inspection points, increasing detection probability for each node.

### Memory Threshold Analysis

| VRAM Limit | Max A100s | Max Model | Nodes for 10^24 | Cost |
|:--|:--|:--|:--|:--|
| No limit | 48 | 240B | 5 | $4M |
| 2 TB | 25 | 125B | 8 | $3M |
| 1 TB | 12 | 60B | 16 | $3M |
| 512 GB | 6 | 30B | 31 | $3M |

*   Same pattern as compute threshold: cost unchanged, model size constrained.
*   512 GB limit catches 6 of 14 representative legitimate systems (all DGX, 8-GPU cloud instances).
*   1 TB limit catches 0 legitimate systems while still constraining the 48x A100 exploit.

### Countermeasure Effectiveness Summary

| Countermeasure | vs Non-State | vs State | Reg. Burden | Recommended? |
|:--|:--|:--|:--|:--|
| Lower CCC threshold (to 4-8) | Low-Med | None | Moderate | **Consider** |
| Memory threshold (1 TB) | Low-Med | None | Low | **Yes** |
| Bandwidth caps | None | None | Moderate | No (technically ineffective) |
| Traffic monitoring at GPU sites | Low-Med | Low | Low | **Yes** (supplementary) |
| TEE/Remote attestation | **High** | Low | Medium | **Yes** (medium-term) |
| Orchestration regulation | None | None | Low | No (unenforceable) |
| Model possession redefinition | Medium | Low | None | **Yes** |
| Enhanced chip tracking | **Med-High** | Low | Low | **Yes** |
| Whistleblower programs | **Med-High** | Low | None | **Yes** |

Note: "Reg. Burden" refers to the registration, reporting, and inspection obligations imposed on legitimate computing systems — not bans or confiscation. Systems caught by lower thresholds must be registered and may be subject to inspections, but continue to operate normally.

### Conclusions

*   **No single modification closes the loophole.** The fundamental challenge is that DiLoCo's cost depends on total GPUs, not per-node size, so threshold-based countermeasures cannot increase evasion cost. However, they can increase inspection surface and constrain model quality.
*   **Against non-state actors:** TEE-based attestation + enhanced chip tracking + whistleblower programs is the most effective combination. Lower thresholds and traffic monitoring provide supplementary value.
*   **Against state actors:** Only diplomatic, intelligence, and financial instruments are effective. Technical countermeasures are insufficient against actors with domestic chip manufacturing and classified procurement.
*   **Recommended package:** 1 TB VRAM threshold, TEE attestation mandate, model possession redefinition, enhanced whistleblower bounties, utilization reporting, consider lowering CCC threshold to 4-8 H100-eq, traffic monitoring (not caps) at GPU facilities.
