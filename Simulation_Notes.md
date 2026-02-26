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
