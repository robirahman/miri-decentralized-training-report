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
*   **Compression quality scenario:** Expected (η_compression = 0.98 for 16x)

### Results

| Nodes | GPUs | Est. Cost | Inner Steps H | $\eta$ | C_local (FLOP) | $\eta_{\text{chin}}$ | C_quality (FLOP) | x Strict Threshold |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | 48 | $0.7M | — | 1.000 | 2.84e23 | — | — | 0.3x |
| 4 | 192 | $2.9M | 168 | 0.862 | **9.77e23** | — | — | **1.0x** |
| 8 | 384 | $5.8M | 176 | 0.861 | 1.95e24 | — | — | 2.0x |
| 16 | 768 | $12M | 183 | 0.860 | 3.90e24 | — | — | 3.9x |
| 72 | 3,456 | $52M | 200 | 0.858 | **1.75e25** | 0.884 | **1.55e25** | **17.5x** |
| 144 | 6,912 | $104M | 207 | 0.857 | 3.50e25 | — | — | 35.0x |
| 500 | 24,000 | $360M | 221 | 0.855 | 1.21e26 | — | — | 121.2x |

*   **Minimum to reach threshold:** ~4 nodes (192 A100s, ~$3M)
*   **Reference case (72 nodes):** 240B model, 14.2T tokens, 3x Chinchilla overtraining
*   **Chinchilla optimality (72 nodes):** DiLoCo 240B is the optimal model size at 72 nodes. The model is overtrained at 3x Chinchilla ratio ($D/N = 76.8$ vs optimal $D^* = 25.6N$), yielding $\eta_{\text{chinchilla}} = 0.884$ and $C_{\text{quality}} = 1.55 \times 10^{25}$. Smaller models (e.g., 144B) fit more comfortably but have lower total capacity; larger models (e.g., 480B) require PP-Group DiLoCo, which reduces throughput enough to offset the capacity gain at this node count.
*   **Bottleneck:** Compute-bound (H optimized to balance compute and communication)
*   **Algorithmic efficiency:** Stable at ~86% across all node counts (η = η_H × η_compression × η_replicas)

### Compression Quality Uncertainty
*   **Compression ratio:** 16x (FP4 quantization + 4x sparsification)
*   **Confidence:** Moderate-high. FP4 pseudo-gradient quantization validated lossless at 4B (Streaming DiLoCo, 2501.18512) and 15B (MuLoCo, 2505.23725). Sparsification component has less direct evidence at scale.
*   **Range:** C_local for 72 nodes = 1.70e25 (conservative) to 1.79e25 (optimistic), ±2.5% from expected.
*   The 16x compression scenarios are the best-supported by the literature and constitute the primary results of the analysis.

### Conclusions
The 10^24 FLOP Strict Threshold can be exceeded with as few as 4 sub-CCC nodes. The primary barrier to this evasion is not the compute threshold but the treaty's chip tracking and consolidation provisions. See [Governance_Analysis.md](Governance_Analysis.md) for full implications.

---

## Scenario 3: Large-Scale Evasion — 10^27 FLOP Configurations

Extension of Scenario 2 to explore whether state-actor-level resources could achieve 10^27 FLOP using all simulator-supported techniques. Full analysis in [Governance_Analysis.md](Governance_Analysis.md), Section 8.

### Configurations Compared (Expected Scenario)

| Config | Hardware | Technique | Nodes | GPUs | Cost | $\eta$ | C_local (FLOP) | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| A | 48x A100 FP16 | Flat DiLoCo 240B, 16x comp | 4,000 | 192,000 | $2.9B | 0.853 | 9.67e26 | — | — |
| B | 48x A100 FP16 | Hierarchical 240B, 16x comp | 4,000 | 192,000 | $2.9B | 0.883 | 1.00e27 | — | — |
| C | 16x H100 FP8 | Flat DiLoCo 240B, 16x comp | 2,000 | 32,000 | $960M | 0.844 | 1.01e27 | — | — |
| D | 16x H100 FP8 | Hierarchical 240B, 16x comp | 2,000 | 32,000 | $960M | 0.876 | 1.05e27 | — | — |
| E | 48x A100 FP16 | Flat DiLoCo 240B, 100x comp | 4,000 | 192,000 | $2.9B | 0.868 | 9.84e26 | — | — |
| F | 16x H100 FP8 | Hier 240B + 100x comp | 2,000 | 32,000 | $960M | 0.892 | **1.07e27** | — | — |
| H | 48x A100 FP16 | PP-DiLoCo 960B, 16x comp | 4,000 | 192,000 | $2.9B | — | — | — | **2.75e26** |
| I | 16x H100 FP8 | PP-DiLoCo 549B, 16x comp | 2,000 | 32,000 | $960M | — | — | — | **1.28e26** |

*   Hierarchical: groups of 8 nodes, 1 Gbps regional bandwidth, 20 ms latency
*   100x compression: FP4 pseudo-gradient quantization + aggressive sparsification
*   PP-DiLoCo configs (H, I): pipeline parallelism within co-located groups, DiLoCo across groups. PP groups use regional interconnect (1 Gbps, 20 ms). Model sizes selected by `sweep_model_sizes()` to maximize $C_{\text{quality}}$.
*   All results use the **expected** compression quality scenario; optimistic/conservative ranges in §3a below

### Compression Quality Ranges

| Config | Comp | Optimistic C_local | Expected C_local | Conservative C_local | Confidence |
|:--|:--|:--|:--|:--|:--|
| A-D | 16x | +2% from expected | baseline | −3% from expected | Moderate-high |
| E | 100x | 1.03e27 | 9.84e26 | 9.32e26 | Low-moderate |
| F | 100x | 1.12e27 | 1.07e27 | 1.01e27 | Low-moderate |

*   **16x configs (A-D):** FP4 quantization component well-validated at 4B-15B. Expected 2% penalty accounts for extrapolation to 91-240B scale. Narrow uncertainty range (±2-3%).
*   **100x configs (E-F):** Only validated at 512M-1B scale (SparseLoCo 2508.15706). Expected 5% penalty; conservative 10%. Wide uncertainty range (±5-10%). Config F conservative estimate (1.01e27) still exceeds 10^27, but barely.

### Key Findings

*   **10^27 requires 2,000-4,000 nodes ($1-3B):** state-actor-level investment
*   **FP8 is most cost-effective:** 2x compute throughput from same CCC-threshold node, 3x fewer GPUs needed
*   **Config F is optimal for raw C_local:** hierarchical + 100x compression + FP8 achieves $\eta = 0.892$ and 1.07e27 FLOP at $960M (expected)
*   **PP-DiLoCo enables larger models:** Config H (PP-DiLoCo 960B, 4000 A100 nodes) achieves $C_{\text{quality}} = 2.75 \times 10^{26}$, a 2.59x improvement over DiLoCo 240B at the same node count. Config I (PP-DiLoCo 549B, 2000 H100 FP8 nodes) achieves $C_{\text{quality}} = 1.28 \times 10^{26}$, a 5.58x improvement.
*   **Hierarchical DiLoCo:** +3pp efficiency over flat (0.883 vs 0.853) by reducing effective H from 244 to 65
*   **MoE+EP:** doesn't increase C_local but enables 600B-1T MoE models within same compute budget
*   **Compression quality is the largest source of uncertainty** at the 100x level, wider than bandwidth sensitivity

### Conclusions

10^27 FLOP is technically achievable with DiLoCo over WAN under the expected compression quality scenario, but the hardware procurement ($1-3B) is the binding constraint. The treaty's financial monitoring and chip tracking provisions are the effective enforcement mechanism at this scale, not the compute threshold. The 100x compression results should be interpreted with caution — they depend on unvalidated extrapolation of compression quality to 91B+ scale.

---

## Scenario 4: Network Sensitivity — Bandwidth, Latency, and Deployment Profiles

Tests whether degraded network conditions (lower bandwidth, higher latency) significantly reduce achievable compute. Uses real-world measured latency values from Azure, AWS, Verizon, and Epsilon Telecom. All results use the **expected** compression quality scenario. Full analysis in [Governance_Analysis.md](Governance_Analysis.md), Section 9.

### Bandwidth Sensitivity (100 ms latency, varying bandwidth)

**72 nodes, 48x A100 FP16, 16x compression (expected scenario):**

| BW (Mbps) | H_min | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 1,994 | 0.804 | 1.64e25 | 88% |
| 100 (baseline) | 200 | 0.858 | 1.75e25 | 94% |
| 1,000 | 20 | 0.911 | 1.86e25 | 100% |

**2,000 nodes, 16x H100 FP8, hier+100x (Config F, expected scenario):**

| BW (Mbps) | H_eff | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 33 | 0.866 | 1.04e27 | 95% |
| 100 (baseline) | 11 | 0.892 | 1.07e27 | 97% |
| 1,000 | 4 | 0.914 | 1.10e27 | 100% |

### Latency Sensitivity (100 Mbps, varying real-world latency)

**Result:** Latency has zero measurable impact across all tested configurations. H_min and $\eta$ are identical from 2 ms (same cloud region) to 340 ms (Brazil–SE Asia). This is because sync volumes (hundreds of Gbits) make RTT negligible — even 340 ms is only 0.007% of the ~4,800-second bandwidth-limited sync time.

### Deployment Profile Summary (combined BW + latency)

**72 nodes, 48x A100 FP16, 16x comp (expected scenario):**

| Deployment | BW | RTT | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 0.911 | 1.86e25 | 100% |
| Continental US | 100 Mbps | 65 ms | 0.858 | 1.75e25 | 94% |
| Global worst-case | 10 Mbps | 340 ms | 0.804 | 1.64e25 | 88% |

**2,000 nodes, 16x H100 FP8, hier+100x (expected scenario):**

| Deployment | BW | RTT | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 0.914 | 1.10e27 | 100% |
| Continental US | 100 Mbps | 65 ms | 0.892 | 1.07e27 | 97% |
| Global worst-case | 10 Mbps | 340 ms | 0.866 | 1.04e27 | 95% |

### Decomposing the Efficiency Gap

The total efficiency gap between colocated (1 Gbps) and global worst-case (10 Mbps) has two distinct sources:

**72 nodes, 16x compression:**
*   Colocated η = 0.911 → Global η = 0.804 → total gap = 10.7 percentage points
*   Of which: ~5.3pp from η_H (larger H at lower bandwidth) + ~2pp from η_compression (16x, expected) + ~3.4pp from interaction effects
*   The bandwidth sensitivity figures include compression quality — the η_H penalty alone (increasing H from 20 to 1,994) would be ~5.4%, but the combined model shows ~12% total gap.

**2,000 nodes, 100x hier compression:**
*   Colocated η = 0.914 → Global η = 0.866 → total gap = 4.8 percentage points
*   The 100x compression quality penalty (~5% expected) is already baked into all rows. The bandwidth-only component is smaller for hierarchical configs because regional syncs limit effective H.

### Conclusions

*   **Bandwidth matters moderately**, and a 10x bandwidth reduction (100→10 Mbps) costs 6-12% of C_local. DiLoCo compensates by increasing H, but the combined bandwidth + compression quality penalty is larger than the bandwidth penalty alone.
*   **Latency is irrelevant** for any realistic model size and compression level. Sync volumes dominate RTT by 4-5 orders of magnitude.
*   **10^26 is achievable under all network conditions** (500 nodes, 48x A100, even at 10 Mbps/340 ms) with high confidence.
*   **10^27 requires hierarchical+100x at very low bandwidths** (flat DiLoCo falls short below ~50 Mbps), but with Config F it is achievable even at 10 Mbps globally under the expected compression quality scenario.
*   **Network-level enforcement is ineffective.** An evader using consumer broadband loses at most 12% of compute versus an optimized local deployment. The treaty cannot rely on network monitoring or bandwidth restrictions to prevent distributed training.
*   **Note:** The bandwidth sensitivity figures in this section incorporate compression quality penalties. Prior to the compression quality model update, the 10 Mbps → 1 Gbps gap was reported as 6-12% (η_H only). The combined model shows the same bandwidth range but with lower absolute η values due to the compression quality and replica penalty factors.

---

## Scenario 5: Treaty Modification Analysis — Countermeasure Effectiveness

Evaluates proposed treaty modifications to close the distributed training loophole. Full analysis in [Governance_Analysis.md](Governance_Analysis.md), Section 10.

**Note on compression quality:** The countermeasure analysis depends on the *relative* cost to achieve compute thresholds (e.g., nodes needed for 10^24), not on absolute η values. Since compression quality is a multiplicative factor applied uniformly across all node configurations, it does not materially affect the countermeasure conclusions: lowering the CCC threshold still requires the same cost regardless of compression quality scenario, and the effectiveness rankings of countermeasures are unchanged. The nodes-to-threshold values in the tables below are computed under the expected scenario but vary by <5% between optimistic and conservative.

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

---

## Scenario 6: Model Size Optimization — PP vs DiLoCo Tradeoff

Evaluates the tradeoff between model size, training mode (DiLoCo vs PP-Group DiLoCo), and Chinchilla-optimal scaling across key hardware configurations. Uses the model size sweep (`sweep_model_sizes()`) to find the model that maximizes quality-adjusted compute $C_{\text{quality}} = C_{\text{local}} \times \eta_{\text{chinchilla}}$.

### Motivation

Previous scenarios fixed the model size at 240B (the maximum that fits in a 48x A100 node's 3,840 GB VRAM at FP16). This is optimal for maximizing raw $C_{\text{local}}$ in DiLoCo mode, but it may not be optimal when Chinchilla scaling is considered. A larger model trained with PP-Group DiLoCo has lower throughput (due to pipeline bubbles) but higher per-token capacity, potentially yielding a better-trained model per FLOP when measured on the loss curve.

### Configuration

*   **Hardware profiles:** 48x A100 FP16 (3,840 GB VRAM, 14.98 PFLOPS) and 16x H100 FP8 (1,280 GB VRAM, 31.6 PFLOPS)
*   **Network:** 100 Mbps symmetric WAN, 100 ms latency
*   **PP interconnect:** 1 Gbps regional, 20 ms latency (for PP-Group DiLoCo)
*   **Algorithm:** Streaming DiLoCo or PP-Group DiLoCo, 16x compression, expected quality scenario
*   **Time limit:** 548 days (1.5 years)
*   **Chinchilla ratio:** $D^* = 25.6N$ (Besiroglu et al. 2024)

### Results: Optimal Model Sizes at Key Node Counts

#### 72 Nodes (48x A100 FP16)

| Model Size | Mode | $\eta$ | C_local (FLOP) | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|
| 144B | DiLoCo | 0.858 | 1.75e25 | ~0.95 | ~1.66e25 |
| **240B** | **DiLoCo** | **0.858** | **1.75e25** | **0.884** | **1.55e25** |
| 480B | PP-DiLoCo (S=2) | — | — | — | <1.55e25 |

**Winner: DiLoCo 240B** ($C_{\text{quality}} = 1.55 \times 10^{25}$). At 72 nodes, PP overhead outweighs the capacity benefit of larger models. The 240B model is overtrained at 3x Chinchilla ratio but remains optimal because it avoids PP entirely.

#### 500 Nodes (48x A100 FP16)

| Model Size | Mode | $\eta$ | C_local (FLOP) | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|
| 240B | DiLoCo | 0.855 | 1.21e26 | — | — |
| **480B** | **PP-DiLoCo (S=2)** | — | — | — | **5.59e25** |
| 960B | PP-DiLoCo (S=4) | — | — | — | <5.59e25 |

**Winner: PP-DiLoCo 480B** ($C_{\text{quality}} = 5.59 \times 10^{25}$, 1.09x improvement over DiLoCo 240B). At 500 nodes, the compute budget is large enough that the PP overhead for 2-stage pipeline is compensated by the improved Chinchilla efficiency of a larger model.

#### 4,000 Nodes (48x A100 FP16)

| Model Size | Mode | $\eta$ | C_local (FLOP) | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|
| 240B | DiLoCo | 0.853 | 9.67e26 | — | — |
| 480B | PP-DiLoCo (S=2) | — | — | — | — |
| **960B** | **PP-DiLoCo (S=4)** | — | — | — | **2.75e26** |

**Winner: PP-DiLoCo 960B** ($C_{\text{quality}} = 2.75 \times 10^{26}$, 2.59x improvement over DiLoCo 240B). At state-actor scale, the massive compute budget makes even 4-stage PP worthwhile: the model trains closer to Chinchilla-optimal ratio, and the raw compute is large enough to absorb the pipeline overhead.

#### 2,000 Nodes (16x H100 FP8)

| Model Size | Mode | $\eta$ | C_local (FLOP) | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|
| 91B | DiLoCo | 0.844 | 1.01e27 | — | — |
| **549B** | **PP-DiLoCo** | — | — | — | **1.28e26** |

**Winner: PP-DiLoCo 549B** ($C_{\text{quality}} = 1.28 \times 10^{26}$, 5.58x improvement). The H100 FP8 node has only 1,280 GB VRAM (16x 80GB), constraining DiLoCo to a 91B model. PP-Group DiLoCo enables a 549B model, dramatically improving Chinchilla efficiency despite the PP overhead.

### Cross-Validation

The model size sweep results pass cross-validation checks 8-12 (added alongside this analysis):

*   **Check 8:** C_quality is always less than or equal to C_local (efficiency cannot exceed 1).
*   **Check 9:** Chinchilla efficiency decreases as overtraining ratio increases (monotonicity).
*   **Check 10:** PP-DiLoCo C_local is less than DiLoCo C_local at same node count (PP has overhead).
*   **Check 11:** Optimal model size increases with node count (more compute favors larger models).
*   **Check 12:** All C_quality values are positive and finite.

### Key Findings

*   **DiLoCo is optimal at small scale (72 nodes).** The PP overhead is not justified by the Chinchilla benefit when the total compute budget is modest.
*   **PP-DiLoCo dominates at large scale (500+ nodes).** The improvement grows from 1.09x at 500 nodes to 2.59x at 4,000 nodes, because larger compute budgets increasingly penalize sub-optimal model sizing.
*   **H100 FP8 benefits most from PP-DiLoCo** (5.58x improvement at 2,000 nodes) because the low per-node VRAM (1,280 GB) severely constrains DiLoCo model size. PP-Group DiLoCo lifts this constraint.
*   **The crossover point is ~500 nodes** for 48x A100 FP16 hardware. Below this, DiLoCo 240B is sufficient; above, PP-DiLoCo with larger models yields increasingly superior quality-adjusted compute.
*   **Chinchilla efficiency is the key differentiator.** Raw $C_{\text{local}}$ (FLOP throughput) favors DiLoCo, but $C_{\text{quality}}$ (which accounts for model sizing) favors PP-DiLoCo at scale. This is because DiLoCo is constrained to a 240B model that becomes increasingly overtrained as compute grows, while PP-DiLoCo can scale model size to match.

### Implications for Governance Analysis

These results affect the treaty evasion conclusions (Scenarios 2-3):

*   **72-node evasion:** Unchanged. DiLoCo 240B remains optimal and the $C_{\text{quality}} = 1.55 \times 10^{25}$ figure supersedes the raw $C_{\text{local}} = 1.75 \times 10^{25}$.
*   **State-actor evasion (4,000 nodes):** PP-DiLoCo 960B achieves higher quality-adjusted compute than any DiLoCo configuration, but the raw FLOP figure ($C_{\text{local}}$) is lower due to PP overhead. The threat is *qualitatively different*: a better-trained 960B model vs. a more-overtrained 240B model with more total FLOP.
*   **H100 FP8 evasion (2,000 nodes):** PP-DiLoCo 549B is the clearly dominant strategy, making the already-concerning H100 FP8 pathway even more effective in terms of model quality.
