# Technical Comparison: MIRI Decentralized Training Simulator vs. Douillard DiLoCo Simulator

This document outlines how the MIRI simulator incorporates and extends the research logic found in Arthur Douillard's [DiLoCo Bandwidth Simulator](https://arthurdouillard.com/diloco/index.html).

## 1. Core Methodology Mapping

| Feature | Douillard Simulator | MIRI Simulator |
| :--- | :--- | :--- |
| **Throughput Model** | Compute Utilization (CU) = $\frac{T_{comp}}{T_{comp} + T_{stall}}$ | Step Time = $\max(T_{comp}, T_{comm})$. Identical to ideal Streaming DiLoCo. |
| **Bandwidth Logic** | Linear bits/bandwidth calculation. | Linear bits/bandwidth calculation with adjustable **Compression** factor. |
| **Precision Support** | Selectable bytes per param (BF16, FP8, FP4). | **Added:** Dynamic precision selector that adjusts both Comm volume and VRAM footprint. |
| **Algorithmic Penalty** | Not explicitly in the UI (referenced in papers). | **Included:** $\eta = 1 - \alpha \cdot \log_{10}(H)$ based on *Charles et al. (2025)*. |
| **Architecture** | Single-tier Data Parallel (DiLoCo). | Multi-tier: Hierarchical Sync, Pipeline Parallelism (PP), and Data Parallelism. |

## 2. Shared Assumptions & Special Case Alignment

The MIRI simulator replicates the Douillard simulator as a special case when:
- **Use Hierarchy** is unchecked.
- **Model Size** fits within single-node VRAM (`isSharded = false`).
- **Precision** settings match.

In this state, the "Bottleneck" indicator and the "Total Training Time" in our simulator align with the "Compute Utilization" trends in Douillard's tool.

## 3. Extensions in the MIRI Simulator

### A. Algorithmic Efficiency ($\eta$)
While Douillard's web tool focuses on *throughput* (how fast you can push bits), the MIRI simulator incorporates the *quality* of training. Based on Douillard's recent Scaling Laws research, we apply a penalty for large $H$ values, reflecting that decentralized steps are not 100% equivalent to synchronous steps.

### B. Memory-Triggered Mode Switching
Our simulator includes a sharding trigger. In Douillard's tool, you can simulate a 1000B model on a single node regardless of memory. Our simulator detects if `Model_Size * bytesPerParam > Node_VRAM` and automatically switches to **Pipeline Parallelism**, introducing activation-sync latency penalties which are the primary bottleneck for models $> 1$T parameters.

### C. Hierarchical "Cluster of Clusters"
We allow modeling of regional high-speed interconnects (e.g., nodes in the same city) nested within a global WAN. This addresses the "SPES Protocol" style of MoE and DiLoCo training, which is not covered in the base DiLoCo simulator.

## 4. Conclusion
The MIRI simulator acts as a comprehensive superset of the research included in Douillard's simulator. It captures the networking trade-offs of DiLoCo while adding the critical engineering constraints of memory sharding and the algorithmic constraints of long-interval synchronization.
