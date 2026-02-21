# Research TODO: Papers to Study for Model Refinement

The following papers have been identified as high-priority sources to improve the accuracy of the Decentralized Training Simulator, particularly for modeling the $10^{26}$ and $10^{27}$ FLOP regimes.

---

## 1. Memory & Precision (Refining VRAM Logic)
*   **[ZeRO: Memory Optimizations for Deep Learning (Rajbhandari et al.)](https://arxiv.org/abs/1910.02054)**
    *   **Why:** To refine the "bytes per parameter" assumption. ZeRO-3 allows reducing memory states from ~16 bytes/param to ~2.1 bytes/param by sharding states across all nodes.
    *   **Impact on Simulator:** Add a "ZeRO-3" toggle that lowers VRAM requirements but adds a communication penalty for "parameter fetching" during the forward pass.
*   **[FP8 Training (NVIDIA/Transformer Engine)](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)**
    *   **Why:** Modern hardware (H100/GH200) uses FP8 for pre-training. 
    *   **Impact on Simulator:** Adjust the default precision from 2 bytes (FP16) to 1 byte (FP8), effectively doubling the "free" bandwidth and halving weight-sync volume.

## 2. Advanced Compression (Refining Bandwidth Logic)
*   **[MuLoCo: Muon inner optimizer for DiLoCo (2025)](https://github.com/lucidrains/muon)**
    *   **Why:** Investigates using the Muon optimizer to achieve 8x less communication than standard AdamW-DiLoCo.
    *   **Impact on Simulator:** Add a "Muon Optimizer" preset to the compression settings.
*   **[DeMo: Decoupled Momentum (Nous Research, 2024)](https://arxiv.org/abs/2411.19870)**
    *   **Why:** Proposes a way to decouple momentum states to allow for even more infrequent synchronization.
    *   **Impact on Simulator:** Allows modeling of even higher "Inner Steps" without the typical algorithmic penalty.

## 3. Robustness & Reality (Refining Throughput Logic)
*   **[On the Utility of Gradient Compression (Agarwal et al. 2021)](https://arxiv.org/abs/2102.04013)**
    *   **Why:** Provides empirical bounds on where compression starts to harm model convergence.
    *   **Impact on Simulator:** Add a "Warning" state when the user sets compression so high (e.g., >100x) that the model is unlikely to converge.
*   **Straggler & Failure Modeling (General Networking Theory)**
    *   **Why:** Real-world internet training is limited by the *slowest* node (the "Straggler").
    *   **Impact on Simulator:** Add a "Node Reliability" or "P99 Latency" slider. This would apply a statistical penalty to the `commSec` results, reflecting that synchronization takes as long as the slowest node's VPN spike.

---

## 4. The "Internet Scale" Frontier
*   **[DisTrO: Distributed Training Over-The-Internet (Nous Research)](https://nousresearch.com/distro/)**
    *   **Why:** This is currently the most aggressive claim for WAN training (100x-1000x bandwidth reduction).
    *   **Impact on Simulator:** Foundational for justifying the feasibility of $10^{27}$ FLOP runs. We need to study their technical releases to validate if these ratios hold for 10T+ parameter models.
