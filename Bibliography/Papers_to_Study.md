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

---

## 5. Recently Identified Research (February 2026 Literature Review)

These papers were identified during a systematic literature review and are directly relevant to the simulator's current modeling gaps.

### Pipeline Parallelism Modeling
*   **[Zero Bubble Pipeline Parallelism (Qi et al., 2024)](https://arxiv.org/abs/2401.10241)**
    *   **Why:** Achieves zero pipeline bubbles by splitting backward into input-gradient and weight-gradient phases. Foundation for DeepSeek-V3's DualPipe. Now standard practice at frontier labs.
    *   **Impact on Simulator:** Our GPipe bubble formula $(M + S - 1)$ is pessimistic. Could add a "bubble efficiency" parameter (1.0 for GPipe, ~0.0 for zero-bubble schedules) to the PP-Group DiLoCo mode.
*   **[DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437)**
    *   **Why:** DualPipe eliminates pipeline bubbles and fully overlaps EP communication with compute. FP8 training at 671B MoE scale. Real training cost: 2.788M H800 GPU-hours.
    *   **Impact on Simulator:** Provides real-world data to validate the simulator's PP + MoE modeling. The compute-communication overlap for EP is not modeled in our simulator.

### Hierarchical & Async DiLoCo
*   **[HALoS: Hierarchical Asynchronous Local SGD (Kim et al., 2025)](https://arxiv.org/abs/2506.04531)**
    *   **Why:** Directly models hierarchical async Local SGD for geo-distributed LLM training. 7.5x faster convergence than synchronous baselines. Includes their own simulator.
    *   **Impact on Simulator:** Validates or challenges our hierarchical DiLoCo model and the $\sqrt{H_{\text{regional}}}$ effective-H heuristic. Their simulator could be used for cross-validation.
*   **[Asynchronous Local-SGD Training for Language Modeling (Liu, Douillard et al., 2024)](https://arxiv.org/abs/2401.09135)**
    *   **Why:** Comprehensive DeepMind study of async vs. sync Local SGD efficiency for LLM pretraining. Introduces Delayed Nesterov outer optimizer and Dynamic Local Updates.
    *   **Impact on Simulator:** Provides empirical data to calibrate the algorithmic efficiency penalty $\alpha$. Could replace our engineering estimate with published measurements.

### Communication & Reliability
*   **[SPARTA: Sparse Parameter Averaging for DiLoCo (2025)](https://openreview.net/pdf?id=stFPf3gzq1)**
    *   **Why:** 1000x+ communication reduction by exchanging only sparse parameter subsets. Allows H=10,000 while improving perplexity by 14.3%.
    *   **Impact on Simulator:** Would dramatically change bandwidth requirements if validated at scale. Could be modeled as an extreme compression preset.
*   **[Distributed Training under Packet Loss (Weintraub et al., 2025)](https://arxiv.org/abs/2507.07114)**
    *   **Why:** First framework for training over unreliable connections. 10% random packet loss causes only 0.8% perplexity change on LLaMA-2 7B.
    *   **Impact on Simulator:** Our simulator assumes reliable TCP. This paper shows WAN packet loss has minimal impact, which is reassuring for our modeling assumptions.
