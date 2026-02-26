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

## 5. Recently Identified Research (February 2026 Literature Review) — REVIEWED

These papers were identified during a systematic literature review and compared against the simulator's modeling assumptions. None of these techniques have been validated at the scale the simulator targets ($10^{26}$–$10^{27}$ FLOPs, 100B+ parameters, 50+ WAN nodes), so they are not incorporated into the simulator. They are documented in `Simulator_Documentation.md`, Appendix B as potential future improvements that policymakers should be aware of.

### Pipeline Parallelism Modeling
*   **[Zero Bubble Pipeline Parallelism (Qi et al., 2024)](https://arxiv.org/abs/2401.10241)** ✅ Reviewed
    *   **Finding:** ZB-2p achieves <1% bubble vs. our GPipe formula's 11–27%. 10–30% speedup on PP compute. WAN-applicable (local computation reordering only). Tested at 28B, 8 stages.
    *   **Verdict:** Simulator is conservative. Documented as known conservative assumption (Appendix B.1).
*   **[DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437)** ✅ Reviewed
    *   **Finding:** DualPipe fully hides EP All-to-All communication, but requires NVLink/InfiniBand. At WAN latency (100ms), overlap is impossible — our additive EP latency model is correct for WAN.
    *   **Verdict:** Simulator is accurate for its WAN use case. Documented as datacenter-only technique (Appendix B.5).

### Hierarchical & Async DiLoCo
*   **[HALoS: Hierarchical Asynchronous Local SGD (Kim et al., 2025)](https://arxiv.org/abs/2506.04531)** ✅ Reviewed
    *   **Finding:** Achieves near-zero algorithmic penalty with hierarchical momentum ($\beta_{\text{local}}, \beta_{\text{global}}$), vs. our $\sqrt{H_R}$ heuristic predicting 85–88% efficiency. Only tested at 70M scale.
    *   **Verdict:** Simulator's hierarchy heuristic is conservative by ~3–5%. Documented in Appendix B.3.
*   **[Asynchronous Local-SGD Training for Language Modeling (Liu, Douillard et al., 2024)](https://arxiv.org/abs/2401.09135)** ✅ Reviewed
    *   **Finding:** Delayed Nesterov + Dynamic Local Updates fully close the async gap (effective $\alpha \to 0$). DyLU also mitigates straggler effects for heterogeneous hardware. Tested at ≤150M, ≤16 workers.
    *   **Verdict:** Simulator's $\alpha=0.08$ is reasonable for standard sync DiLoCo but 5–12% too pessimistic for DN+DyLU-enhanced versions. Documented in Appendix B.2.

### Communication & Reliability
*   **[SPARTA: Sparse Parameter Averaging for DiLoCo (2025)](https://openreview.net/pdf?id=stFPf3gzq1)** ✅ Reviewed
    *   **Finding:** 1000× compression via continuous sparse exchange. At $H=10{,}000$: 14.3% perplexity *improvement* (regularization effect). Decouples $H$ from convergence — a relationship the simulator treats as fundamental. Only tested at 124M, ≤8 nodes; paper states "doesn't scale well beyond 16 nodes."
    *   **Verdict:** Potentially large gap (25–35 pp) but highly speculative at scale. Documented with caveats in Appendix B.4.
*   **[Distributed Training under Packet Loss (Weintraub et al., 2025)](https://arxiv.org/abs/2507.07114)** ✅ Reviewed
    *   **Finding:** 10% packet loss → 0.8% perplexity degradation on LLaMA-2 7B. Loss-tolerant UDP could reduce straggler overhead by 10–15%. Tested at 7B, 64 GPUs.
    *   **Verdict:** Validates simulator's reliability assumptions. TCP retransmission effects partially captured by straggler factor. Documented in Appendix B.6.
