# Project Bibliography

Annotated bibliography for the MIRI Decentralized Training Simulator project. For each paper: main findings, relevance to large-scale decentralized training, and whether its results are reflected in the simulator.

---

## 1. Scaling Laws & Compute Accounting

**Scaling Laws for Neural Language Models (Kaplan et al., 2020)**
https://arxiv.org/abs/2001.08361
- **Findings:** Established power-law relationships between model size, dataset size, compute budget, and language model loss. Introduced the 6 FLOPs/parameter/token approximation (2N forward, 4N backward) for transformer compute.
- **Relevance:** The 6N rule is the standard accounting method for estimating total training compute, used by all frontier labs.
- **In simulator:** Yes. The simulator uses C = 6PD as its core compute estimate (Section 1.1).

**Training Compute-Optimal Large Language Models — "Chinchilla" (Hoffmann et al., 2022)**
https://arxiv.org/abs/2203.15556
- **Findings:** Revised Kaplan's scaling laws to show that models and datasets should be scaled equally — for a fixed compute budget, the optimal model is much smaller (and the dataset much larger) than Kaplan suggested. Defined "Chinchilla-optimal" ratios (~20 tokens per parameter).
- **Relevance:** Determines the optimal model size for a given compute budget, which directly sets the parameters and tokens inputs to the simulator.
- **In simulator:** Indirectly. Users choose model/dataset size; Chinchilla ratios inform sensible defaults (e.g., 144B params / 12T tokens for the 10^25 FLOP configuration).

**Calculating the Computational Complexity of Transformers (Casson, 2023)**
https://www.adamcasson.com/posts/transformer-flops
- **Findings:** Detailed FLOP breakdown showing attention logit computation, softmax, and other non-matmul operations are <3% of total FLOPs for models above 175B parameters.
- **Relevance:** Confirms the 6N approximation is accurate at the scales the simulator targets.
- **In simulator:** Yes. Justifies the simulator's use of 6N without an attention correction term (Section 1.1).

**The Longest Training Run (Epoch AI, 2023)**
https://epoch.ai/blog/the-longest-training-run
- **Findings:** Derives L = 1/(g·ln(10)) years, where g is the combined OOM/year growth rate in hardware, software, and investment. Any training run longer than L is better served by waiting for improved conditions.
- **Relevance:** Sets an economic upper bound on training run duration, regardless of technical feasibility.
- **In simulator:** Yes. The simulator computes maximum training duration from user-adjustable growth rates (Section 7).

**Introducing the Distributed Training Interactive Simulator (Epoch AI)**
https://epoch.ai/blog/introducing-the-distributed-training-interactive-simulator
- **Findings:** Web-based tool for estimating communication overhead in distributed training with DiLoCo.
- **Relevance:** Predecessor simulator that this project extends with memory-triggered mode switching, pipeline parallelism, hierarchical topology, and algorithmic efficiency penalties.
- **In simulator:** The MIRI simulator is a superset of this tool's functionality.

**Epoch AI — AI Models Database**
https://epoch.ai/data/trends
- **Findings:** Database of MFU/HFU measurements across 80+ publicly reported training runs.
- **Relevance:** Empirical validation source for the simulator's MFU assumptions.
- **In simulator:** Used to calibrate the default 40% MFU and validate the MFU/HFU = 0.8 ratio.

---

## 2. Large-Scale Training Systems & MFU Benchmarks

**PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)**
https://arxiv.org/abs/2204.02311
- **Findings:** Trained a 540B parameter model on 6144 TPU v4 chips. Achieved 46.2% MFU and 57.8% HFU, giving a MFU/HFU ratio of 0.799.
- **Relevance:** The gold-standard MFU benchmark for large-scale training. The MFU/HFU ratio reveals the overhead of activation recomputation.
- **In simulator:** Yes. Source of the default 40% MFU, the MFU/HFU = 0.8 ratio, and the Global MFU / HFU calculations (Sections 1.3, 6.1, 6.2).

**The Llama 3 Herd of Models (Dubey et al., 2024)**
https://arxiv.org/abs/2407.21783
- **Findings:** LLaMA-3.1 405B achieved 38–43% MFU during training on Meta's GPU clusters.
- **Relevance:** Modern MFU benchmark confirming that 40% is a reasonable default even for well-optimized runs on latest hardware.
- **In simulator:** Yes. Validates the 40% MFU default (Section 1.3).

**Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al., 2022)**
https://arxiv.org/abs/2205.05198
- **Findings:** Well-optimized Megatron-LM achieves 40–55% MFU on A100 clusters with selective activation recomputation.
- **Relevance:** Establishes the achievable MFU range for optimized training frameworks.
- **In simulator:** Yes. Used to set the 30–60% MFU range and justify the 60% warning threshold (Section 1.3).

**DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model (2024)**
https://arxiv.org/abs/2405.04434
- **Findings:** 236B MoE with only 21B active parameters per token, using multi-head latent attention and DeepSeekMoE architecture. Demonstrates that very large MoE models can be trained with modest per-token compute.
- **Relevance:** Validates the simulator's separation of total parameters (for memory) and active parameters (for compute) in MoE models. Shows expert parallelism working at production scale.
- **In simulator:** Yes. Cited as evidence for the EP memory reduction model (Section 2.4).

**DeepSeek-V3 Technical Report (DeepSeek AI, 2024)**
https://arxiv.org/abs/2412.19437
- **Findings:** 671B MoE trained with FP8 mixed precision on H800 GPUs. Introduced DualPipe: bidirectional pipeline parallelism that fully overlaps both PP and EP All-to-All communication with compute by dedicating 20 of 132 SMs to communication kernels. Achieved ~40% BF16-equivalent MFU. Real training cost: 2.788M H800 GPU-hours.
- **Relevance:** DualPipe eliminates pipeline bubbles and hides EP communication entirely, but requires NVLink (160 GB/s, ~1μs latency) and InfiniBand. At WAN latency (100ms), this overlap is physically impossible.
- **In simulator:** Not directly. The simulator's additive EP latency model is correct for WAN settings. DualPipe is documented in Appendix B.5 as a datacenter-only technique that does not apply to the WAN scenarios the simulator models.

---

## 3. DiLoCo & Local SGD

**Local SGD Converges Fast and Communicates Little (Stich, 2019)**
https://arxiv.org/abs/1805.09767
- **Findings:** Provides theoretical convergence bounds for Local SGD, showing that convergence rate degrades sub-linearly with the number of local steps between synchronizations. The key result: Local SGD with H local steps achieves convergence rates comparable to mini-batch SGD, with communication reduced by a factor of H.
- **Relevance:** Theoretical foundation for DiLoCo's approach of training independently for many steps before synchronizing. Supports the claim that large H values are feasible.
- **In simulator:** Yes. Supports the logarithmic (sub-linear) form of the efficiency decay η = 1 − α·log₁₀(H) (Section 4.2).

**Cooperative SGD: A Unified Framework for Communication-Efficient SGD (Wang & Joshi, 2021)**
https://jmlr.org/papers/volume22/20-147/20-147.pdf
- **Findings:** Analyzes convergence of local SGD with periodic averaging, showing that more frequent partial averaging reduces variance of divergence sub-linearly.
- **Relevance:** Provides theoretical intuition for the hierarchical DiLoCo model, where regional syncs partially anchor local weight drift.
- **In simulator:** Yes. Referenced as theoretical support for the hierarchical effective-H heuristic H_eff = H_inner · H_regional^0.5 (Section 4.4).

**DiLoCo: Distributed Low-Communication Training of Language Models (Douillard et al., 2023)**
https://arxiv.org/abs/2311.08105
- **Findings:** Core algorithm for decentralized training. Each worker trains independently for H inner steps using a standard optimizer (AdamW), then synchronizes pseudo-gradients (Δθ = θ_local − θ₀) and applies an outer optimizer (Nesterov momentum). Tested H up to 500 with only modest loss degradation.
- **Relevance:** The fundamental algorithm that the simulator models. Defines the communication pattern, pseudo-gradient format, and outer optimization loop.
- **In simulator:** Yes. The simulator's Mode A (DiLoCo) directly implements this algorithm's communication and compute model (Sections 3.1, 4.2).

**Streaming DiLoCo with Overlapping Communication and Computation (Douillard et al., 2025)**
https://arxiv.org/abs/2501.18512
- **Findings:** Achieves 95% compute utilization by staggering parameter fragment synchronization across the inner-step window. Communication is overlapped with the next block of H inner compute steps. Introduces FP4 (E3M0) pseudo-gradient compression for up to 100× total bandwidth reduction. Documents 66% additional memory overhead (reducible to ~2% via CPU offloading at 100B+ scale).
- **Relevance:** Source of the streaming overlap model and the most aggressive published compression ratios for DiLoCo.
- **In simulator:** Yes. The simulator's streaming mode uses T_outer = max(compute, comm), directly from this paper. The 66% memory overhead is noted as a known limitation not modeled (Sections 3.1, 8, 10).

**Scaling Laws for DiLoCo (Charles et al., 2025)**
https://arxiv.org/abs/2503.09799
- **Findings:** Systematic study of how DiLoCo efficiency scales with model size and H. Key finding: evaluation loss increases with H, but the rate of increase is less pronounced for larger models. Does not publish a specific η = 1 − α·log(H) formula.
- **Relevance:** Directly supports two simulator assumptions: (1) logarithmic efficiency decay with H, and (2) α decreasing with model size.
- **In simulator:** Yes. Primary evidence for the alpha-scaling-with-model-size formula α = 0.08 · 1/(1 + log₁₀(P/10⁹)/5) (Sections 4.2, 4.3).

**DiPaCo: Distributed Path Composition (Douillard et al., 2024)**
https://arxiv.org/abs/2403.10616
- **Findings:** Alternative decentralized architecture where the model is a collection of shared modules, and each input is routed through a "path" (subset of modules). Only 150M parameters executed per input out of 256 possible. Combined with DiLoCo-style optimization.
- **Relevance:** Represents an extreme form of conditional computation for decentralized training — an alternative architecture to the dense/MoE models the simulator focuses on.
- **In simulator:** No. DiPaCo is an alternative architecture not modeled by the simulator, which focuses on dense and standard MoE models. DiPaCo's path-based routing would require a fundamentally different communication and compute model.

**Asynchronous Local-SGD Training for Language Modeling (Liu, Douillard et al., 2024)**
https://arxiv.org/abs/2401.09135
- **Findings:** Comprehensive Google DeepMind study comparing async vs. sync Local SGD for LLM pretraining. Introduces Delayed Nesterov (DN): standard Nesterov momentum compounds incorrectly in async mode, so DN applies momentum only every N server iterations. Also introduces Dynamic Local Updates (DyLU): each worker adjusts its H proportionally to its speed, preventing stale gradients from slow workers. With DN + DyLU, async DiLoCo at H=50 matches sync quality (perplexity 41.13 vs 41.35 for 20M params).
- **Relevance:** Shows that the algorithmic penalty from infrequent synchronization can be nearly eliminated with the right outer optimizer. Also shows that heterogeneous hardware need not cause straggler penalties if slow workers do fewer steps.
- **In simulator:** No. DN and DyLU have only been tested at ≤150M parameters with ≤16 workers. The simulator's α=0.08 is reasonable for standard sync DiLoCo at the scales it targets (100B+). DN+DyLU are documented in Appendix B.2 as a potential future improvement not yet validated at scale.

**OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training (Prime Intellect, 2024)**
https://arxiv.org/abs/2407.07852
- **Findings:** Open-source DiLoCo implementation using the Hivemind library, demonstrated across two continents. Reports 90–95% compute utilization and documents real-world fault tolerance behavior (node dropout, rejoining).
- **Relevance:** Validates that DiLoCo over WAN achieves the high compute utilization the simulator predicts, and demonstrates practical fault tolerance.
- **In simulator:** Indirectly. Validates the simulator's DiLoCo over WAN assumptions. The fault tolerance mechanisms (node dropout handling) are not explicitly modeled.

**HALoS: Hierarchical Asynchronous Local SGD for Geo-Distributed LLM Training (Kim et al., 2025)**
https://arxiv.org/abs/2506.04531
- **Findings:** Uses local parameter servers within regions and a global parameter server across regions, with separate hierarchical momentum terms (β_local and β_global). Matches synchronous SGD quality — near-zero algorithmic penalty. Achieves 7.5× wall-clock speedup over synchronous baselines by eliminating synchronization barriers. Includes their own geo-distributed training simulator.
- **Relevance:** Suggests the simulator's sqrt(H_regional) heuristic for hierarchical DiLoCo is conservative, and that near-zero penalty is achievable with the right optimizer.
- **In simulator:** No. Only tested at 70M scale (Pythia/LLaMA/Qwen-70M). Whether hierarchical momentum maintains zero penalty at 144B is unknown. Documented in Appendix B.3 as a potential future improvement.

**SPARTA: Sparse Parameter Averaging for DiLoCo (2025)**
https://openreview.net/pdf?id=stFPf3gzq1
- **Findings:** Exchanges only 0.1–0.5% of parameters continuously between DiLoCo sync boundaries. At H=10,000 with 0.1% exchange: 1000× communication reduction AND 14.3% perplexity improvement over DiLoCo-alone. The sparse exchange acts as a regularizer preventing local model drift, decoupling H from convergence quality.
- **Relevance:** If validated at scale, would fundamentally change the simulator's bandwidth requirements and efficiency predictions at high H. The decoupling of H from convergence contradicts the simulator's monotonically-decaying efficiency model.
- **In simulator:** No. Only tested at 124M parameters, 2–8 nodes. Paper explicitly states "doesn't scale well beyond 16 nodes." Extrapolation to 144B / 72 nodes is highly speculative. Documented in Appendix B.4 as a potential future improvement with strong caveats.

**DeMo: Decoupled Momentum Optimizer (Nous Research, 2024)**
https://arxiv.org/abs/2411.19870
- **Findings:** Decouples momentum states across workers, allowing more infrequent synchronization without the typical algorithmic penalty from stale momentum.
- **Relevance:** Could allow higher effective H values in DiLoCo-style training.
- **In simulator:** No. Not validated at the scale the simulator targets. The simulator's alpha parameter already accommodates a range of optimizer behaviors through its model-size scaling.

**MuLoCo: Muon Inner Optimizer for DiLoCo (lucidrains, 2025)**
https://github.com/lucidrains/muon
- **Findings:** Uses the Muon optimizer as DiLoCo's inner optimizer, achieving 8× less communication than standard AdamW-DiLoCo.
- **Relevance:** Would reduce the simulator's communication volume estimates if Muon becomes the standard inner optimizer.
- **In simulator:** No. The simulator models communication based on pseudo-gradient size, which is independent of the inner optimizer. Muon's communication reduction comes from faster convergence (fewer outer steps needed), which the simulator captures through the user-adjustable compression ratio.

**Muon is Scalable for LLM Training — "Moonlight" (Liu, Su et al., 2025)**
https://arxiv.org/abs/2502.16982
- **Findings:** Demonstrates Muon optimizer achieves ~2× computational efficiency vs. AdamW at scale (3B/16B MoE on 5.7T tokens). Suggests Muon could become the standard optimizer for large-scale training.
- **Relevance:** If Muon replaces AdamW as the standard optimizer, all compute projections (including the simulator's) would need to account for ~2× higher algorithmic efficiency.
- **In simulator:** No. The simulator uses the standard 6N FLOP accounting which assumes AdamW. If Muon becomes standard, the effective compute per token would change, but this would be captured by adjusting the total tokens or compute target rather than modifying the simulator's internal model.

---

## 4. Pipeline Parallelism

**GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al., 2019)**
https://arxiv.org/abs/1811.06965
- **Findings:** Introduced micro-batch pipelining for training large models across multiple devices. The standard bubble formula: (M + S − 1) total micro-batch slots, of which (S − 1) are bubble.
- **Relevance:** The canonical pipeline parallelism schedule. Sets the baseline for pipeline bubble overhead that all later work improves upon.
- **In simulator:** Yes. The simulator uses the GPipe bubble formula for PP-Group DiLoCo and PP-over-WAN modes (Section 3.3).

**SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient (Ryabinin et al., 2023)**
https://arxiv.org/abs/2301.11913
- **Findings:** Adaptive pipeline parallelism over unreliable internet connections. At 100ms RTT, GPT-3-scale models retain ~60% utilization (72% with compression). Handles node failures through dynamic rerouting.
- **Relevance:** Demonstrates that pipeline parallelism over WAN is feasible, though severely limited by per-micro-batch latency. Motivates the PP-Group DiLoCo approach where WAN is used for infrequent DiLoCo sync instead of per-micro-batch PP handoffs.
- **In simulator:** Yes. Cited as evidence for the PP-over-WAN mode and as motivation for PP-Group DiLoCo (Sections 3.3, 3.4).

**DiLoCoX: A Scalable Approach to Distributed Training of Large Language Models (Chen et al., 2025)**
https://arxiv.org/abs/2506.21263
- **Findings:** Combines intra-group pipeline parallelism (fast interconnect) with inter-group DiLoCo (slow WAN). On a 107B model across 160 A800 GPUs over 1 Gbps interconnect, achieved 357× throughput improvement over standard AllReduce.
- **Relevance:** Primary evidence that grouping nodes into PP clusters and running DiLoCo across groups is dramatically faster than pure PP over WAN.
- **In simulator:** Yes. Primary evidence for the simulator's PP-Group DiLoCo mode (Mode C, Section 3.4).

**Zero Bubble Pipeline Parallelism (Qi et al., 2024)**
https://arxiv.org/abs/2401.10241
- **Findings:** Splits the backward pass into input-gradient (B) and weight-gradient (W) phases, rescheduling W to fill bubble slots. ZB-2p achieves <1% bubble across all tested configurations. Measured 23–30% throughput improvement over 1F1B on models from 1.5B to 28.3B with 8 pipeline stages. Now standard practice at frontier labs.
- **Relevance:** Makes the simulator's GPipe bubble formula (11–27% bubble) pessimistic by 10–30% on the compute portion of PP step time.
- **In simulator:** No. The simulator uses the GPipe formula, which is conservative. Zero-bubble scheduling is a local computation reordering that doesn't change communication patterns, so it would only improve the compute portion of PP step time. Over WAN, communication latency typically dominates regardless. Documented in Appendix B.1 as a potential future improvement.

**Pipeline Parallelism with Controllable Memory (Qi et al., 2024)**
https://arxiv.org/abs/2405.15362
- **Findings:** Framework for PP schedules that trade off activation memory vs. throughput. Reduces peak activation memory to 1/2 or 1/3 of 1F1B without throughput loss, and outperforms 1F1B by 7–55%.
- **Relevance:** Could change the simulator's PP memory calculations and sharding trigger, since less activation memory means potentially fewer pipeline stages needed.
- **In simulator:** No. The simulator's memory model focuses on model state (weights, gradients, optimizer), not activation memory. Activation memory is a secondary concern at the scales modeled, and the paper's techniques have not been tested in WAN-distributed PP settings.

---

## 5. Mixture of Experts & Expert Parallelism

**Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2022)**
https://arxiv.org/abs/2101.03961
- **Findings:** Simplified MoE routing to top-1 expert selection, demonstrating that MoE models decouple total parameter count from per-token compute. Showed 7× speedups over dense models at fixed compute budgets.
- **Relevance:** Established the fundamental MoE principle the simulator uses: total parameters determine memory, active parameters determine compute.
- **In simulator:** Yes. The simulator separates P_total (for memory) from P_active (for compute) in MoE mode (Section 1.4).

**Mixtral of Experts (Jiang et al., 2024)**
https://arxiv.org/abs/2401.04088
- **Findings:** Open-weight MoE model with 47B total parameters but only 13B active per token (8 experts, top-2 routing). Demonstrated competitive quality with much lower inference compute than comparable dense models.
- **Relevance:** Practical reference for MoE parameter ratios (total/active ≈ 3.6×) used to validate the simulator's MoE modeling.
- **In simulator:** Yes. Referenced as a practical MoE scaling example (Section 1.4).

**GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020)**
https://arxiv.org/abs/2006.16668
- **Findings:** Standard expert parallelism design distributing experts across devices. All-to-All communication for token routing scales with network hops. EP reduces per-device memory proportionally to the number of EP-participating devices.
- **Relevance:** Establishes that expert parallelism reduces per-node memory by sharding experts, and that EP communication is latency-dominated for small token payloads.
- **In simulator:** Yes. Evidence for both the EP latency model (Section 1.5) and the EP memory reduction formula (Section 2.4).

**Tutel: Adaptive Mixture-of-Experts at Scale (Hwang et al., 2023)**
https://arxiv.org/abs/2206.03382
- **Findings:** Optimized All-to-All communication for expert-parallel MoE training, adapting to dynamic network conditions.
- **Relevance:** Confirms that All-to-All latency dominates in expert-parallel settings, especially as network diameter increases.
- **In simulator:** Yes. Supporting evidence for the latency-dominated EP communication model (Section 1.5).

**SPES Protocol: Expert-Sharded MoE for Decentralized Training (Prime Intellect, 2025)**
https://github.com/PrimeIntellect-ai/spes
- **Findings:** Each node stores only shared parameters plus its local experts, avoiding per-forward-pass All-to-All communication entirely. Instead, tokens are only routed to local experts. Achieves 35–65% communication reduction vs. standard DiLoCo.
- **Relevance:** Primary evidence for the simulator's EP memory reduction model, where per-node memory = (P_shared + P_experts/N_EP) × β.
- **In simulator:** Yes. The simulator's EP memory reduction formula directly models this approach (Section 2.4).

---

## 6. Memory & Precision Optimization

**ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)**
https://arxiv.org/abs/1910.02054
- **Findings:** Detailed breakdown of training memory: 16 bytes per parameter for mixed-precision AdamW (2B weights + 2B gradients + 4B master weights + 4B momentum + 4B variance). ZeRO stages 1–3 progressively shard these across data-parallel ranks.
- **Relevance:** The canonical source for training memory accounting. ZeRO sharding reduces per-GPU memory but not per-node aggregate memory (since the simulator models entire nodes).
- **In simulator:** Yes. Source of the 16/14/13 bytes-per-parameter model for FP16/FP8/FP4 (Section 2.1). ZeRO intra-node sharding is noted as not changing the sharding trigger.

**Mixed Precision Training (Micikevicius et al., 2017)**
https://arxiv.org/abs/1710.03740
- **Findings:** FP32 master weights are required for numerical stability when training with reduced precision (FP16). Loss scaling prevents underflow in FP16 gradients.
- **Relevance:** Explains why FP32 master weights (4 bytes) are mandatory in the memory budget even when compute uses FP16/FP8/FP4.
- **In simulator:** Yes. Justifies the 4-byte FP32 master weight component in all precision modes (Section 2.1).

**8-bit Optimizers via Block-wise Quantization (Dettmers et al., 2022)**
https://arxiv.org/abs/2110.02861
- **Findings:** 8-bit optimizer states (momentum and variance) are feasible with dynamic quantization, reducing optimizer memory by 2×. However, FP32 optimizer states remain standard practice for pre-training stability.
- **Relevance:** Shows that the simulator's assumption of FP32 optimizer states is conservative — 8-bit states could reduce memory — but this is not yet standard practice for frontier pre-training.
- **In simulator:** Yes, conservatively. The simulator uses FP32 optimizer states. 8-bit states are noted as feasible but not modeled (Section 2.1).

**FP8 Training — NVIDIA Transformer Engine**
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
- **Findings:** H100/GH200 hardware natively supports FP8 compute via Transformer Engine, effectively doubling throughput and halving communication volume compared to FP16.
- **Relevance:** FP8 is becoming the default precision for frontier training, which the simulator supports.
- **In simulator:** Yes. The simulator supports FP8 precision (1-byte weights/gradients + FP32 optimizer = 14 bytes/param) and auto-adjusts communication volume (Sections 2.1, 8.1).

---

## 7. Gradient Compression & Communication

**signSGD: Compressed Optimisation for Non-Convex Problems (Bernstein et al., 2018)**
https://arxiv.org/abs/1802.04434
- **Findings:** Compressing gradients to their sign (1 bit per parameter) can preserve convergence with appropriate learning rate scaling.
- **Relevance:** Establishes that aggressive gradient compression is theoretically sound, supporting the simulator's modeling of high compression ratios.
- **In simulator:** Yes. Supports the simulator's assumption that high compression ratios (up to 16× default) are feasible (Section 8.2).

**Error Feedback Fixes SignSGD and other Gradient Compression Schemes (Karimireddy et al., 2019)**
https://arxiv.org/abs/1901.09847
- **Findings:** Error feedback accumulation (carrying forward the compression error to the next round) ensures convergence even under very aggressive compression. Without error feedback, many compression schemes diverge.
- **Relevance:** Provides the theoretical guarantee that the high compression ratios modeled in the simulator (4-bit quantization + 25% sparsification = 16×) can converge.
- **In simulator:** Yes. Foundation for the simulator's compression ratio modeling (Section 8.2).

**1-bit Adam: Communication Efficient Large-Scale Training (Tang et al., 2021)**
https://arxiv.org/abs/2102.02888
- **Findings:** Extends 1-bit compression to Adam optimizer with error compensation, achieving up to 5× communication reduction with minimal loss degradation.
- **Relevance:** Additional evidence that extreme gradient compression is practical for large-scale training.
- **In simulator:** Indirectly. Supports the feasibility of the simulator's compression settings but is not specifically cited.

**CocktailSGD: Fine-tuning Foundation Models over 500Kbps Networks (Wang et al., 2023)**
https://arxiv.org/abs/2307.03718
- **Findings:** Combines multiple compression techniques (quantization, sparsification, low-rank) to enable training over extremely slow networks (500 Kbps).
- **Relevance:** Demonstrates that training is feasible even at bandwidths far below the simulator's default 100 Mbps, relevant to extreme WAN scenarios.
- **In simulator:** Indirectly. Validates that the simulator's WAN bandwidth assumptions (100 Mbps) are generous compared to the minimum feasible bandwidth.

**On the Utility of Gradient Compression in Distributed Training Systems (Agarwal et al., 2021)**
https://arxiv.org/abs/2102.04013
- **Findings:** Provides empirical bounds on where gradient compression starts to harm model convergence. Very high compression ratios (>100×) increasingly risk convergence failure.
- **Relevance:** Could inform a compression warning threshold in the simulator (e.g., warning when compression exceeds 100×).
- **In simulator:** Not directly. The simulator allows arbitrary compression ratios without a convergence warning. A future version could add a warning based on this paper's findings.

---

## 8. Straggler Mitigation & Unreliable Networks

**More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server (Ho et al., 2013)**
https://proceedings.neurips.cc/paper/2013/hash/b7bb35b9c6ca2aee2df08cf09d7016c2-Abstract.html
- **Findings:** Bounded-staleness SGD allows workers to proceed without waiting for the slowest worker, up to a staleness threshold. Convergence degrades with staleness.
- **Relevance:** Theoretical basis for the simulator's threshold aggregation strategy (proceeding with 90% of workers, discarding stragglers) and its associated 15% efficiency penalty.
- **In simulator:** Yes. Supports the threshold strategy penalty of 1.15× in the straggler mitigation options (Section 4.5).

**Revisiting Distributed Synchronous SGD (Chen et al., 2016)**
https://arxiv.org/abs/1604.00981
- **Findings:** Empirically confirms that the expected delay of the slowest worker scales logarithmically with the number of workers, consistent with extreme value theory for lognormal distributions.
- **Relevance:** Justifies the simulator's logarithmic straggler factor f(n) = 1 + 0.05·log₂(n).
- **In simulator:** Yes. Evidence for the straggler scaling formula (Section 5.1).

**PyTorch DDP Straggler Mitigation**
https://pytorch.org/blog/straggler-mitigation/
- **Findings:** Practical straggler mitigation techniques including profiling, load balancing, and gradient compression to reduce the impact of slow workers.
- **Relevance:** Confirms that straggler effects are a real concern in distributed training and provides practical context for the simulator's straggler model.
- **In simulator:** Yes. Referenced as supporting evidence for the straggler model (Section 5.1).

**Distributed Training under Packet Loss (Weintraub, Banner & Orda, 2025)**
https://arxiv.org/abs/2507.07114
- **Findings:** First framework for distributed training over unreliable connections using unbiased gradient aggregation over whatever packets arrive. 10% random packet loss causes only 0.8% perplexity degradation on LLaMA-2 7B. At 20%: 2.6%. At 30%: 3.8%. Argues that TCP retransmissions (not packet loss itself) inflate tail latencies in WAN training.
- **Relevance:** Shows WAN training is more robust to network issues than assumed, and that loss-tolerant UDP could reduce straggler penalties by 10–15% at the cost of ~1% convergence penalty.
- **In simulator:** No. The simulator assumes reliable TCP. This paper's findings largely validate the simulator's approach — the straggler factor partially captures TCP retransmission effects. A loss-tolerant protocol could reduce straggler overhead, but the technique has only been tested at 7B/64 GPUs. Documented in Appendix B.6 as a potential future improvement.

---

## 9. Decentralized Training at Scale

**Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixtures of Experts — "Learning@home" (Ryabinin & Gusev, 2020)**
https://arxiv.org/abs/2002.04013
- **Findings:** Early foundational work proposing collaborative/decentralized training over the internet using volunteer compute and mixture-of-experts architectures.
- **Relevance:** Established the research direction of internet-scale distributed training that DiLoCo and subsequent work builds upon.
- **In simulator:** Not directly. Historical context for the field.

**Distributed Deep Learning in Open Collaborations (Diskin et al., 2021)**
https://arxiv.org/abs/2106.10207
- **Findings:** Fully asynchronous decentralized training approach where workers contribute gradients without strict synchronization barriers.
- **Relevance:** Represents an alternative to DiLoCo's synchronous outer optimization. The simulator only models synchronous protocols.
- **In simulator:** No. Referenced as a known limitation — fully asynchronous methods are not modeled (Section 10).

**INTELLECT-1: Launching the First 10B Parameter Decentralized Training Run (Jaghouar et al., 2024)**
https://arxiv.org/abs/2412.01152
- **Findings:** First 10B parameter model trained in a decentralized manner across USA, Europe, and Asia. MFU measurements: 41.4% (USA only), 37.1% (USA+Europe), 36.2% (3 continents). Baseline without communication overhead: 43.3%.
- **Relevance:** The most directly relevant empirical validation of the simulator's predictions. Confirms that decentralized WAN training achieves MFU in the range the simulator predicts, with measurable per-tier degradation matching the hierarchical model.
- **In simulator:** Yes. Key empirical calibration point for MFU under WAN conditions and hierarchical topology (Sections 1.3, 3.2).

**INTELLECT-2 Technical Report (Prime Intellect)**
https://primeintellect.ai/intellect-2
- **Findings:** Follow-up to INTELLECT-1 with larger scale decentralized training.
- **Relevance:** Extends INTELLECT-1's empirical validation to larger scale.
- **In simulator:** Not directly, but validates the approach at increasing scale.

**The Future of Large Language Model Pre-training is Federated — "Photon" (Sani, Iacob, Cao et al., 2024)**
https://arxiv.org/abs/2405.10853
- **Findings:** Federated LLM pre-training up to 7B parameters showing high resilience to statistical and hardware heterogeneity. Convergence is robust to partial worker participation (nodes dropping in and out).
- **Relevance:** Demonstrates robustness properties of decentralized training that the simulator's deterministic model doesn't capture — in practice, training can continue even with node failures.
- **In simulator:** Not directly. The simulator assumes all nodes are always available. Photon's resilience findings are reassuring for the simulator's reliability assumptions.

**Beyond A Single AI Cluster: A Survey of Decentralized LLM Training (Dong, Jiang, Lu et al., 2025)**
https://arxiv.org/abs/2503.11023
- **Findings:** Comprehensive survey categorizing community-driven and organizational decentralized training efforts. Covers Local SGD, DiLoCo, pipeline parallelism over WAN, expert parallelism, compression, and fault tolerance.
- **Relevance:** Provides a broad landscape of the techniques the simulator models, useful for ensuring the simulator covers the most important approaches.
- **In simulator:** Indirectly. Used during the literature review to identify gaps in the simulator's coverage.

**DisTrO: Distributed Training Over-The-Internet (Nous Research)**
https://nousresearch.com/distro/
- **Findings:** Claims 100–1000× bandwidth reduction for WAN training through novel optimization techniques.
- **Relevance:** If validated, would dramatically change the feasibility of large-scale WAN training. However, detailed technical publications have not been released for independent verification.
- **In simulator:** No. Claims are not yet supported by peer-reviewed results at scale. The simulator's compression settings (up to 16× default) reflect published, validated compression ratios.

**SparseLoCo (Sarfi et al., 2025)**
https://arxiv.org/abs/2508.13077
- **Findings:** Sparse local optimization for communication-efficient distributed training.
- **Relevance:** Additional approach to reducing communication in distributed training.
- **In simulator:** No. The simulator's compression parameter captures the general effect of communication reduction techniques without modeling specific approaches.

---

## 10. Other Resources

**DiLoCo Bandwidth Simulator (Arthur Douillard)**
https://arthurdouillard.com/diloco/index.html
- **Description:** Web-based tool for estimating DiLoCo communication overhead. The MIRI simulator is a superset of this tool, extending it with memory-triggered mode switching, pipeline parallelism, hierarchical topology, MoE support, and algorithmic efficiency penalties.
- **In simulator:** The MIRI simulator replicates this tool as a special case when hierarchy is disabled and the model fits in single-node VRAM.

**Covenant-72B (Covenant AI)**
https://huggingface.co/CovenantAI/Covenant-72B
- **Description:** Example of a 72B model trained using decentralized methods. Demonstrates practical feasibility of decentralized training at non-trivial scale.
- **In simulator:** Not directly. Existence proof for decentralized training at the ~70B scale.

**Protocol Models (Pluralis Research)**
https://pluralis.ai
- **Description:** Protocol-based decentralized model training project where no single node holds the full model weights — they are sharded and encrypted across the network.
- **Relevance:** Relevant to governance considerations around "distributed possession" of model weights.
- **In simulator:** Not directly. Relevant to the project's broader governance analysis rather than the simulator's technical modeling.
