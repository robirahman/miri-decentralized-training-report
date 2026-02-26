# Bibliography

Annotated bibliography for the MIRI Decentralized Training Simulator project. Organized by topic area. Each entry notes where it is cited in the project and its relevance to the simulator's modeling.

Papers marked **NEW** were identified during a February 2026 literature review and are not yet cited in the simulator documentation or code.

---

## 1. Scaling Laws & Compute Accounting

**Scaling Laws for Neural Language Models (Kaplan et al., 2020)**
- URL: https://arxiv.org/abs/2001.08361
- Cited in: Simulator_Documentation.md (Section 1.1)
- Relevance: Established the 6 FLOPs/parameter/token standard used by the simulator for all compute estimates.

**Training Compute-Optimal Large Language Models — "Chinchilla" (Hoffmann et al., 2022)**
- URL: https://arxiv.org/abs/2203.15556
- Cited in: Simulator_Documentation.md (Section 1.1), 10^25 FLOP configuration.md
- Relevance: Defines compute-optimal scaling ratios for model size vs. dataset size. Validates the 6N approximation.

**Calculating the Computational Complexity of Transformers (Casson, 2023)**
- URL: https://www.adamcasson.com/posts/transformer-flops
- Cited in: Simulator_Documentation.md (Section 1.1)
- Relevance: Shows attention FLOPs are <3% of total at 175B+ parameters, confirming 6N is accurate at scale.

**The Longest Training Run (Epoch AI, 2023)**
- URL: https://epoch.ai/blog/the-longest-training-run
- Cited in: Simulator_Documentation.md (Section 7), App.tsx
- Relevance: Source of the maximum training run duration formula L = 1/g and growth rate estimates.

**Introducing the Distributed Training Interactive Simulator (Epoch AI)**
- URL: https://epoch.ai/blog/introducing-the-distributed-training-interactive-simulator
- Cited in: miri-vs-douillard-simulator-comparison.md
- Relevance: Predecessor simulator that this project extends.

---

## 2. Large-Scale Training Systems & MFU Benchmarks

**PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)**
- URL: https://arxiv.org/abs/2204.02311
- Cited in: Simulator_Documentation.md (Sections 1.3, 6.1, 6.2)
- Relevance: Key MFU benchmark (46.2% MFU, 57.8% HFU on TPU v4). Source of the MFU/HFU = 0.8 ratio.

**The Llama 3 Herd of Models (Dubey et al., 2024)**
- URL: https://arxiv.org/abs/2407.21783
- Cited in: Simulator_Documentation.md (Section 1.3)
- Relevance: LLaMA-3.1 405B achieved 38-43% MFU, validating the simulator's 40% default.

**Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al., 2022)**
- URL: https://arxiv.org/abs/2205.05198
- Cited in: Simulator_Documentation.md (Section 1.3)
- Relevance: Well-optimized Megatron-LM achieves 40-55% MFU on A100 clusters.

**DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)**
- URL: https://arxiv.org/abs/2405.04434
- Cited in: Simulator_Documentation.md (Section 2.4)
- Relevance: 236B MoE with 21B active parameters, demonstrating expert parallelism at scale.

**NEW: DeepSeek-V3 Technical Report (DeepSeek AI, 2024)**
- URL: https://arxiv.org/abs/2412.19437
- Cited in: Not yet cited
- Relevance: Introduces DualPipe (zero-bubble bidirectional PP) and FP8 mixed-precision training at 671B MoE scale. Provides real-world training cost data (2.788M H800 GPU-hours). DualPipe's compute-communication overlap is directly relevant to the simulator's PP bubble modeling.

**Epoch AI — AI Models Database**
- URL: https://epoch.ai/data/trends
- Cited in: Hardware utilization data insight.md
- Relevance: Source of MFU/HFU measurements across 80+ models, used for empirical validation.

---

## 3. DiLoCo & Local SGD

**Local SGD Converges Fast and Communicates Little (Stich, 2019)**
- URL: https://arxiv.org/abs/1805.09767
- Cited in: Simulator_Documentation.md (Section 4.2), Papers_to_Study.md
- Relevance: Theoretical convergence bounds for local SGD showing sub-linear degradation with local steps. Supports the logarithmic efficiency decay assumption.

**Cooperative SGD: A Unified Framework for the Design and Analysis of Communication-Efficient SGD (Wang & Joshi, 2021)**
- URL: https://jmlr.org/papers/volume22/20-147/20-147.pdf
- Cited in: Simulator_Documentation.md (Section 4.4)
- Relevance: Analyzes convergence of local SGD with periodic averaging. Referenced for the hierarchical effective-H heuristic.

**DiLoCo: Distributed Low-Communication Training of Language Models (Douillard et al., 2023)**
- URL: https://arxiv.org/abs/2311.08105
- Cited in: Simulator_Documentation.md (Sections 3.1, 4.2), miri-vs-douillard-simulator-comparison.md, 10^25 FLOP configuration.md
- Relevance: Core algorithm the simulator models. Defines pseudo-gradient communication and outer optimization with Nesterov momentum. Tested H up to 500 with modest degradation.

**Streaming DiLoCo with Overlapping Communication and Computation (Douillard et al., 2025)**
- URL: https://arxiv.org/abs/2501.18512
- Cited in: Simulator_Documentation.md (Sections 3.1, 8)
- Relevance: Achieves 95% compute utilization by overlapping communication with computation. Source of the max(compute, comm) streaming model. Documents 66% memory overhead.

**Scaling Laws for DiLoCo (Charles et al., 2025)**
- URL: https://arxiv.org/abs/2503.09799
- Cited in: Simulator_Documentation.md (Sections 4.2, 4.3)
- Relevance: Shows efficiency loss increases with H but is less pronounced for larger models. Directly supports the simulator's alpha-scaling-with-model-size assumption.

**NEW: DiPaCo: Distributed Path Composition (Douillard et al., 2024)**
- URL: https://arxiv.org/abs/2403.10616
- Cited in: Not yet cited
- Relevance: Alternative decentralized architecture distributing computation by "paths" through shared modules, combined with DiLoCo-style optimization. Only 150M params executed per input out of 256 possible. Represents an extreme form of conditional computation for decentralized training.

**NEW: Asynchronous Local-SGD Training for Language Modeling (Liu, Douillard et al., 2024)**
- URL: https://arxiv.org/abs/2401.09135
- Cited in: Not yet cited
- Relevance: Comprehensive empirical study of async vs. sync Local SGD for LLM pretraining from Google DeepMind. Introduces Delayed Nesterov outer optimizer and Dynamic Local Updates. Provides empirical data to calibrate the simulator's algorithmic efficiency penalty.

**NEW: OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training (Prime Intellect, 2024)**
- URL: https://arxiv.org/abs/2407.07852
- Cited in: Not yet cited
- Relevance: Open-source DiLoCo implementation using Hivemind, demonstrated across two continents. Reports 90-95% compute utilization and documents real-world fault tolerance behavior. Validates the simulator's DiLoCo over WAN assumptions.

**NEW: HALoS: Hierarchical Asynchronous Local SGD for Geo-Distributed LLM Training (Kim et al., 2025)**
- URL: https://arxiv.org/abs/2506.04531
- Cited in: Not yet cited
- Relevance: Hierarchical local parameter servers within regions + global server across regions, achieving 7.5x faster convergence than synchronous baselines. Directly models the WAN/LAN hierarchy the simulator assumes. Includes their own geo-distributed training simulator.

**NEW: SPARTA: Sparse Parameter Averaging for DiLoCo (2025)**
- URL: https://openreview.net/pdf?id=stFPf3gzq1
- Cited in: Not yet cited
- Relevance: Achieves 1000x+ communication reduction by exchanging only a sparse subset of parameters at each sync. Allows increasing sync interval from H=100 to H=10,000 while improving perplexity by 14.3%. Could dramatically change the simulator's bandwidth requirements.

**DeMo: Decoupled Momentum Optimizer (Nous Research, 2024)**
- URL: https://arxiv.org/abs/2411.19870
- Cited in: Papers_to_Study.md
- Relevance: Decouples momentum states to allow more infrequent synchronization without the typical algorithmic penalty.

**MuLoCo: Muon Inner Optimizer for DiLoCo (lucidrains, 2025)**
- URL: https://github.com/lucidrains/muon
- Cited in: Papers_to_Study.md
- Relevance: Using the Muon optimizer as DiLoCo's inner optimizer achieves 8x less communication than standard AdamW-DiLoCo.

---

## 4. Pipeline Parallelism

**GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al., 2019)**
- URL: https://arxiv.org/abs/1811.06965
- Cited in: Simulator_Documentation.md (Section 3.3)
- Relevance: Source of the standard pipeline bubble formula (M + S - 1) used in the simulator's PP mode.

**SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient (Ryabinin et al., 2023)**
- URL: https://arxiv.org/abs/2301.11913
- Cited in: Simulator_Documentation.md (Sections 3.3, 3.4)
- Relevance: Adaptive pipeline parallelism over unreliable internet. At 100ms RTT, GPT-3-scale models retain ~60% utilization (72% with compression).

**DiLoCoX: A Scalable Approach to Distributed Training of Large Language Models (Chen et al., 2025)**
- URL: https://arxiv.org/abs/2506.21263
- Cited in: Simulator_Documentation.md (Section 3.4)
- Relevance: Combines intra-group PP with inter-group DiLoCo. 357x throughput improvement on 107B model over 1 Gbps. Primary evidence for the simulator's PP-Group DiLoCo mode.

**NEW: Zero Bubble Pipeline Parallelism (Qi et al., 2024)**
- URL: https://arxiv.org/abs/2401.10241
- Cited in: Not yet cited
- Relevance: Achieves zero pipeline bubbles by splitting backward pass into input-gradient and weight-gradient phases. Foundation for DeepSeek-V3's DualPipe. The simulator's GPipe bubble formula is pessimistic compared to modern zero-bubble schedules.

**NEW: Pipeline Parallelism with Controllable Memory (Qi et al., 2024)**
- URL: https://arxiv.org/abs/2405.15362
- Cited in: Not yet cited
- Relevance: Framework for PP schedules with controllable activation memory: reduces peak memory to 1/2 or 1/3 of 1F1B without throughput loss. Outperforms 1F1B by 7-55%. Relevant to the simulator's PP memory calculations.

---

## 5. Mixture of Experts & Expert Parallelism

**Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2022)**
- URL: https://arxiv.org/abs/2101.03961
- Cited in: Simulator_Documentation.md (Section 1.4)
- Relevance: Established that MoE models decouple parameter count from per-token compute.

**Mixtral of Experts (Jiang et al., 2024)**
- URL: https://arxiv.org/abs/2401.04088
- Cited in: Simulator_Documentation.md (Section 1.4)
- Relevance: Practical MoE scaling reference: 47B total params, 13B active per token.

**GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020)**
- URL: https://arxiv.org/abs/2006.16668
- Cited in: Simulator_Documentation.md (Sections 1.5, 2.4)
- Relevance: Standard expert parallelism design. All-to-All latency scaling with network hops. EP reduces per-device memory proportionally.

**Tutel: Adaptive Mixture-of-Experts at Scale (Hwang et al., 2023)**
- URL: https://arxiv.org/abs/2206.03382
- Cited in: Simulator_Documentation.md (Section 1.5)
- Relevance: All-to-All latency optimization in expert-parallel settings.

**SPES Protocol: Expert-Sharded MoE for Decentralized Training (Prime Intellect, 2025)**
- URL: https://github.com/PrimeIntellect-ai/spes
- Cited in: Simulator_Documentation.md (Sections 2.4, 3.2)
- Relevance: Each node stores only shared params + local experts, avoiding per-forward-pass All-to-All. Achieves 35-65% communication reduction vs. DiLoCo. Primary evidence for the simulator's EP memory reduction model.

---

## 6. Memory & Precision Optimization

**ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)**
- URL: https://arxiv.org/abs/1910.02054
- Cited in: Simulator_Documentation.md (Section 2.1), Papers_to_Study.md
- Relevance: Establishes the 16 bytes/param breakdown for mixed-precision AdamW training. Source of the simulator's memory model.

**Mixed Precision Training (Micikevicius et al., 2017)**
- URL: https://arxiv.org/abs/1710.03740
- Cited in: Simulator_Documentation.md (Section 2.1)
- Relevance: Demonstrates FP32 master weights are required for numerical stability in mixed-precision training.

**8-bit Optimizers via Block-wise Quantization (Dettmers et al., 2022)**
- URL: https://arxiv.org/abs/2110.02861
- Cited in: Simulator_Documentation.md (Section 2.1)
- Relevance: 8-bit optimizer state compression is feasible but FP32 remains standard for pre-training stability.

**FP8 Training Documentation (NVIDIA Transformer Engine)**
- URL: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
- Cited in: Papers_to_Study.md
- Relevance: Modern H100/GH200 hardware uses FP8 compute, effectively doubling bandwidth and halving sync volume.

---

## 7. Gradient Compression & Communication

**signSGD: Compressed Optimisation for Non-Convex Problems (Bernstein et al., 2018)**
- URL: https://arxiv.org/abs/1802.04434
- Cited in: Simulator_Documentation.md (Section 8.2)
- Relevance: Demonstrates aggressive gradient compression with preserved convergence.

**Error Feedback Fixes SignSGD and other Gradient Compression Schemes (Karimireddy et al., 2019)**
- URL: https://arxiv.org/abs/1901.09847
- Cited in: Simulator_Documentation.md (Section 8.2)
- Relevance: Error feedback mechanism ensures convergence under aggressive compression. Foundation for the simulator's compression ratio modeling.

**1-bit Adam: Communication Efficient Large-Scale Training (Tang et al., 2021)**
- URL: https://arxiv.org/abs/2102.02888
- Cited in: Bibliography/*.url
- Relevance: Extreme gradient compression to 1 bit per parameter with error compensation.

**CocktailSGD: Fine-tuning Foundation Models over 500Kbps Networks (Wang et al., 2023)**
- URL: https://arxiv.org/abs/2307.03718
- Cited in: Bibliography/*.url
- Relevance: Compression techniques for extremely slow network training. Relevant to the simulator's WAN bandwidth modeling.

**On the Utility of Gradient Compression in Distributed Training Systems (Agarwal et al., 2021)**
- URL: https://arxiv.org/abs/2102.04013
- Cited in: Papers_to_Study.md
- Relevance: Empirical bounds on where compression starts to harm convergence. Could inform a compression warning threshold in the simulator.

---

## 8. Straggler Mitigation & Unreliable Networks

**More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server (Ho et al., 2013)**
- URL: https://proceedings.neurips.cc/paper/2013/hash/b7bb35b9c6ca2aee2df08cf09d7016c2-Abstract.html
- Cited in: Simulator_Documentation.md (Section 4.5)
- Relevance: Bounded-staleness SGD literature. Shows degradation from stale gradients, supporting the threshold strategy penalty.

**Revisiting Distributed Synchronous SGD (Chen et al., 2016)**
- URL: https://arxiv.org/abs/1604.00981
- Cited in: Simulator_Documentation.md (Section 5.1)
- Relevance: Empirical confirmation of logarithmic straggler scaling with node count.

**PyTorch DDP Straggler Mitigation**
- URL: https://pytorch.org/blog/straggler-mitigation/
- Cited in: Simulator_Documentation.md (Section 5.1)
- Relevance: Practical straggler mitigation techniques from the PyTorch team.

**NEW: Distributed Training under Packet Loss (Weintraub, Banner & Orda, 2025)**
- URL: https://arxiv.org/abs/2507.07114
- Cited in: Not yet cited
- Relevance: First framework for distributed training over unreliable connections with unbiased gradient aggregation. 10% random packet loss causes only 0.8% perplexity change on LLaMA-2 7B. Highly relevant to WAN modeling where packet loss is a reality.

---

## 9. Decentralized Training at Scale

**Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixtures of Experts — "Learning@home" (Ryabinin & Gusev, 2020)**
- URL: https://arxiv.org/abs/2002.04013
- Cited in: Bibliography/*.url
- Relevance: Early foundational work on collaborative/decentralized training over the internet.

**Distributed Deep Learning in Open Collaborations (Diskin et al., 2021)**
- URL: https://arxiv.org/abs/2106.10207
- Cited in: Simulator_Documentation.md (Section 10)
- Relevance: Fully asynchronous decentralized training approach. Not modeled in the simulator but referenced as a known limitation.

**INTELLECT-1: Launching the First 10B Parameter Decentralized Training Run (Jaghouar et al., 2024)**
- URL: https://arxiv.org/abs/2412.01152
- Cited in: Simulator_Documentation.md (Sections 1.3, 3.2)
- Relevance: Demonstrated decentralized training across USA, Europe, and Asia. MFU data: 41.4% (USA), 37.1% (USA+Europe), 36.2% (3 continents). Key empirical validation for the simulator.

**INTELLECT-2 Technical Report (Prime Intellect)**
- URL: https://primeintellect.ai/intellect-2
- Cited in: Bibliography/*.url
- Relevance: Follow-up to INTELLECT-1 with larger scale decentralized training.

**NEW: The Future of Large Language Model Pre-training is Federated — "Photon" (Sani, Iacob, Cao et al., 2024)**
- URL: https://arxiv.org/abs/2405.10853
- Cited in: Not yet cited
- Relevance: Federated LLM pre-training up to 7B parameters showing high resilience to statistical and hardware heterogeneity. Convergence is robust to partial worker participation.

**NEW: Beyond A Single AI Cluster: A Survey of Decentralized LLM Training (Dong, Jiang, Lu et al., 2025)**
- URL: https://arxiv.org/abs/2503.11023
- Cited in: Not yet cited
- Relevance: Comprehensive survey categorizing community-driven and organizational decentralized training efforts. Covers the full landscape of techniques the simulator models.

---

## 10. Other Resources

**DisTrO: Distributed Training Over-The-Internet (Nous Research)**
- URL: https://nousresearch.com/distro/
- Cited in: Papers_to_Study.md
- Relevance: Claims 100-1000x bandwidth reduction. Foundational for assessing feasibility of 10^27 FLOP runs.

**SparseLoCo (Sarfi et al., 2025)**
- URL: https://arxiv.org/abs/2508.13077
- Cited in: Bibliography/*.url
- Relevance: Sparse local optimization for communication-efficient distributed training.

**Covenant-72B (Covenant AI)**
- URL: https://huggingface.co/CovenantAI/Covenant-72B
- Cited in: Bibliography/*.url
- Relevance: Example of a decentralized-trained model.

**Protocol Models (Pluralis Research)**
- URL: https://pluralis.ai
- Cited in: Bibliography/*.url
- Relevance: Protocol-based decentralized model training project.

**DiLoCo Bandwidth Simulator (Arthur Douillard)**
- URL: https://arthurdouillard.com/diloco/index.html
- Cited in: miri-vs-douillard-simulator-comparison.md
- Relevance: Baseline simulator that the MIRI project extends and compares against.

**NEW: Muon is Scalable for LLM Training — "Moonlight" (Liu, Su et al., 2025)**
- URL: https://arxiv.org/abs/2502.16982
- Cited in: Not yet cited
- Relevance: Demonstrates Muon optimizer achieves ~2x computational efficiency vs. AdamW at scale (3B/16B MoE on 5.7T tokens). Relevant to MuLoCo and to all compute projections if Muon becomes the standard inner optimizer for DiLoCo.
