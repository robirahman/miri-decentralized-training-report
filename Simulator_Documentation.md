# Decentralized Training Simulator: Technical Documentation

This document provides a detailed breakdown of the mathematical models, variables, empirical assumptions, and supporting evidence used in the Decentralized Training Simulator.

---

## 1. Core Model & Compute

The simulator estimates the total computational effort and per-step compute time for a pre-training run.

### 1.1 Total Compute

$$C = 6 \cdot P \cdot D$$

*   $P$ = model parameters (active parameters for MoE).
*   $D$ = total training tokens.

**Assumption: 6 FLOPs per parameter per token.** This counts the dense matrix multiplications in the forward pass ($2P$ FLOPs) and backward pass ($4P$ FLOPs: $2P$ for activation gradients, $2P$ for weight gradients). It excludes non-matmul operations such as attention logit computation ($QK^T$), softmax, layer norms, and embedding lookups.

**Evidence:**
*   Established by [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) and used as the standard in [Hoffmann et al. (2022, "Chinchilla")](https://arxiv.org/abs/2203.15556).
*   [Casson (2023)](https://www.adamcasson.com/posts/transformer-flops) shows the excluded attention FLOPs ($O(\text{seq\_len} \cdot d_{\text{model}})$) are <3% of total FLOPs for models above 175B parameters, making $6P$ an accurate approximation at scale.
*   Used as the standard accounting method in the PaLM, LLaMA, and GPT-4 technical reports.

**Limitation:** For small models (<13B), attention FLOPs can exceed 10% of total compute, making $6P$ an undercount. The simulator is designed for large-scale runs where this approximation holds.

### 1.2 Step Duration (Compute Only)

$$T_{\text{comp}} = \frac{6 \cdot P_{\text{active}} \cdot B_{\text{local}}}{\text{Node\_PFLOPS} \cdot \text{MFU}} + T_{\text{EP}}$$

*   $P_{\text{active}}$ = active parameters per token. Equal to total $P$ for dense models; equal to routed-expert parameters for MoE.
*   $B_{\text{local}}$ = local batch size in tokens.
*   $\text{MFU}$ = Model FLOP Utilization (fraction of peak hardware FLOPs achieved).
*   $T_{\text{EP}}$ = Expert Parallelism All-to-All latency (MoE only; zero for dense).

### 1.3 Model FLOP Utilization (MFU)

**Assumption: Default 40%. Range 30–60%.**

**Evidence:**
*   [PaLM 540B (Chowdhery et al. 2022)](https://arxiv.org/abs/2204.02311): 46.2% MFU on TPU v4 pods.
*   LLaMA-3.1 405B: 38–43% MFU reported during training ([Dubey et al. 2024](https://arxiv.org/abs/2407.21783)).
*   [INTELLECT-1 (Jaghouar et al. 2024)](https://arxiv.org/abs/2412.01152), decentralized across continents: 41.4% (USA only), 37.1% (USA+Europe), 36.2% (3 continents). Baseline without communication overhead: 43.3%.
*   General guidance from [Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198): well-optimized Megatron-LM achieves 40–55% MFU on A100 clusters.

The simulator displays a warning when MFU is set above 60%, which is rarely achieved in practice.

### 1.4 MoE Compute

For Mixture of Experts models, only the active (routed) parameters contribute to per-token FLOPs. The simulator uses $P_{\text{active}}$ for compute and $P_{\text{total}}$ for memory.

**Evidence:**
*   [Fedus et al. (2022, "Switch Transformer")](https://arxiv.org/abs/2101.03961): established that MoE models decouple parameter count from per-token compute.
*   [Jiang et al. (2024, "Mixtral")](https://arxiv.org/abs/2401.04088): Mixtral 8×7B has 47B total parameters but only 13B active per token.

### 1.5 Expert Parallelism Latency

$$T_{\text{EP}} = 2 \cdot \text{Latency} \cdot L_{\text{MoE}}$$

Each MoE layer requires two All-to-All communications (dispatch tokens to experts, combine results). With $L_{\text{MoE}}$ MoE layers, this adds $2 \cdot L_{\text{MoE}}$ latency round-trips per step. When expert parallelism is regional (within a cluster), regional latency is used instead of WAN latency.

**Assumption:** The All-to-All communication is latency-dominated (not bandwidth-dominated) because token routing payloads are small relative to WAN round-trip times.

**Evidence:**
*   [Lepikhin et al. (2020, "GShard")](https://arxiv.org/abs/2006.16668) and [Hwang et al. (2023, "Tutel")](https://arxiv.org/abs/2206.03382): All-to-All latency scales with network hops in expert-parallel settings.

---

## 2. Memory & Sharding Logic

Determines whether a model fits on a single node or must be sharded using Pipeline Parallelism.

### 2.1 Memory Required

$$M_{\text{req}} = P \cdot \beta$$

where $\beta$ (bytes per parameter) depends on precision:

| Precision | Weights | Gradients | FP32 Master Weights | Optimizer $m$ (FP32) | Optimizer $v$ (FP32) | **Total $\beta$** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP16/BF16 | 2 B | 2 B | 4 B | 4 B | 4 B | **16 B** |
| FP8 | 1 B | 1 B | 4 B | 4 B | 4 B | **14 B** |
| FP4 | 0.5 B | 0.5 B | 4 B | 4 B | 4 B | **13 B** |

**Assumption: Mixed-precision training with AdamW optimizer.** The optimizer maintains FP32 master weights, first moment ($m$), and second moment ($v$) regardless of the compute precision. Activations are excluded from $\beta$ because they scale with batch size and sequence length, not purely with parameter count.

**Evidence:**
*   [Rajbhandari et al. (2019, "ZeRO")](https://arxiv.org/abs/1910.02054): establishes the 16 bytes/param breakdown for FP16 mixed-precision AdamW as the standard model-state memory accounting.
*   [Micikevicius et al. (2017)](https://arxiv.org/abs/1710.03740): demonstrates that FP32 master weights are required for numerical stability in mixed-precision training.
*   FP8/FP4 optimizer states remain in FP32 per current practice. [Dettmers et al. (2022)](https://arxiv.org/abs/2110.02861) showed 8-bit optimizer states are feasible with dynamic quantization, but FP32 remains standard for pre-training stability.

**Note:** ZeRO and FSDP distribute model states across GPUs within a node but do not reduce the total memory footprint per model replica. Since the simulator models per-node aggregate VRAM, intra-node sharding strategies do not change the sharding trigger.

### 2.2 Sharding Trigger

If $M_{\text{req}} > \text{Node\_VRAM}$, the simulator switches from Data Parallel to Pipeline Parallel mode.

$$S = \left\lceil \frac{M_{\text{req}}}{\text{Node\_VRAM}} \right\rceil$$

where $S$ is the number of pipeline stages (each assigned to a different node).

### 2.3 MoE Memory

For MoE models, memory is determined by **total** parameters $P_{\text{total}}$ (all experts must be stored), while compute uses **active** parameters $P_{\text{active}}$. This means MoE models require more memory per useful FLOP than dense models, with the ratio being $P_{\text{total}} / P_{\text{active}}$.

### 2.4 Expert Parallelism Memory Reduction

When Expert Parallelism (EP) is enabled, experts are sharded across participating nodes. Each node stores only the shared (non-expert) parameters plus its fraction of the expert parameters:

$$M_{\text{node}} = \left(P_{\text{shared}} + \frac{P_{\text{experts}}}{N_{\text{EP}}}\right) \times \beta$$

where:
*   $P_{\text{shared}} = P_{\text{active}}$ (attention, embeddings, routing layers — parameters active on every token).
*   $P_{\text{experts}} = P_{\text{total}} - P_{\text{active}}$ (expert feed-forward parameters).
*   $N_{\text{EP}}$ = EP degree: $N$ (all nodes) for global EP, or $N_{\text{group}}$ for regional EP.

This can dramatically reduce per-node memory. For example, a 600B MoE with 100B shared and 500B expert params across 72 nodes: $M_{\text{node}} = (100\text{B} + 500\text{B}/72) \times 16 \approx 1{,}711$ GB, versus $600\text{B} \times 16 = 9{,}600$ GB without EP.

**Evidence:**
*   [SPES Protocol (Prime Intellect, 2025)](https://github.com/PrimeIntellect-ai/spes): expert-sharded MoE for decentralized training. Each node trains only on tokens routed to its local experts, avoiding per-forward-pass All-to-All communication. Achieves 35–65% communication reduction vs. standard DiLoCo.
*   [Lepikhin et al. (2020, "GShard")](https://arxiv.org/abs/2006.16668): standard EP distributes experts across devices, reducing per-device memory proportionally.
*   [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434): uses expert parallelism to train a 236B MoE across devices with only 21B active parameters per token.

**Implication:** When EP reduces per-node memory below Node VRAM, the simulator avoids pipeline parallelism entirely and uses DiLoCo mode, which is dramatically more efficient over WAN.

---

## 3. Communication Models

### 3.1 Mode A: Data Parallel (DiLoCo)

Used when the model fits in a single node's VRAM. Each node trains independently for $H$ inner steps, then synchronizes pseudo-gradients (parameter deltas) with all other nodes.

#### Communication Volume per Sync

$$V_{\text{bits}} = \frac{P \cdot B_{\text{precision}}}{C_r}$$

*   $B_{\text{precision}}$: bits per pseudo-gradient value, determined by compute precision (16 for FP16, 8 for FP8, 4 for FP4). Pseudo-gradients are cast to compute precision before transmission.
*   $C_r$: additional compression ratio (from quantization, sparsification, or both), applied on top of the precision-based reduction.

**Assumption:** Pseudo-gradients (the difference $\theta_{\text{local}} - \theta_{\text{original}}$) have the same dimensionality as the model parameters — one scalar per parameter — so the communication volume is identical whether sending weights or pseudo-gradients.

**Evidence:**
*   [Douillard et al. (2023, "DiLoCo")](https://arxiv.org/abs/2311.08105): defines DiLoCo's outer optimization as communicating pseudo-gradients $\Delta\theta = \theta_{\text{local}} - \theta_0$.
*   [Douillard et al. (2025, "Streaming DiLoCo")](https://arxiv.org/abs/2501.18512): confirms FP16 pseudo-gradient communication with optional further compression to FP4 (E3M0), achieving up to 100× total bandwidth reduction when combined with infrequent synchronization.

#### Sync Time (Non-Hierarchical)

$$T_{\text{sync}} = \left(\frac{2 \cdot V_{\text{bits}}}{\text{Bandwidth}} + \text{Latency}\right) \times f_{\text{straggler}}(N)$$

The factor of 2 accounts for the full round-trip: each node uploads its pseudo-gradients and downloads the averaged result.

#### Effective Outer Step Time (Streaming DiLoCo)

$$T_{\text{outer}} = \max(H \cdot T_{\text{comp}},\ T_{\text{sync}})$$

When streaming is enabled, communication is overlapped with the next block of $H$ inner compute steps. The effective time per outer step is the maximum of compute and communication. When streaming is disabled, the two are summed.

**Evidence:**
*   [Douillard et al. (2025, "Streaming DiLoCo")](https://arxiv.org/abs/2501.18512): reports 95% compute utilization by staggering parameter fragment synchronization across the inner-step window.

**Limitation:** The $\max()$ model omits several second-order overheads documented in the Streaming DiLoCo paper: (1) brief blocking periods when receiving aggregated updates (~2–5% of compute time), (2) 66% additional memory overhead for storing original parameters, outer optimizer state, and Nesterov momentum buffers (reducible to ~2% via CPU offloading for 100B+ models), and (3) minor compute cost for gradient compression. These are small enough that 95% compute utilization is achievable in practice.

#### Total Training Time (Non-Hierarchical)

$$T_{\text{total}} = \frac{D}{B_{\text{local}} \cdot N_{\text{eff}} \cdot H} \cdot T_{\text{outer}}$$

where $D$ = total tokens, $N_{\text{eff}}$ = effective nodes, $H$ = inner steps.

### 3.2 Mode A′: Hierarchical DiLoCo

Models a two-tier "cluster of clusters" topology with fast regional interconnects nested inside a slower global WAN.

#### Regional Sync (every $H_{\text{inner}}$ steps)

$$T_{\text{regional}} = \left(\frac{2 \cdot V_{\text{bits}}}{\text{BW}_{\text{regional}}} + \text{Lat}_{\text{regional}}\right) \times f_{\text{straggler}}(N_{\text{group}})$$

#### Global Sync (every $H_{\text{inner}} \times H_{\text{regional}}$ steps)

$$T_{\text{global}} = \left(\frac{2 \cdot V_{\text{bits}}}{\text{BW}_{\text{WAN}}} + \text{Lat}_{\text{WAN}}\right) \times f_{\text{straggler}}(G)$$

where $G = N_{\text{eff}} / N_{\text{group}}$ is the number of regional groups. Only group leaders participate in global sync.

#### Cycle Time

With streaming:

$$T_{\text{regional\_cycle}} = \max(H_{\text{inner}} \cdot T_{\text{comp}},\ T_{\text{regional}})$$

$$T_{\text{global\_cycle}} = \max(H_{\text{regional}} \cdot T_{\text{regional\_cycle}},\ T_{\text{global}})$$

Without streaming, $\max$ is replaced by addition.

**Evidence:**
*   The two-tier structure is used in the [SPES Protocol (Prime Intellect)](https://github.com/PrimeIntellect-ai/spes) for decentralized MoE training, where regional clusters sync frequently over LAN and global clusters sync infrequently over WAN.
*   [INTELLECT-1 (Jaghouar et al. 2024)](https://arxiv.org/abs/2412.01152): demonstrated hierarchical communication across USA, Europe, and Asia with measurable per-tier utilization degradation.

### 3.3 Mode B: Pipeline Parallel (PP)

Used when the model exceeds a single node's VRAM. The model is partitioned across $S$ pipeline stages, each on a different node. Activations and gradients are communicated every micro-batch.

#### Activation Size (per layer boundary)

$$A = B_{\text{local}} \cdot h \cdot \frac{B_{\text{precision}}}{8} \text{ bytes}$$

*   $B_{\text{local}}$ = local batch size **in tokens** (not sequences). This equals $\text{num\_sequences} \times \text{seq\_length}$ (e.g. $32 \times 4096 = 131{,}072$).
*   $h$ = hidden dimension.
*   $B_{\text{precision}} / 8$ = bytes per element (2 for FP16, 1 for FP8, 0.5 for FP4).

Since $B_{\text{local}}$ is measured in tokens, the formula correctly accounts for sequence length. The activation tensor at a pipeline boundary is shaped $[\text{num\_sequences}, \text{seq\_length}, h]$, and the total number of elements is $\text{num\_sequences} \times \text{seq\_length} \times h = B_{\text{local}} \times h$.

**Heuristic: Hidden dimension $h \approx 0.03 \cdot \sqrt{P}$.** This is derived from the standard transformer parameter formula $P \approx 12 \cdot L \cdot h^2$, where $L$ (number of layers) also scales with $h$, yielding $h \propto \sqrt{P}$.

**Validation against known architectures:**

| Model | $P$ | Predicted $h = 0.03\sqrt{P}$ | Actual $h$ | Error |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA-2 70B | 70B | 7,937 | 8,192 | −3% |
| GPT-3 175B | 175B | 12,550 | 12,288 | +2% |
| LLaMA-3.1 405B | 405B | 19,092 | 16,384 | +17% |

The heuristic is accurate within ~20% across the range of models likely to require pipeline parallelism (>70B parameters).

#### PP Step Time

$$T_{\text{step}} = (M + S - 1) \cdot (T_{\text{micro\_comp}} + (T_{\text{micro\_comm}} + \text{Latency}) \cdot f_{\text{straggler}})$$

*   $M$ = number of micro-batches.
*   $S$ = number of pipeline stages.
*   $(M + S - 1)$ = total micro-batch slots including pipeline bubble overhead.

**Evidence:**
*   This is the standard GPipe pipeline bubble formula from [Huang et al. (2019)](https://arxiv.org/abs/1811.06965).
*   In WAN settings, the latency term dominates because every micro-batch handover incurs a network round-trip, as noted in [Ryabinin et al. (2023, "SWARM Parallelism")](https://arxiv.org/abs/2301.11913).

### 3.4 Mode C: PP-Group DiLoCo

Used when the model exceeds single-node VRAM and there are enough nodes to form multiple pipeline groups. Instead of running a single pipeline across all WAN-connected nodes, nodes are grouped into small PP clusters of size $S$ (the number of pipeline stages needed), and DiLoCo is run across the $G = \lfloor N / S \rfloor$ groups.

#### Intra-Group PP

Each group of $S$ nodes runs standard pipeline parallelism using the GPipe bubble formula:

$$T_{\text{PP\_step}} = (M + S - 1) \cdot (T_{\text{micro\_comp}} + (T_{\text{micro\_comm}} + \text{Lat}_{\text{PP}}) \cdot f_{\text{straggler}}(S))$$

When hierarchical mode is enabled, PP groups use **regional** latency and bandwidth for intra-group communication (nodes in a group are co-located in the same region). Otherwise, PP handoffs use WAN latency.

#### Inter-Group DiLoCo

Groups synchronize pseudo-gradients every $H$ inner steps, identically to Mode A:

$$T_{\text{sync}} = \left(\frac{2 \cdot V_{\text{bits}}}{\text{BW}_{\text{WAN}}} + \text{Lat}_{\text{WAN}}\right) \times f_{\text{straggler}}(G)$$

With streaming enabled:

$$T_{\text{outer}} = \max(H \cdot T_{\text{PP\_step}},\ T_{\text{sync}})$$

#### Total Training Time

$$T_{\text{total}} = \frac{D}{B_{\text{local}} \cdot G \cdot H} \cdot T_{\text{outer}}$$

Each group processes $B_{\text{local}}$ tokens per step (the pipeline splits the model, not the data).

#### Fallback: Pure PP over WAN

When $G < 2$ (not enough nodes to form multiple groups), the simulator falls back to a single pipeline across all nodes with no DiLoCo parallelism. This is extremely slow and triggers a prominent warning.

**Evidence:**
*   [DiLoCoX (Chen et al., 2025)](https://arxiv.org/abs/2506.21263): combines intra-group PP (fast interconnect) with inter-group DiLoCo (slow WAN). On a 107B model across 160 A800 GPUs over 1 Gbps interconnect, achieved 357× throughput improvement over standard AllReduce.
*   [SWARM Parallelism (Ryabinin et al., 2023)](https://arxiv.org/abs/2301.11913): adaptive PP over unreliable internet. At 100ms RTT, GPT-3-scale models retain ~60% utilization (72% with compression).

**Why PP-Group DiLoCo is far superior to PP-over-WAN:** In pure PP-over-WAN, every micro-batch handover pays WAN latency. With $M=8$ micro-batches and $S=3$ stages, there are $(8+3-1)=10$ sequential latency penalties per step. In PP-Group DiLoCo, the WAN is only used for periodic DiLoCo sync (every $H=128$ steps), while PP handoffs within a group can use faster regional interconnect.

---

## 4. Algorithmic Efficiency

Decentralized training with infrequent synchronization is not 100% compute-equivalent to synchronous data-parallel training. The simulator models this as a multiplicative efficiency penalty.

### 4.1 Efficiency Equation

$$\eta = \max\!\left(0.4,\;\frac{1 - \alpha \cdot \log_{10}(H_{\text{eff}})}{P_{\text{strategy}}}\right)$$

*   $H_{\text{eff}}$ = effective inner steps between global syncs.
*   $\alpha$ = sensitivity coefficient (base value ~0.08, reduced for larger models).
*   $P_{\text{strategy}}$ = straggler strategy penalty (1.15 for threshold aggregation, 1.0 otherwise).
*   Floor of 40% prevents nonsensical near-zero efficiency.

**Interpretation:** Efficiency $\eta < 1$ means the model requires $1/\eta$ times more tokens (and wall-clock time) to reach the same loss as fully-synchronous training.

### 4.2 Logarithmic Decay with $H$

**Assumption (modeling choice):** Efficiency decays logarithmically with inner steps. This means doubling $H$ causes a fixed additive penalty, not a multiplicative one — large $H$ values (100–500) are feasible with only modest degradation.

**Evidence:**
*   [Douillard et al. (2023, "DiLoCo")](https://arxiv.org/abs/2311.08105): tested $H$ up to 500 inner steps with only modest loss degradation, consistent with slow (sub-linear) decay.
*   [Charles et al. (2025)](https://arxiv.org/abs/2503.09799): shows that evaluation loss increases with $H$, but the rate of increase is less pronounced for larger models. The paper does not publish a specific $\eta = 1 - \alpha \log H$ formula — the logarithmic functional form is a simulator modeling choice that is consistent with, but not directly derived from, these empirical results.
*   [Stich (2019, "Local SGD Converges Fast")](https://arxiv.org/abs/1805.09767): provides theoretical convergence bounds for local SGD showing convergence rate degrades sub-linearly with local steps.

### 4.3 Alpha Scaling with Model Size

$$\alpha = 0.08 \cdot \frac{1}{1 + \frac{\log_{10}(P / 10^9)}{5}}$$

**Assumption:** Larger models are more robust to delayed synchronization, so $\alpha$ decreases with model size.

**Evidence:**
*   [Charles et al. (2025)](https://arxiv.org/abs/2503.09799): "outside of $H=1$, evaluation loss increases with $H$" but the degradation is "less pronounced for larger models." This directly supports a decreasing $\alpha$ with scale.
*   The specific functional form and the base value 0.08 are engineering estimates calibrated to match observed DiLoCo results at the 1B–10B scale, not published constants.

### 4.4 Hierarchical Effective $H$

$$H_{\text{eff}} = H_{\text{inner}} \cdot H_{\text{regional}}^{0.5}$$

**Assumption (heuristic):** Regional syncs partially "anchor" local weight drift, reducing the effective divergence interval. The $\sqrt{}$ relationship models a moderate benefit: less than full multiplicative accumulation ($H_{\text{inner}} \cdot H_{\text{regional}}$), but more than simple additive ($H_{\text{inner}} + H_{\text{regional}}$).

**Evidence:** This is a heuristic with limited direct empirical support. The intuition is drawn from:
*   [Wang & Joshi (2021, "Cooperative SGD")](https://jmlr.org/papers/volume22/20-147/20-147.pdf): analyzes convergence of local SGD with periodic averaging, though does not model two-tier hierarchies specifically.
*   General convergence theory for local SGD suggests that more frequent partial averaging reduces the variance of the divergence, which scales sub-linearly.
*   The $\sqrt{}$ exponent should be treated as a tunable heuristic, not a literature-derived constant.

### 4.5 Threshold Strategy Penalty

**Assumption:** Threshold aggregation (proceeding with the fastest 90% of nodes, discarding stragglers) introduces a 15% token-efficiency penalty because the excluded gradients introduce staleness.

This is an engineering estimate. Asynchronous/bounded-staleness SGD literature ([Ho et al. 2013](https://proceedings.neurips.cc/paper/2013/hash/b7bb35b9c6ca2aee2df08cf09d7016c2-Abstract.html)) shows degradation from stale gradients, but the specific 15% value is not derived from a published measurement at this scale.

---

## 5. Straggler & Congestion Model

### 5.1 Straggler Factor

$$f_{\text{straggler}}(n) = 1 + 0.05 \cdot \log_2(n)$$

**Assumption:** In synchronous training, the group must wait for the slowest node. The expected delay of the maximum of $n$ workers scales logarithmically with $n$.

**Evidence:**
*   From extreme value theory: if worker completion times are i.i.d. lognormal, the expected maximum grows as $O(\sqrt{\log n})$, which is well-approximated by a $\log n$ scaling for practical ranges of $n$.
*   Empirically confirmed by [Chen et al. (2016)](https://arxiv.org/abs/1604.00981) and discussed in [PyTorch DDP straggler mitigation](https://pytorch.org/blog/straggler-mitigation/).
*   The coefficient 0.05 is an engineering estimate. For $n = 72$: factor = 1.31 (31% overhead), which is plausible for heterogeneous WAN environments. The coefficient is tunable.

### 5.2 Mitigation Strategies

| Strategy | Effect | Modeling |
| :--- | :--- | :--- |
| **None** | Full straggler penalty | $f(n)$ applied directly |
| **Threshold (90% cutoff)** | $f(n) = 1.0$ (no sync delay) | But 15% algorithmic efficiency penalty |
| **Backup Workers (10% extra)** | 70% reduction in straggler delay | $N_{\text{eff}} = N / 1.1$; $f' = 1 + 0.3(f-1)$ |

---

## 6. Global Utilization Metrics

### 6.1 Global MFU

$$\text{Global MFU} = \frac{6 \cdot P_{\text{active}} \cdot D}{N \cdot \text{PFLOPS} \cdot 10^{15} \cdot T_{\text{effective}}}$$

This is the end-to-end utilization of the entire cluster across the full training run, accounting for communication stalls, pipeline bubbles, straggler delays, and algorithmic inefficiency. It will always be lower than the per-node base MFU because it includes all system-level overheads.

### 6.2 Hardware FLOPs Utilization (HFU)

$$\text{HFU} = \frac{\text{Global MFU}}{0.8}$$

**Assumption:** MFU is approximately 80% of HFU. The difference is that HFU counts all hardware FLOPs including activation recomputation (rematerialization), while MFU counts only the "useful" forward + backward FLOPs.

**Evidence:**
*   [PaLM 540B (Chowdhery et al. 2022)](https://arxiv.org/abs/2204.02311): MFU = 46.2%, HFU = 57.8%, giving a ratio of MFU/HFU = 0.799, matching the 0.8 assumption closely.

**Limitation:** The 0.8 ratio depends on the activation recomputation strategy. With no recomputation, MFU = HFU (ratio 1.0). With full recomputation, the ratio drops to ~0.67. The 0.8 value corresponds to selective recomputation, which is the common practice for large models. This could be made configurable in a future version.

---

## 7. Maximum Training Run Duration

Based on [Epoch AI's "The Longest Training Run" analysis](https://epoch.ai/blog/the-longest-training-run).

### 7.1 Formula

$$L = \frac{1}{(g_H + g_S + g_I) \cdot \ln(10)} \text{ years}$$

*   $g_H$ = hardware efficiency growth rate (OOM/year, i.e., $\log_{10}$ scale).
*   $g_S$ = software/algorithmic efficiency growth rate (OOM/year).
*   $g_I$ = investment/spending growth rate (OOM/year).

**Interpretation:** If technology and budgets are improving at combined rate $g$ (OOM/year), then any training run longer than $L$ years would be better served by waiting for improved conditions and running a shorter, more efficient job. The $\ln(10)$ conversion is required because the Epoch AI derivation uses natural-log growth rates internally while the simulator stores rates in $\log_{10}$ (OOM/year) units.

### 7.2 Default Growth Rates

| Factor | Default (×/yr) | OOM/yr | Source |
| :--- | :--- | :--- | :--- |
| Hardware ($g_H$) | 1.37× | 0.137 | Conservative estimate; Epoch AI estimates ~1.9×/yr |
| Software ($g_S$) | 3.0× | 0.477 | Epoch AI estimates ~3.5×/yr for algorithmic efficiency |
| Investment ($g_I$) | 3.5× | 0.544 | Epoch AI estimates ~2.8×/yr for capital investment |

**Evidence:**
*   [Epoch AI (2023), "The Longest Training Run"](https://epoch.ai/blog/the-longest-training-run): derives the $L = 1/g$ formula and estimates growth rates from historical data on compute costs, algorithmic improvements, and AI investment trends.
*   The simulator's defaults are user-adjustable and represent one plausible set of assumptions. Users should calibrate these to their own estimates.

---

## 8. Precision & Compression

### 8.1 Precision Effect on Communication

Lower compute precision reduces communication volume for both pseudo-gradients (Section 3.1) and pipeline activations (Section 3.3). The $B_{\text{precision}}$ parameter in those formulas is determined by the precision setting:

| Precision | $B_{\text{precision}}$ (bits) | Bytes/Value | Bandwidth Savings vs FP16 |
| :--- | :--- | :--- | :--- |
| FP16/BF16 | 16 | 2 | 1× (baseline) |
| FP8 | 8 | 1 | 2× |
| FP4 | 4 | 0.5 | 4× |

This precision-based reduction is **independent of** the user-specified compression ratio $C_r$. The total effective compression from FP16 baseline is $(16 / B_{\text{precision}}) \times C_r$.

### 8.2 Additional Compression

On top of precision, the user can apply further compression via quantization and sparsification of pseudo-gradients. A compression ratio of 16× corresponds to, e.g., 4-bit quantization (4× from FP16) combined with 25% sparsification (4×), with error-feedback accumulation to minimize accuracy loss.

**Evidence:**
*   [Douillard et al. (2025, "Streaming DiLoCo")](https://arxiv.org/abs/2501.18512): achieves 100× total bandwidth reduction by combining infrequent synchronization with FP4 (E3M0) quantization.
*   [Bernstein et al. (2018, "signSGD")](https://arxiv.org/abs/1802.04434) and [Karimireddy et al. (2019, "Error Feedback")](https://arxiv.org/abs/1901.09847): demonstrate that aggressive gradient compression with error feedback preserves convergence.

---

## 9. Variables Reference Table

| Variable | Unit | Default | Description |
| :--- | :--- | :--- | :--- |
| `parameters` | Billion ($10^9$) | 144B | Total model parameter count. For MoE, includes all experts. |
| `activeParams` | Billion ($10^9$) | 24B | Active parameters per token (MoE only). |
| `tokens` | Trillion ($10^{12}$) | 12T | Total training dataset size. |
| `numNodes` | Integer | 72 | Number of geographically distributed training replicas. |
| `pflopsPerNode` | PFLOPS ($10^{15}$) | 32 | Peak FP16 performance per node (e.g., 16× GH200). |
| `vramPerNode` | GB | 2304 | Total HBM per node (e.g., 16× 144 GB). |
| `bandwidthMbps` | Mbps | 100 | Symmetric WAN upload/download per node. |
| `latencyMs` | ms | 100 | WAN round-trip time between nodes. |
| `mfu` | Fraction | 0.40 | Base Model FLOP Utilization. |
| `innerSteps` | Integer | 128 | Local SGD steps between synchronizations. |
| `compression` | Factor | 16× | Pseudo-gradient compression ratio. |
| `localBatch` | Tokens | 131,072 | Tokens per local training step (= num\_sequences × seq\_length). |
| `microBatches` | Integer | 8 | Micro-batches for pipeline parallelism. |
| `precision` | Enum | FP16 | Compute precision (FP16, FP8, FP4). |
| `streamingEnabled` | Boolean | true | Overlap communication with compute. |
| `nodesPerGroup` | Integer | 8 | Nodes per regional cluster (hierarchical mode). |
| `regionalBandwidth` | Mbps | 1000 | Intra-group bandwidth. |
| `regionalLatency` | ms | 20 | Intra-group round-trip time. |
| `regionalSteps` | Integer | 16 | Regional sync cycles per global sync. |

---

## 10. Known Limitations & Future Work

1.  **PP-Group DiLoCo assumes homogeneous groups.** All PP groups are assumed to have equal compute power and interconnect. In practice, heterogeneous node capabilities would cause some groups to be faster than others, increasing the straggler penalty at the DiLoCo sync boundary.
2.  **Streaming DiLoCo memory overhead:** The simulator does not model the ~66% additional memory requirement for Streaming DiLoCo (original weights buffer + outer optimizer state). This could trigger sharding in cases the simulator currently classifies as data-parallel.
3.  **Hierarchical compression:** The simulator uses the same compression ratio for both regional and global tiers. In practice, regional sync over LAN could use less aggressive compression.
4.  **Heterogeneous hardware:** All nodes are assumed identical. Real decentralized networks may have heterogeneous compute speeds, which would increase straggler effects beyond the logarithmic model.
5.  **Asynchronous methods:** The simulator only models synchronous protocols (DiLoCo, hierarchical DiLoCo). Fully asynchronous approaches ([Diskin et al. 2021](https://arxiv.org/abs/2106.10207)) are not modeled.
6.  **PP mode only models forward activations.** Real pipeline parallelism also sends gradients backward through the pipeline, roughly doubling communication per micro-batch.
7.  **EP memory model is simplified.** The EP memory reduction assumes a clean split between shared and expert parameters ($P_{\text{shared}} = P_{\text{active}}$). Real MoE architectures may have routing layers, load-balancing buffers, and token-drop buffers that add overhead beyond this estimate.

---

## Appendix A: When the Model Doesn't Fit — Developer Decision Tree

The simulator's purpose is to answer: *given these exact hardware constraints (node VRAM, WAN bandwidth, WAN latency, node count), how long does this training run take?* When the model exceeds single-node VRAM, the simulator automatically applies the best available strategy: EP memory reduction (MoE), PP-Group DiLoCo, or pure PP-over-WAN as a last resort.

This appendix explains the full decision tree and what the simulator does at each stage.

### Decision Tree

```
Model fits on one node? (P × β ≤ VRAM)
├── YES → Mode A: DiLoCo (fast, efficient)
│
└── NO →
    ├── MoE with EP enabled?
    │   ├── YES → Compute per-node memory: (P_shared + P_experts/N_EP) × β
    │   │   ├── Fits on one node? → Mode A: DiLoCo (EP saved the day!)
    │   │   └── Still too large → continue to PP-Group below
    │   └── NO → continue to PP-Group below
    │
    ├── Enough nodes for multiple PP groups? (N / ppStages ≥ 2)
    │   ├── YES → Mode C: PP-Group DiLoCo
    │   │   ├── Group S nodes into PP clusters
    │   │   ├── Run DiLoCo across N/S groups
    │   │   └── Hierarchy enabled? PP uses regional interconnect
    │   │
    │   └── NO → Pure PP over WAN (last resort, extremely slow)
    │
    └── Developer should also consider:
        ├── Switch to lower precision (FP8/FP4)
        │   └── Reduces β from 16 → 14 or 13, may allow DiLoCo
        ├── Train a smaller model
        │   └── Max size = VRAM / β (e.g., 2304 GB / 16 = 144B)
        └── Use MoE + EP to get memory below the threshold
```

### Mode Comparison

| Mode | When Used | WAN Hit Per Step | Typical Overhead |
| :--- | :--- | :--- | :--- |
| A: DiLoCo | Model fits on one node | Every H steps (sync only) | Low (communication-overlapped) |
| C: PP-Group DiLoCo | Model sharded, N/S ≥ 2 groups | PP within group + sync every H steps | Moderate (PP bubble + periodic sync) |
| PP over WAN | Model sharded, N/S < 2 | Every micro-batch | Extreme (WAN latency dominates) |

### Why PP-Group DiLoCo Is Superior to Pure PP-over-WAN

Consider a 300B model on 72 nodes with 2304 GB VRAM each (FP16): $300\text{B} \times 16 = 4{,}800$ GB, requiring $S=3$ PP stages.

**Pure PP over WAN (old behavior):** A single 3-stage pipeline across all 72 nodes. Each of the $(M + S - 1) = 10$ micro-batch slots pays 100ms WAN latency = 1 second of pure idle time per step. The pipeline only utilizes 24 nodes (72/3), wasting 48 nodes.

**PP-Group DiLoCo (new behavior):** $72 / 3 = 24$ groups of 3 nodes each. Each group runs a 3-stage pipeline. All 72 nodes are utilized. WAN latency only affects the DiLoCo sync every 128 steps, not every micro-batch. With hierarchical mode, PP handoffs use 20ms regional latency instead of 100ms WAN.

### Guidance for Simulator Users

1.  **Explore EP for MoE models.** If your model is MoE, enabling Expert Parallelism may reduce per-node memory enough to avoid PP entirely — the biggest performance win.
2.  **Use hierarchical mode with PP-Group DiLoCo.** When the model must be sharded, enabling hierarchical mode lets PP groups use fast regional interconnect, dramatically reducing per-micro-batch latency.
3.  **Explore the boundary.** Toggle precision or model size to find the largest model that fits on one node. The performance cliff at the sharding boundary is significant, so staying in DiLoCo mode (even with a smaller model) is often better than PP-Group DiLoCo with a larger model.

---

## Appendix B: Potential Future Improvements to Decentralized Training

The techniques documented below are architectural and algorithmic improvements that have been demonstrated in recent research but have **not been tested at large scale in distributed WAN settings**. They are not incorporated into the simulator because their effectiveness at the $10^{26}$–$10^{27}$ FLOP regimes targeted by this analysis remains unvalidated.

They are documented here so that policymakers are aware that decentralized training capabilities could improve in the future as these techniques mature and are validated at scale. Each entry notes the scale at which the technique has been tested, which is typically 2–3 orders of magnitude smaller than the simulator's target regime.

All findings are from the February 2026 literature review (see `Bibliography.md`).

### B.1 Zero Bubble Pipeline Scheduling

**Applies to:** PP-Group DiLoCo (Mode C) and PP-over-WAN (Mode B).

**Simulator assumption:** GPipe bubble formula — $(S - 1)$ of $(M + S - 1)$ micro-batch slots are idle. For $M=8, S=3$: 22.2% bubble overhead.

**What the literature shows:** [Qi et al. (2024), "Zero Bubble Pipeline Parallelism"](https://arxiv.org/abs/2401.10241) splits the backward pass into input-gradient (B) and weight-gradient (W) phases, rescheduling W to fill bubble slots. ZB-2p achieves <1% bubble across all tested configurations. Measured 23–30% throughput improvement over 1F1B on models from 1.5B to 28.3B with 8 pipeline stages. Now standard practice at frontier labs (foundation for DeepSeek-V3's DualPipe).

**Quantified gap:**

| PP Stages ($S$) | Simulator Bubble | ZB-2p Bubble | Compute Speedup |
|:--|:--|:--|:--|
| 2, $M=8$ | 11.1% | ~2% | 1.10× |
| 3, $M=8$ | 22.2% | ~3% | 1.25× |
| 4, $M=8$ | 27.3% | ~4% | 1.32× |

**WAN-applicable:** Yes. Zero-bubble scheduling is a local computation reordering that does not change the communication pattern.

**Caveat:** Over WAN, communication latency (not bubble) typically dominates. The speedup applies only to the compute portion of the PP step time. End-to-end improvement depends on the compute/communication ratio.

**Evidence scale:** 28.3B parameters, 8 pipeline stages.

### B.2 Delayed Nesterov + Dynamic Local Updates

**Applies to:** Algorithmic efficiency penalty (Section 4).

**Simulator assumption:** $\alpha \approx 0.08$ (scaled by model size), giving ~88% efficiency at $H=128$ for 144B parameters.

**What the literature shows:** [Liu, Douillard et al. (2024), "Asynchronous Local-SGD Training for Language Modeling"](https://arxiv.org/abs/2401.09135) demonstrates two techniques that nearly eliminate the DiLoCo algorithmic penalty:

*   **Delayed Nesterov (DN):** Standard Nesterov momentum compounds incorrectly in async mode when workers arrive sequentially. DN fixes this by applying momentum only every $N$ server iterations ($N$ = worker count), using pure gradient descent between. This alone closes most of the async gap.
*   **Dynamic Local Updates (DyLU):** Each worker adjusts its $H$ proportionally to its speed: $w.\text{steps} = \lfloor v(w) / \max(v(w')) \cdot H \rfloor$. Slow workers do fewer local steps, preventing stale gradients.

At $H=50$, naive async DiLoCo has ~7% perplexity degradation vs. sync (44.27 vs 41.35 for 20M params). With DN + DyLU, the gap is fully closed (41.13 vs 41.35), slightly *beating* synchronous training.

**Quantified gap:** The simulator's effective $\alpha$ could drop to near zero with these techniques, overestimating the penalty by 5–12%.

**On the straggler model:** DyLU also makes the straggler factor $f(n) = 1 + 0.05 \cdot \log_2(n)$ overly pessimistic for heterogeneous hardware, since DyLU prevents slow workers from blocking fast ones. Potential 10–30% improvement in mixed-hardware settings.

**WAN-applicable:** Yes.

**Evidence scale:** 150M parameters, ≤16 workers. The simulator targets 144B+; whether the alpha reduction persists at 1000× larger scale is unknown.

### B.3 Hierarchical Momentum (HALoS)

**Applies to:** Hierarchical effective $H$ (Section 4.4).

**Simulator assumption:** $H_{\text{eff}} = H_{\text{inner}} \cdot H_{\text{regional}}^{0.5}$, predicting 85–88% efficiency.

**What the literature shows:** [Kim et al. (2025), "HALoS: Hierarchical Asynchronous Local SGD"](https://arxiv.org/abs/2506.04531) uses local parameter servers within regions and a global parameter server across regions, with separate hierarchical momentum terms ($\beta_{\text{local}}$ and $\beta_{\text{global}}$). HALoS **matches synchronous SGD quality** — meaning near-zero algorithmic penalty. Achieves 7.5× wall-clock speedup over synchronous baselines by eliminating synchronization barriers.

**Quantified gap:** The $\sqrt{H_{\text{regional}}}$ heuristic is conservative. HALoS suggests hierarchy can achieve near-zero penalty with the right optimizer, vs. the simulator's prediction of 85–88% efficiency. Gap: ~3–5%.

**WAN-applicable:** Yes.

**Evidence scale:** 70M parameters only (Pythia/LLaMA/Qwen-70M). Whether hierarchical momentum maintains zero penalty at 144B is unknown.

### B.4 Sparse Parameter Averaging (SPARTA)

**Applies to:** Algorithmic efficiency at high $H$ (Section 4) and communication volume (Section 3.1).

**Simulator assumption:** Higher $H$ = worse efficiency per $\eta = 1 - \alpha \cdot \log_{10}(H)$. Communication payload = $P \times 16 / C_r$.

**What the literature shows:** [SPARTA (2025)](https://openreview.net/pdf?id=stFPf3gzq1) exchanges only 0.1–0.5% of parameters continuously (not at DiLoCo sync boundaries). At $H=10{,}000$ with 0.1% exchange: **1000× communication reduction** AND **14.3% perplexity improvement** over DiLoCo-alone at $H=10{,}000$.

The improvement is possible because sparse weight averaging acts as a regularizer preventing local model drift, enabling higher learning rates (2×) and larger $H$ values without the expected degradation. This **decouples** $H$ from convergence quality — a relationship the simulator treats as fundamental.

**Quantified gap:** At $H=10{,}000$ for 124M parameters, the simulator predicts ~61% efficiency. SPARTA achieves quality better than DiLoCo-alone, suggesting ~85–95% effective efficiency. That is a 25–35 percentage point gap.

**WAN-applicable:** Yes.

**Critical caveats:**
*   Only tested at 124M parameters, 2–8 nodes.
*   Paper explicitly states "doesn't scale well beyond 16 nodes."
*   Extrapolation to 144B / 72 nodes is highly speculative.
*   The regularization mechanism may behave differently at scale.

### B.5 DualPipe Compute-Communication Overlap

**Applies to:** EP All-to-All latency (Section 1.5).

**Simulator assumption:** EP latency is additive: $T_{\text{EP}} = 2 \cdot \text{Latency} \cdot L_{\text{MoE}}$.

**What the literature shows:** [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) feeds micro-batches from both pipeline ends simultaneously and dedicates 20 of 132 SMs to communication kernels. Result: "both all-to-all and PP communication can be fully hidden during execution."

**WAN-applicable:** No. DualPipe's overlap requires NVLink (160 GB/s, ~1μs latency) and InfiniBand (50 GB/s, ~5μs). At WAN latency (100ms), communication cannot be hidden behind ~30ms of MLP compute. **The simulator's additive EP latency model is correct for WAN settings.**

**Evidence scale:** 671B MoE (DeepSeek-V3) on H800 GPUs.

### B.6 Loss-Tolerant Training

**Applies to:** Straggler model (Section 5.1).

**Simulator assumption:** Reliable TCP connections; straggler factor based on compute heterogeneity.

**What the literature shows:** [Weintraub et al. (2025), "Distributed Training under Packet Loss"](https://arxiv.org/abs/2507.07114) demonstrates that 10% packet loss causes only 0.8% perplexity degradation on LLaMA-2 7B. At 20%: 2.6%. At 30%: 3.8%. Uses unbiased gradient aggregation over whatever packets arrive, plus bounded-drift parameter broadcasts.

The paper argues that TCP retransmissions are the primary problem for WAN training — they inflate tail latencies. A loss-tolerant UDP approach could reduce effective straggler penalties at the cost of a small (~1%) convergence penalty.

**Quantified gap:** A developer using loss-tolerant protocols could see 10–15% reduction in straggler overhead.

**WAN-applicable:** Yes.

**Verdict:** This largely validates the simulator's approach. The simulator's straggler factor partially captures TCP retransmission effects. WAN training is more robust to network issues than one might fear.

**Evidence scale:** 7B parameters, 64 GPUs.

### B.7 Summary

| Technique | Simulator Gap | Speedup vs. Simulator | WAN? | Evidence Scale |
|:--|:--|:--|:--|:--|
| Zero Bubble PP (B.1) | GPipe bubble 11–27%; ZB-2p <1% | 10–30% on PP compute | Yes | 28B, 8 stages |
| Delayed Nesterov + DyLU (B.2) | $\alpha=0.08$ too pessimistic with proper async optimizer | 5–15% | Yes | 150M, ≤16 workers |
| HALoS hierarchy (B.3) | $\sqrt{H_R}$ heuristic conservative | 3–5% | Yes | 70M only |
| SPARTA sparse averaging (B.4) | Efficiency formula too pessimistic at high $H$ | 30–50% (speculative) | Yes | 124M, ≤8 nodes |
| DualPipe EP overlap (B.5) | EP latency additive vs. fully hidden | Large (datacenter only) | **No** | 671B |
| Loss-tolerant training (B.6) | Straggler model doesn't capture UDP benefits | 10–15% on straggler | Yes | 7B, 64 GPUs |

**Implications for policy analysis:** None of these techniques have been validated at the scale the simulator targets ($10^{26}$–$10^{27}$ FLOPs, 100B+ parameters, 50+ WAN-connected nodes). The simulator's estimates reflect techniques that are proven at scale today. However, policymakers should be aware that decentralized training capabilities are an active area of research. If techniques like zero-bubble PP scheduling, optimized async DiLoCo, sparse parameter averaging, or loss-tolerant protocols are successfully scaled, future developers could plausibly train 15–30% faster than the simulator currently predicts for pipeline-parallel workloads, or 5–15% faster for pure DiLoCo workloads. The SPARTA results hint at even larger gains at very high synchronization intervals, but the evidence is preliminary (124M parameters, ≤8 nodes). These improvements would narrow the gap between decentralized and datacenter training, not eliminate it.
