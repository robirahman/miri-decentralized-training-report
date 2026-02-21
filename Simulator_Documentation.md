# Decentralized Training Simulator: Technical Documentation

This document provides a detailed breakdown of the mathematical models, variables, and empirical assumptions used in the Decentralized Training Simulator.

---

## 1. Core Model & Compute
The simulator estimates the total computational effort required for a pre-training run.

### Equations
*   **Total Compute ($C$):** $C = 6 \cdot P \cdot D$
    *   *Definition:* Total floating-point operations (FLOPs) required.
    *   *Evidence:* The "6 FLOPs per parameter per token" is the standard heuristic for the forward and backward pass of a Transformer model, established by [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) and [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556).
*   **Step Duration (Compute Only):** $T_{comp} = \frac{6 \cdot P \cdot B_{local}}{Node\_PFLOPS \cdot MFU}$
    *   *Variable $B_{local}$:* Local batch size (tokens).
    *   *Variable $MFU$:* Model FLOP Utilization.
    *   *Assumption:* Standard MFU for optimized clusters is 40-60%. In decentralized settings with heterogeneous hardware, this is likely lower (30-40%).

---

## 2. Memory & Sharding Logic
Determines whether a model can fit on a single node or must be sharded using Pipeline Parallelism (PP).

### Equations
*   **Memory Required ($M_{req}$):** $M_{req} = P \cdot 12$
    *   *Assumption:* Training requires memory for weights (2 bytes in FP16), gradients (2 bytes), and optimizer states (8 bytes for AdamW). Techniques like ZeRO-1/2/3 or FSDP can shard these, but 12 bytes/param is a realistic "middle ground" for large models using some memory optimization.
*   **Sharding Trigger:** If $M_{req} > Node\_VRAM$, mode switches to **Pipeline Parallelism**.
    *   *PP Stages ($S$):* $S = \lceil M_{req} / Node\_VRAM ceil$.

---

## 3. Communication Models

### Mode A: Data Parallel (DiLoCo)
Used when the model fits in a single node's VRAM. Synchronizes weights infrequently.

*   **Comm Volume per Sync:** $V_{bits} = \frac{P \cdot 16}{Compression}$
    *   *Logic:* Each node sends its local weights (or pseudo-gradients) and receives the global average. 16 bits per param (FP16).
*   **Sync Time:** $T_{sync} = \max(H \cdot T_{comp}, \frac{2 \cdot V_{bits}}{Bandwidth} + Latency)$
    *   *Logic:* DiLoCo uses "Streaming" to overlap compute and communication. The effective time is the maximum of the two blocks if the network is fast enough to hide the transfer during the $H$ inner steps.
    *   *Evidence:* [Douillard et al. (2023)](https://arxiv.org/abs/2311.08105).

### Mode B: Pipeline Parallel (PP)
Used for models $>1$T parameters. Synchronizes activations/gradients every micro-batch.

*   **Activation Size ($A$):** $A = B_{local} \cdot h \cdot 2 	ext{ bytes}$
    *   *Heuristic:* Hidden dimension $h \approx 0.004 \cdot \sqrt{P}$. Derived from scaling laws where $h$ grows with the square root of parameter count.
*   **PP Step Time:** $T_{step} = (M + S - 1) \cdot (T_{micro\_comp} + T_{micro\_comm} + Latency)$
    *   *Variable $M$:* Number of micro-batches.
    *   *Logic:* This is the standard "Pipeline Bubble" formula. In WAN settings, the $Latency$ component dominates because every micro-batch handover between nodes incurs a network ping.
    *   *Evidence:* [Huang et al. (GPipe)](https://arxiv.org/abs/1811.06965).

---

## 4. Algorithmic Efficiency
Distributed training is not 100% compute-efficient compared to a single monolithic cluster.

### Equation
*   **Efficiency ($\eta$):** $\eta = 1 - \alpha \cdot \log_{10}(H_{total})$
    *   *Variable $H_{total}$:* Total inner steps between global weight averages.
    *   *Variable $\alpha$:* Sensitivity coefficient (~0.08). It reduces for larger models, which are empirically more robust to local divergence.
    *   *Assumption:* Large values of $H$ (e.g., >500) lead to "weight divergence," where the model takes more tokens to reach the same loss as a synchronous model.
    *   *Evidence:* [Charles et al. (2025)](https://arxiv.org/abs/2503.04269) on DiLoCo scaling laws.

---

## 5. Hierarchical Synchronization
Models a "Cluster of Clusters" (e.g., multiple regional data centers).

### Logic
*   **Regional Sync:** High bandwidth, low latency. Occurs every $H_{inner}$ steps.
*   **Global Sync:** Low bandwidth, high latency. Occurs every $H_{regional} \cdot H_{inner}$ steps.
*   **Impact:** Regional syncs keep "islands" of nodes tightly coupled, reducing the algorithmic penalty that would otherwise occur if global sync was the only mechanism.
*   *Evidence:* Used in the [SPES Protocol](https://github.com/PrimeIntellect-ai/spes) for MoE training.

---

## 6. Variables Reference Table

| Variable | Unit | Description |
| :--- | :--- | :--- |
| `parameters` | Billion ($10^9$) | Total model parameter count. |
| `tokens` | Trillion ($10^{12}$) | Total training dataset size. |
| `Node PFLOPS` | $10^{15}$ | Peak FP16 performance per node. |
| `WAN Bandwidth` | Mbps | Symmetric upload/download speed of the node. |
| `WAN Latency` | ms | Round-trip time (ping) between nodes. |
| `Inner Steps` | Integer | Number of local SGD steps before a synchronization. |
| `Compression` | Factor ($x$) | Reduction in communication volume (Quantization/Sparsification). |
| `MFU` | % | Fraction of peak theoretical FLOPs actually achieved. |
