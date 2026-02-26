# Governance Analysis: Distributed Training Below the Registration Threshold

## 1. Scenario

The [MIRI treaty proposal](https://arxiv.org/abs/2511.10783) defines a **covered chip cluster (CCC)** as any set of AI chips with aggregate compute capacity exceeding **16 H100-equivalents** (15,840 TFLOPS FP16). All CCCs must be registered and monitored. Training runs above **10^24 FLOP** (the "Strict Threshold") are prohibited.

This analysis examines a specific evasion scenario: an actor operates multiple unregistered nodes, each below the CCC compute threshold, connected via ordinary internet (100 Mbps WAN) and running the DiLoCo distributed training protocol. The treaty explicitly captures such distributed training: the CCC definition includes clusters "networked to perform workloads together," regardless of interconnect bandwidth. However, **detecting** such a run is much harder than detecting a traditional data center, because each individual node is indistinguishable from legitimate sub-threshold compute use.

**Question:** How much local-equivalent compute can the evader achieve in the maximum training duration?

**Scope: pretraining only.** This analysis focuses exclusively on the pretraining phase — the initial large-scale training run that produces a base model. Pretraining is the phase most amenable to detection because it requires coordinating all hardware simultaneously over an extended period, creating observable patterns in chip procurement, power consumption, and network traffic. Post-training procedures (reinforcement learning from human feedback, RL rollouts, instruction tuning) are excluded because they are (a) substantially less compute-intensive than pretraining and (b) highly parallelizable across independent, small-scale compute instances. An evader could perform post-training on commodity hardware without the concentrated GPU clusters that the treaty's monitoring mechanisms are designed to detect. Detection efforts should therefore focus on the pretraining phase.

## 2. Time Limit

Under the treaty's restrictions on hardware concentration, AI research, and compute spending, the growth rates for hardware efficiency ($g_H$), software efficiency ($g_S$), and investment ($g_I$) are all substantially reduced. Using the maximum training duration formula from [Epoch AI](https://epoch.ai/blog/the-longest-training-run):

$$L = \frac{1}{(g_H + g_S + g_I) \cdot \ln(10)}$$

With slower growth rates summing to ~0.29 OOM/year (compared to ~1.16 OOM/year without the treaty), the maximum rational training duration extends to approximately **1.5 years** (548 days). This is the longest run an evader would undertake before newer hardware and algorithms would make a fresh start more efficient. For comparison, without the treaty, the maximum duration would be ~4.5 months.

## 3. Hardware Configuration

The CCC threshold limits **compute capacity** (FLOP/s), not memory. An evader would select hardware that maximizes the memory-to-compute ratio, enabling the largest possible model on a single unregistered node.

| Node Configuration | Peak FP16 | H100-equiv | VRAM | Max Dense Model (FP16) |
|:--|:--|:--|:--|:--|
| **48x A100 80GB** | 14.98 PFLOPS | **15.1** | **3,840 GB** | **240B** |
| 16x GH200 NVL2 | 15.84 PFLOPS | 16.0 | 2,304 GB | 144B |
| 16x H100 SXM | 15.84 PFLOPS | 16.0 | 1,280 GB | 80B |

(A100 SXM 80GB: 312 TFLOPS FP16. 48 x 312 = 14,976 TFLOPS = 15.1 H100-equivalents, under the 16-equivalent threshold.)

The **48x A100 80GB** configuration is optimal for flat DiLoCo: nearly the same compute as 16x H100 but **3x the VRAM**, enabling a 240B-parameter model to fit entirely on one node without model parallelism. However, as shown in Section 5.4, an evader could also use **PP-Group DiLoCo** to train models larger than single-node VRAM by co-locating pipeline stages on regional interconnects. This trades fewer DiLoCo groups for a closer-to-optimal model size, which becomes advantageous at large scale (500+ nodes).

## 4. Training Protocol and Compression Assumptions

### 4.1 Protocol Configuration

Each node runs as an independent DiLoCo worker:

| Parameter | Value | Rationale |
|:--|:--|:--|
| Mode | Streaming DiLoCo (flat or PP-Group) | Overlaps communication with compute |
| Inner Steps (H) | ~168-229 (optimized per N) | Minimum for compute-bound operation |
| Pseudo-gradient compression | 16x (default) or 100x (aggressive) | 4-bit quantization + sparsification of weight deltas |
| Activation compression (PP only) | 4x (4-bit quantization) | Compress hidden-state tensors between PP stages |
| Local Batch | 131,072 tokens | 32 sequences x 4,096 seq length |
| Model | 240B dense (flat) or up to ~960B (PP-Group) | Flat: max single-node; PP: larger via pipeline |
| Precision | BF16/FP16 | Standard mixed-precision training |
| MFU | 40% | Empirically supported for distributed training |

**Key insight:** The minimum inner steps $H$ to keep the system compute-bound is independent of model size. Both the sync time and the compute time per step scale linearly with $P$ (parameters), so $P$ cancels in the ratio:

$$H_{\min} = \left\lceil \frac{T_{\text{sync}}}{T_{\text{comp}}} \right\rceil = \left\lceil \frac{2 \cdot P \cdot 16 / (C_r \cdot BW) \cdot f(N)}{6 \cdot P \cdot B / (\text{PFLOPS} \cdot \text{MFU})} \right\rceil \approx 152 \cdot f(N)$$

### 4.2 Efficiency Model

The simulator computes local-equivalent FLOPs as $C_{\text{local}} = N \cdot \text{FLOPS}_{\text{eff}} \cdot T_{\text{wall}} \cdot \eta$, where $\eta$ is a combined efficiency factor. For flat DiLoCo (model fits on one node):

$$\eta = \eta_H \times \eta_{\text{pg-compression}} \times \eta_{\text{replicas}}$$

For PP-Group DiLoCo (model sharded across pipeline stages):

$$\eta = \eta_H \times \eta_{\text{pg-compression}} \times \eta_{\text{replicas}} \times \eta_{\text{act-compression}}$$

The simulator also computes **quality-adjusted compute** $C_{\text{quality}} = C_{\text{local}} \times \eta_{\text{Chinchilla}}$, which accounts for the quality cost of training a non-optimally-sized model (Section 5.4).

**Sync interval penalty ($\eta_H$):** The primary efficiency loss from using large H (many inner steps between synchronization). More inner steps cause replicas to diverge further from each other, reducing the quality of the averaged pseudo-gradient:

$$\eta_H = \max\!\left(0.4,\; 1 - \alpha \cdot \log_{10}(H)\right), \quad \alpha = \frac{0.08}{1 + \log_{10}(P/10^9) / 5}$$

The $\alpha$ coefficient decreases with model size, reflecting the empirical finding from the [DiLoCo Scaling Laws](https://arxiv.org/abs/2503.09799) paper that larger models are more robust to infrequent synchronization. This is the dominant efficiency factor, accounting for 8-16% loss depending on H and model size.

**Pseudo-gradient compression quality ($\eta_{\text{pg-compression}}$):** A multiplicative penalty from quantizing and sparsifying the pseudo-gradients (weight deltas) before transmission during DiLoCo synchronization. This occurs every H inner steps and is the only compression type in flat DiLoCo mode. Parameterized by compression ratio:

| Pseudo-gradient Compression | Optimistic | Expected | Conservative | Evidence |
|:--|:--|:--|:--|:--|
| 1x (none) | 1.00 | 1.00 | 1.00 | Baseline |
| 4x (FP4 only) | 1.00 | 1.00 | 0.99 | Lossless at 4B ([Streaming DiLoCo](https://arxiv.org/abs/2501.18512)) and 15B ([MuLoCo](https://arxiv.org/abs/2505.23725)) |
| 16x (FP4 + 4x sparse) | 1.00 | 0.98 | 0.95 | FP4 component validated; sparsification at 25% tested at 512M ([SparseLoCo](https://arxiv.org/abs/2508.15706)) |
| 100x (2-bit + TopK or FP4 + 25x) | 0.99 | 0.95 | 0.90 | Validated only at 512M-1B; significant extrapolation to 100B+ |

The "expected" scenario is used as the primary estimate throughout this analysis, with optimistic/conservative ranges noted where they materially affect conclusions. See [Compression Quality](Compression_Quality.md) for the full literature review underlying these estimates.

**Activation compression quality ($\eta_{\text{act-compression}}$, PP-Group DiLoCo only):** In PP-Group DiLoCo, hidden-state activation tensors are transferred between pipeline stages at every micro-batch. These can be compressed (e.g., 4-bit quantization), but unlike pseudo-gradient errors (which average across replicas), activation errors accumulate through the pipeline — each stage boundary introduces error in both the forward and backward passes:

$$\eta_{\text{act-compression}} = q_{\text{per-boundary}}^{2(S-1)}$$

where $S$ is the number of pipeline stages and $q_{\text{per-boundary}}$ is the per-boundary quality factor:

| Activation Compression | Optimistic | Expected | Conservative | Evidence |
|:--|:--|:--|:--|:--|
| 1x (none) | 1.00 | 1.00 | 1.00 | Baseline |
| 2x (FP8) | 1.00 | 1.00 | 0.995 | Universally near-lossless ([COAT](https://arxiv.org/abs/2410.19313), [SWARM](https://arxiv.org/abs/2301.11913)) |
| 4x (4-bit adaptive) | 1.00 | 0.995 | 0.98 | Near-lossless ([TAH-Quant](https://arxiv.org/abs/2506.06984), [GACT](https://proceedings.mlr.press/v162/liu22v.html)) |
| 10x (structural) | 0.995 | 0.98 | 0.95 | Requires subspace methods; validated at 8B ([Protocol Models](https://arxiv.org/abs/2504.01943)) |

The default activation compression ratio is **4x** (4-bit quantization), which is well-validated in the literature. At 4x with 2 pipeline stages, $\eta_{\text{act}} = 0.995^2 = 0.990$; at 4x with 4 stages, $\eta_{\text{act}} = 0.995^6 = 0.970$. See [Compression Quality §6](Compression_Quality.md#6-activation-compression-evidence-for-pp-group-diloco) for the literature review.

**Replica count penalty ($\eta_{\text{replicas}}$):** Averaging pseudo-gradients across many replicas introduces noise. Based on the [DiLoCo Scaling Laws](https://arxiv.org/abs/2503.09799) empirical data (M=8 costs ~1.2% at 2.4B parameters, with the penalty decreasing at larger model sizes), this factor is modeled as:

$$\eta_{\text{replicas}} = \max\!\left(0.85,\; 1 - 0.005 \cdot \frac{\min(2.4, P_B)}{P_B} \cdot \log_2(N)\right)$$

where $P_B$ is the model size in billions and $N$ is the number of replicas. For the primary 240B configuration, this penalty is negligible (<0.1% at 72 nodes, <0.3% at 500 nodes) because the denominator scales with model size. It becomes material only for very small models at very large replica counts.

### 4.3 Uncertainty and Validation Status

The simulator's predictions carry different levels of confidence depending on the configuration:

| Configuration | Confidence | Basis |
|:--|:--|:--|
| 16x pseudo-gradient compression, 4-72 nodes | **High** | FP4 validated lossless at 15B; DiLoCo tested at 10B; replica counts modest |
| 16x pseudo-gradient compression, 500+ nodes | **Medium** | Compression well-validated; replica count extrapolated (largest test: M=16 at 15B) |
| 100x pseudo-gradient compression, any scale | **Low-Medium** | Only validated at 512M-1B; requires error feedback (not in all implementations); significant extrapolation |
| PP-Group DiLoCo with 4x activation compression | **Medium** | FP8/4-bit activation compression well-validated; PP bubble formula standard; depth accumulation uncertain at >4 stages |
| 2000+ nodes, any compression | **Low** | No empirical data at this replica count; Epoch AI projects ~6x FLOP penalty at 10,000 nodes |

## 5. Results

### 5.1 Primary Configuration: 48x A100 80GB (240B model)

All results use the **expected** compression quality scenario (Section 4.2). Optimistic values (no compression penalty) and conservative values are shown in parentheses where they differ materially.

| Nodes | GPUs | Est. Cost | H | $\eta$ | C_local (FLOP) | x Threshold | OT Ratio | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | 48 | $0.7M | 1 | 1.000 | 2.84 x 10^23 | 0.3x | 0.0x | 1.000 | 2.84 x 10^23 |
| 4 | 192 | $2.9M | 168 | 0.862 | **9.77 x 10^23** | **1.0x** | 0.1x | 0.781 | 7.64 x 10^23 |
| 8 | 384 | $5.8M | 176 | 0.861 | 1.95 x 10^24 | 2.0x | 0.3x | 0.926 | 1.81 x 10^24 |
| 16 | 768 | $12M | 183 | 0.860 | 3.90 x 10^24 | 3.9x | 0.5x | 1.000 | 3.90 x 10^24 |
| 32 | 1,536 | $23M | 191 | 0.859 | 7.79 x 10^24 | 7.8x | 1.0x | 0.998 | 7.78 x 10^24 |
| 72 | 3,456 | $52M | 200 | 0.858 | **1.75 x 10^25** | **17.5x** | 2.3x | 0.884 | **1.55 x 10^25** |
| 144 | 6,912 | $104M | 207 | 0.857 | 3.50 x 10^25 | 35.0x | 4.6x | 0.726 | 2.54 x 10^25 |
| 500 | 24,000 | $360M | 221 | 0.855 | 1.21 x 10^26 | 121.2x | 16.0x | 0.421 | 5.11 x 10^25 |

The **overtraining ratio** (OT) is the ratio of actual training tokens to the Chinchilla-optimal token count ($D^* = 25.6 \times N$, per the [corrected Chinchilla scaling law](https://arxiv.org/abs/2404.10102)). The **Chinchilla efficiency** $\eta_{\text{chin}}$ quantifies the compute-allocation penalty from training a non-optimally-sized model relative to Chinchilla-optimal. **C_quality** $= C_{\text{local}} \times \eta_{\text{chin}}$ is the quality-adjusted compute — the amount of optimally-allocated compute that would produce the same model quality.

**Note on overtraining in practice:** The Chinchilla scaling law minimizes the loss of a single forward pass per training FLOP. In practice, developers intentionally overtrain models because each trained model is used for many inference calls. A moderately overtrained smaller model provides better cost-efficiency at inference time than a Chinchilla-optimal larger model, because inference cost scales with model size. When a model's lifetime inference compute approximately equals its training compute, moderate overtraining (2-10x) is optimal from a total-cost perspective. The $\eta_{\text{chin}}$ metric is therefore a conservative (pessimistic) measure of effective compute: it overstates the practical cost of overtraining, particularly in the 2-10x range where production models typically operate. Only extreme overtraining (>50x) represents unambiguous waste regardless of inference amortization.

At 72 nodes, C_quality = 1.55 x 10^25, reflecting the mild overtraining (2.3x) of a 240B model — well within the range that developers actively prefer. At 500 nodes, overtraining reaches 16x and $\eta_{\text{chin}}$ drops to 0.42. While 16x overtraining is beyond what developers typically choose, the real quality penalty is smaller than $\eta_{\text{chin}}$ suggests due to inference amortization. Nonetheless, at very large node counts the overtraining ratio becomes extreme (128x at 4,000 nodes), motivating the PP-Group DiLoCo analysis in Section 5.4, where larger models reduce overtraining.

The 72-node reference point shows $\eta = 0.858$ under the expected scenario (optimistic: 0.875; conservative: 0.831). The corresponding C_local range is **1.70-1.79 x 10^25 FLOP** — the conclusion that 72 nodes exceeds the Strict Threshold by ~17x is robust across all compression quality assumptions.

The **algorithmic efficiency** ($\eta$) is stable at 85-86% across all node counts. Under the optimistic scenario (no compression quality penalty), efficiency would be 87-88%. The 2% difference reflects the expected cost of 16x pseudo-gradient compression extrapolated to 240B scale.

### 5.2 Model Quality Analysis

The evader trains a **240B-parameter dense model** in flat DiLoCo mode. Using the corrected Chinchilla-optimal token count $D^* = 25.6 \times N$ ([Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102)):

| Nodes | Total Tokens | Chinchilla Tokens (240B) | Overtraining Ratio | Assessment |
|:--|:--|:--|:--|:--|
| 4 | 0.8T | 6.1T | 0.1x | **Under-trained** (would use a smaller model) |
| 16 | 3.2T | 6.1T | 0.5x | Below optimal |
| 32 | 6.3T | 6.1T | 1.0x | Near Chinchilla-optimal |
| 72 | 14.2T | 6.1T | 2.3x | **Moderately overtrained** (industry-standard; preferred for production) |
| 144 | 28.4T | 6.1T | 4.6x | Overtrained (still practical; comparable to LLaMA-3 at ~10x) |
| 500 | 98.5T | 6.1T | 16.0x | Heavily overtrained (beyond typical practice; $\eta_{\text{chin}} = 0.42$) |

At the **72-node** reference point: 240B params trained on 14.2T tokens with 2.3x overtraining is not a penalty — it is in fact the preferred regime for production deployments, where inference amortization makes moderate overtraining cost-optimal. This is comparable to how production models like LLaMA-3 are trained (which used ~10x overtraining). The $\eta_{\text{chin}} = 0.88$ represents a conservative lower bound; accounting for inference amortization, the effective penalty of 2.3x overtraining is negligible.

At **500+ nodes**, overtraining reaches 16x, which is beyond what developers typically choose but not catastrophic — the real quality penalty is smaller than $\eta_{\text{chin}} = 0.42$ suggests. However, at 4,000 nodes the ratio reaches 128x, which represents genuine waste even accounting for inference amortization. The Chinchilla-optimal model for 500 nodes' compute budget is ~960B — far larger than the 240B that fits on a single node. This motivates PP-Group DiLoCo (Section 5.4), where pipeline stages allow training larger models.

For small node counts (N < 16), the evader would train a proportionally smaller model to stay near Chinchilla-optimal. For example, at N=4 with C_local = 10^24 FLOP, the Chinchilla-optimal model is ~29B params on ~0.7T tokens, which fits easily within the 240B memory budget.

### 5.3 Comparison to Existing 10^25 Baseline

The [existing baseline scenario](Past%20MIRI%20work/10%5E25%20FLOP%20decentralized%20training%20configuration.md) used 72 nodes of 16x GH200 (~15.8 PFLOPS/node FP16, 2,304 GB) to train a 144B model on 12T tokens (1.02 x 10^25 FLOP) in 331 days. It was communication-bound.

This analysis differs in three ways:
1. **Similar compute per node** (~15 vs ~15.8 PFLOPS) — both near the CCC threshold boundary
2. **Higher memory per node** (3,840 vs 2,304 GB) — enables a larger 240B model (vs 144B)
3. **Longer time** (548 vs 331 days) — the treaty scenario allows 1.5 years

Result: with similar per-node compute but more VRAM and a longer training window, DiLoCo's compute-bound operation yields **1.75 x 10^25 local-equivalent FLOP** (expected) — a 1.7x improvement over the baseline's 1.02 x 10^25 raw compute. The improvement comes from DiLoCo's compute-bound operation (vs the baseline's communication-bound regime) and the longer training window.

### 5.4 PP-Group DiLoCo and Chinchilla Optimality

The flat DiLoCo results in Section 5.1 all train the largest model that fits on a single node (240B for 48x A100). At large node counts, this leads to increasingly extreme overtraining: 16x at 500 nodes, and 128x at 4,000 nodes. While moderate overtraining (2-10x) is preferred in practice due to inference amortization, overtraining beyond ~50x represents genuine waste. The Chinchilla-optimal model for 500 nodes' compute budget is ~960B — far larger than what fits on one node.

**PP-Group DiLoCo** addresses this by grouping $S$ co-located nodes into pipeline stages that collectively hold a larger model, then running DiLoCo across $G = N/S$ groups over WAN. The co-located pipeline groups use regional interconnect (1 Gbps, 20 ms latency), while DiLoCo synchronization uses WAN as before. Activation tensors passed between pipeline stages are compressed using 4-bit quantization (4x compression, $\eta_{\text{act}} = 0.995^{2(S-1)}$).

**The fundamental tradeoff:** PP-Group DiLoCo trains a larger (closer-to-optimal) model but with fewer DiLoCo groups, reducing throughput. The GPipe pipeline bubble also adds overhead: $(S-1)/(M+S-1)$ fraction of compute is wasted in the bubble, where $M=8$ micro-batches.

**Model size sweep results (expected scenario):**

| Nodes | Best Mode | Model | PP Stages | Groups | OT | $\eta$ | $\eta_{\text{chin}}$ | C_quality | vs Flat 240B |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 72 | DiLoCo | 240B | 1 | 72 | 2.3x | 0.858 | 0.884 | 1.55 x 10^25 | 1.0x |
| 500 | PP-DiLoCo | 480B | 2 | 250 | 2.1x | 0.852 | 0.902 | 5.59 x 10^25 | 1.09x |
| 4,000 | PP-DiLoCo | 960B | 4 | 1,000 | 2.8x | 0.832 | 0.833 | **2.75 x 10^26** | **2.59x** |
| 2,000 (H100 FP8) | PP-DiLoCo | 549B | 6 | 333 | 4.8x | 0.816 | 0.710 | **1.28 x 10^26** | **5.58x** |

**Key findings:**

1. **At 72 nodes, flat DiLoCo is still optimal.** The 2.3x overtraining is mild, and splitting 72 nodes into 36 pipeline groups halves throughput, which is not compensated by better Chinchilla efficiency.

2. **At 500 nodes, PP-DiLoCo gives a modest 9% improvement** by training a 480B model (2 stages, 250 groups) instead of 240B (500 groups). The overtraining drops from 16x to 2.1x.

3. **At 4,000 nodes, PP-DiLoCo gives 2.6x improvement.** Training a 960B model (4 stages, 1,000 groups) produces 2.75 x 10^26 quality-adjusted FLOP versus 1.06 x 10^26 for flat DiLoCo at 240B. The flat approach overtrains at 128x Chinchilla tokens — well beyond the ~10x range where inference amortization provides a practical benefit, and into genuinely wasteful territory.

4. **For H100 FP8 nodes, PP-DiLoCo is essential.** The max single-node model at FP8 is only 91B, leading to extreme overtraining (934x) at 2,000 nodes. PP-DiLoCo with 6 stages trains a 549B model, improving C_quality by **5.6x**.

5. **The optimal model size is NOT the Chinchilla-optimal model.** The Chinchilla-optimal model for 4,000 A100 nodes is ~2,700B, which would require 12 pipeline stages and suffer 58% bubble overhead. The practical optimum (960B, 4 stages) balances model size against pipeline efficiency.

**Implication for governance:** PP-Group DiLoCo substantially increases the quality of models an evader could train at large scale. An evader with 4,000 nodes could train a 960B model that achieves 2.75 x 10^26 quality-adjusted FLOP — compared to 1.06 x 10^26 for a naive flat approach. However, PP-Group DiLoCo requires co-locating pipeline stages (S nodes within one facility or metro area), which partially re-creates the physical clustering that the treaty's monitoring mechanisms are designed to detect.

## 6. Governance Implications

### 6.1 The Threshold is Easily Exceeded

**As few as 4 unregistered nodes** (~200 A100 GPUs, ~$3M) are sufficient to match the 10^24 FLOP Strict Threshold in 1.5 years. This conclusion is robust: under the conservative compression quality scenario, 4 nodes still produce 9.52 x 10^23 FLOP (~1.0x threshold). This is a modest investment — comparable to a single research lab's annual GPU budget. At 72 nodes ($52M), the evader achieves **17.5x** the threshold (range: 17.0-17.9x), training a 240B-parameter model that would be competitive with frontier models.

### 6.2 The CCC Interconnect Criterion Is Irrelevant for DiLoCo

The treaty's CCC definition includes clusters with "inter-node aggregate bandwidth greater than 25 Gbit/s." DiLoCo training requires only **100 Mbps** between nodes — 250x below this threshold. The synchronization traffic (every ~50-90 minutes, compressed pseudo-gradients) is indistinguishable from routine internet traffic in bandwidth terms. The definition does also capture clusters "networked to perform workloads together," but enforcing this requires detecting the workload itself, not just the network topology.

### 6.3 Detection Challenges

DiLoCo training creates a distinctive network "heartbeat": synchronization traffic at regular intervals. However, with Streaming DiLoCo, this traffic is spread continuously rather than bursty:

- **Average bandwidth per node:** Each sync exchanges 240 Gbit (30 GB) compressed pseudo-gradients in each direction. With streaming enabled and H optimized for compute-bound operation, the sync data is spread over the full ~100-minute outer step, resulting in ~38 Mbps average upload + ~38 Mbps average download. This is comparable to a media server or cloud backup service.
- **Traffic is encrypted:** Standard VPN tunnels obscure content. The sustained ~76 Mbps throughput is not unusual for internet-connected servers.
- **Nodes are geographically distributed:** No physical clustering, power draw anomaly, or thermal signature to detect via satellite.
- **Each node is individually legitimate:** A 48-GPU cluster is common in research labs, gaming companies, and cloud providers.
- **Potential fingerprint:** The sync traffic has a highly regular cadence (every ~100 minutes, identical payload sizes). Traffic analysis at the ISP level could detect this pattern, but it requires monitoring encrypted flow metadata across many nodes simultaneously.

### 6.4 The Treaty Paradoxically Extends the Evasion Window

Without the treaty, the maximum rational training duration is ~4.5 months (rapid hardware/software/investment growth makes longer runs obsolete). The treaty's restrictions on research and compute spending slow these growth rates, extending the maximum duration to ~1.5 years. This gives the evader **4x more training time**, partially offsetting the need to use smaller, unregistered nodes.

### 6.5 Hardware Optimization and PP-Group DiLoCo

A naive analysis might assume the evader is limited to 16 H100s (80B model). But the CCC threshold is on compute, not memory. By choosing high-memory, lower-compute GPUs (48x A100 80GB), the evader fits a **240B model** on each node. Pipeline parallelism over WAN (sending activations between globally distributed nodes) remains impractical (~49 seconds per micro-batch at 100 Mbps), but **PP-Group DiLoCo** — co-locating pipeline stages on regional interconnect — enables training models up to ~960B by grouping 2-4 nodes into pipeline clusters. As shown in Section 5.4, this produces up to **2.6x more quality-adjusted compute** at 4,000 nodes by training a closer-to-Chinchilla-optimal model, though it requires physical co-location of pipeline stages.

### 6.6 Scaling Is Nearly Linear

The local-equivalent compute scales almost perfectly linearly with the number of nodes. The DiLoCo algorithmic efficiency ($\eta \approx 86\%$ expected) and the straggler overhead ($f(N) \approx 1.1-1.5$) are both modest and slow-varying. A well-funded actor with access to 500+ nodes could achieve over **100x** the Strict Threshold.

### 6.7 Enforcement Time Pressure

The 1.5-year maximum training duration (Section 2) is a theoretical optimum based on technology growth rates. In practice, the treaty's enforcement mechanisms create pressure to train *faster* than the theoretical maximum:

- **Whistleblower programs:** The treaty establishes financial incentives for reporting violations. A 1.5-year operation requires maintaining operational security across dozens or hundreds of sites for an extended period. Every employee, contractor, facility operator, and GPU supplier who becomes aware of the operation is a potential whistleblower. The cumulative probability of a leak increases with time.
- **International inspections:** Treaty signatories conduct both scheduled and triggered inspections of compute facilities. Over 1.5 years, each node site faces multiple inspection cycles. Even if individual inspections have a low detection probability, the compound probability over 18 months across many sites becomes significant.
- **Operational security failures:** Hardware failures requiring replacement, network outages, power interruptions, and personnel turnover all create detection opportunities. A 48-GPU cluster operating at near-full utilization for 18 months will experience multiple hardware failures requiring maintenance or procurement.

The practical effect: an evader would likely target **6-12 months** rather than the full 1.5 years, accepting lower total compute in exchange for reduced detection risk. Since $C_{\text{local}}$ scales linearly with time, this reduces the achievable compute proportionally:

| Duration | C_local (72 nodes, expected) | x Strict Threshold |
|:--|:--|:--|
| 6 months | 5.84 x 10^24 | 5.8x |
| 1 year | 1.17 x 10^25 | 11.7x |
| 1.5 years | 1.75 x 10^25 | 17.5x |

Even at 6 months, the 72-node configuration still exceeds the Strict Threshold by **5.8x** — sufficient to train a meaningful model (240B dense, ~4.7T tokens).

### 6.8 Financial Detection

The treaty's chip tracking provisions require all AI chips to be registered with serial numbers. Acquiring large quantities of unregistered GPUs is a supply-chain challenge that scales with the investment:

| Scale | Hardware Cost | GPUs | Financial Detectability |
|:--|:--|:--|:--|
| 4 nodes | ~$3M | 192 A100s | **Low** — within reach of individuals, startups, or research labs. Comparable to routine equipment purchases. |
| 72 nodes | ~$52M | 3,456 A100s | **Moderate** — significant capital expenditure, but not unusual for mid-size tech companies. Could be structured across multiple procurement channels. |
| 500 nodes | ~$360M | 24,000 A100s | **High** — detectable via financial intelligence: large wire transfers, unusual GPU procurement patterns, supply chain monitoring of chip manufacturers. |
| 4,000 nodes | ~$2.9B | 192,000 A100s | **Very high** — comparable to a national defense procurement. Would create visible distortions in the GPU market. |

**Forensic accounting methods** include monitoring GPU manufacturer shipment records, tracking large purchases from cloud providers or distributors, and flagging unusual patterns in semiconductor trade data. The treaty's chip registration requirement makes this particularly effective: every unaccounted chip represents a potential violation.

**The state-actor exception:** The most significant gap in financial detection is a **state actor with classified budgets**. A nation-state could acquire GPUs through military or intelligence procurement channels that are exempt from civilian financial transparency requirements. If the state has domestic chip manufacturing capability (e.g., China's SMIC or a future domestic fab), it could produce unregistered chips entirely outside the treaty's tracking system. Pre-existing military GPU stockpiles accumulated before treaty ratification would also evade tracking.

For non-state actors, the financial barrier is the primary practical constraint: it is straightforward to hide a $3M GPU purchase but extremely difficult to hide $3B without state-level resources.

## 7. Summary of Sub-CCC Evasion

| Metric | Value |
|:--|:--|
| **Minimum nodes to reach 10^24 FLOP** | **4 nodes** (~200 A100s, ~$3M) |
| **Reference scenario (72 nodes, flat DiLoCo)** | **1.75 x 10^25 C_local** (17.5x threshold), **1.55 x 10^25 C_quality** |
| **Optimistic/conservative range (72 nodes)** | 1.70-1.79 x 10^25 C_local (17.0-17.9x) |
| **Model trained (72 nodes, flat)** | 240B dense, 14.2T tokens, 2.3x overtraining |
| **Best quality at 4,000 nodes (PP-DiLoCo)** | **2.75 x 10^26 C_quality** (960B, 4 PP stages, 2.6x better than flat) |
| **Algorithmic efficiency** | ~86% expected (range: 83-88%) |
| **Confidence in 16x pseudo-gradient compression** | High (FP4 validated to 15B; penalty ~2% expected) |
| **Bottleneck** | Network bandwidth at low H; compute-bound when H is optimized |
| **Detection** | Challenging: low-bandwidth encrypted traffic, no physical signature |
| **Enforcement time pressure** | 6 months still yields 5.8x threshold at 72 nodes |
| **Financial barrier** | $3M (easy to hide) to $52M (moderate) to $3B (state-actor) |

The simulator predicts that the treaty's FLOP threshold can be significantly exceeded using distributed training with unregistered sub-CCC nodes. The primary defense against this evasion is not the compute threshold itself, but rather the treaty's chip tracking and consolidation provisions — the ability to prevent actors from accumulating sufficient GPU hardware in the first place.

## 8. Scaling to 10^27 FLOP and Beyond

The analysis in Sections 5-7 focuses on modest-scale evasion (4-500 nodes). This section explores whether much larger training runs — on the order of **10^27 FLOP** — are achievable using all techniques supported by the simulator: hierarchical DiLoCo, FP8 precision, aggressive pseudo-gradient compression, PP-Group DiLoCo, and Mixture-of-Experts with Expert Parallelism.

### 8.1 Configurations

Nine configurations are compared, all using sub-CCC nodes over 100 Mbps WAN. All values use the **expected** compression quality scenario. The table now includes **C_quality** (quality-adjusted compute accounting for Chinchilla-optimality deviation):

| Config | Hardware | Nodes | Model | $\eta$ | C_local | $\eta_{\text{chin}}$ | C_quality |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **A** | 48x A100 FP16, flat DiLoCo, 16x | 4,000 | 240B dense | 0.853 | 9.67 x 10^26 | 0.110 | 1.06 x 10^26 |
| **B** | 48x A100 FP16, hierarchical, 16x | 4,000 | 240B dense | 0.883 | 1.00 x 10^27 | 0.110 | 1.10 x 10^26 |
| **C** | 16x H100 FP8, flat DiLoCo, 16x | 2,000 | 91B dense | 0.844 | 1.01 x 10^27 | 0.023 | 2.29 x 10^25 |
| **D** | 16x H100 FP8, hierarchical, 16x | 2,000 | 91B dense | 0.876 | 1.05 x 10^27 | 0.023 | 2.37 x 10^25 |
| **E** | 48x A100 FP16, flat DiLoCo, 100x | 4,000 | 240B dense | 0.868 | 9.84 x 10^26 | 0.110 | 1.08 x 10^26 |
| **F** | 16x H100 FP8, hier + 100x comp | 2,000 | 91B dense | 0.892 | 1.07 x 10^27 | 0.023 | 2.42 x 10^25 |
| **G** | 48x A100 FP16, MoE+EP 600B/100B | 4,000 | 600B MoE | 0.829 | 6.32 x 10^26 | 0.039 | 2.44 x 10^25 |
| **H** | 48x A100 FP16, PP-DiLoCo 16x | 4,000 | **960B PP-4x1000** | 0.832 | 3.30 x 10^26 | 0.833 | **2.75 x 10^26** |
| **I** | 16x H100 FP8, PP-DiLoCo 16x | 2,000 | **480B PP-6x333** | 0.816 | 1.71 x 10^26 | 0.656 | **1.12 x 10^26** |

Hierarchical configurations use groups of 8 nodes with 1 Gbps regional interconnect and 20 ms latency (co-located nodes within the same facility or metro area). PP-Group DiLoCo configurations use co-located pipeline groups on regional interconnect, with 4x activation compression. The 100x pseudo-gradient compression ratio corresponds to FP4 quantization combined with aggressive sparsification, as demonstrated in [Streaming DiLoCo](https://arxiv.org/abs/2501.18512).

**Compression quality uncertainty for 100x configurations (E, F):** Under the optimistic scenario, Config F achieves 1.12 x 10^27; under the conservative scenario, it drops to 1.01 x 10^27. The 100x compression assumptions carry significant uncertainty (see [Compression Quality](Compression_Quality.md)), so these estimates should be treated as bounds rather than point predictions.

### 8.2 Key Findings

**10^27 FLOP requires 2,000-4,000 sub-CCC nodes ($1-3B in hardware).** This is unambiguously a state-actor-level investment. No non-state actor could plausibly acquire 32,000-192,000 GPUs without detection through the treaty's financial monitoring and chip tracking provisions (Section 6.8).

**C_local vs C_quality: a critical distinction.** The table reveals a dramatic divergence between raw compute (C_local) and quality-adjusted compute (C_quality). Configs A-F achieve 10^27 C_local, but their C_quality ranges from only 2.3 x 10^25 (Config C) to 1.1 x 10^26 (Config B). The reason: all flat DiLoCo configurations at 2,000-4,000 nodes massively overtrain their fixed-size models — Config A trains 128x more tokens than Chinchilla-optimal for 240B. Note that $\eta_{\text{chin}}$ is a conservative metric that overstates the practical cost of overtraining (see Section 5.1 for discussion of inference amortization); however, at the extreme levels seen here (128x-934x), the overtraining is genuinely excessive even accounting for inference. **PP-Group DiLoCo (Config H) achieves the highest C_quality** (2.75 x 10^26) by training a 960B model closer to Chinchilla-optimal, despite lower raw throughput.

**PP-Group DiLoCo is the most effective strategy at scale.** Config H (PP-DiLoCo, 4 stages, 960B model, 1,000 groups) achieves 2.75 x 10^26 C_quality — **2.5x more than the best flat DiLoCo** (Config B, 1.10 x 10^26) and **11x more than the best H100 FP8 flat configuration** (Config D, 2.37 x 10^25). The cost is lower raw throughput (C_local = 3.30 x 10^26 vs 1.00 x 10^27) due to fewer groups and PP bubble overhead, but the larger model makes far better use of each FLOP.

**FP8 precision is cost-effective but benefits greatly from PP.** The CCC threshold is defined by FP16 capacity, but FP8 yields **2x throughput** and FP8 pseudo-gradients are half the size. Config C achieves 10^27 C_local with only 2,000 nodes ($960M) — but its C_quality is only 2.3 x 10^25 due to extreme overtraining of the small 91B model (934x Chinchilla tokens). Config I (PP-DiLoCo, 480B) improves C_quality by **4.9x** to 1.12 x 10^26, making H100 FP8 with PP competitive with A100 flat DiLoCo.

**Hierarchical DiLoCo improves efficiency by 3 percentage points.** At N=4,000, flat DiLoCo achieves $\eta = 0.853$ with $H_{\min} = 244$, while hierarchical DiLoCo achieves $\eta = 0.883$ with $H_{\text{eff}} = 65$. The improvement comes from regional syncs over 1 Gbps LAN, which keep $H_{\text{inner}}$ low (18 steps) while the global sync interval is hidden behind regional cycles.

**100x pseudo-gradient compression reduces $H_{\min}$ dramatically but carries more uncertainty.** With 100x compression (vs 16x baseline), sync volume drops from 240 Gbit to 38 Gbit per direction, reducing $H_{\min}$ from 244 to 39 at N=4,000. This improves $\eta_H$ from 0.871 to 0.914. However, the pseudo-gradient compression quality factor ($\eta_{\text{pg-compression}} = 0.95$ expected, vs 0.98 for 16x) partially offsets this gain. The net expected efficiency for Config E is 0.868 — an improvement over Config A (0.853) but not as dramatic as the $\eta_H$ improvement alone would suggest. The 100x compression assumption is based on limited empirical evidence (see [Compression Quality](Compression_Quality.md)) and represents the least certain component of the analysis.

**The combined best case for raw throughput (Config F)** uses hierarchical DiLoCo, 100x pseudo-gradient compression, and FP8 H100 nodes to achieve **1.07 x 10^27 C_local** (expected). However, its C_quality is only 2.42 x 10^25 due to extreme overtraining of the 91B model. **For quality-adjusted compute, Config H (PP-DiLoCo 960B)** is optimal at 2.75 x 10^26 C_quality.

### 8.3 Mixture of Experts + Expert Parallelism

MoE with Expert Parallelism (EP) does not increase $C_{\text{local}}$ — total compute is determined by active parameters, not total parameters. However, it enables training a **much larger model** within the same compute budget:

| Config | Total Params | Active Params | Per-Node Memory | Fits (48x A100) | EP Overhead |
|:--|:--|:--|:--|:--|:--|
| Dense 240B | 240B | 240B | 3,840 GB | Yes | — |
| MoE 600B | 600B | 100B | 1,711 GB (at N=72) | Yes | 32.8% |
| MoE 1T | 1,000B | 150B | 2,589 GB (at N=72) | Yes | 24.5% |

EP distributes expert parameters across all nodes, reducing per-node memory from $P_{\text{total}} \cdot \beta$ to $(P_{\text{shared}} + P_{\text{experts}}/N) \cdot \beta$. A 600B MoE model (100B shared, 500B expert) at N=72 requires only 1,711 GB per node — well within the 3,840 GB budget.

The trade-off is a **32.8% compute overhead** from EP All-to-All communication ($T_{\text{EP}} = 2 \times 0.1\text{s} \times 32\text{ layers} = 6.4\text{s}$ per step, added to the 13.1s compute time for 100B active params). This reduces $C_{\text{local}}$ to **6.32 x 10^26** at 4,000 nodes — about 35% less than the dense 240B configuration. However, the resulting 600B MoE model may be more capable per FLOP than a 240B dense model, following the trend established by models like Mixtral and DeepSeek-V2.

### 8.4 Reaching 10^28 FLOP

For completeness, 10^28 FLOP would require approximately:
- **~20,000 nodes** of 16x H100 FP8 (~320,000 H100s, ~$9.5B), or
- **~40,000 nodes** of 48x A100 FP16 (~1,900,000 A100s, ~$29B)

These are resources comparable to a major nation's annual military procurement budget. The straggler factor at N=20,000 is $f(20000) = 1.71$, and $\eta$ remains above 0.84 — the physics of DiLoCo does not prevent scaling to this level, but the economic and logistical requirements make this exclusively a state-actor scenario. Note that at 20,000 replicas, the replica count penalty becomes non-trivial (~1-2%), though this is dwarfed by other uncertainties at this scale.

### 8.5 Summary: Scale and Actor Type

| Compute Scale (C_quality) | Nodes | Cost | Best Mode | Model | Actor Type |
|:--|:--|:--|:--|:--|:--|
| **10^24** (Strict Threshold) | 4 A100 | $3M | Flat DiLoCo | 240B | Research lab, wealthy individual |
| **10^25** (frontier-competitive) | 72 A100 | $52M | Flat DiLoCo | 240B | Well-funded startup, small state |
| **10^26** (C_quality) | 500 A100 | $360M | PP-DiLoCo | 480B | Large corporation, medium state |
| **~3 x 10^26** (C_quality) | 4,000 A100 | $2.9B | PP-DiLoCo | 960B | Major state actor only |
| **10^27** (C_local) | 2,000-4,000 | $1-3B | Flat/Hier DiLoCo | 91-240B | Major state actor only |

Note the distinction between C_local (raw compute) and C_quality (quality-adjusted). While several configurations achieve 10^27 C_local, their C_quality is only ~10^26 due to severe overtraining. An evader optimizing for model quality rather than raw FLOP would prefer PP-DiLoCo (Config H: 960B, C_quality = 2.75 x 10^26) over flat DiLoCo (Config A: 240B, C_quality = 1.06 x 10^26).

The compute threshold in the treaty (10^24 FLOP) is designed to catch training runs at the **research lab** level and above. Distributed training with DiLoCo makes this threshold porous at the 4-72 node scale ($3-52M), where financial detection is challenging and the pseudo-gradient compression assumptions are well-validated. At the 10^27+ scale, both the treaty's financial monitoring mechanisms and the compression quality uncertainty become relevant — the hardware procurement itself is the hardest step to conceal, and the achievable compute carries wider error bars.

## 9. Network Sensitivity Analysis

The analysis in Sections 5-8 assumes a baseline network of 100 Mbps symmetric WAN with 100 ms RTT. This section examines how sensitive the results are to degraded network conditions — lower bandwidth, higher latency, and realistic deployment profiles across different geographical regions. All results include compression quality penalties (expected scenario).

**Central question:** Does DiLoCo's performance degrade significantly under realistic or adversarial network conditions?

### 9.1 Real-World Network Parameters

All latency values are from measured RTT data (Azure Network Latency Statistics June 2025, Verizon IP Latency, Epsilon Telecom, AWS inter-region via CloudPing.co):

| Scenario | Measured RTT | Source |
|:--|:--|:--|
| Same cloud region (cross-AZ) | 1-2 ms | AWS, Azure intra-region P50 |
| Same continent (Europe) | ~20 ms | London-Frankfurt (Epsilon 20ms, Azure 17ms) |
| Continental US | ~65 ms | NYC-LA (Epsilon 64ms, Azure East-West 71ms) |
| Transatlantic | ~75 ms | NYC-London (Verizon 70ms, Epsilon 75ms, Azure 79ms) |
| Transpacific | ~105 ms | LA-Tokyo (AWS 105ms, Epsilon 100ms) |
| US East Coast-Asia | ~230 ms | Virginia-Singapore (AWS 224ms) |
| Global worst-case | ~340 ms | Brazil-SE Asia (Azure 332-343ms) |

### 9.2 Bandwidth Sensitivity

Bandwidth was swept from 10 Mbps to 1 Gbps while holding latency at 100 ms. Results include compression quality penalty (expected scenario).

**72 nodes, 48x A100 FP16, 16x compression (targeting 10^25):**

| BW (Mbps) | H_min | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 1,994 | 0.804 | 1.64 x 10^25 | 88% |
| 25 | 798 | 0.826 | 1.69 x 10^25 | 91% |
| 50 | 399 | 0.842 | 1.72 x 10^25 | 92% |
| **100** | **200** | **0.858** | **1.75 x 10^25** | **94%** |
| 250 | 80 | 0.879 | 1.79 x 10^25 | 96% |
| 1,000 | 20 | 0.911 | 1.86 x 10^25 | 100% |

**2,000 nodes, 16x H100 FP8, hierarchical + 100x compression (Config F, expected):**

| BW (Mbps) | H_eff | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 33 | 0.866 | 1.04 x 10^27 | 95% |
| 25 | 21 | 0.877 | 1.05 x 10^27 | 96% |
| 50 | 15 | 0.885 | 1.06 x 10^27 | 97% |
| **100** | **11** | **0.892** | **1.07 x 10^27** | **98%** |
| 500 | 5 | 0.910 | 1.09 x 10^27 | 99% |
| 1,000 | 4 | 0.914 | 1.10 x 10^27 | 100% |

**Key finding:** Reducing bandwidth from 1 Gbps to 10 Mbps — a 100x degradation — costs **12% of C_local** for the 72-node flat configuration and **5% for Config F** (hierarchical+100x). These percentages include both the $\eta_H$ penalty from larger H and the compression quality penalty. The bandwidth sensitivity itself (the difference between 10 Mbps and 1 Gbps at fixed compression quality) accounts for about 6-8 percentage points; the compression quality penalty adds another 2-5 percentage points compared to the lossless assumption.

**Comparison to prior estimates:** Earlier versions of this analysis reported "3-6% reduction" from bandwidth degradation. This was based on a model that assumed lossless compression. With compression quality accounted for, the combined total reduction from optimal (1 Gbps, lossless) to degraded (10 Mbps, expected quality) is **10-14%** for 16x compression and **14-18%** for 100x compression. The qualitative conclusion — that DiLoCo is bandwidth-insensitive — remains valid, but the magnitude is larger than initially reported.

### 9.3 Latency Sensitivity

Latency was swept across all real-world scenarios from 2 ms (same cloud region) to 340 ms (Brazil-SE Asia) while holding bandwidth at 100 Mbps.

**Result: Latency has virtually no effect on any configuration.**

For all configurations tested — 72 nodes flat, 500 nodes flat, 2,000 nodes flat, and 2,000 nodes hierarchical+100x — the C_local values are **identical** across all latency scenarios (2 ms to 340 ms). The H values do not change.

**Why latency is irrelevant:** The sync time formula is $T_{\text{sync}} = 2 \cdot V_{\text{bits}} / BW + \text{latency}$. For a 240B model with 16x compression at 100 Mbps:

$$V_{\text{bits}} = 240 \times 10^9 \times 16 / 16 = 240 \text{ Gbit}$$
$$T_{\text{sync,bandwidth}} = 2 \times 240 \times 10^9 / (100 \times 10^6) = 4{,}800 \text{ seconds}$$

The bandwidth term is **4,800 seconds**. Even the global worst-case latency of 340 ms is only **0.007%** of the sync time. Latency is negligible compared to the time required to transfer billions of pseudo-gradient values, even with 16x compression.

### 9.4 Combined Deployment Profiles

Realistic network conditions combine both bandwidth and latency. The table below shows results for deployment profiles that an evader might realistically use. All values use the expected compression quality scenario.

**72 nodes, 48x A100 FP16, 16x compression (10^25 target, ~$52M):**

| Deployment | BW | RTT | H | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 20 | 0.911 | 1.86 x 10^25 | 100% |
| Same country (US) | 500 Mbps | 35 ms | 40 | 0.895 | 1.83 x 10^25 | 98% |
| Continental US | 100 Mbps | 65 ms | 200 | 0.858 | 1.75 x 10^25 | 94% |
| Transatlantic | 100 Mbps | 75 ms | 200 | 0.858 | 1.75 x 10^25 | 94% |
| Transpacific | 50 Mbps | 105 ms | 399 | 0.842 | 1.72 x 10^25 | 92% |
| Global adversarial | 25 Mbps | 230 ms | 798 | 0.826 | 1.69 x 10^25 | 91% |
| Global worst-case | 10 Mbps | 340 ms | 1,994 | 0.804 | 1.64 x 10^25 | 88% |

**2,000 nodes, 16x H100 FP8, hier+100x (Config F, 10^27 target, ~$960M):**

| Deployment | BW | RTT | H_eff | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 4 | 0.914 | 1.10 x 10^27 | 100% |
| Same country (US) | 500 Mbps | 35 ms | 5 | 0.910 | 1.09 x 10^27 | 99% |
| Continental US | 100 Mbps | 65 ms | 11 | 0.892 | 1.07 x 10^27 | 97% |
| Transatlantic | 100 Mbps | 75 ms | 11 | 0.892 | 1.07 x 10^27 | 97% |
| Transpacific | 50 Mbps | 105 ms | 15 | 0.885 | 1.06 x 10^27 | 97% |
| Global adversarial | 25 Mbps | 230 ms | 21 | 0.877 | 1.05 x 10^27 | 96% |
| Global worst-case | 10 Mbps | 340 ms | 33 | 0.866 | 1.04 x 10^27 | 95% |

### 9.5 Achievability of 10^26 and 10^27 Under Degraded Networks

**10^26 FLOP under degraded networks (500 nodes, 48x A100 FP16, 16x comp, 1.5 years, expected):**

| Network Conditions | C_local | Achievable? |
|:--|:--|:--|
| 100 Mbps, 100 ms (baseline) | 1.21 x 10^26 | Yes |
| 50 Mbps, 105 ms (transpacific) | 1.18 x 10^26 | Yes |
| 25 Mbps, 230 ms (global) | 1.16 x 10^26 | Yes |
| 10 Mbps, 340 ms (worst-case) | 1.13 x 10^26 | **Yes** |

10^26 FLOP is achievable under **all tested network conditions**, including the global worst-case with expected compression quality. Even under the conservative compression scenario, 500 nodes at 10 Mbps still produce ~1.10 x 10^26 FLOP.

**10^27 FLOP under degraded networks (2,000 nodes, 16x H100 FP8, expected):**

| Network Conditions | C_local (flat, 16x) | C_local (hier+100x) | Achievable? |
|:--|:--|:--|:--|
| 100 Mbps, 100 ms | 1.01 x 10^27 | 1.07 x 10^27 | Yes |
| 50 Mbps, 105 ms | 9.91 x 10^26 | 1.06 x 10^27 | Marginal (flat) / **Yes** (hier) |
| 25 Mbps, 230 ms | 9.72 x 10^26 | 1.05 x 10^27 | No (flat) / **Yes** (hier) |
| 10 Mbps, 340 ms | 9.44 x 10^26 | 1.04 x 10^27 | No (flat) / **Yes** (hier) |

With the expected compression quality scenario, flat 16x DiLoCo at 2,000 nodes achieves 10^27 only at 100 Mbps or above. Hierarchical+100x (Config F) achieves 10^27 under all conditions. Under the conservative scenario, Config F at 10 Mbps drops to ~9.8 x 10^26 — just below 10^27, making this the most uncertain result in the analysis.

### 9.6 Implications for Treaty Enforcement

The sensitivity analysis reveals three governance-critical findings:

**1. DiLoCo is bandwidth-insensitive by design.** The protocol absorbs low bandwidth by increasing the synchronization interval $H$. Because efficiency depends on $\log_{10}(H)$, a 10x increase in $H$ (needed to compensate for 10x lower bandwidth) costs only a constant reduction in $\eta$. Across the full range from 10 Mbps to 1 Gbps, C_local varies by only 12% (flat, 16x) to 5% (hierarchical+100x). Including compression quality, the total gap between optimal and worst-case is **12-14%** for 16x and **14-18%** for 100x.

**2. Latency is irrelevant to distributed training.** The sync volumes in frontier-scale training (Gbits to Tbits) make round-trip latency negligible. Even the Brazil-to-Southeast-Asia worst case (340 ms) has zero measurable impact. Geographical distribution is free — an evader needs no special network infrastructure.

**3. Network-level enforcement is ineffective.** Even if bandwidth restrictions were perfectly enforced, the 12-14% reduction in achievable compute is small relative to the order-of-magnitude exceedance of the Strict Threshold. An evader using consumer broadband loses at most 12-14% of compute versus an optimized local deployment — a minor penalty that does not change whether the threshold is exceeded or by how much. The effective enforcement mechanisms remain chip tracking, financial monitoring, and physical detection of GPU concentrations.

## 10. Treaty Modifications to Close the Distributed Training Loophole

The preceding sections establish three facts: (1) the CCC threshold is porous — DiLoCo enables sub-CCC nodes to collectively exceed the Strict Threshold with modest investment; (2) network conditions barely matter — even 10 Mbps consumer broadband suffices; (3) the existing enforcement mechanisms (chip tracking, financial monitoring) are the primary barriers, not the compute threshold itself. This section evaluates specific treaty modifications that could narrow or close the distributed training loophole.

### 10.1 Lowering the CCC Compute Threshold

The most direct response is to lower the CCC registration threshold from 16 H100-equivalents. The table below shows the maximum node configuration that fits under each threshold (using A100 80GB, the optimal VRAM/compute hardware):

| CCC Threshold | Max GPUs/node | PFLOPS | VRAM | Max Model | H100-equiv |
|:--|:--|:--|:--|:--|:--|
| **16 (current)** | **48x A100** | **14.98** | **3,840 GB** | **240B** | **15.1** |
| 8 | 25x A100 | 7.80 | 2,000 GB | 125B | 7.9 |
| 4 | 12x A100 | 3.74 | 960 GB | 60B | 3.8 |
| 2 | 6x A100 | 1.87 | 480 GB | 30B | 1.9 |
| 1 | 3x A100 | 0.94 | 240 GB | 15B | 0.9 |

**Nodes and cost to reach each compute target (A100 FP16, 16x compression, 1.5 years, expected):**

| CCC Threshold | Nodes for 10^24 | Cost | Nodes for 10^25 | Cost | Nodes for 10^26 | Cost |
|:--|:--|:--|:--|:--|:--|:--|
| 16 (current) | 5 | $4M | 42 | $30M | 413 | $297M |
| 8 | 8 | $3M | 79 | $30M | 783 | $294M |
| 4 | 16 | $3M | 161 | $29M | 1,611 | $290M |
| 2 | 32 | $3M | 317 | $29M | 3,186 | $287M |
| 1 | 63 | $3M | 627 | $28M | 6,309 | $284M |

**The central finding is that lowering the CCC threshold barely increases the cost of evasion.** Reaching 10^24 FLOP costs approximately $3M regardless of whether the threshold is 16 or 1 H100-equivalent. The evader simply redistributes the same total number of GPUs into more, smaller nodes. The total hardware cost is almost unchanged because the same aggregate compute power is needed.

The impact is on **model quality**, not cost. A 3x A100 node (under 1 H100-eq threshold) can only hold a 15B model, compared to 240B at the current threshold. This forces the evader to either (a) train a much smaller model, or (b) use PP-Group DiLoCo to shard a larger model across co-located pipeline stages (Section 5.4). Option (b) is viable with regional interconnect (1 Gbps, 20 ms) but requires co-locating multiple nodes — partially re-creating the physical clustering the treaty monitors. At the 1 H100-eq threshold, even training a 60B model would require 4 pipeline stages, incurring 27% bubble overhead. A 15B model trained on ~3.5T tokens (at 10^24 FLOP) is still a meaningful AI system, but the lower threshold substantially limits model capability.

**Collateral damage of lowering the threshold:**

| Threshold | Legitimate systems requiring registration |
|:--|:--|
| 16 (current) | 0 of 14 representative systems |
| 8 | 0 (the 8x H100 systems are exactly at 8.0, not over) |
| 4 | 4 systems: AWS p5.48xlarge, DGX H100, Princeton AI cluster, molecular dynamics clusters |
| 2 | 7 systems: adds DGX A100, AI lab servers, AWS p4d |
| 1 | 11 of 14 systems: includes nearly all multi-GPU research workstations |

At a threshold of 4 H100-equiv, every 8x H100 node would require registration and be subject to potential inspection. At threshold=1, even a researcher with three A100s would need to register. This would not ban or confiscate any hardware, but the reporting requirements and inspection obligations would expand the treaty's administrative footprint.

However, a lower threshold does have an indirect benefit: it **increases the surface area for detection**. If an evader must manage 63 individual nodes (at threshold=1) rather than 5 larger nodes (at threshold=16), each becomes a point where inspectors could detect training activity. The model quality penalty is also significant — forcing the evader from a 240B model to a 15B model substantially reduces the threat.

**Assessment:** Lowering the CCC compute threshold is **modestly effective through indirect mechanisms** (increased inspection surface, model quality degradation) but **does not increase evasion cost**. A moderate reduction to **4-8 H100-equivalents** may represent a reasonable compromise.

### 10.2 Adding Memory to the CCC Definition

The current CCC threshold is defined solely by compute capacity. Section 6.5 identified the resulting exploit: an evader can select hardware that maximizes VRAM while staying under the compute threshold (48x A100 80GB = 3,840 GB VRAM at 15.1 H100-equiv), enabling large models (240B) that would be impossible on compute-equivalent H100 nodes (1,280 GB VRAM).

A **memory-based registration threshold** would require CCC registration when aggregate accelerator VRAM exceeds a specified limit.

**Impact of VRAM thresholds (combined with 16 H100-eq compute threshold):**

| VRAM Limit | Max A100s | VRAM | Max Model | Nodes for 10^24 | Cost | Nodes for 10^25 | Cost |
|:--|:--|:--|:--|:--|:--|:--|:--|
| No limit (current) | 48 | 3,840 GB | 240B | 5 | $4M | 42 | $30M |
| 2 TB | 25 | 2,000 GB | 125B | 8 | $3M | 79 | $30M |
| 1 TB | 12 | 960 GB | 60B | 16 | $3M | 161 | $29M |
| 512 GB | 6 | 480 GB | 30B | 32 | $3M | 317 | $29M |

Same pattern as lowering the compute threshold: costs barely change, but model size shrinks. The memory threshold **closes the specific A100 exploit** but does not prevent evasion because the evader simply uses more, smaller nodes.

**Collateral at each level:** A 1 TB limit catches 0 of 14 representative legitimate systems. A 512 GB limit catches 6 (all 8-GPU A100/H100 systems), bringing standard DGX systems and cloud instances under registration — these are standard equipment at institutions that have the resources to comply.

**Assessment:** A memory threshold of **1-2 TB** is a reasonable, low-impact addition that closes the 48x A100 exploit without subjecting standard 8-GPU systems to registration.

### 10.3 Bandwidth Restrictions

Section 9 established that DiLoCo is inherently bandwidth-insensitive. Could the treaty mandate bandwidth restrictions on sub-CCC nodes?

**Scenario: 10 Mbps upload cap on all unregistered nodes.** From Section 9.2 (expected scenario), the impact:

| Configuration | C_local at 100 Mbps | C_local at 10 Mbps | Reduction |
|:--|:--|:--|:--|
| 72 nodes, A100 FP16, flat 16x | 1.75 x 10^25 | 1.64 x 10^25 | 6% |
| 2,000 nodes, H100 FP8, hier+100x | 1.07 x 10^27 | 1.04 x 10^27 | 3% |

A 10 Mbps cap reduces evasion capability by only 3-6% from baseline. The fundamental problem is DiLoCo's logarithmic bandwidth sensitivity: the evader needs 10x more inner steps but loses only one $\log_{10}$ unit of efficiency.

**Implementation options:** (a) Monitoring WAN connections at registered GPU facilities — lightest touch, provides a detection signal; (b) mandating ISP throttling for GPU facilities — more intrusive, constrains legitimate inference and cloud computing; (c) requiring network registration for AI chip facilities. Option (a) has value as a supplementary detection mechanism (Section 10.4) even though bandwidth *caps* are technically ineffective.

**Assessment:** Bandwidth *caps* are **technically ineffective** — DiLoCo's logarithmic sensitivity to bandwidth means no realistic restriction produces a meaningful reduction in achievable compute. Traffic *monitoring* at GPU facilities has supplementary detection value.

### 10.4 Traffic Fingerprinting and ISP Monitoring

DiLoCo creates a distinctive network signature: pseudo-gradient synchronization at regular intervals, with identical payload sizes. Could treaty-mandated ISP monitoring detect this pattern?

**The sync fingerprint:** At 100 Mbps with a 240B model and 16x compression, each sync transfers ~30 GB per direction. With streaming DiLoCo, this is spread over ~100 minutes (~38 Mbps average). The traffic is continuous (streaming eliminates bursts), bidirectional (~76 Mbps total), encrypted via VPN, and has a regular cadence.

**Detection feasibility:** The regular cadence and fixed payload size are distinctive features. However: streaming eliminates the bursty pattern; traffic shaping defeats detection (jitter, padding, cover traffic); the false positive rate for sustained bidirectional encrypted traffic at 50-100 Mbps describes millions of legitimate servers. Mesh detection (correlating N distributed nodes) is even harder.

**Assessment:** Traffic fingerprinting is a **useful supplementary detection signal** when scoped to registered GPU facilities, but not a reliable primary mechanism. It is best deployed as part of a multi-source intelligence approach, providing corroborating evidence for investigations triggered by other means.

### 10.5 Hardware-Level Enforcement: TEE and Remote Attestation

Modern AI accelerators include **Trusted Execution Environment (TEE)** capabilities through NVIDIA Confidential Computing. This could be repurposed for treaty enforcement: every AI chip cryptographically attests its workload to the CTB, signed by a hardware-embedded key.

**Strengths:** Eliminates plausible deniability; scales with chip tracking; per-chip enforcement without network monitoring.

**Limitations:** (1) Requires manufacturer cooperation (currently only NVIDIA); (2) firmware integrity can potentially be defeated by state-level actors with semiconductor analysis tools; (3) pre-existing and unregistered chips lack attestation firmware; (4) privacy and sovereignty concerns about real-time workload reporting; (5) chain-of-trust problem (who verifies the TEE firmware?).

**Assessment:** TEE-based attestation is the **most technically promising** countermeasure for non-state actors. It should be considered a **medium-term** enforcement mechanism (3-5 years to deploy), requiring cooperation from all major chip manufacturers.

### 10.6 Orchestration Layer Regulation

Banning WAN-based distributed training software. **Assessment: Unenforceable and counterproductive.** The knowledge is published, implementation is trivial (~100 lines of PyTorch), and the techniques (federated learning, local SGD) are broadly useful in legitimate applications. Precedent: 1990s encryption export controls failed for similar reasons.

### 10.7 Redefining Model Possession

In distributed DiLoCo training, each node holds a complete copy of the model weights. The treaty could redefine possession to include "distributed possession" and "control of orchestration" as violation triggers.

**Assessment:** A **necessary but insufficient** legal clarification. It closes a definitional gap but does nothing to solve the detection problem.

### 10.8 Enhanced Chip Tracking and Physical Inspections

Strengthened enforcement through: (a) mandatory utilization reporting for all registered AI chips (DiLoCo shows distinctive 100% utilization pattern); (b) random physical inspections of AI chip facilities; (c) enhanced whistleblower programs with financial bounties; (d) power monitoring (though 15-20 kW per sub-CCC node is within normal commercial range).

**Assessment:** Moderately effective and the lowest-collateral enforcement approach. Utilization reporting + whistleblower programs directly increase the cost and risk of prolonged distributed training.

### 10.9 Chip Activation Gates

Unlike TEE attestation (Section 10.5), which *reports* what a chip is doing, chip activation gates *prevent* unregistered chips from functioning. The mechanism: all AI chips require an online activation handshake with a manufacturer-controlled server before they can execute training workloads, similar to software product activation. Chips that have not been registered with the international agency cannot activate.

**Strengths:** Stronger than TEE — a chip that never activates cannot train anything. Scales with manufacturing: every new chip ships locked. Combined with chip tracking (Section 10.8), creates a comprehensive lifecycle from manufacture to deployment.

**Limitations:** (1) Requires cooperation from all major chip manufacturers (same barrier as TEE); (2) pre-existing and stockpiled chips lack activation firmware — a legacy fleet problem that diminishes over time but creates a multi-year window; (3) state actors with domestic fabs can produce chips without activation gates; (4) activation servers become critical infrastructure (availability, sovereignty, and censorship concerns); (5) can potentially be bypassed through firmware modification, though this requires significant expertise.

**Assessment:** **High effectiveness against non-state actors** when combined with chip tracking. A natural complement to TEE attestation — activation gates prevent unauthorized use, while TEE monitors authorized use. Requires the same manufacturer cooperation as TEE and should be pursued on the same timeline.

### 10.10 Foundry-Level Serialization

Section 10.8's chip tracking operates at the point of sale. Foundry-level serialization operates at the point of manufacture: all advanced semiconductor fabs (TSMC, Samsung, Intel, GlobalFoundries) would be required to report every AI-capable chip produced, with unique serial numbers assigned at the wafer level, directly to the international agency.

**Strengths:** Closes the gap between chips manufactured and chips sold. Point-of-sale tracking can be evaded through intermediaries, shell companies, or informal markets; point-of-manufacture tracking cannot, because there are only ~4 fabs worldwide capable of producing frontier AI chips. Creates a complete census of AI chips in existence.

**Limitations:** (1) Requires international cooperation from fab host nations (Taiwan, South Korea, US, Japan) — but these are likely treaty signatories; (2) domestic fabs in non-signatory states (e.g., China's SMIC) are exempt; (3) defining "AI-capable chip" is non-trivial — the same silicon can serve AI, HPC, and graphics workloads; (4) older-generation chips (pre-serialization) remain untracked.

**Assessment:** **Medium-high effectiveness.** The small number of advanced fabs makes this unusually tractable for an international verification regime. Should be a **high priority** for treaty implementation, as it provides the foundation for all downstream chip tracking.

### 10.11 Cloud Provider Obligations

The analysis in Sections 5-8 assumes the evader owns hardware. An alternative evasion path — not previously analyzed — is renting GPUs from cloud providers in small increments across multiple providers and accounts.

**Two complementary measures:**

1. **Cross-provider GPU-hour reporting with KYC:** Require all cloud GPU providers (AWS, GCP, Azure, Lambda, CoreWeave, Oracle, etc.) to report cumulative GPU-hours per customer entity to the international agency. Know-your-customer (KYC) requirements prevent the same entity from splitting across accounts. Cross-referencing reports across providers catches an evader distributing rentals.

2. **Sustained utilization flagging:** Require hosting providers and colocation facilities to flag any customer sustaining >90% GPU utilization for more than 48 hours. DiLoCo training is inherently compute-bound and runs for months — this is a distinctive operational signature that differentiates training from inference or intermittent research workloads.

**Limitations:** (1) Only effective if all major cloud providers participate — a single non-compliant provider in a non-signatory jurisdiction creates a gap; (2) KYC can be evaded through shell companies, though this adds cost and detection risk; (3) legitimate long-running workloads (climate simulation, molecular dynamics, rendering farms) would trigger utilization flags, requiring follow-up investigation.

**Assessment:** **Medium-high effectiveness against non-state actors** who rent rather than own hardware. **Strongly recommended** — closes a significant evasion path not addressed by hardware-focused measures. Ineffective against state actors who use owned hardware.

### 10.12 Energy-Side Detection

GPU clusters have distinctive power consumption profiles: sustained near-constant high draw, 24/7, for months. Utility companies could be required to flag step-function increases in power consumption at facilities known or suspected to house computing equipment.

**Scale analysis:**

| Configuration | Power Draw | Detectability |
|:--|:--|:--|
| 4 nodes (48x A100 each) | ~60-80 kW | **Low** — within normal commercial range |
| 72 nodes | ~1-1.5 MW | **Marginal** — comparable to a medium office building; detectable if correlated with GPU procurement intelligence |
| 500 nodes (distributed) | ~7-10 MW total, ~15-20 kW per site | **Low per site** — individual sites are unremarkable |
| 500 nodes (PP co-located) | ~7-10 MW concentrated | **Medium** — comparable to a small data center; requires industrial power infrastructure |

**Assessment:** **Low-medium effectiveness** as a standalone measure. Power monitoring is most useful as a **corroborating signal** when combined with other intelligence (chip procurement records, financial monitoring, tip-offs). The key scenario where power detection adds value is PP-Group DiLoCo co-location, where multiple nodes must be in the same facility, concentrating power draw. For fully distributed flat DiLoCo, individual node power draw is unremarkable.

### 10.13 Model Provenance and Capability-Based Detection

Rather than detecting the training process, detect the *result*. Two complementary approaches:

1. **Mandatory model provenance certificates:** Require all publicly deployed AI models to carry a cryptographic certificate linking them to a registered training run, including the training compute used, hardware, and duration. Models without valid certificates are presumptive evidence of illegal training. Similar to chain-of-custody requirements for controlled substances or certificates of origin in trade.

2. **Capability-based suspicion triggers:** Maintain a registry of authorized training runs and their expected capability levels. If a model appears with capabilities that exceed what any registered training run could produce — for example, a model demonstrating GPT-4-level performance that no registered run had sufficient compute to produce — this triggers an investigation.

**Strengths:** Operates on the output side, bypassing all hardware-level and network-level evasion. Doesn't prevent training but makes *deploying* the result risky, which reduces the incentive to evade in the first place. Capability-based detection works even if the training process was completely undetectable.

**Limitations:** (1) Doesn't prevent training, only deployment — an evader who wants a model for private use (e.g., state intelligence agency) is unaffected; (2) capability evaluation is imprecise — it's difficult to establish exact capability thresholds; (3) open-source and private deployment channels are hard to monitor; (4) models can be fine-tuned or distilled to obscure their origin; (5) provenance certificates can be forged if the signing infrastructure is compromised.

**Assessment:** **Medium effectiveness.** Most valuable as a deterrent against commercial deployment of illegally trained models. Should be **recommended** as part of a comprehensive package — it addresses a gap that no hardware-focused measure can fill.

### 10.14 Active Intelligence Operations

Standard intelligence tradecraft applied to the distributed training detection problem:

1. **Honeypot nodes:** Operate decoy DiLoCo nodes that advertise availability to join distributed training runs — for example, on forums, dark-web markets, or through intermediaries where an evader might recruit third-party compute. Analogous to law enforcement honeypots on illicit marketplaces. When an evader's orchestrator contacts the honeypot, the agency learns the training configuration, model architecture, and network topology.

2. **Checkpoint forensics:** If an evader's model weights are obtained through any channel (whistleblower, infiltration, interception, or public release), the optimizer state and weight distribution can be forensically analyzed to reconstruct the training configuration — approximate number of replicas, H value, compression artifacts, total FLOP, and training duration. This enables post-hoc attribution even if the training process was undetected.

3. **Infiltration of recruitment networks:** A non-state evader operating 72+ nodes likely needs to recruit personnel and procure hardware through networks that can be infiltrated. The larger the operation, the more human touchpoints exist.

**Assessment:** **Low-medium effectiveness against non-state actors** (useful as a supplementary intelligence tool), **medium effectiveness against state actors** (where infiltration and signals intelligence are standard tools and often the only effective lever). These are operational capabilities, not treaty provisions — they would be funded and directed by national intelligence agencies rather than specified in treaty text.

### 10.15 Synthesis: Recommended Treaty Modifications

No single countermeasure closes the distributed training loophole. Effectiveness depends critically on the **actor type**:

**Against non-state actors ($3M-$360M, 4-500 nodes):**

| Countermeasure | Effectiveness | Registration Burden | Recommendation |
|:--|:--|:--|:--|
| Lower CCC threshold (to 4-8) | Low-Medium | Moderate | **Consider** — increases inspection surface |
| Memory threshold (1-2 TB) | Low-Medium | Low | **Recommended** — closes A100 exploit |
| Bandwidth caps | None | Moderate | Not recommended (technically ineffective) |
| Traffic monitoring at GPU sites | Low-Medium | Low | **Recommended** — supplementary signal |
| TEE/Remote attestation | **High** | Medium | **Strongly recommended** (medium-term) |
| Orchestration regulation | None | Low | Not recommended (unenforceable) |
| Model possession redefinition | Medium | None | **Recommended** — legal clarification |
| Enhanced chip tracking | **Medium-High** | Low | **Strongly recommended** |
| Whistleblower programs | **Medium-High** | None | **Strongly recommended** |
| Chip activation gates | **High** | Medium | **Strongly recommended** (medium-term, with TEE) |
| Foundry-level serialization | **Medium-High** | Low | **Recommended** — point-of-manufacture tracking |
| Cloud provider KYC/reporting | **Medium-High** | Medium | **Strongly recommended** — closes cloud rental path |
| Energy monitoring | Low-Medium | Low | **Consider** — supplementary signal at scale |
| Model provenance certificates | **Medium** | Low | **Recommended** — output-side deterrent |
| Honeypot operations | Low-Medium | None | **Consider** — intelligence tool |

Note: "Registration Burden" refers to the registration, reporting, and inspection obligations imposed on legitimate computing systems — not bans or confiscation. Systems caught by lower thresholds must be registered and may be subject to inspections, but continue to operate normally.

**Against state actors ($1B+, 2,000+ nodes):**

| Countermeasure | Effectiveness | Notes |
|:--|:--|:--|
| Lower CCC threshold | None | Same cost, more nodes |
| TEE/Remote attestation | Low | State actors can manufacture unattested chips |
| Enhanced chip tracking | Low | Classified procurement, domestic fabs |
| Financial monitoring | **Medium** | Detectable above ~$1B for most states |
| Chip activation gates | Low | State actors can manufacture unactivated chips |
| Foundry serialization | Low-Medium | Requires international fab cooperation; domestic fabs exempt |
| Cloud provider reporting | None | State actors use owned hardware |
| Model provenance | Low | State actors deploy internally |
| Active intelligence | **Medium** | Infiltration and SIGINT are standard tools |
| Diplomatic/intelligence | **Medium-High** | The only effective lever at state scale |

**Recommended package:** (1) 1 TB VRAM threshold; (2) TEE attestation mandate with chip activation gates; (3) foundry-level serialization; (4) model possession redefinition; (5) enhanced whistleblower bounties; (6) utilization reporting; (7) cloud provider KYC and GPU-hour reporting; (8) model provenance certificates for public deployment; (9) consider lowering CCC threshold to 4-8 H100-eq; (10) traffic monitoring (not caps) at GPU facilities.

**The hard truth:** Even with all recommended modifications, the distributed training loophole cannot be fully closed against a determined state actor with domestic chip manufacturing, classified procurement, and sovereign territory. Against such actors, the treaty's enforcement ultimately depends on diplomatic, intelligence, and economic instruments — the same tools used in nuclear nonproliferation, with the same fundamental limitations.
