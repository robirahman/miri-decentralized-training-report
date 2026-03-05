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
| **50x A100 80GB** | 15.60 PFLOPS | **15.8** | **4,000 GB** | **250B** |
| 16x GH200 NVL2 | 15.84 PFLOPS | 16.0 | 2,304 GB | 144B |
| 16x H100 SXM | 15.84 PFLOPS | 16.0 | 1,280 GB | 80B |

(A100 SXM 80GB: 312 TFLOPS FP16. 50 x 312 = 15,600 TFLOPS = 15.8 H100-equivalents, under the 16-equivalent threshold.)

The **50x A100 80GB** configuration is optimal for flat DiLoCo: nearly the same compute as 16x H100 but **3x the VRAM**, enabling a 250B-parameter model to fit entirely on one node without model parallelism. However, as shown in Section 5.4, an evader could also use **PP-Group DiLoCo** to train models larger than single-node VRAM by co-locating pipeline stages on regional interconnects. This trades fewer DiLoCo groups for a closer-to-optimal model size, which becomes advantageous at large scale (500+ nodes).

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
| Model | 250B dense (flat) or up to ~1000B (PP-Group) | Flat: max single-node; PP: larger via pipeline |
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

**Replica count penalty ($\eta_{\text{replicas}}$):** Averaging pseudo-gradients across many replicas introduces noise. Based on the [DiLoCo Scaling Laws](https://arxiv.org/abs/2503.09799) empirical data (Table 4: M=8 costs ~1.1% loss penalty at 2.4B parameters, with the penalty decreasing at larger model sizes), this factor is modeled as:

$$\eta_{\text{replicas}} = \max\!\left(0,\; 1 - 0.005 \cdot \frac{\min(2.4, P_B)}{P_B} \cdot \log_2(N)\right)$$

where $P_B$ is the model size in billions and $N$ is the number of replicas. For the primary 250B configuration, this penalty is negligible (<0.1% at 72 nodes, <0.3% at 500 nodes) because the denominator scales with model size. It becomes material only for very small models at very large replica counts.

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

### 5.1 Primary Configuration: 50x A100 80GB (250B model)

All results use the **expected** compression quality scenario (Section 4.2). Optimistic values (no compression penalty) and conservative values are shown in parentheses where they differ materially.

| Nodes | GPUs | Est. Cost | H | $\eta$ | C_local (FLOP) | x Threshold | OT Ratio | $\eta_{\text{chin}}$ | C_quality (FLOP) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | 50 | $0.8M | 1 | 1.000 | 2.95 x 10^23 | 0.3x | 0.0x | 1.000 | 2.95 x 10^23 |
| 4 | 200 | $3.0M | 175 | 0.861 | **1.02 x 10^24** | **1.0x** | 0.1x | 0.772 | 7.86 x 10^23 |
| 8 | 400 | $6.0M | 183 | 0.860 | 2.03 x 10^24 | 2.0x | 0.2x | 0.920 | 1.87 x 10^24 |
| 16 | 800 | $12M | 191 | 0.859 | 4.06 x 10^24 | 4.1x | 0.5x | 1.000 | 4.06 x 10^24 |
| 32 | 1,600 | $24M | 199 | 0.858 | 8.11 x 10^24 | 8.1x | 1.0x | 1.000 | 8.11 x 10^24 |
| 72 | 3,600 | $54M | 208 | 0.857 | **1.82 x 10^25** | **18.2x** | 2.2x | 0.891 | **1.62 x 10^25** |
| 144 | 7,200 | $108M | 216 | 0.856 | 3.64 x 10^25 | 36.4x | 4.4x | 0.736 | 2.68 x 10^25 |
| 500 | 25,000 | $375M | 230 | 0.854 | 1.26 x 10^26 | 126.2x | 15.4x | 0.430 | 5.43 x 10^25 |

The **overtraining ratio** (OT) is the ratio of actual training tokens to the Chinchilla-optimal token count ($D^* = 25.6 \times N$, per the [corrected Chinchilla scaling law](https://arxiv.org/abs/2404.10102)). The **Chinchilla efficiency** $\eta_{\text{chin}}$ quantifies the compute-allocation penalty from training a non-optimally-sized model relative to Chinchilla-optimal. **C_quality** $= C_{\text{local}} \times \eta_{\text{chin}}$ is the quality-adjusted compute — the amount of optimally-allocated compute that would produce the same model quality.

**Note on overtraining in practice:** The Chinchilla scaling law minimizes the loss of a single forward pass per training FLOP. In practice, developers intentionally overtrain models because each trained model is used for many inference calls. A moderately overtrained smaller model provides better cost-efficiency at inference time than a Chinchilla-optimal larger model, because inference cost scales with model size. When a model's lifetime inference compute approximately equals its training compute, moderate overtraining (2-10x) is optimal from a total-cost perspective. The $\eta_{\text{chin}}$ metric is therefore a conservative (pessimistic) measure of effective compute: it overstates the practical cost of overtraining, particularly in the 2-10x range where production models typically operate. Only extreme overtraining (>50x) represents unambiguous waste regardless of inference amortization.

At 72 nodes, C_quality = 1.62 x 10^25, reflecting the mild overtraining (2.2x) of a 250B model — well within the range that developers actively prefer. At 500 nodes, overtraining reaches 15.4x and $\eta_{\text{chin}}$ drops to 0.43. While 15x overtraining is beyond what developers typically choose, the real quality penalty is smaller than $\eta_{\text{chin}}$ suggests due to inference amortization. Nonetheless, at very large node counts the overtraining ratio becomes extreme (123x at 4,000 nodes), motivating the PP-Group DiLoCo analysis in Section 5.4, where larger models reduce overtraining.

The 72-node reference point shows $\eta = 0.857$ under the expected scenario (optimistic: 0.874; conservative: 0.831). Varying compression quality alone gives a C_local range of 1.77-1.86 x 10^25 FLOP (~18x). However, this narrow range understates the true uncertainty because it holds several other uncertain quantities fixed. Accounting for realistic variation in MFU (30-45% vs. the assumed 40%), hardware availability over 18 months, the straggler model coefficient (an engineering estimate, not empirically calibrated at this scale), effective network bandwidth, and DiLoCo convergence behavior at 250B (extrapolated from experiments at ≤10B parameters), the plausible range widens to approximately **7-23x the Strict Threshold**. The core conclusion — that 72 nodes exceeds the threshold by a large margin — is robust even at the pessimistic end of this range.

The **algorithmic efficiency** ($\eta$) is stable at 85-86% across all node counts. Under the optimistic scenario (no compression quality penalty), efficiency would be 87-88%. The 2% difference reflects the expected cost of 16x pseudo-gradient compression extrapolated to 250B scale.

### 5.2 Model Quality Analysis

The evader trains a **250B-parameter dense model** in flat DiLoCo mode. Using the corrected Chinchilla-optimal token count $D^* = 25.6 \times N$ ([Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102)):

| Nodes | Total Tokens | Chinchilla Tokens (250B) | Overtraining Ratio | Assessment |
|:--|:--|:--|:--|:--|
| 4 | 0.8T | 6.4T | 0.1x | **Under-trained** (would use a smaller model) |
| 16 | 3.2T | 6.4T | 0.5x | Below optimal |
| 32 | 6.3T | 6.4T | 1.0x | Near Chinchilla-optimal |
| 72 | 14.2T | 6.4T | 2.2x | **Moderately overtrained** (industry-standard; preferred for production) |
| 144 | 28.4T | 6.4T | 4.4x | Overtrained (still practical; comparable to LLaMA-3 at ~10x) |
| 500 | 98.5T | 6.4T | 15.4x | Heavily overtrained (beyond typical practice; $\eta_{\text{chin}} = 0.43$) |

At the **72-node** reference point: 250B params trained on 14.2T tokens with 2.2x overtraining is not a penalty — it is in fact the preferred regime for production deployments, where inference amortization makes moderate overtraining cost-optimal. This is comparable to how production models like LLaMA-3 are trained (which used ~10x overtraining). The $\eta_{\text{chin}} = 0.89$ represents a conservative lower bound; accounting for inference amortization, the effective penalty of 2.2x overtraining is negligible.

At **500+ nodes**, overtraining reaches 15x, which is beyond what developers typically choose but not catastrophic — the real quality penalty is smaller than $\eta_{\text{chin}} = 0.43$ suggests. However, at 4,000 nodes the ratio reaches 123x, which represents genuine waste even accounting for inference amortization. The Chinchilla-optimal model for 500 nodes' compute budget is ~980B — far larger than the 250B that fits on a single node. This motivates PP-Group DiLoCo (Section 5.4), where pipeline stages allow training larger models.

For small node counts (N < 16), the evader would train a proportionally smaller model to stay near Chinchilla-optimal. For example, at N=4 with C_local = 10^24 FLOP, the Chinchilla-optimal model is ~29B params on ~0.7T tokens, which fits easily within the 250B memory budget.

### 5.3 Comparison to Existing 10^25 Baseline

The [existing baseline scenario](Past%20MIRI%20work/10%5E25%20FLOP%20decentralized%20training%20configuration.md) used 72 nodes of 16x GH200 (~15.8 PFLOPS/node FP16, 2,304 GB) to train a 144B model on 12T tokens (1.02 x 10^25 FLOP) in 331 days. It was communication-bound.

This analysis differs in three ways:
1. **Similar compute per node** (~15.6 vs ~15.8 PFLOPS) — both near the CCC threshold boundary
2. **Higher memory per node** (4,000 vs 2,304 GB) — enables a larger 250B model (vs 144B)
3. **Longer time** (548 vs 331 days) — the treaty scenario allows 1.5 years

Result: with similar per-node compute but more VRAM and a longer training window, DiLoCo's compute-bound operation yields **1.82 x 10^25 local-equivalent FLOP** (expected) — a 1.8x improvement over the baseline's 1.02 x 10^25 raw compute. The improvement comes from DiLoCo's compute-bound operation (vs the baseline's communication-bound regime) and the longer training window.

### 5.4 PP-Group DiLoCo and Chinchilla Optimality

The flat DiLoCo results in Section 5.1 all train the largest model that fits on a single node (250B for 50x A100). At large node counts, this leads to increasingly extreme overtraining: 15x at 500 nodes, and 123x at 4,000 nodes. While moderate overtraining (2-10x) is preferred in practice due to inference amortization, overtraining beyond ~50x represents genuine waste. The Chinchilla-optimal model for 500 nodes' compute budget is ~980B — far larger than what fits on one node.

**PP-Group DiLoCo** addresses this by grouping $S$ co-located nodes into pipeline stages that collectively hold a larger model, then running DiLoCo across $G = N/S$ groups over WAN. The co-located pipeline groups use regional interconnect (1 Gbps, 20 ms latency), while DiLoCo synchronization uses WAN as before. Activation tensors passed between pipeline stages are compressed using 4-bit quantization (4x compression, $\eta_{\text{act}} = 0.995^{2(S-1)}$).

**The fundamental tradeoff:** PP-Group DiLoCo trains a larger (closer-to-optimal) model but with fewer DiLoCo groups, reducing throughput. The GPipe pipeline bubble also adds overhead: $(S-1)/(M+S-1)$ fraction of compute is wasted in the bubble, where $M=8$ micro-batches.

**Model size sweep results (expected scenario):**

| Nodes | Best Mode | Model | PP Stages | Groups | OT | $\eta$ | $\eta_{\text{chin}}$ | C_quality | vs Flat 250B |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 72 | DiLoCo | 250B | 1 | 72 | 2.2x | 0.857 | 0.891 | 1.62 x 10^25 | 1.0x |
| 500 | PP-DiLoCo | 500B | 2 | 250 | 2.0x | 0.851 | 0.911 | 5.83 x 10^25 | 1.07x |
| 4,000 | PP-DiLoCo | 1000B | 4 | 1,000 | 2.7x | 0.831 | 0.844 | **2.87 x 10^26** | **2.52x** |
| 2,000 (H100 FP8) | PP-DiLoCo | 549B | 6 | 333 | 4.8x | 0.816 | 0.710 | **1.28 x 10^26** | **5.58x** |

**Key findings:**

1. **At 72 nodes, flat DiLoCo is still optimal.** The 2.2x overtraining is mild, and splitting 72 nodes into 36 pipeline groups halves throughput, which is not compensated by better Chinchilla efficiency.

2. **At 500 nodes, PP-DiLoCo gives a modest 7% improvement** by training a 500B model (2 stages, 250 groups) instead of 250B (500 groups). The overtraining drops from 15x to 2.0x.

3. **At 4,000 nodes, PP-DiLoCo gives 2.5x improvement.** Training a 1000B model (4 stages, 1,000 groups) produces 2.87 x 10^26 quality-adjusted FLOP versus 1.14 x 10^26 for flat DiLoCo at 250B. The flat approach overtrains at 123x Chinchilla tokens — well beyond the ~10x range where inference amortization provides a practical benefit, and into genuinely wasteful territory.

4. **For H100 FP8 nodes, PP-DiLoCo is essential.** The max single-node model at FP8 is only 91B, leading to extreme overtraining (934x) at 2,000 nodes. PP-DiLoCo with 6 stages trains a 549B model, improving C_quality by **5.6x**.

5. **The optimal model size is NOT the Chinchilla-optimal model.** The Chinchilla-optimal model for 4,000 A100 nodes is ~2,800B, which would require 12 pipeline stages and suffer 58% bubble overhead. The practical optimum (1000B, 4 stages) balances model size against pipeline efficiency.

**Implication for governance:** PP-Group DiLoCo substantially increases the quality of models an evader could train at large scale. An evader with 4,000 nodes could train a 1000B model that achieves 2.87 x 10^26 quality-adjusted FLOP — compared to 1.14 x 10^26 for a naive flat approach. However, PP-Group DiLoCo requires co-locating pipeline stages (S nodes within one facility or metro area), which partially re-creates the physical clustering that the treaty's monitoring mechanisms are designed to detect.

## 6. Governance Implications

### 6.1 The Threshold is Easily Exceeded

**As few as 4 unregistered nodes** (~200 A100 GPUs, ~$3M) are sufficient to match the 10^24 FLOP Strict Threshold in 1.5 years. This conclusion is robust: under the conservative compression quality scenario, 4 nodes still produce ~1.0 x 10^24 FLOP (~1.0x threshold). This is a modest investment by industry standards — comparable to a mid-size tech company's annual ML compute budget, though it substantially exceeds what most academic research labs can afford. At 72 nodes ($54M), the evader achieves an estimated **~7-23x** the threshold (central estimate 18.2x; range driven by uncertainties in MFU, hardware availability, straggler effects, and DiLoCo convergence at scale), training a 250B-parameter model that would be competitive with frontier models.

### 6.2 The CCC Interconnect Criterion Is Irrelevant for DiLoCo

The treaty's CCC definition includes clusters with "inter-node aggregate bandwidth greater than 25 Gbit/s." DiLoCo training requires only **100 Mbps** between nodes — 250x below this threshold. The synchronization traffic (every ~50-90 minutes, compressed pseudo-gradients) is indistinguishable from routine internet traffic in bandwidth terms. The definition does also capture clusters "networked to perform workloads together," but enforcing this requires detecting the workload itself, not just the network topology.

### 6.3 Detection Challenges

DiLoCo training creates a distinctive network "heartbeat": synchronization traffic at regular intervals. However, with Streaming DiLoCo, this traffic is spread continuously rather than bursty:

- **Average bandwidth per node:** Each sync exchanges 250 Gbit (31 GB) compressed pseudo-gradients in each direction. With streaming enabled and H optimized for compute-bound operation, the sync data is spread over the full ~100-minute outer step, resulting in ~39 Mbps average upload + ~39 Mbps average download. This is comparable to a media server or cloud backup service.
- **Traffic is encrypted but fingerprintable:** Standard VPN tunnels obscure payload content but not traffic metadata (packet timing, sizes, volume patterns). Research demonstrates >93% accuracy at identifying applications through encrypted VPN tunnels using metadata alone. With Streaming DiLoCo, the traffic presents as a constant ~78 Mbps symmetric flow rather than periodic bursts, which eliminates the most obvious fingerprint. However, several features remain distinctive even through encryption: the extreme constancy of bandwidth over days or weeks (unlike normal internet traffic, which has characteristic micro-variations), near-perfect upload/download symmetry (~39 Mbps in each direction, a pattern produced by almost no common legitimate application), and correlation across multiple endpoints (72 nodes all maintaining identical bandwidth profiles with correlated start/stop times). The sustained ~78 Mbps throughput is not unusual for an individual internet-connected server, but the combination of constancy, symmetry, and multi-node correlation is.
- **Nodes are geographically distributed:** No physical clustering, power draw anomaly, or thermal signature to detect via satellite.
- **Each node is individually legitimate:** A 50-GPU cluster is common in research labs, gaming companies, and cloud providers.
- **Traffic fingerprinting is a well-studied attack class:** Encrypted traffic classification is a mature research field with demonstrated high accuracy even against VPN and Tor traffic. While Streaming DiLoCo eliminates the periodic synchronization "heartbeat" that would be trivial to detect, the remaining features (constant symmetric bandwidth, multi-endpoint correlation, packet-level uniformity) shift the detection problem rather than eliminating it. Multi-endpoint correlation — observing that dozens of geographically distributed nodes maintain identical bandwidth profiles — is the hardest feature for an evader to disguise, and would be visible to an enforcement body with access to ISP-level flow data across multiple jurisdictions.

### 6.4 The Treaty Paradoxically Extends the Evasion Window

Without the treaty, the maximum rational training duration is ~4.5 months (rapid hardware/software/investment growth makes longer runs obsolete). The treaty's restrictions on research and compute spending slow these growth rates, extending the maximum duration to ~1.5 years. This gives the evader **4x more training time**, partially offsetting the need to use smaller, unregistered nodes.

### 6.5 Hardware Optimization and PP-Group DiLoCo

A naive analysis might assume the evader is limited to 16 H100s (80B model). But the CCC threshold is on compute, not memory. By choosing high-memory, lower-compute GPUs (50x A100 80GB), the evader fits a **250B model** on each node. Pipeline parallelism over WAN (sending activations between globally distributed nodes) remains impractical (~49 seconds per micro-batch at 100 Mbps), but **PP-Group DiLoCo** — co-locating pipeline stages on regional interconnect — enables training models up to ~1000B by grouping 2-4 nodes into pipeline clusters. As shown in Section 5.4, this produces up to **2.5x more quality-adjusted compute** at 4,000 nodes by training a closer-to-Chinchilla-optimal model, though it requires physical co-location of pipeline stages.

### 6.6 Scaling Is Nearly Linear

The local-equivalent compute scales almost perfectly linearly with the number of nodes. The DiLoCo algorithmic efficiency ($\eta \approx 86\%$ expected) and the straggler overhead ($f(N) \approx 1.1-1.5$) are both modest and slow-varying. A well-funded actor with access to 500+ nodes could achieve over **100x** the Strict Threshold.

### 6.7 Enforcement Time Pressure

The 1.5-year maximum training duration (Section 2) is a theoretical optimum based on technology growth rates. In practice, the treaty's enforcement mechanisms create pressure to train *faster* than the theoretical maximum:

- **Whistleblower programs:** The treaty establishes financial incentives for reporting violations. A 1.5-year operation requires maintaining operational security across dozens or hundreds of sites for an extended period. Every employee, contractor, facility operator, and GPU supplier who becomes aware of the operation is a potential whistleblower. The cumulative probability of a leak increases with time.
- **International inspections:** Treaty signatories conduct both scheduled and triggered inspections of compute facilities. Over 1.5 years, each node site faces multiple inspection cycles. Even if individual inspections have a low detection probability, the compound probability over 18 months across many sites becomes significant.
- **Operational security failures:** Hardware failures requiring replacement, network outages, power interruptions, and personnel turnover all create detection opportunities. A 50-GPU cluster operating at near-full utilization for 18 months will experience multiple hardware failures requiring maintenance or procurement.

The practical effect: an evader would likely target **6-12 months** rather than the full 1.5 years, accepting lower total compute in exchange for reduced detection risk. Since $C_{\text{local}}$ scales linearly with time, this reduces the achievable compute proportionally:

| Duration | C_local (72 nodes, expected) | x Strict Threshold |
|:--|:--|:--|
| 6 months | 6.07 x 10^24 | 6.1x |
| 1 year | 1.21 x 10^25 | 12.1x |
| 1.5 years | 1.82 x 10^25 | 18.2x |

Even at 6 months, the 72-node configuration still exceeds the Strict Threshold by **6.1x** — sufficient to train a meaningful model (250B dense, ~4.7T tokens).

### 6.8 Financial Detection

The treaty's chip tracking provisions require all AI chips to be registered with serial numbers. Acquiring large quantities of unregistered GPUs is a supply-chain challenge that scales with the investment:

| Scale | Hardware Cost | GPUs | Financial Detectability |
|:--|:--|:--|:--|
| 4 nodes | ~$3M | 200 A100s | **Low** — within reach of individuals, startups, or research labs. Comparable to routine equipment purchases. |
| 72 nodes | ~$54M | 3,600 A100s | **Moderate** — significant capital expenditure, but not unusual for mid-size tech companies. Could be structured across multiple procurement channels. |
| 500 nodes | ~$375M | 25,000 A100s | **High** — detectable via financial intelligence: large wire transfers, unusual GPU procurement patterns, supply chain monitoring of chip manufacturers. |
| 4,000 nodes | ~$3.0B | 200,000 A100s | **Very high** — comparable to a national defense procurement. Would create visible distortions in the GPU market. |

**Forensic accounting methods** include monitoring GPU manufacturer shipment records, tracking large purchases from cloud providers or distributors, and flagging unusual patterns in semiconductor trade data. The treaty's chip registration requirement makes this particularly effective: every unaccounted chip represents a potential violation.

**The state-actor exception:** The most significant gap in financial detection is a **state actor with classified budgets**. A nation-state could acquire GPUs through military or intelligence procurement channels that are exempt from civilian financial transparency requirements. If the state has domestic chip manufacturing capability (e.g., China's SMIC or a future domestic fab), it could produce unregistered chips entirely outside the treaty's tracking system. Pre-existing military GPU stockpiles accumulated before treaty ratification would also evade tracking.

For non-state actors, the financial barrier is the primary practical constraint: it is straightforward to hide a $3M GPU purchase but extremely difficult to hide $3B without state-level resources.

## 7. Summary of Sub-CCC Evasion

| Metric | Value |
|:--|:--|
| **Minimum nodes to reach 10^24 FLOP** | **4 nodes** (~200 A100s, ~$3M) |
| **Reference scenario (72 nodes, flat DiLoCo)** | **1.82 x 10^25 C_local** (18.2x threshold), **1.62 x 10^25 C_quality** |
| **Uncertainty range (72 nodes)** | ~7-23x threshold (central estimate 18.2x; range reflects MFU, uptime, straggler, and convergence uncertainties) |
| **Model trained (72 nodes, flat)** | 250B dense, 14.2T tokens, 2.2x overtraining |
| **Best quality at 4,000 nodes (PP-DiLoCo)** | **2.87 x 10^26 C_quality** (1000B, 4 PP stages, 2.5x better than flat) |
| **Algorithmic efficiency** | ~86% expected (range: 83-88%) |
| **Confidence in 16x pseudo-gradient compression** | High (FP4 validated to 15B; penalty ~2% expected) |
| **Bottleneck** | Network bandwidth at low H; compute-bound when H is optimized |
| **Detection** | Challenging: encrypted traffic with no physical signature, but traffic fingerprinting via metadata (constancy, symmetry, multi-node correlation) is feasible |
| **Enforcement time pressure** | 6 months still yields 6.1x threshold at 72 nodes |
| **Financial barrier** | $3M (easy to hide) to $54M (moderate) to $3B (state-actor) |

The simulator predicts that the treaty's FLOP threshold can be significantly exceeded using distributed training with unregistered sub-CCC nodes. The primary defense against this evasion is not the compute threshold itself, but rather the treaty's chip tracking and consolidation provisions — the ability to prevent actors from accumulating sufficient GPU hardware in the first place.

## 8. Scaling to 10^27 FLOP and Beyond

The analysis in Sections 5-7 focuses on modest-scale evasion (4-500 nodes). This section explores whether much larger training runs — on the order of **10^27 FLOP** — are achievable using all techniques supported by the simulator: hierarchical DiLoCo, FP8 precision, aggressive pseudo-gradient compression, PP-Group DiLoCo, and Mixture-of-Experts with Expert Parallelism.

### 8.1 Configurations

Nine configurations are compared, all using sub-CCC nodes over 100 Mbps WAN. All values use the **expected** compression quality scenario. The table now includes **C_quality** (quality-adjusted compute accounting for Chinchilla-optimality deviation):

| Config | Hardware | Nodes | Model | $\eta$ | C_local | $\eta_{\text{chin}}$ | C_quality |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **A** | 50x A100 FP16, flat DiLoCo, 16x | 4,000 | 250B dense | 0.852 | 1.01 x 10^27 | 0.113 | 1.14 x 10^26 |
| **B** | 50x A100 FP16, hierarchical, 16x | 4,000 | 250B dense | 0.882 | 1.04 x 10^27 | 0.113 | 1.18 x 10^26 |
| **C** | 16x H100 FP8, flat DiLoCo, 16x | 2,000 | 91B dense | 0.844 | 1.01 x 10^27 | 0.023 | 2.29 x 10^25 |
| **D** | 16x H100 FP8, hierarchical, 16x | 2,000 | 91B dense | 0.876 | 1.05 x 10^27 | 0.023 | 2.37 x 10^25 |
| **E** | 50x A100 FP16, flat DiLoCo, 100x | 4,000 | 250B dense | 0.867 | 1.02 x 10^27 | 0.113 | 1.16 x 10^26 |
| **F** | 16x H100 FP8, hier + 100x comp | 2,000 | 91B dense | 0.892 | 1.07 x 10^27 | 0.023 | 2.42 x 10^25 |
| **G** | 50x A100 FP16, MoE+EP 600B/100B | 4,000 | 600B MoE | 0.828 | 6.49 x 10^26 | 0.038 | 2.45 x 10^25 |
| **H** | 50x A100 FP16, PP-DiLoCo 16x | 4,000 | **960B PP-4x1000** | 0.831 | 3.36 x 10^26 | 0.829 | **2.78 x 10^26** |
| **I** | 16x H100 FP8, PP-DiLoCo 16x | 2,000 | **480B PP-6x333** | 0.816 | 1.71 x 10^26 | 0.656 | **1.12 x 10^26** |

Hierarchical configurations use groups of 8 nodes with 1 Gbps regional interconnect and 20 ms latency (co-located nodes within the same facility or metro area). PP-Group DiLoCo configurations use co-located pipeline groups on regional interconnect, with 4x activation compression. The 100x pseudo-gradient compression ratio corresponds to FP4 quantization combined with aggressive sparsification, as demonstrated in [Streaming DiLoCo](https://arxiv.org/abs/2501.18512).

**Compression quality uncertainty for 100x configurations (E, F):** Under the optimistic scenario, Config F achieves 1.12 x 10^27; under the conservative scenario, it drops to 1.01 x 10^27. The 100x compression assumptions carry significant uncertainty (see [Compression Quality](Compression_Quality.md)), so these estimates should be treated as bounds rather than point predictions.

### 8.2 Key Findings

**10^27 FLOP requires 2,000-4,000 sub-CCC nodes ($1-3B in hardware).** This is unambiguously a state-actor-level investment. No non-state actor could plausibly acquire 32,000-192,000 GPUs without detection through the treaty's financial monitoring and chip tracking provisions (Section 6.8).

**C_local vs C_quality: a critical distinction.** The table reveals a dramatic divergence between raw compute (C_local) and quality-adjusted compute (C_quality). Configs A-F achieve 10^27 C_local, but their C_quality ranges from only 2.3 x 10^25 (Config C) to 1.2 x 10^26 (Config B). The reason: all flat DiLoCo configurations at 2,000-4,000 nodes massively overtrain their fixed-size models — Config A trains 123x more tokens than Chinchilla-optimal for 250B. Note that $\eta_{\text{chin}}$ is a conservative metric that overstates the practical cost of overtraining (see Section 5.1 for discussion of inference amortization); however, at the extreme levels seen here (123x-934x), the overtraining is genuinely excessive even accounting for inference. **PP-Group DiLoCo (Config H) achieves the highest C_quality** (2.78 x 10^26) by training a 960B model closer to Chinchilla-optimal, despite lower raw throughput.

**PP-Group DiLoCo is the most effective strategy at scale.** Config H (PP-DiLoCo, 4 stages, 960B model, 1,000 groups) achieves 2.78 x 10^26 C_quality — **2.4x more than the best flat DiLoCo** (Config B, 1.18 x 10^26) and **12x more than the best H100 FP8 flat configuration** (Config D, 2.37 x 10^25). The cost is lower raw throughput (C_local = 3.36 x 10^26 vs 1.04 x 10^27) due to fewer groups and PP bubble overhead, but the larger model makes far better use of each FLOP.

**FP8 precision is cost-effective but benefits greatly from PP.** The CCC threshold is defined by FP16 capacity, but FP8 yields **2x throughput** and FP8 pseudo-gradients are half the size. Config C achieves 10^27 C_local with only 2,000 nodes ($960M) — but its C_quality is only 2.3 x 10^25 due to extreme overtraining of the small 91B model (934x Chinchilla tokens). Config I (PP-DiLoCo, 480B) improves C_quality by **4.9x** to 1.12 x 10^26, making H100 FP8 with PP competitive with A100 flat DiLoCo.

**Hierarchical DiLoCo improves efficiency by 3 percentage points.** At N=4,000, flat DiLoCo achieves $\eta = 0.852$ with $H_{\min} = 254$, while hierarchical DiLoCo achieves $\eta = 0.882$ with $H_{\text{eff}} = 69$. The improvement comes from regional syncs over 1 Gbps LAN, which keep $H_{\text{inner}}$ low (18 steps) while the global sync interval is hidden behind regional cycles.

**100x pseudo-gradient compression reduces $H_{\min}$ dramatically but carries more uncertainty.** With 100x compression (vs 16x baseline), sync volume drops from 250 Gbit to 40 Gbit per direction, reducing $H_{\min}$ from 254 to 41 at N=4,000. This improves $\eta_H$ from 0.871 to 0.914. However, the pseudo-gradient compression quality factor ($\eta_{\text{pg-compression}} = 0.95$ expected, vs 0.98 for 16x) partially offsets this gain. The net expected efficiency for Config E is 0.867 — an improvement over Config A (0.852) but not as dramatic as the $\eta_H$ improvement alone would suggest. The 100x compression assumption is based on limited empirical evidence (see [Compression Quality](Compression_Quality.md)) and represents the least certain component of the analysis.

**The combined best case for raw throughput (Config F)** uses hierarchical DiLoCo, 100x pseudo-gradient compression, and FP8 H100 nodes to achieve **1.07 x 10^27 C_local** (expected). However, its C_quality is only 2.42 x 10^25 due to extreme overtraining of the 91B model. **For quality-adjusted compute, Config H (PP-DiLoCo 960B)** is optimal at 2.78 x 10^26 C_quality.

### 8.3 Mixture of Experts + Expert Parallelism

MoE with Expert Parallelism (EP) does not increase $C_{\text{local}}$ — total compute is determined by active parameters, not total parameters. However, it enables training a **much larger model** within the same compute budget:

| Config | Total Params | Active Params | Per-Node Memory | Fits (50x A100) | EP Overhead |
|:--|:--|:--|:--|:--|:--|
| Dense 250B | 250B | 250B | 4,000 GB | Yes | — |
| MoE 600B | 600B | 100B | 1,711 GB (at N=72) | Yes | 33.7% |
| MoE 1T | 1,000B | 150B | 2,589 GB (at N=72) | Yes | 24.5% |

EP distributes expert parameters across all nodes, reducing per-node memory from $P_{\text{total}} \cdot \beta$ to $(P_{\text{shared}} + P_{\text{experts}}/N) \cdot \beta$. A 600B MoE model (100B shared, 500B expert) at N=72 requires only 1,711 GB per node — well within the 4,000 GB budget.

The trade-off is a **33.7% compute overhead** from EP All-to-All communication ($T_{\text{EP}} = 2 \times 0.1\text{s} \times 32\text{ layers} = 6.4\text{s}$ per step, added to the 13.1s compute time for 100B active params). This reduces $C_{\text{local}}$ to **6.49 x 10^26** at 4,000 nodes — about 36% less than the dense 250B configuration. However, the resulting 600B MoE model may be more capable per FLOP than a 250B dense model, following the trend established by models like Mixtral and DeepSeek-V2.

### 8.4 Reaching 10^28 FLOP

For completeness, 10^28 FLOP would require approximately:
- **~20,000 nodes** of 16x H100 FP8 (~320,000 H100s, ~$9.5B), or
- **~40,000 nodes** of 50x A100 FP16 (~2,000,000 A100s, ~$30B)

These are resources comparable to a major nation's annual military procurement budget. The straggler factor at N=20,000 is $f(20000) = 1.71$, and $\eta$ remains above 0.84 — the physics of DiLoCo does not prevent scaling to this level, but the economic and logistical requirements make this exclusively a state-actor scenario. Note that at 20,000 replicas, the replica count penalty becomes non-trivial (~1-2%), though this is dwarfed by other uncertainties at this scale.

### 8.5 Summary: Scale and Actor Type

| Compute Scale (C_quality) | Nodes | Cost | Best Mode | Model | Actor Type |
|:--|:--|:--|:--|:--|:--|
| **10^24** (Strict Threshold) | 4 A100 | $3M | Flat DiLoCo | 250B | Research lab, wealthy individual |
| **10^25** (frontier-competitive) | 72 A100 | $54M | Flat DiLoCo | 250B | Well-funded startup, small state |
| **10^26** (C_quality) | 500 A100 | $375M | PP-DiLoCo | 500B | Large corporation, medium state |
| **~3 x 10^26** (C_quality) | 4,000 A100 | $3.0B | PP-DiLoCo | 1000B | Major state actor only |
| **10^27** (C_local) | 2,000-4,000 | $1-3B | Flat/Hier DiLoCo | 91-250B | Major state actor only |

Note the distinction between C_local (raw compute) and C_quality (quality-adjusted). While several configurations achieve 10^27 C_local, their C_quality is only ~10^26 due to severe overtraining. An evader optimizing for model quality rather than raw FLOP would prefer PP-DiLoCo (Config H: 960B, C_quality = 2.78 x 10^26) over flat DiLoCo (Config A: 250B, C_quality = 1.14 x 10^26).

The compute threshold in the treaty (10^24 FLOP) is designed to catch training runs at the **research lab** level and above. Distributed training with DiLoCo makes this threshold porous at the 4-72 node scale ($3-54M), where financial detection is challenging and the pseudo-gradient compression assumptions are well-validated. At the 10^27+ scale, both the treaty's financial monitoring mechanisms and the compression quality uncertainty become relevant — the hardware procurement itself is the hardest step to conceal, and the achievable compute carries wider error bars.

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

**72 nodes, 50x A100 FP16, 16x compression (targeting 10^25):**

| BW (Mbps) | H_min | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 2,077 | 0.804 | 1.71 x 10^25 | 88% |
| 25 | 831 | 0.825 | 1.75 x 10^25 | 91% |
| 50 | 416 | 0.841 | 1.79 x 10^25 | 93% |
| **100** | **208** | **0.857** | **1.82 x 10^25** | **94%** |
| 250 | 84 | 0.878 | 1.87 x 10^25 | 97% |
| 1,000 | 21 | 0.910 | 1.93 x 10^25 | 100% |

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

**Why latency is irrelevant:** The sync time formula is $T_{\text{sync}} = 2 \cdot V_{\text{bits}} / BW + \text{latency}$. For a 250B model with 16x compression at 100 Mbps:

$$V_{\text{bits}} = 250 \times 10^9 \times 16 / 16 = 250 \text{ Gbit}$$
$$T_{\text{sync,bandwidth}} = 2 \times 250 \times 10^9 / (100 \times 10^6) = 5{,}000 \text{ seconds}$$

The bandwidth term is **5,000 seconds**. Even the global worst-case latency of 340 ms is only **0.007%** of the sync time. Latency is negligible compared to the time required to transfer billions of pseudo-gradient values, even with 16x compression.

### 9.4 Combined Deployment Profiles

Realistic network conditions combine both bandwidth and latency. The table below shows results for deployment profiles that an evader might realistically use. All values use the expected compression quality scenario.

**72 nodes, 50x A100 FP16, 16x compression (10^25 target, ~$54M):**

| Deployment | BW | RTT | H | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 21 | 0.910 | 1.93 x 10^25 | 100% |
| Same country (US) | 500 Mbps | 35 ms | 42 | 0.894 | 1.90 x 10^25 | 98% |
| Continental US | 100 Mbps | 65 ms | 208 | 0.857 | 1.82 x 10^25 | 94% |
| Transatlantic | 100 Mbps | 75 ms | 208 | 0.857 | 1.82 x 10^25 | 94% |
| Transpacific | 50 Mbps | 105 ms | 416 | 0.841 | 1.79 x 10^25 | 93% |
| Global adversarial | 25 Mbps | 230 ms | 831 | 0.825 | 1.75 x 10^25 | 91% |
| Global worst-case | 10 Mbps | 340 ms | 2,077 | 0.804 | 1.71 x 10^25 | 89% |

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

**10^26 FLOP under degraded networks (500 nodes, 50x A100 FP16, 16x comp, 1.5 years, expected):**

| Network Conditions | C_local | Achievable? |
|:--|:--|:--|
| 100 Mbps, 100 ms (baseline) | 1.26 x 10^26 | Yes |
| 50 Mbps, 105 ms (transpacific) | 1.24 x 10^26 | Yes |
| 25 Mbps, 230 ms (global) | 1.21 x 10^26 | Yes |
| 10 Mbps, 340 ms (worst-case) | 1.18 x 10^26 | **Yes** |

10^26 FLOP is achievable under **all tested network conditions**, including the global worst-case with expected compression quality. Even under the conservative compression scenario, 500 nodes at 10 Mbps still produce ~1.14 x 10^26 FLOP.

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

**1. The model predicts DiLoCo is bandwidth-insensitive.** The protocol absorbs low bandwidth by increasing the synchronization interval $H$. Because efficiency depends on $\log_{10}(H)$, a 10x increase in $H$ (needed to compensate for 10x lower bandwidth) costs only a constant reduction in $\eta$. Across the full range from 10 Mbps to 1 Gbps, the model predicts C_local varies by only 12% (flat, 16x) to 5% (hierarchical+100x). Including compression quality, the predicted total gap between optimal and worst-case is **12-14%** for 16x and **14-18%** for 100x. However, this prediction relies on extrapolations not yet validated at the target scale: no published distributed training project has operated below ~500 Mbps (INTELLECT-1 used 500 Mbps-4 Gbps), and at 10 Mbps the 72-node scenario requires H≈2,000 inner steps — beyond the H≤1,000 range tested in DiLoCo scaling experiments. The 16x compression stack (FP4 quantization + sparsification) has also not been validated as a combined system at 100B+ scale. The qualitative conclusion — that DiLoCo is far less bandwidth-sensitive than data-parallel training — is robust, but the precise 12-14% figure should be treated as a model estimate rather than an empirical result.

**2. Latency is irrelevant to distributed training.** The sync volumes in frontier-scale training (Gbits to Tbits) make round-trip latency negligible. Even the Brazil-to-Southeast-Asia worst case (340 ms) has zero measurable impact. Geographical distribution is free — an evader needs no special network infrastructure.

**3. Network-level enforcement is ineffective.** Even if bandwidth restrictions were perfectly enforced, the 12-14% reduction in achievable compute is small relative to the order-of-magnitude exceedance of the Strict Threshold. An evader using consumer broadband loses at most 12-14% of compute versus an optimized local deployment — a minor penalty that does not change whether the threshold is exceeded or by how much. The effective enforcement mechanisms remain chip tracking, financial monitoring, and physical detection of GPU concentrations.

## 10. Treaty Modifications to Close the Distributed Training Loophole

The preceding sections establish three facts: (1) the CCC threshold is porous — DiLoCo enables sub-CCC nodes to collectively exceed the Strict Threshold with modest investment; (2) the model predicts network conditions have limited impact — even 10 Mbps consumer broadband is estimated to reduce efficiency by only 12-14%, though this relies on compression and scaling assumptions not yet validated beyond 15B parameters; (3) the existing enforcement mechanisms (chip tracking, financial monitoring) are the primary barriers, not the compute threshold itself. This section evaluates specific treaty modifications that could narrow or close the distributed training loophole.

### 10.1 Lowering the CCC Compute Threshold

The most direct response is to lower the CCC registration threshold from 16 H100-equivalents. The table below shows the maximum node configuration that fits under each threshold (using A100 80GB, the optimal VRAM/compute hardware):

| CCC Threshold | Max GPUs/node | PFLOPS | VRAM | Max Model | H100-equiv |
|:--|:--|:--|:--|:--|:--|
| **16 (current)** | **50x A100** | **15.60** | **4,000 GB** | **250B** | **15.8** |
| 8 | 25x A100 | 7.80 | 2,000 GB | 125B | 7.9 |
| 4 | 12x A100 | 3.74 | 960 GB | 60B | 3.8 |
| 2 | 6x A100 | 1.87 | 480 GB | 30B | 1.9 |
| 1 | 3x A100 | 0.94 | 240 GB | 15B | 0.9 |

**Optimal evasion configuration at each CCC threshold (A100 FP16, 16x compression, 1.5 years, all links 100 Mbps):**

The optimizer searches over flat DiLoCo, hierarchical DiLoCo, and PP-Group DiLoCo (pipeline parallelism over WAN) at various model sizes, targeting C_quality (Chinchilla-adjusted compute) rather than raw FLOP. This penalizes configurations that massively overtrain small models.

| CCC Threshold | C_quality target | Nodes | Cost | Best Mode | Model | OT Ratio |
|:--|:--|:--|:--|:--|:--|:--|
| **16 (current)** | **10^24** | **4** | **$3M** | **Flat DiLoCo** | **125B** | **0.5x** |
| 16 (current) | 10^25 | 41 | $31M | Flat DiLoCo | 250B | 1.3x |
| 16 (current) | 10^26 | 2,424 | $1.8B | PP-DiLoCo (2 stages, 10x act.) | 500B | 4.2x |
| **8** | **10^24** | **8** | **$3M** | **Flat DiLoCo** | **94B** | **0.9x** |
| 8 | 10^25 | 132 | $50M | Hierarchical | 125B | 8.1x |
| 8 | 10^26 | 6,876 | $2.6B | PP-DiLoCo (4 stages, 10x act.) | 500B | 4.7x |
| **4** | **10^24** | **18** | **$3M** | **Flat DiLoCo** | **60B** | **2.3x** |
| 4 | 10^25 | 735 | $132M | PP-DiLoCo (3 stages, 10x act.) | 180B | 2.9x |
| 4 | 10^26 | 26,752 | $4.8B | PP-DiLoCo (8 stages, 10x act.) | 480B | 6.8x |
| **2** | **10^24** | **84** | **$8M** | **Flat DiLoCo** | **30B** | **21.5x** |
| 2 | 10^25 | 2,304 | $207M | PP-DiLoCo (6 stages, 10x act.) | 180B | 3.4x |
| 2 | 10^26 | — | — | Not achievable | — | — |
| **1** | **10^24** | **234** | **$11M** | **PP-DiLoCo (3 stages, 10x act.)** | **45B** | **5.5x** |
| 1 | 10^25 | 9,224 | $415M | PP-DiLoCo (8 stages, 10x act.) | 120B | 15.0x |
| 1 | 10^26 | — | — | Not achievable | — | — |

*Note: configurations marked "10x act." use 10x activation compression, which reduces pipeline communication overhead but has lower empirical confidence than 4x (validated only at 8B via subspace decomposition; see Compression_Quality.md). With the well-validated 4x activation compression, costs at thresholds ≤4 would be 1.2-1.8x higher.*

**The central finding is that lowering the CCC threshold substantially increases the cost of evasion when measured by quality-adjusted compute (C_quality).** Reaching 10^24 C_quality costs $3M at the current threshold but $11M at threshold=1 (4x increase). For 10^25, costs escalate from $31M to $415M (13x increase). For 10^26, evasion becomes infeasible at thresholds below 8 H100-equivalents within 100,000 nodes.

The cost increase has two drivers: (1) **smaller nodes train smaller models**, leading to severe overtraining at high FLOP counts, which the Chinchilla quality adjustment penalizes; and (2) **PP-Group DiLoCo over 100 Mbps WAN** incurs pipeline bubble overhead from activation transfer between stages, requiring more total hardware to compensate. Even with aggressive (10x) activation compression to reduce this overhead, costs still increase substantially at lower thresholds. At threshold=1, even reaching 10^24 C_quality requires 234 nodes using PP-DiLoCo to shard a 45B model across 3 pipeline stages — a far more complex and detectable operation than the 4-node flat DiLoCo configuration at the current threshold.

At lower thresholds, the optimizer selects PP-Group DiLoCo with activation compression to train models larger than single-node VRAM, accepting pipeline overhead and compression quality loss to avoid the worse penalty of extreme overtraining. This forces the evader to coordinate pipeline stages across WAN, which both increases cost and creates a more distinctive operational footprint (tightly synchronized multi-node communication patterns).

**Collateral damage of lowering the threshold:**

| Threshold | Legitimate systems requiring registration |
|:--|:--|
| 16 (current) | 0 of 14 representative systems |
| 8 | 0 (the 8x H100 systems are exactly at 8.0, not over) |
| 4 | 4 systems: AWS p5.48xlarge, DGX H100, Princeton AI cluster, molecular dynamics clusters |
| 2 | 7 systems: adds DGX A100, AI lab servers, AWS p4d |
| 1 | 11 of 14 systems: includes nearly all multi-GPU research workstations |

At a threshold of 4 H100-equiv, every 8x H100 node would require registration and be subject to potential inspection. At threshold=1, even a researcher with three A100s would need to register. This would not ban or confiscate any hardware, but the reporting requirements and inspection obligations would expand the treaty's administrative footprint.

A lower threshold also **increases the surface area for detection**. If an evader must manage 234 nodes using PP-DiLoCo (at threshold=1) rather than 5 larger nodes (at threshold=16), each becomes a point where inspectors could detect training activity. The combination of higher cost, greater operational complexity, and increased detection surface makes threshold reduction a meaningful deterrent.

**Assessment:** Lowering the CCC compute threshold is **moderately effective** — it increases evasion cost (3-13x for 10^24 to 10^25 targets), degrades model quality, forces more complex and detectable training configurations (PP-DiLoCo over WAN), and increases the inspection surface. A reduction to **4-8 H100-equivalents** would substantially constrain evasion while keeping collateral impact manageable.

### 10.2 Adding Memory to the CCC Definition

The current CCC threshold is defined solely by compute capacity. Section 6.5 identified the resulting exploit: an evader can select hardware that maximizes VRAM while staying under the compute threshold (50x A100 80GB = 4,000 GB VRAM at 15.8 H100-equiv), enabling large models (250B) that would be impossible on compute-equivalent H100 nodes (1,280 GB VRAM).

A **memory-based registration threshold** would require CCC registration when aggregate accelerator VRAM exceeds a specified limit.

**Optimal evasion configuration at each VRAM threshold** (quality-adjusted, searching over flat DiLoCo, hierarchical DiLoCo, and PP-Group DiLoCo at various model sizes; all links 100 Mbps / 100 ms):

| VRAM Limit | Target C_quality | Best Mode | Model | Nodes | Cost | OT Ratio | eta |
|:--|:--|:--|:--|:--|:--|:--|:--|
| No limit (current) | 10^24 | Flat DiLoCo | 125B | 4 | $3M | 0.5x | 0.856 |
| No limit (current) | 10^25 | Flat DiLoCo | 250B | 41 | $31M | 1.3x | 0.858 |
| 2 TB | 10^24 | Flat DiLoCo | 94B | 8 | $3M | 0.9x | 0.869 |
| 2 TB | 10^25 | Hier 16x8 | 125B | 132 | $50M | 8.1x | 0.870 |
| 1 TB | 10^24 | Flat DiLoCo | 60B | 18 | $3M | 2.3x | 0.883 |
| 1 TB | 10^25 | PP-DiLoCo (3 stages, act 10x) | 180B | 735 | $132M | 2.9x | 0.821 |
| 512 GB | 10^24 | Flat DiLoCo | 30B | 84 | $8M | 21.5x | 0.892 |
| 512 GB | 10^25 | PP-DiLoCo (6 stages, act 10x) | 180B | 2,304 | $207M | 3.4x | 0.732 |

At 10^25 C_quality, the cost of evasion rises from **$31M** (no limit) to **$50M** (2 TB) to **$132M** (1 TB) to **$207M** (512 GB) — a **4-7x increase** at the most restrictive thresholds. The evader faces a dilemma: flat DiLoCo with a small model suffers severe Chinchilla-suboptimality (e.g., 21.5x overtraining at 30B), while PP-Group DiLoCo over 100 Mbps WAN incurs substantial pipeline bubble overhead and activation compression losses that compound across stage boundaries. The memory threshold **closes the high-VRAM A100 exploit** and forces the evader into more costly and detectable configurations.

**Collateral at each level:** A 1 TB limit catches 0 of 14 representative legitimate systems. A 512 GB limit catches 6 (all 8-GPU A100/H100 systems), bringing standard DGX systems and cloud instances under registration — these are standard equipment at institutions that have the resources to comply.

**Assessment:** A memory threshold of **1-2 TB** is the most effective single countermeasure identified in this analysis. It closes the high-VRAM A100 exploit, forces evasion costs up by 2-4x at 10^25 C_quality, and has zero collateral impact on legitimate systems at the 1 TB level.

### 10.3 Bandwidth Restrictions

Section 9 established that DiLoCo is inherently bandwidth-insensitive. Could the treaty mandate bandwidth restrictions on sub-CCC nodes?

**Scenario: 10 Mbps upload cap on all unregistered nodes.** From Section 9.2 (expected scenario), the impact:

| Configuration | C_local at 100 Mbps | C_local at 10 Mbps | Reduction |
|:--|:--|:--|:--|
| 72 nodes, A100 FP16, flat 16x | 1.75 x 10^25 | 1.64 x 10^25 | 6% |
| 2,000 nodes, H100 FP8, hier+100x | 1.07 x 10^27 | 1.04 x 10^27 | 3% |

The model predicts a 10 Mbps cap reduces evasion capability by only 3-6% from baseline. This estimate assumes compression ratios validated at much smaller scale (≤15B parameters) hold at 250B, and that H≈2,000 inner steps incur only the logarithmic penalty predicted by the efficiency formula. The actual penalty could be larger, but the fundamental dynamic — DiLoCo's logarithmic bandwidth sensitivity, where the evader needs 10x more inner steps but loses only one $\log_{10}$ unit of efficiency — means bandwidth caps alone are unlikely to prevent threshold exceedance.

**Implementation options:** (a) Monitoring WAN connections at registered GPU facilities — lightest touch, provides a detection signal; (b) mandating ISP throttling for GPU facilities — more intrusive, constrains legitimate inference and cloud computing; (c) requiring network registration for AI chip facilities. Option (a) has value as a supplementary detection mechanism (Section 10.4) even though bandwidth *caps* are technically ineffective.

**Assessment:** Bandwidth *caps* are **technically ineffective** — DiLoCo's logarithmic sensitivity to bandwidth means no realistic restriction produces a meaningful reduction in achievable compute. Traffic *monitoring* at GPU facilities has supplementary detection value.

### 10.4 Traffic Fingerprinting and ISP Monitoring

DiLoCo creates a distinctive network signature: pseudo-gradient synchronization at regular intervals, with identical payload sizes. Could treaty-mandated ISP monitoring detect this pattern?

**The sync fingerprint:** At 100 Mbps with a 250B model and 16x compression, each sync transfers ~31 GB per direction. With streaming DiLoCo, this is spread over ~100 minutes (~39 Mbps average). The traffic is continuous (streaming eliminates bursts), bidirectional (~78 Mbps total), encrypted via VPN, and has a regular cadence.

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
| 4 nodes (50x A100 each) | ~60-80 kW | **Low** — within normal commercial range |
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
