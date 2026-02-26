# Governance Analysis: Distributed Training Below the Registration Threshold

## 1. Scenario

The [MIRI treaty proposal](https://arxiv.org/abs/2511.10783) defines a **covered chip cluster (CCC)** as any set of AI chips with aggregate compute capacity exceeding **16 H100-equivalents** (15,840 TFLOPS FP16). All CCCs must be registered and monitored. Training runs above **10^24 FLOP** (the "Strict Threshold") are prohibited.

This analysis examines a specific evasion scenario: an actor operates multiple unregistered nodes, each below the CCC compute threshold, connected via ordinary internet (100 Mbps WAN) and running the DiLoCo distributed training protocol. The treaty explicitly captures such distributed training: the CCC definition includes clusters "networked to perform workloads together," regardless of interconnect bandwidth. However, **detecting** such a run is much harder than detecting a traditional data center, because each individual node is indistinguishable from legitimate sub-threshold compute use.

**Question:** How much local-equivalent compute can the evader achieve in the maximum training duration?

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

The **48x A100 80GB** configuration is optimal: nearly the same compute as 16x H100 but **3x the VRAM**, enabling a 240B-parameter model to fit entirely on one node without model parallelism. This avoids the severe latency penalties of pipeline parallelism over WAN.

## 4. Training Protocol and Compression Assumptions

### 4.1 Protocol Configuration

Each node runs as an independent DiLoCo worker:

| Parameter | Value | Rationale |
|:--|:--|:--|
| Mode | Streaming DiLoCo | Overlaps communication with compute |
| Inner Steps (H) | ~168-229 (optimized per N) | Minimum for compute-bound operation |
| Compression | 16x (default) or 100x (aggressive) | 4-bit quantization + sparsification |
| Local Batch | 131,072 tokens | 32 sequences x 4,096 seq length |
| Model | 240B dense (on 48x A100) | Largest model that fits in memory |
| Precision | BF16/FP16 | Standard mixed-precision training |
| MFU | 40% | Empirically supported for distributed training |

**Key insight:** The minimum inner steps $H$ to keep the system compute-bound is independent of model size. Both the sync time and the compute time per step scale linearly with $P$ (parameters), so $P$ cancels in the ratio:

$$H_{\min} = \left\lceil \frac{T_{\text{sync}}}{T_{\text{comp}}} \right\rceil = \left\lceil \frac{2 \cdot P \cdot 16 / (C_r \cdot BW) \cdot f(N)}{6 \cdot P \cdot B / (\text{PFLOPS} \cdot \text{MFU})} \right\rceil \approx 152 \cdot f(N)$$

### 4.2 Efficiency Model

The simulator computes local-equivalent FLOPs as $C_{\text{local}} = N \cdot \text{FLOPS}_{\text{eff}} \cdot T_{\text{wall}} \cdot \eta$, where $\eta$ is a combined efficiency factor with three components:

$$\eta = \eta_H \times \eta_{\text{compression}} \times \eta_{\text{replicas}}$$

**Sync interval penalty ($\eta_H$):** The primary efficiency loss from using large H (many inner steps between synchronization). More inner steps cause replicas to diverge further from each other, reducing the quality of the averaged pseudo-gradient:

$$\eta_H = \max\!\left(0.4,\; 1 - \alpha \cdot \log_{10}(H)\right), \quad \alpha = \frac{0.08}{1 + \log_{10}(P/10^9) / 5}$$

The $\alpha$ coefficient decreases with model size, reflecting the empirical finding from the [DiLoCo Scaling Laws](https://arxiv.org/abs/2503.09799) paper that larger models are more robust to infrequent synchronization. This is the dominant efficiency factor, accounting for 8-16% loss depending on H and model size.

**Compression quality ($\eta_{\text{compression}}$):** A multiplicative penalty from quantizing and sparsifying the pseudo-gradients before transmission. This factor is parameterized by compression ratio and estimated from the literature under three scenarios:

| Compression | Optimistic | Expected | Conservative | Evidence |
|:--|:--|:--|:--|:--|
| 1x (none) | 1.00 | 1.00 | 1.00 | Baseline |
| 4x (FP4 only) | 1.00 | 1.00 | 0.99 | Lossless at 4B ([Streaming DiLoCo](https://arxiv.org/abs/2501.18512)) and 15B ([MuLoCo](https://arxiv.org/abs/2505.23725)) |
| 16x (FP4 + 4x sparse) | 1.00 | 0.98 | 0.95 | FP4 component validated; sparsification at 25% tested at 512M ([SparseLoCo](https://arxiv.org/abs/2508.15706)) |
| 100x (2-bit + TopK or FP4 + 25x) | 0.99 | 0.95 | 0.90 | Validated only at 512M-1B; significant extrapolation to 100B+ |

The "expected" scenario is used as the primary estimate throughout this analysis, with optimistic/conservative ranges noted where they materially affect conclusions. See Section 11 for the full literature review underlying these estimates.

**Replica count penalty ($\eta_{\text{replicas}}$):** Averaging pseudo-gradients across many replicas introduces noise. Based on the [DiLoCo Scaling Laws](https://arxiv.org/abs/2503.09799) empirical data (M=8 costs ~1.2% at 2.4B parameters, with the penalty decreasing at larger model sizes), this factor is modeled as:

$$\eta_{\text{replicas}} = \max\!\left(0.85,\; 1 - 0.005 \cdot \frac{\min(2.4, P_B)}{P_B} \cdot \log_2(N)\right)$$

where $P_B$ is the model size in billions and $N$ is the number of replicas. For the primary 240B configuration, this penalty is negligible (<0.1% at 72 nodes, <0.3% at 500 nodes) because the denominator scales with model size. It becomes material only for very small models at very large replica counts.

### 4.3 Uncertainty and Validation Status

The simulator's predictions carry different levels of confidence depending on the configuration:

| Configuration | Confidence | Basis |
|:--|:--|:--|
| 16x compression, 4-72 nodes | **High** | FP4 validated lossless at 15B; DiLoCo tested at 10B; replica counts modest |
| 16x compression, 500+ nodes | **Medium** | Compression well-validated; replica count extrapolated (largest test: M=16 at 15B) |
| 100x compression, any scale | **Low-Medium** | Only validated at 512M-1B; requires error feedback (not in all implementations); significant extrapolation |
| 2000+ nodes, any compression | **Low** | No empirical data at this replica count; Epoch AI projects ~6x FLOP penalty at 10,000 nodes |

## 5. Results

### 5.1 Primary Configuration: 48x A100 80GB (240B model)

All results use the **expected** compression quality scenario (Section 4.2). Optimistic values (no compression penalty) and conservative values are shown in parentheses where they differ materially.

| Nodes | GPUs | Est. Cost | Inner Steps H | $\eta$ | C_local (FLOP) | x Strict Threshold |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | 48 | $0.7M | 1 (no DiLoCo) | 1.000 | 2.84 x 10^23 | 0.3x |
| 4 | 192 | $2.9M | 168 | 0.862 | **9.77 x 10^23** | **1.0x** |
| 8 | 384 | $5.8M | 176 | 0.861 | 1.95 x 10^24 | 2.0x |
| 16 | 768 | $12M | 183 | 0.860 | 3.90 x 10^24 | 3.9x |
| 32 | 1,536 | $23M | 191 | 0.859 | 7.79 x 10^24 | 7.8x |
| 72 | 3,456 | $52M | 200 | 0.858 | **1.75 x 10^25** | **17.5x** |
| 144 | 6,912 | $104M | 207 | 0.857 | 3.50 x 10^25 | 35.0x |
| 500 | 24,000 | $360M | 221 | 0.855 | 1.21 x 10^26 | 121.2x |

The 72-node reference point shows $\eta = 0.858$ under the expected scenario (optimistic: 0.875; conservative: 0.831). The corresponding C_local range is **1.70-1.79 x 10^25 FLOP** — the conclusion that 72 nodes exceeds the Strict Threshold by ~17x is robust across all compression quality assumptions.

The **algorithmic efficiency** ($\eta$) is stable at 85-86% across all node counts. Under the optimistic scenario (no compression quality penalty), efficiency would be 87-88%. The 2% difference reflects the expected cost of 16x gradient compression extrapolated to 240B scale.

### 5.2 Model Quality Analysis

The evader trains a **240B-parameter dense model**. At various node counts:

| Nodes | Total Tokens | Chinchilla Tokens (240B) | Overtraining Ratio | Assessment |
|:--|:--|:--|:--|:--|
| 4 | 0.8T | 4.8T | 0.2x | **Under-trained** (would use a smaller model) |
| 16 | 3.2T | 4.8T | 0.7x | Near Chinchilla-optimal |
| 32 | 6.3T | 4.8T | 1.3x | Mildly overtrained |
| 72 | 14.2T | 4.8T | 3.0x | **Moderately overtrained** (comparable to LLaMA-3) |
| 144 | 28.4T | 4.8T | 5.9x | Overtrained (diminishing returns) |

At the **72-node** reference point: 240B params trained on 14.2T tokens with 3.0x overtraining is a highly effective training configuration, comparable to how production models like LLaMA-3 are trained (which used ~10x overtraining).

For small node counts (N < 16), the evader would train a proportionally smaller model to stay near Chinchilla-optimal. For example, at N=4 with C_local = 10^24 FLOP, the Chinchilla-optimal model is ~29B params on ~0.6T tokens, which fits easily within the 240B memory budget.

### 5.3 Comparison to Existing 10^25 Baseline

The [existing baseline scenario](Past%20MIRI%20work/10%5E25%20FLOP%20decentralized%20training%20configuration.md) used 72 nodes of 16x GH200 (32 PFLOPS/node, 2,304 GB) to train a 144B model on 12T tokens (1.02 x 10^25 FLOP) in 331 days. It was communication-bound.

This analysis differs in three ways:
1. **Lower compute per node** (15 vs 32 PFLOPS) — stays under the CCC threshold
2. **Higher memory per node** (3,840 vs 2,304 GB) — enables a larger 240B model
3. **Longer time** (548 vs 331 days) — the treaty scenario allows 1.5 years

Result: even with half the per-node compute, the longer time window and compute-bound operation yield **1.75 x 10^25 local-equivalent FLOP** (expected) — comparable to the baseline's 1.02 x 10^25 raw compute.

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

### 6.5 Hardware Optimization Defeats the Memory Constraint

A naive analysis might assume the evader is limited to 16 H100s (80B model). But the CCC threshold is on compute, not memory. By choosing high-memory, lower-compute GPUs (48x A100 80GB), the evader fits a **240B model** on each node — close to the ~400B Chinchilla-optimal model for this compute budget. Pipeline parallelism across WAN nodes is impractical (activation transfers take ~47 seconds per micro-batch at 100 Mbps), but this hardware optimization largely eliminates the need for it.

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
| **Reference scenario (72 nodes)** | **1.75 x 10^25 FLOP** (17.5x threshold, expected) |
| **Optimistic/conservative range (72 nodes)** | 1.70-1.79 x 10^25 FLOP (17.0-17.9x) |
| **Model trained (72 nodes)** | 240B dense, 14.2T tokens, 3x overtraining |
| **Algorithmic efficiency** | ~86% expected (range: 83-88%) |
| **Confidence in 16x compression** | High (FP4 validated to 15B; compression penalty ~2% expected) |
| **Bottleneck** | Network bandwidth at low H; compute-bound when H is optimized |
| **Detection** | Challenging: low-bandwidth encrypted traffic, no physical signature |
| **Enforcement time pressure** | 6 months still yields 5.8x threshold at 72 nodes |
| **Financial barrier** | $3M (easy to hide) to $52M (moderate) to $3B (state-actor) |

The simulator predicts that the treaty's FLOP threshold can be significantly exceeded using distributed training with unregistered sub-CCC nodes. The primary defense against this evasion is not the compute threshold itself, but rather the treaty's chip tracking and consolidation provisions — the ability to prevent actors from accumulating sufficient GPU hardware in the first place.

## 8. Scaling to 10^27 FLOP and Beyond

The analysis in Sections 5-7 focuses on modest-scale evasion (4-500 nodes). This section explores whether much larger training runs — on the order of **10^27 FLOP** — are achievable using all techniques supported by the simulator: hierarchical DiLoCo, FP8 precision, aggressive gradient compression, and Mixture-of-Experts with Expert Parallelism.

### 8.1 Configurations

Six configurations are compared, all using sub-CCC nodes over 100 Mbps WAN. All values use the **expected** compression quality scenario:

| Config | Hardware | Nodes | GPUs | Cost | Model | $\eta$ | C_local (FLOP) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **A** | 48x A100 FP16, flat DiLoCo, 16x comp | 4,000 | 192,000 | $2.9B | 240B dense | 0.853 | 9.67 x 10^26 |
| **B** | 48x A100 FP16, hierarchical, 16x comp | 4,000 | 192,000 | $2.9B | 240B dense | 0.883 | 1.00 x 10^27 |
| **C** | 16x H100 FP8, flat DiLoCo, 16x comp | 2,000 | 32,000 | $960M | 91B dense | 0.844 | 1.01 x 10^27 |
| **D** | 16x H100 FP8, hierarchical, 16x comp | 2,000 | 32,000 | $960M | 91B dense | 0.876 | 1.05 x 10^27 |
| **E** | 48x A100 FP16, flat DiLoCo, 100x comp | 4,000 | 192,000 | $2.9B | 240B dense | 0.868 | 9.84 x 10^26 |
| **F** | 16x H100 FP8, hier + 100x comp | 2,000 | 32,000 | $960M | 91B dense | 0.892 | **1.07 x 10^27** |

Hierarchical configurations use groups of 8 nodes with 1 Gbps regional interconnect and 20 ms latency (co-located nodes within the same facility or metro area). The 100x compression ratio corresponds to FP4 pseudo-gradient quantization combined with aggressive sparsification, as demonstrated in [Streaming DiLoCo](https://arxiv.org/abs/2501.18512).

**Compression quality uncertainty for 100x configurations (E, F):** Under the optimistic scenario, Config F achieves 1.12 x 10^27; under the conservative scenario, it drops to 1.01 x 10^27. The 100x compression assumptions carry significant uncertainty (see Section 11), so these estimates should be treated as bounds rather than point predictions.

### 8.2 Key Findings

**10^27 FLOP requires 2,000-4,000 sub-CCC nodes ($1-3B in hardware).** This is unambiguously a state-actor-level investment. No non-state actor could plausibly acquire 32,000-192,000 GPUs without detection through the treaty's financial monitoring and chip tracking provisions (Section 6.8).

**FP8 precision is the most cost-effective approach.** The CCC threshold is defined by FP16 computational capacity, but FP8 compute yields **2x throughput** from the same hardware, and FP8 pseudo-gradients are half the size, further reducing communication overhead. A node of 16x H100 SXM has 15,840 TFLOPS FP16 (exactly 16 H100-equivalents) but produces 31,680 TFLOPS of effective FP8 compute and transmits pseudo-gradients at 8 bits instead of 16. Config C achieves 10^27 FLOP with only 2,000 nodes and 32,000 H100s ($960M) — 3x fewer GPUs and 3x lower cost than the A100-based configs. The trade-off is a smaller model (91B vs 240B) due to lower VRAM per H100 node (1,280 GB at $\beta = 14$ bytes/param for FP8).

**Hierarchical DiLoCo improves efficiency by 3 percentage points.** At N=4,000, flat DiLoCo achieves $\eta = 0.853$ with $H_{\min} = 244$, while hierarchical DiLoCo achieves $\eta = 0.883$ with $H_{\text{eff}} = 65$. The improvement comes from regional syncs over 1 Gbps LAN, which keep $H_{\text{inner}}$ low (18 steps) while the global sync interval is hidden behind regional cycles.

**100x compression reduces $H_{\min}$ dramatically but carries more uncertainty.** With 100x compression (vs 16x baseline), sync volume drops from 240 Gbit to 38 Gbit per direction, reducing $H_{\min}$ from 244 to 39 at N=4,000. This improves $\eta_H$ from 0.871 to 0.914. However, the compression quality factor ($\eta_{\text{compression}} = 0.95$ expected, vs 0.98 for 16x) partially offsets this gain. The net expected efficiency for Config E is 0.868 — an improvement over Config A (0.853) but not as dramatic as the $\eta_H$ improvement alone would suggest. The 100x compression assumption is based on limited empirical evidence (see Section 11) and represents the least certain component of the analysis.

**The combined best case (Config F)** uses hierarchical DiLoCo, 100x compression, and FP8 H100 nodes to achieve **1.07 x 10^27 FLOP** (expected). This configuration achieves $\eta = 0.892$ — the highest efficiency of any scenario — by minimizing the effective synchronization interval ($H_{\text{eff}} = 11$) while applying both hierarchical and compression benefits.

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

| Compute Scale | Nodes (A100) | Cost | Actor Type | Compression Confidence |
|:--|:--|:--|:--|:--|
| **10^24** (Strict Threshold) | 4 | $3M | Research lab, wealthy individual | High (16x) |
| **10^25** (frontier-competitive) | 72 | $52M | Well-funded startup, small state | High (16x) |
| **10^26** | 500 | $360M | Large corporation, medium state | High (16x) |
| **10^27** | 2,000-4,000 | $1-3B | Major state actor only | Medium (16x) / Low-Medium (100x) |
| **10^28** | 20,000-40,000 | $10-29B | Superpower-level resources | Low |

The compute threshold in the treaty (10^24 FLOP) is designed to catch training runs at the **research lab** level and above. Distributed training with DiLoCo makes this threshold porous at the 4-72 node scale ($3-52M), where financial detection is challenging and the compression assumptions are well-validated. At the 10^27+ scale, both the treaty's financial monitoring mechanisms and the compression quality uncertainty become relevant — the hardware procurement itself is the hardest step to conceal, and the achievable compute carries wider error bars.

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

The impact is on **model quality**, not cost. A 3x A100 node (under 1 H100-eq threshold) can only hold a 15B model, compared to 240B at the current threshold. This forces the evader to either (a) train a much smaller model, or (b) use pipeline parallelism over WAN to shard a larger model across nodes — which incurs severe latency penalties (Section 6.5 estimated ~47 seconds per micro-batch for activation transfers at 100 Mbps). Neither option is attractive, but a 15B model trained on ~3.5T tokens (at 10^24 FLOP) is still a meaningful AI system.

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

### 10.9 Synthesis: Recommended Treaty Modifications

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

Note: "Registration Burden" refers to the registration, reporting, and inspection obligations imposed on legitimate computing systems — not bans or confiscation. Systems caught by lower thresholds must be registered and may be subject to inspections, but continue to operate normally.

**Against state actors ($1B+, 2,000+ nodes):**

| Countermeasure | Effectiveness | Notes |
|:--|:--|:--|
| Lower CCC threshold | None | Same cost, more nodes |
| TEE/Remote attestation | Low | State actors can manufacture unattested chips |
| Enhanced chip tracking | Low | Classified procurement, domestic fabs |
| Financial monitoring | **Medium** | Detectable above ~$1B for most states |
| Diplomatic/intelligence | **Medium-High** | The only effective lever at state scale |

**Recommended package:** (1) 1 TB VRAM threshold; (2) TEE attestation mandate; (3) model possession redefinition; (4) enhanced whistleblower bounties; (5) utilization reporting; (6) consider lowering CCC threshold to 4-8 H100-eq; (7) traffic monitoring (not caps) at GPU facilities.

**The hard truth:** Even with all recommended modifications, the distributed training loophole cannot be fully closed against a determined state actor with domestic chip manufacturing, classified procurement, and sovereign territory. Against such actors, the treaty's enforcement ultimately depends on diplomatic, intelligence, and economic instruments — the same tools used in nuclear nonproliferation, with the same fundamental limitations.

## 11. Compression Quality: Evidence, Extrapolation, and Open Questions

The efficiency model in Section 4.2 includes a compression quality factor that reduces C_local by 2-10% depending on the compression ratio and scenario. This section presents the evidence underlying those estimates, identifies what is validated versus extrapolated, and quantifies the remaining uncertainty.

### 11.1 What the Simulator Assumes

The simulator's compression quality model applies three multiplicative factors to efficiency:

$$\eta = \eta_H \times \eta_{\text{compression}} \times \eta_{\text{replicas}}$$

The sync interval penalty ($\eta_H$) is the dominant term and is well-calibrated against empirical data (DiLoCo Scaling Laws, 2503.09799). The compression quality ($\eta_{\text{compression}}$) and replica penalty ($\eta_{\text{replicas}}$) are estimated from the literature with varying levels of confidence.

**What the simulator does NOT model:**
- Error feedback (the mechanism for accumulating compression residuals) — whether it is used or needed
- Interaction effects between compression, replica count, and H
- Compression-induced outlier accumulation over long training runs
- The specific choice of compressor (TopK vs random-K, linear vs statistical quantization)

### 11.2 Empirical Evidence by Compression Method

| Paper | Scale | Compression | Quality Impact | Notes |
|:--|:--|:--|:--|:--|
| [Streaming DiLoCo](https://arxiv.org/abs/2501.18512) (DeepMind, 2025) | 500M-4B, M=2, H=100 | FP4 (E3M0) pseudo-grads | **None detected** | No error feedback; 400x total bandwidth reduction |
| [DiLoCo Scaling Laws](https://arxiv.org/abs/2503.09799) (DeepMind, 2025) | 35M-10B, M=1-8 | None (tests H only) | M=8 at 2.4B: +1.2% loss | Penalty decreases with model size |
| [MuLoCo](https://arxiv.org/abs/2505.23725) (2025) | 150M-15B, K=8-16 | 8-bit, 4-bit, 2-bit | 4-bit: lossless; 2-bit+EF: near-lossless | Error feedback critical at 2-bit |
| [SparseLoCo](https://arxiv.org/abs/2508.15706) (2025) | 512M-2B, R=8-16 | TopK 3% + 2-bit (~50-100x) | **Beats vanilla DiLoCo** at 3% density | Error feedback essential; regularizing effect |
| [INTELLECT-1](https://arxiv.org/abs/2412.01152) (Prime Intellect, 2024) | 10B, 14 nodes | int8 pseudo-grads (400x total) | Negligible | Real-world WAN validation |
| [DiLoCoX](https://arxiv.org/abs/2506.21263) (0G Labs, 2025) | 107B, 20 nodes | Low-rank + int4 | 0.3 loss gap vs AllReduce | First 100B+ DiLoCo experiment |
| [Deep Gradient Compression](https://arxiv.org/abs/1712.01887) (Lin, 2018) | ResNet-50, DeepSpeech | Up to 600x | Lossless with error feedback | Vision/speech models, not LLMs |
| [Aragani et al.](https://arxiv.org/abs/2502.07634) (2025) | LSTMs, transformers | TopK/DGC up to 5000x | 50x can improve via regularization; >5000x degrades | |

### 11.3 What Is Validated vs. Extrapolated

**Well-validated (high confidence):**
- FP4 (4-bit) pseudo-gradient quantization is lossless at up to 4B parameters with H=100 and 2 replicas (Streaming DiLoCo)
- 4-bit quantization is lossless at up to 15B parameters with 16 replicas (MuLoCo)
- DiLoCo's efficiency penalty decreases with model size (confirmed 35M-10B, DiLoCo Scaling Laws)
- int8 compression works in practice over real WAN at 10B (INTELLECT-1)

**Partially validated (medium confidence):**
- 16x compression (FP4 + 4x sparsification): the FP4 component is well-validated; the sparsification component is tested at 512M (SparseLoCo) but not at 100B+
- DiLoCo at 100B+ scale: DiLoCoX demonstrates feasibility but shows a 0.3 loss gap versus AllReduce

**Extrapolated (low-medium confidence):**
- 100x compression at 100B+ scale: only validated at 512M-1B; extrapolation spans ~100x in model size
- 2000+ replicas: largest empirical test is M=16 (MuLoCo); [Epoch AI projects](https://epoch.ai/gradient-updates/how-far-can-decentralized-training-over-the-internet-scale) that 10,000 nodes would require ~6x FLOP for equivalent quality
- H=200-2000 combined with aggressive compression: largest tested is H=125 with compression (DiLoCoX at 107B)

### 11.4 Key Risk Factors

**Error feedback:** At 2-bit compression and below, error feedback (accumulating the difference between the original and compressed value for the next sync) is critical for convergence. MuLoCo and SparseLoCo both find that error feedback is essential at aggressive compression ratios. However, the Streaming DiLoCo paper does **not** use error feedback, and its lossless results at FP4 may not extend to more aggressive compression without it. Error feedback adds memory overhead (one full model copy for the error buffer — ~3.8 TB additional for a 240B model) and is not included in the simulator's memory model.

**Compounding unknowns:** The scenarios in Section 8 combine multiple factors that have not been tested jointly: (a) large models (91-240B), (b) many replicas (2,000+), (c) high H values (200-2000), and (d) aggressive compression (100x). Each factor has been studied somewhat independently, but the **interaction** is unknown. Even if each factor individually causes <5% degradation, compounding could produce larger effects.

**Scale-dependent outliers:** Quantization during training is known to cause outlier accumulation over long training runs. At 100B+ scale with trillions of tokens, this could be worse than at 1B scale with billions of tokens. Recent work on FP4 training ([Quartet](https://arxiv.org/abs/2505.14669), 2025) shows that most loss gap arises from forward-pass quantization, not gradient quantization — but this applies to per-step gradients, not DiLoCo pseudo-gradients.

### 11.5 Impact on Analysis Conclusions

| Conclusion | Sensitivity to Compression Quality | Robust? |
|:--|:--|:--|
| 4 nodes exceed 10^24 (16x comp) | Low — conservative scenario still gives 9.52e23 | **Yes** |
| 72 nodes achieve ~17x threshold (16x comp) | Low — range is 17.0-17.9x | **Yes** |
| 500 nodes achieve 10^26 (16x comp) | Low — conservative gives 1.18e26 | **Yes** |
| Config F achieves 10^27 (100x comp) | **Medium** — expected 1.07e27, conservative 1.01e27 | **Marginal** |
| 10^27 at 10 Mbps + 100x (Config F) | **High** — conservative drops to ~9.8e26 | **Uncertain** |
| Bandwidth reduction costs "only 3-6%" | **Medium** — actual total gap is 12-14% with compression quality | **Revised** |
| Cost of evasion is $3M for 10^24 | None — cost is hardware, not compression-dependent | **Yes** |
| Treaty modifications analysis | None — countermeasure effectiveness is independent | **Yes** |

The most important revision is to the bandwidth sensitivity headline: the frequently cited "3-6% reduction" referred only to the $\eta_H$ penalty from larger H values, not the total efficiency gap including compression quality. With compression quality included, the total expected gap between optimal (1 Gbps, no compression penalty) and degraded (10 Mbps, expected compression quality) is **12-14%** for 16x compression and **14-18%** for 100x. The qualitative conclusion — that DiLoCo is robust to bandwidth constraints — remains valid, but the quantitative magnitude is larger.

The core governance conclusions (Sections 6-7) are robust across all compression quality scenarios because they rely on 16x compression, which is well-validated. The 10^27 scenarios (Section 8) carry meaningful uncertainty, particularly Config F's 100x compression assumption, and should be interpreted as estimates with a +-10% error bar rather than precise predictions.
