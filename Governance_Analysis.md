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

## 4. Training Protocol

Each node runs as an independent DiLoCo worker:

| Parameter | Value | Rationale |
|:--|:--|:--|
| Mode | Streaming DiLoCo | Overlaps communication with compute |
| Inner Steps (H) | ~168-229 (optimized per N) | Minimum for compute-bound operation |
| Compression | 16x | 4-bit quantization + 25% sparsification |
| Local Batch | 131,072 tokens | 32 sequences x 4,096 seq length |
| Model | 240B dense (on 48x A100) | Largest model that fits in memory |
| Precision | BF16/FP16 | Standard mixed-precision training |
| MFU | 40% | Empirically supported for distributed training |

**Key insight:** The minimum inner steps $H$ to keep the system compute-bound is independent of model size. Both the sync time and the compute time per step scale linearly with $P$ (parameters), so $P$ cancels in the ratio:

$$H_{\min} = \left\lceil \frac{T_{\text{sync}}}{T_{\text{comp}}} \right\rceil = \left\lceil \frac{2 \cdot P \cdot 16 / (C_r \cdot BW) \cdot f(N)}{6 \cdot P \cdot B / (\text{PFLOPS} \cdot \text{MFU})} \right\rceil \approx 152 \cdot f(N)$$

## 5. Results

### 5.1 Primary Configuration: 48x A100 80GB (240B model)

| Nodes | GPUs | Est. Cost | Straggler f(N) | Inner Steps H | Efficiency $\eta$ | C_local (FLOP) | x Strict Threshold |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | 48 | $0.7M | 1.000 | 1 (no DiLoCo) | 1.000 | 2.84 x 10^23 | 0.3x |
| 4 | 192 | $2.9M | 1.100 | 168 | 0.879 | **9.97 x 10^23** | **1.0x** |
| 8 | 384 | $5.8M | 1.150 | 176 | 0.878 | 1.99 x 10^24 | 2.0x |
| 16 | 768 | $12M | 1.200 | 183 | 0.877 | 3.98 x 10^24 | 4.0x |
| 32 | 1,536 | $23M | 1.250 | 191 | 0.876 | 7.95 x 10^24 | 8.0x |
| 72 | 3,456 | $52M | 1.308 | 200 | 0.875 | **1.79 x 10^25** | **17.9x** |
| 144 | 6,912 | $104M | 1.358 | 207 | 0.874 | 3.57 x 10^25 | 35.7x |
| 500 | 24,000 | $360M | 1.448 | 221 | 0.873 | 1.24 x 10^26 | 123.8x |

The **algorithmic efficiency** ($\eta$) remains remarkably stable at 87-88% across all node counts. This is because $\eta$ depends on $\log_{10}(H)$, and $H_{\min}$ varies only from 168 to 221 as nodes scale from 4 to 500.

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

Result: even with half the per-node compute, the longer time window and compute-bound operation yield **1.79 x 10^25 local-equivalent FLOP** — comparable to the baseline's 1.02 x 10^25 raw compute.

## 6. Governance Implications

### 6.1 The Threshold is Easily Exceeded

**As few as 4 unregistered nodes** (~200 A100 GPUs, ~$3M) are sufficient to match the 10^24 FLOP Strict Threshold in 1.5 years. This is a modest investment — comparable to a single research lab's annual GPU budget. At 72 nodes ($52M), the evader achieves nearly **18x** the threshold, training a 240B-parameter model that would be competitive with frontier models.

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

The local-equivalent compute scales almost perfectly linearly with the number of nodes. The DiLoCo algorithmic efficiency ($\eta \approx 87\%$) and the straggler overhead ($f(N) \approx 1.1-1.5$) are both modest and slow-varying. A well-funded actor with access to 500+ nodes could achieve over **100x** the Strict Threshold.

### 6.7 Enforcement Time Pressure

The 1.5-year maximum training duration (Section 2) is a theoretical optimum based on technology growth rates. In practice, the treaty's enforcement mechanisms create pressure to train *faster* than the theoretical maximum:

- **Whistleblower programs:** The treaty establishes financial incentives for reporting violations. A 1.5-year operation requires maintaining operational security across dozens or hundreds of sites for an extended period. Every employee, contractor, facility operator, and GPU supplier who becomes aware of the operation is a potential whistleblower. The cumulative probability of a leak increases with time.
- **International inspections:** Treaty signatories conduct both scheduled and triggered inspections of compute facilities. Over 1.5 years, each node site faces multiple inspection cycles. Even if individual inspections have a low detection probability, the compound probability over 18 months across many sites becomes significant.
- **Operational security failures:** Hardware failures requiring replacement, network outages, power interruptions, and personnel turnover all create detection opportunities. A 48-GPU cluster operating at near-full utilization for 18 months will experience multiple hardware failures requiring maintenance or procurement.

The practical effect: an evader would likely target **6-12 months** rather than the full 1.5 years, accepting lower total compute in exchange for reduced detection risk. Since $C_{\text{local}}$ scales linearly with time, this reduces the achievable compute proportionally:

| Duration | C_local (72 nodes) | x Strict Threshold |
|:--|:--|:--|
| 6 months | 5.96 x 10^24 | 6.0x |
| 1 year | 1.19 x 10^25 | 11.9x |
| 1.5 years | 1.79 x 10^25 | 17.9x |

Even at 6 months, the 72-node configuration still exceeds the Strict Threshold by **6x** — sufficient to train a meaningful model (240B dense, ~4.7T tokens).

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
| **Reference scenario (72 nodes)** | **1.79 x 10^25 FLOP** (18x threshold) |
| **Model trained (72 nodes)** | 240B dense, 14.2T tokens, 3x overtraining |
| **Algorithmic efficiency** | ~87% (stable across all node counts) |
| **Bottleneck** | Network bandwidth at low H; compute-bound when H is optimized |
| **Detection** | Challenging: low-bandwidth encrypted traffic, no physical signature |
| **Enforcement time pressure** | 6 months still yields 6x threshold at 72 nodes |
| **Financial barrier** | $3M (easy to hide) to $52M (moderate) to $3B (state-actor) |

The simulator predicts that the treaty's FLOP threshold can be significantly exceeded using distributed training with unregistered sub-CCC nodes. The primary defense against this evasion is not the compute threshold itself, but rather the treaty's chip tracking and consolidation provisions — the ability to prevent actors from accumulating sufficient GPU hardware in the first place.

## 8. Scaling to 10^27 FLOP and Beyond

The analysis in Sections 5-7 focuses on modest-scale evasion (4-500 nodes). This section explores whether much larger training runs — on the order of **10^27 FLOP** — are achievable using all techniques supported by the simulator: hierarchical DiLoCo, FP8 precision, aggressive gradient compression, and Mixture-of-Experts with Expert Parallelism.

### 8.1 Configurations

Six configurations are compared, all using sub-CCC nodes over 100 Mbps WAN:

| Config | Hardware | Nodes | GPUs | Cost | Model | $\eta$ | C_local (FLOP) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **A** | 48x A100 FP16, flat DiLoCo, 16x comp | 4,000 | 192,000 | $2.9B | 240B dense | 0.871 | 9.87 x 10^26 |
| **B** | 48x A100 FP16, hierarchical, 16x comp | 4,000 | 192,000 | $2.9B | 240B dense | 0.902 | 1.02 x 10^27 |
| **C** | 16x H100 FP8, flat DiLoCo, 16x comp | 2,000 | 32,000 | $960M | 91B dense | 0.862 | 1.03 x 10^27 |
| **D** | 16x H100 FP8, hierarchical, 16x comp | 2,000 | 32,000 | $960M | 91B dense | 0.896 | 1.07 x 10^27 |
| **E** | 48x A100 FP16, flat DiLoCo, 100x comp | 4,000 | 192,000 | $2.9B | 240B dense | 0.914 | 1.04 x 10^27 |
| **F** | 16x H100 FP8, hier + 100x comp | 2,000 | 32,000 | $960M | 91B dense | 0.941 | **1.13 x 10^27** |

Hierarchical configurations use groups of 8 nodes with 1 Gbps regional interconnect and 20 ms latency (co-located nodes within the same facility or metro area). The 100x compression ratio corresponds to FP4 pseudo-gradient quantization combined with aggressive sparsification, as demonstrated in [Streaming DiLoCo](https://arxiv.org/abs/2501.18512).

### 8.2 Key Findings

**10^27 FLOP requires 2,000-4,000 sub-CCC nodes ($1-3B in hardware).** This is unambiguously a state-actor-level investment. No non-state actor could plausibly acquire 32,000-192,000 GPUs without detection through the treaty's financial monitoring and chip tracking provisions (Section 6.8).

**FP8 precision is the most cost-effective approach.** The CCC threshold is defined by FP16 computational capacity, but FP8 compute yields **2x throughput** from the same hardware, and FP8 pseudo-gradients are half the size, further reducing communication overhead. A node of 16x H100 SXM has 15,840 TFLOPS FP16 (exactly 16 H100-equivalents) but produces 31,680 TFLOPS of effective FP8 compute and transmits pseudo-gradients at 8 bits instead of 16. Config C achieves 10^27 FLOP with only 2,000 nodes and 32,000 H100s ($960M) — 3x fewer GPUs and 3x lower cost than the A100-based configs. The trade-off is a smaller model (91B vs 240B) due to lower VRAM per H100 node (1,280 GB at $\beta = 14$ bytes/param for FP8).

**Hierarchical DiLoCo improves efficiency by 3 percentage points.** At N=4,000, flat DiLoCo achieves $\eta = 0.871$ with $H_{\min} = 244$, while hierarchical DiLoCo achieves $\eta = 0.902$ with $H_{\text{eff}} = 65$. The improvement comes from regional syncs over 1 Gbps LAN, which keep $H_{\text{inner}}$ low (18 steps) while the global sync interval is hidden behind regional cycles. This is a modest but meaningful improvement that compounds over 1.5 years of training.

**100x compression reduces $H_{\min}$ dramatically.** With 100x compression (vs 16x baseline), sync volume drops from 240 Gbit to 38 Gbit per direction, reducing $H_{\min}$ from 244 to 39 at N=4,000. This improves $\eta$ from 0.871 to 0.914. The algorithmic cost of such aggressive compression is not modeled by the simulator, but the [Streaming DiLoCo paper](https://arxiv.org/abs/2501.18512) demonstrates that FP4 quantization of pseudo-gradients preserves training quality.

**The combined best case (Config F)** uses hierarchical DiLoCo, 100x compression, and FP8 H100 nodes to achieve **1.11 x 10^27 FLOP** at the lowest cost ($960M). This configuration achieves $\eta = 0.923$ — the highest efficiency of any scenario — by minimizing the effective synchronization interval ($H_{\text{eff}} = 22$).

### 8.3 Mixture of Experts + Expert Parallelism

MoE with Expert Parallelism (EP) does not increase $C_{\text{local}}$ — total compute is determined by active parameters, not total parameters. However, it enables training a **much larger model** within the same compute budget:

| Config | Total Params | Active Params | Per-Node Memory | Fits (48x A100) | EP Overhead |
|:--|:--|:--|:--|:--|:--|
| Dense 240B | 240B | 240B | 3,840 GB | Yes | — |
| MoE 600B | 600B | 100B | 1,711 GB (at N=72) | Yes | 32.8% |
| MoE 1T | 1,000B | 150B | 2,589 GB (at N=72) | Yes | 24.5% |

EP distributes expert parameters across all nodes, reducing per-node memory from $P_{\text{total}} \cdot \beta$ to $(P_{\text{shared}} + P_{\text{experts}}/N) \cdot \beta$. A 600B MoE model (100B shared, 500B expert) at N=72 requires only 1,711 GB per node — well within the 3,840 GB budget.

The trade-off is a **32.8% compute overhead** from EP All-to-All communication ($T_{\text{EP}} = 2 \times 0.1\text{s} \times 32\text{ layers} = 6.4\text{s}$ per step, added to the 13.1s compute time for 100B active params). This reduces $C_{\text{local}}$ to **6.45 x 10^26** at 4,000 nodes — about 35% less than the dense 240B configuration. However, the resulting 600B MoE model may be more capable per FLOP than a 240B dense model, following the trend established by models like Mixtral and DeepSeek-V2.

### 8.4 Reaching 10^28 FLOP

For completeness, 10^28 FLOP would require approximately:
- **~20,000 nodes** of 16x H100 FP8 (~320,000 H100s, ~$9.5B), or
- **~40,000 nodes** of 48x A100 FP16 (~1,900,000 A100s, ~$29B)

These are resources comparable to a major nation's annual military procurement budget. The straggler factor at N=20,000 is $f(20000) = 1.71$, and $\eta$ remains above 0.84 — the physics of DiLoCo does not prevent scaling to this level, but the economic and logistical requirements make this exclusively a state-actor scenario.

### 8.5 Summary: Scale and Actor Type

| Compute Scale | Nodes (A100) | Cost | Actor Type |
|:--|:--|:--|:--|
| **10^24** (Strict Threshold) | 4 | $3M | Research lab, wealthy individual |
| **10^25** (frontier-competitive) | 72 | $52M | Well-funded startup, small state |
| **10^26** | 500 | $360M | Large corporation, medium state |
| **10^27** | 2,000-4,000 | $1-3B | Major state actor only |
| **10^28** | 20,000-40,000 | $10-29B | Superpower-level resources |

The compute threshold in the treaty (10^24 FLOP) is designed to catch training runs at the **research lab** level and above. Distributed training with DiLoCo makes this threshold porous at the 4-72 node scale ($3-52M), where financial detection is challenging. At the 10^27+ scale, the treaty's financial monitoring and chip tracking provisions become the effective enforcement mechanism, since the hardware procurement itself is the hardest step to conceal.

## 9. Network Sensitivity Analysis

The analysis in Sections 5-8 assumes a baseline network of 100 Mbps symmetric WAN with 100 ms RTT. This section examines how sensitive the results are to degraded network conditions — lower bandwidth, higher latency, and realistic deployment profiles across different geographical regions. The central question is: **does DiLoCo's performance degrade significantly under realistic or adversarial network conditions?**

### 9.1 Real-World Network Parameters

All latency values are from measured RTT data (Azure Network Latency Statistics June 2025, Verizon IP Latency, Epsilon Telecom, AWS inter-region via CloudPing.co):

| Scenario | Measured RTT | Source |
|:--|:--|:--|
| Same cloud region (cross-AZ) | 1-2 ms | AWS, Azure intra-region P50 |
| Same continent (Europe) | ~20 ms | London–Frankfurt (Epsilon 20ms, Azure 17ms) |
| Continental US | ~65 ms | NYC–LA (Epsilon 64ms, Azure East-West 71ms) |
| Transatlantic | ~75 ms | NYC–London (Verizon 70ms, Epsilon 75ms, Azure 79ms) |
| Transpacific | ~105 ms | LA–Tokyo (AWS 105ms, Epsilon 100ms) |
| US East Coast–Asia | ~230 ms | Virginia–Singapore (AWS 224ms) |
| Global worst-case | ~340 ms | Brazil–SE Asia (Azure 332-343ms) |

Bandwidth tiers available for distributed training:

| Tier | Typical Speed | Availability |
|:--|:--|:--|
| Consumer broadband (cable) | 20 Mbps upload | Widely available |
| Consumer fiber | 100-1000 Mbps symmetric | Increasingly common |
| Business DIA | 1-10 Gbps | Available in metro areas |
| Enterprise leased line | 10-100 Gbps | Major DCs only |

### 9.2 Bandwidth Sensitivity

Bandwidth was swept from 10 Mbps to 1 Gbps while holding latency at 100 ms. Results for representative configurations:

**72 nodes, 48x A100 FP16, 16x compression (targeting 10^25):**

| BW (Mbps) | H_min | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 1,994 | 0.821 | 1.68 x 10^25 | 88% |
| 25 | 798 | 0.843 | 1.72 x 10^25 | 91% |
| 50 | 399 | 0.859 | 1.75 x 10^25 | 92% |
| **100** | **200** | **0.875** | **1.79 x 10^25** | **94%** |
| 250 | 80 | 0.897 | 1.83 x 10^25 | 96% |
| 500 | 40 | 0.913 | 1.86 x 10^25 | 98% |
| 1,000 | 20 | 0.929 | 1.90 x 10^25 | 100% |

**2,000 nodes, 16x H100 FP8, 16x compression (targeting 10^27):**

| BW (Mbps) | H_min | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 2,495 | 0.805 | 9.66 x 10^26 | 88% |
| 25 | 998 | 0.828 | 9.93 x 10^26 | 90% |
| 50 | 499 | 0.845 | 1.01 x 10^27 | 92% |
| **100** | **250** | **0.862** | **1.03 x 10^27** | **94%** |
| 500 | 50 | 0.902 | 1.08 x 10^27 | 98% |
| 1,000 | 25 | 0.920 | 1.10 x 10^27 | 100% |

**2,000 nodes, 16x H100 FP8, hierarchical + 100x compression (Config F):**

| BW (Mbps) | H_eff | $\eta$ | C_local (FLOP) | % of best |
|:--|:--|:--|:--|:--|
| 10 | 33 | 0.913 | 1.10 x 10^27 | 95% |
| 25 | 21 | 0.924 | 1.11 x 10^27 | 96% |
| 50 | 15 | 0.932 | 1.12 x 10^27 | 97% |
| **100** | **11** | **0.941** | **1.13 x 10^27** | **98%** |
| 500 | 5 | 0.959 | 1.15 x 10^27 | 99% |
| 1,000 | 4 | 0.964 | 1.16 x 10^27 | 100% |

**Key finding:** Reducing bandwidth from 100 Mbps to 10 Mbps — a 10x degradation — costs only **6% of C_local** for flat DiLoCo and **3% for hierarchical+100x**. DiLoCo compensates by increasing $H$: at 10 Mbps, $H_{\min}$ rises to ~2,000-2,500 (flat) or ~33 (hierarchical+100x), which increases $\log_{10}(H)$ and thus reduces $\eta$, but only modestly due to the logarithmic dependence.

### 9.3 Latency Sensitivity

Latency was swept across all real-world scenarios from 2 ms (same cloud region) to 340 ms (Brazil–SE Asia) while holding bandwidth at 100 Mbps.

**Result: Latency has virtually no effect on any configuration.**

For all configurations tested — 72 nodes flat, 500 nodes flat, 2,000 nodes flat, and 2,000 nodes hierarchical+100x — the C_local values are **identical** across all latency scenarios (2 ms to 340 ms). The H values do not change.

**Why latency is irrelevant:** The sync time formula is $T_{\text{sync}} = 2 \cdot V_{\text{bits}} / BW + \text{latency}$. For a 240B model with 16x compression at 100 Mbps:

$$V_{\text{bits}} = 240 \times 10^9 \times 16 / 16 = 240 \text{ Gbit}$$
$$T_{\text{sync,bandwidth}} = 2 \times 240 \times 10^9 / (100 \times 10^6) = 4{,}800 \text{ seconds}$$

The bandwidth term is **4,800 seconds**. Even the global worst-case latency of 340 ms is only **0.007%** of the sync time. Latency is negligible compared to the time required to transfer billions of pseudo-gradient values, even with 16x compression.

Even in the most latency-favorable scenario — small model (91B), FP8, 100x compression, 1 Gbps bandwidth:

$$V_{\text{bits}} = 91 \times 10^9 \times 8 / 100 = 7.28 \text{ Gbit}$$
$$T_{\text{sync,bandwidth}} = 2 \times 7.28 \times 10^9 / (1 \times 10^9) = 14.6 \text{ seconds}$$

At 14.6 seconds, 340 ms latency is still only 2.3% of sync time — below the threshold where it would change $H_{\min}$. **Latency only becomes relevant for models under ~1B parameters or with compression ratios exceeding 1,000x**, neither of which is realistic for frontier-scale training.

### 9.4 Combined Deployment Profiles

Realistic network conditions combine both bandwidth and latency. The table below shows results for deployment profiles that an evader might realistically use, ranging from co-located nodes in the same city to maximally distributed nodes across the globe.

**72 nodes, 48x A100 FP16, 16x compression (10^25 target, ~$52M):**

| Deployment | BW | RTT | H | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 20 | 0.929 | 1.90 x 10^25 | 100% |
| Same country (US) | 500 Mbps | 35 ms | 40 | 0.913 | 1.86 x 10^25 | 98% |
| Continental US | 100 Mbps | 65 ms | 200 | 0.875 | 1.79 x 10^25 | 94% |
| Transatlantic | 100 Mbps | 75 ms | 200 | 0.875 | 1.79 x 10^25 | 94% |
| Transpacific | 50 Mbps | 105 ms | 399 | 0.859 | 1.75 x 10^25 | 92% |
| Global adversarial | 25 Mbps | 230 ms | 798 | 0.843 | 1.72 x 10^25 | 91% |
| Global worst-case | 10 Mbps | 340 ms | 1,994 | 0.821 | 1.68 x 10^25 | 88% |

**2,000 nodes, 16x H100 FP8, hier+100x (Config F, 10^27 target, ~$960M):**

| Deployment | BW | RTT | H_eff | $\eta$ | C_local | % of best |
|:--|:--|:--|:--|:--|:--|:--|
| Colocated (same metro) | 1 Gbps | 5 ms | 4 | 0.964 | 1.16 x 10^27 | 100% |
| Same country (US) | 500 Mbps | 35 ms | 5 | 0.959 | 1.15 x 10^27 | 99% |
| Continental US | 100 Mbps | 65 ms | 11 | 0.941 | 1.13 x 10^27 | 97% |
| Transatlantic | 100 Mbps | 75 ms | 11 | 0.941 | 1.13 x 10^27 | 97% |
| Transpacific | 50 Mbps | 105 ms | 15 | 0.932 | 1.12 x 10^27 | 97% |
| Global adversarial | 25 Mbps | 230 ms | 21 | 0.924 | 1.11 x 10^27 | 96% |
| Global worst-case | 10 Mbps | 340 ms | 33 | 0.913 | 1.10 x 10^27 | 95% |

### 9.5 Achievability of 10^26 and 10^27 Under Degraded Networks

**10^26 FLOP under degraded networks (500 nodes, 48x A100 FP16, 16x comp, 1.5 years):**

| Network Conditions | C_local | Achievable? |
|:--|:--|:--|
| 100 Mbps, 100 ms (baseline) | 1.24 x 10^26 | Yes |
| 50 Mbps, 105 ms (transpacific) | 1.21 x 10^26 | Yes |
| 25 Mbps, 230 ms (global) | 1.19 x 10^26 | Yes |
| 10 Mbps, 340 ms (worst-case) | 1.16 x 10^26 | **Yes** |

10^26 FLOP is achievable under **all tested network conditions**, including the global worst-case. Even at 10 Mbps with 340 ms latency, 500 nodes of 48x A100 produce 1.16 x 10^26 FLOP — comfortably above the 10^26 target.

**10^27 FLOP under degraded networks (2,000 nodes, 16x H100 FP8, 1.5 years):**

| Network Conditions | C_local (flat, 16x) | C_local (hier+100x) | Achievable? |
|:--|:--|:--|:--|
| 100 Mbps, 100 ms | 1.03 x 10^27 | 1.13 x 10^27 | Yes |
| 50 Mbps, 105 ms | 1.01 x 10^27 | 1.12 x 10^27 | Yes |
| 25 Mbps, 230 ms | 9.93 x 10^26 | 1.11 x 10^27 | Marginal (flat) / **Yes** (hier) |
| 10 Mbps, 340 ms | 9.66 x 10^26 | 1.10 x 10^27 | No (flat) / **Yes** (hier) |

With flat DiLoCo and 16x compression, 10^27 FLOP requires at least ~50 Mbps per node. Below that, flat DiLoCo at 2,000 nodes falls short of 10^27, though it still produces ~9.7 x 10^26 (97% of target). With hierarchical DiLoCo and 100x compression, 10^27 is achievable **even at 10 Mbps with 340 ms latency** — the protocol's combination of regional syncs and aggressive compression makes it nearly network-independent.

### 9.6 Implications for Treaty Enforcement

The sensitivity analysis reveals three governance-critical findings:

**1. DiLoCo is bandwidth-insensitive by design.** The protocol absorbs low bandwidth by increasing the synchronization interval $H$. Because efficiency depends on $\log_{10}(H)$, a 10x increase in $H$ (needed to compensate for 10x lower bandwidth) costs only a constant reduction in $\eta$. Across the full range from 10 Mbps to 1 Gbps, C_local varies by only 12% (flat) to 5% (hierarchical+100x). This means:
- **Network throttling is not an effective countermeasure.** Even if the treaty imposed bandwidth restrictions on sub-CCC nodes, an evader using 10 Mbps consumer broadband upload speeds would lose at most 12% of their achievable compute.
- **Network monitoring faces an asymmetry problem.** The evader's pseudo-gradient traffic, compressed 16-100x and transmitted intermittently (every few minutes to hours), is indistinguishable from ordinary cloud backup, video streaming, or file synchronization traffic.

**2. Latency is irrelevant to distributed training.** The sync volumes in frontier-scale training (Gbits to Tbits) make round-trip latency negligible. Even the Brazil-to-Southeast-Asia worst case (340 ms) has zero measurable impact. This means:
- **Geographical distribution is free.** An evader can spread nodes across continents with no performance penalty, maximizing resistance to physical seizure and jurisdictional enforcement.
- **Submarine cable routing and peering arrangements are irrelevant.** The evader needs no special network infrastructure — ordinary consumer internet suffices.

**3. Hierarchical DiLoCo + compression creates a compounding defense.** With 100x compression and regional clustering, the sync volume drops to ~7 Gbit per direction (for a 91B FP8 model), and the two-tier structure means only group aggregates traverse the WAN. The resulting system achieves $\eta > 0.91$ under *all* tested network conditions, including the worst-case global deployment. An evader using this configuration needs only consumer fiber connections and achieves 95% of the theoretical maximum compute.

**Summary:** The treaty's compute threshold (10^24 FLOP) cannot be enforced through network-level controls. DiLoCo's architecture makes it inherently robust to bandwidth constraints and completely insensitive to latency. The effective enforcement mechanisms remain chip tracking, financial monitoring, and physical detection of GPU concentrations — not network monitoring or bandwidth restrictions.
