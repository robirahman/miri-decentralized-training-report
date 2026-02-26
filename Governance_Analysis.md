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

**Nodes and cost to reach each compute target (A100 FP16, 16x compression, 1.5 years):**

| CCC Threshold | Nodes for 10^24 | Cost | Nodes for 10^25 | Cost | Nodes for 10^26 | Cost |
|:--|:--|:--|:--|:--|:--|:--|
| 16 (current) | 5 | $4M | 41 | $30M | 404 | $291M |
| 8 | 8 | $3M | 77 | $29M | 767 | $288M |
| 4 | 16 | $3M | 157 | $28M | 1,576 | $284M |
| 2 | 31 | $3M | 310 | $28M | 3,108 | $280M |
| 1 | 61 | $3M | 610 | $27M | 6,120 | $275M |

**The central finding is that lowering the CCC threshold barely increases the cost of evasion.** Reaching 10^24 FLOP costs approximately $3M regardless of whether the threshold is 16 or 1 H100-equivalent. This is because DiLoCo's efficiency is nearly constant across node counts: $\eta \approx 87\%$ whether using 5 large nodes or 61 small nodes. The evader simply redistributes the same total number of GPUs into more, smaller nodes. The total hardware cost is almost unchanged because the same aggregate compute power is needed.

The impact is on **model quality**, not cost. A 3x A100 node (under 1 H100-eq threshold) can only hold a 15B model, compared to 240B at the current threshold. This forces the evader to either (a) train a much smaller model, or (b) use pipeline parallelism over WAN to shard a larger model across nodes — which incurs severe latency penalties (Section 6.5 estimated ~47 seconds per micro-batch for activation transfers at 100 Mbps). Neither option is attractive, but a 15B model trained on ~3.5T tokens (at 10^24 FLOP) is still a meaningful AI system.

**With H100 FP8 nodes** (exploiting the FP8/FP16 loophole), the evader fares even better:

| CCC Threshold | Max H100s | FP8 PFLOPS | Max Model (FP8) | Nodes for 10^24 | Cost |
|:--|:--|:--|:--|:--|:--|
| 16 (current) | 16 | 31.68 | 91B | 2 | $1M |
| 8 | 8 | 15.84 | 46B | 4 | $1M |
| 4 | 4 | 7.92 | 23B | 8 | $1M |
| 2 | 2 | 3.96 | 11B | 15 | $1M |
| 1 | 1 | 1.98 | 6B | 29 | $1M |

A single H100 running FP8 produces 1.98 PFLOPS of effective compute — nearly 2 H100-equivalents on the CCC's FP16 metric. Even at a 1 H100-eq threshold, the evader can use individual H100s as single-GPU nodes and reach 10^24 with 29 cards ($870K). The cost is $1M or less at every threshold level.

**Collateral damage of lowering the threshold:**

| Threshold | Legitimate systems requiring registration |
|:--|:--|
| 16 (current) | 0 of 14 representative systems |
| 8 | 0 (the 8x H100 systems are exactly at 8.0, not over) |
| 4 | 4 systems: AWS p5.48xlarge, DGX H100, Princeton AI cluster, molecular dynamics clusters |
| 2 | 7 systems: adds DGX A100, AI lab servers, AWS p4d |
| 1 | 11 of 14 systems: includes nearly all multi-GPU research workstations |

At a threshold of 4 H100-equiv, every 8x H100 node in every university, cloud provider, and research lab worldwide would require registration and be subject to potential inspection. AWS's p5.48xlarge instances and DGX H100 systems — standard equipment in hundreds of institutions — would become CCCs. At a threshold of 1, even a researcher with three A100s would need to register. This would not ban or confiscate any hardware, but the reporting requirements and inspection obligations would substantially expand the treaty's administrative footprint while barely increasing the cost of evasion.

However, a lower threshold does have an indirect benefit: it **increases the surface area for detection**. If an evader must register 61 individual nodes (at threshold=1) rather than 5 larger nodes (at threshold=16), each registered node becomes a point where inspectors could detect training activity. The evader faces a dilemma: register the nodes (exposing them to inspections) or operate them unregistered (violating the treaty on 61 separate counts, increasing whistleblower exposure). The model quality penalty is also significant — forcing the evader from a 240B model to a 15B model substantially reduces the threat.

**Assessment:** Lowering the CCC compute threshold is **modestly effective through indirect mechanisms** (increased inspection surface, model quality degradation) but **does not increase evasion cost**. The regulatory burden on legitimate users scales significantly at lower thresholds. A moderate reduction to **4-8 H100-equivalents** may represent a reasonable compromise: it catches DGX H100 and 8x H100 cloud instances (which are already operated by entities with the resources to comply with registration) while forcing the evader to smaller models and more numerous, inspectable nodes.

### 10.2 Adding Memory to the CCC Definition

The current CCC threshold is defined solely by compute capacity (TFLOPS FP16). Section 6.5 identified the resulting exploit: an evader can select hardware that maximizes VRAM while staying under the compute threshold (48x A100 80GB = 3,840 GB VRAM at 15.1 H100-equiv), enabling large models (240B) that would be impossible on compute-equivalent H100 nodes (1,280 GB VRAM).

A **memory-based registration threshold** would require CCC registration when aggregate accelerator VRAM exceeds a specified limit, regardless of compute capacity.

**Impact of VRAM thresholds (combined with 16 H100-eq compute threshold):**

| VRAM Limit | Max A100s | VRAM | Max Model | Nodes for 10^24 | Cost | Nodes for 10^25 | Cost |
|:--|:--|:--|:--|:--|:--|:--|:--|
| No limit (current) | 48 | 3,840 GB | 240B | 5 | $4M | 41 | $30M |
| 2 TB | 25 | 2,000 GB | 125B | 8 | $3M | 77 | $29M |
| 1 TB | 12 | 960 GB | 60B | 16 | $3M | 157 | $28M |
| 512 GB | 6 | 480 GB | 30B | 31 | $3M | 310 | $28M |
| 256 GB | 3 | 240 GB | 15B | 61 | $3M | 610 | $27M |

The pattern is identical to lowering the compute threshold: costs barely change, but model size shrinks. A 256 GB VRAM limit forces the evader to 3x A100 nodes (15B model), but reaching 10^24 still costs only $3M.

The memory threshold does **close the specific A100 exploit** — preventing an evader from packing 48x A100 80GB into a single unregistered node. But it does not prevent evasion because the evader simply uses more, smaller nodes with the same total GPU count.

**Collateral damage of memory thresholds:**

| VRAM Limit | Legitimate systems requiring registration |
|:--|:--|
| 2 TB | 0 of 14 representative systems |
| 1 TB | 0 |
| 512 GB | 6 systems: all 8x A100/H100 nodes (DGX A100, DGX H100, AI lab servers, AWS p4d/p5, molecular dynamics, Princeton) |
| 256 GB | 10 systems: adds 4x A100 80GB workstations, 8x L40S inference servers |

A 512 GB VRAM limit would bring every DGX system, every 8-GPU cloud instance, and every 8-GPU research server under registration and inspection requirements. These are standard equipment at universities, cloud providers, and research labs — the operators of these systems would need to register them and comply with reporting obligations. This is not confiscation or prohibition, but it does substantially widen the treaty's scope to include routine research infrastructure.

A 2 TB or 1 TB limit would avoid catching standard 8-GPU systems while still constraining the A100 exploit. However, at 1 TB the evader can still use 12x A100 (960 GB, 60B model) — a substantial model by any standard. And an H100 FP8 evader is not constrained at all by memory limits above 1.28 TB, since 16x H100 has only 1,280 GB VRAM.

**Assessment:** A memory threshold of **1-2 TB** is a reasonable, low-impact addition that closes the specific 48x A100 exploit without subjecting standard 8-GPU systems to registration. However, it does not fundamentally prevent distributed training evasion — it merely reduces the maximum model size per node from 240B to 60-125B. The evader compensates with more nodes at nearly the same total cost. A lower threshold of **512 GB** would bring most multi-GPU research systems under registration, which increases the treaty's enforcement surface but also its administrative scope.

### 10.3 Bandwidth Restrictions

Section 9 established that DiLoCo is inherently bandwidth-insensitive. Could the treaty mandate bandwidth restrictions on sub-CCC nodes?

**Scenario: 10 Mbps upload cap on all unregistered nodes.**

From Section 9.2, the impact at 10 Mbps (worst tested):

| Configuration | C_local at 100 Mbps | C_local at 10 Mbps | Reduction |
|:--|:--|:--|:--|
| 72 nodes, A100 FP16, flat 16x | 1.79 x 10^25 | 1.68 x 10^25 | 6% |
| 2,000 nodes, H100 FP8, flat 16x | 1.03 x 10^27 | 9.66 x 10^26 | 6% |
| 2,000 nodes, H100 FP8, hier+100x | 1.13 x 10^27 | 1.10 x 10^27 | 3% |

A 10 Mbps cap reduces evasion capability by only 3-6%. DiLoCo compensates by increasing H from ~200 to ~2,000 inner steps, which reduces $\eta$ from 0.875 to 0.821. The logarithmic dependence on H makes this inherently resilient: the evader needs 10x more inner steps but loses only one $\log_{10}$ unit of efficiency.

**Implementation challenges:** A bandwidth restriction on GPU-equipped facilities could take several forms: (a) requiring registered sub-CCC nodes to have their WAN connections monitored, (b) mandating ISPs to throttle upload bandwidth for known GPU facilities, or (c) requiring facilities with AI chips to register their network connections with the CTB. Option (a) is the lightest — it imposes a monitoring obligation but does not restrict bandwidth, and it could catch evasion attempts through traffic analysis. Options (b) and (c) are more intrusive: they would constrain legitimate remote inference, federated learning, scientific data transfer, and cloud computing at GPU-equipped facilities. These are not catastrophic — the hardware is not banned — but they would impose operational friction on routine research and commercial activities.

**The fundamental problem** is that DiLoCo is bandwidth-insensitive by design. Even if bandwidth restrictions were perfectly enforced, the 3-6% reduction in achievable compute is negligible. The evader can also circumvent bandwidth monitoring by routing training traffic through VPN tunnels or mixing it with cover traffic.

**Assessment:** Bandwidth restrictions are **technically ineffective** — DiLoCo's logarithmic sensitivity to bandwidth means no realistic restriction produces a meaningful reduction in achievable compute. Monitoring WAN connections at GPU facilities has some value as a detection signal (Section 10.4), but bandwidth *caps* provide negligible security benefit relative to the operational constraints they impose on legitimate use.

### 10.4 Traffic Fingerprinting and ISP Monitoring

DiLoCo creates a distinctive network signature: pseudo-gradient synchronization at regular intervals (every ~50-200 minutes), with identical payload sizes (compressed model-sized tensors). Could treaty-mandated ISP monitoring detect this pattern?

**The sync fingerprint:** At 100 Mbps with a 240B model and 16x compression, each sync transfers ~30 GB (240 Gbit) per direction. With streaming DiLoCo, this traffic is spread over the full outer step (~100 minutes), producing a sustained ~38 Mbps average in each direction. The traffic pattern is:
- **Continuous:** Streaming DiLoCo eliminates the bursty pattern that non-streaming DiLoCo would produce
- **Bidirectional:** ~76 Mbps total, comparable to a media server or cloud backup service
- **Encrypted:** VPN tunnels obscure content; only flow metadata (size, timing, destination) is visible
- **Regular cadence:** Sync intervals are highly periodic (e.g., every 100 minutes ± straggler variance)

**Detection feasibility:** The regular cadence and fixed payload size are the most distinctive features. A traffic classifier at the ISP level could look for long-running, encrypted, bidirectional flows of approximately constant bandwidth with periodic structure. However:

1. **Streaming eliminates bursts.** Non-streaming DiLoCo would create obvious ~30 GB bursts every 100 minutes. Streaming spreads this into continuous transfer, making it resemble ordinary sustained traffic (video streaming, cloud sync, CDN nodes).
2. **Traffic shaping defeats detection.** The evader can inject random jitter into sync timing (varying H by a few steps), pad payloads to variable sizes, and multiplex training traffic with cover traffic (video streaming, torrents) to obscure the pattern.
3. **Scale of monitoring required.** The treaty would need ISP cooperation to monitor encrypted flows from servers with GPUs. This could be scoped to registered AI chip facilities (a manageable set) rather than all internet traffic, but the evader's nodes may not be registered.
4. **False positive rate.** Sustained bidirectional encrypted traffic at 50-100 Mbps describes millions of legitimate servers worldwide (game servers, CDN edges, VPN endpoints, corporate backup systems). Even a 0.1% false positive rate would generate many spurious alerts requiring investigation.
5. **Mesh detection is harder than point detection.** Identifying that N geographically distributed servers are participating in the same training run requires correlating flow metadata across N^2 pairwise connections (or N connections to a parameter server), across multiple ISPs and jurisdictions.

**Assessment:** Traffic fingerprinting is a **useful supplementary detection signal** but not a reliable primary mechanism. Against a sophisticated evader using streaming DiLoCo with traffic shaping, it can be defeated by elementary countermeasures. However, mandating traffic monitoring at registered GPU facilities could catch less sophisticated evaders and provide corroborating evidence for investigations triggered by other means (whistleblowers, financial analysis). It is best deployed as part of a multi-source intelligence approach rather than as a standalone enforcement mechanism.

### 10.5 Hardware-Level Enforcement: TEE and Remote Attestation

Modern AI accelerators (NVIDIA H100, H200, Blackwell) include **Trusted Execution Environment (TEE)** capabilities through NVIDIA Confidential Computing. This technology could be repurposed for treaty enforcement:

**Concept:** Every AI chip is required to cryptographically attest its workload to an international monitoring body (the CTB — Compute Treaty Body). The TEE ensures that the attestation is tamper-resistant:
- The chip's firmware reports the kernels being executed, the model architecture, and cumulative FLOP counts
- This attestation is signed by a hardware-embedded key that cannot be extracted or modified without destroying the chip
- The CTB can verify that a chip is running only approved workloads (per the treaty's AI Technique Whitelist, Article II.20)
- If a violation is detected, the chip's attestation keys can be revoked, rendering it unable to participate in monitored workloads

**Strengths:**
- **Eliminates plausible deniability:** A chip running DiLoCo training would be detected by its own hardware, regardless of network obfuscation
- **Scales with chip tracking:** Combined with the treaty's chip registration system, every registered chip becomes a monitor that cannot be deceived by its operator
- **Per-chip enforcement:** Does not require network monitoring, ISP cooperation, or physical inspections

**Limitations:**
1. **Manufacturer cooperation required.** Currently only NVIDIA supports confidential computing on AI accelerators. AMD, Intel, and other manufacturers would need to implement equivalent capabilities. Custom ASICs (Google TPU, Amazon Trainium) would need similar mechanisms.
2. **Firmware integrity.** The security of TEE-based attestation depends on firmware that cannot be modified. Sophisticated actors may be able to extract keys, reflash firmware, or exploit hardware vulnerabilities. The arms race between attestation security and hardware hacking is ongoing, and state-level actors with access to semiconductor analysis tools (FIB, delayering) may be able to defeat current protections.
3. **Pre-existing and unregistered chips.** Chips manufactured before treaty ratification, or manufactured outside the treaty's jurisdiction (e.g., domestic Chinese fabs using SMIC process), would not have the required attestation firmware. The treaty's chip tracking provisions partially address this, but a state actor with pre-existing stockpiles or domestic manufacturing could acquire unattested chips.
4. **Privacy and sovereignty concerns.** Requiring every AI chip to report its workload to an international body raises significant privacy, intellectual property, and national sovereignty objections. Legitimate users may object to having their proprietary research monitored in real-time.
5. **Verification of the TEE itself.** Who verifies that the TEE firmware is correct and has not been backdoored? This creates a chain-of-trust problem that ultimately depends on hardware manufacturers and the CTB acting in good faith.

**Assessment:** TEE-based remote attestation is the **most technically promising** countermeasure for non-state actors. If all major chip manufacturers implement treaty-compliant attestation firmware, and the chip tracking system ensures that only attested chips are in circulation, the plausible deniability problem is effectively eliminated. However, it requires significant industry cooperation, does not address state actors with domestic chip manufacturing, and raises substantial privacy and sovereignty concerns. It should be considered a **medium-term** enforcement mechanism (3-5 years to deploy) rather than an immediate solution.

### 10.6 Orchestration Layer Regulation

The treaty could ban the development, distribution, or use of software specifically designed for WAN-based distributed training (DiLoCo, streaming DiLoCo, hierarchical DiLoCo, etc.). Article VIII already restricts research that "advances toward ASI" or "endangers agreement verifiability."

**Problems:**
1. **The knowledge is published.** DiLoCo was published by Douillard et al. (2023) at Google DeepMind. Streaming DiLoCo, hierarchical DiLoCo, and related techniques are in the open literature. Banning the software cannot unpublish the papers.
2. **Implementation is trivial.** DiLoCo is a simple modification to distributed SGD: run H local steps, then average pseudo-gradients. A competent engineer can implement it from the paper in a few days. No specialized software framework is required — it can be implemented in a few hundred lines of PyTorch.
3. **Dual-use problem.** The same techniques (federated learning, local SGD, gradient compression) are used extensively in privacy-preserving machine learning, mobile device training, and scientific simulation. Banning WAN-optimized training software would cripple legitimate federated learning research and deployment.
4. **Precedent: encryption export controls.** The 1990s attempt to regulate encryption software (treating strong encryption as a munition under ITAR/EAR) is widely regarded as a policy failure. The software was too simple, too broadly useful, and too easily reimplemented to be effectively controlled. DiLoCo is even simpler than modern encryption.

**Assessment: Orchestration layer regulation is unenforceable and counterproductive.** It would chill legitimate distributed computing research while being trivially circumvented by any evader willing to write a few hundred lines of code.

### 10.7 Redefining Model Possession

In distributed DiLoCo training, each node holds a complete copy of the model weights (unlike pipeline parallelism, where weights are sharded). However, the model is never "possessed" in the traditional sense by any single entity if the nodes are operated by different actors or across jurisdictions. The treaty could redefine model possession to include:

- **Distributed possession:** Holding any replica of a model that is part of an ongoing distributed training run
- **Control of orchestration:** Operating the coordination infrastructure (parameter server, communication protocol) for a distributed training run, regardless of whether one "possesses" the model weights

This is primarily a **legal** rather than technical modification. It addresses the scenario where an evader claims that no individual node operator violated the treaty because no individual node exceeded the CCC threshold.

**Assessment:** This is a **necessary but insufficient** legal clarification. It closes a definitional gap in the treaty but does nothing to solve the detection problem — you still need to discover the distributed training run before the possession question becomes relevant.

### 10.8 Enhanced Chip Tracking and Physical Inspections

The treaty already includes chip tracking and inspection provisions (Articles V, VII, X). These could be strengthened:

**a) Mandatory utilization reporting.** All registered AI chips (even sub-CCC) could be required to report utilization metrics (GPU hours, average utilization, power draw) to the CTB at regular intervals. DiLoCo training produces a distinctive utilization pattern: 100% GPU utilization during compute blocks, with periodic brief pauses during synchronization (in non-streaming mode) or ~38 Mbps sustained network I/O (in streaming mode). Anomalous utilization (24/7 high GPU utilization on a sub-CCC node for months) would trigger investigation.

**b) Random physical inspections.** The treaty could authorize unannounced inspections of any facility known to contain AI chips, not just registered CCCs. Inspectors could examine running processes, network connections, and training logs.

**c) Enhanced whistleblower programs.** Section 6.7 noted that operational security over 1.5 years across dozens of sites is difficult. Financial incentives for reporting (e.g., bounties proportional to the scale of the detected violation) could significantly increase the probability of detection through human intelligence rather than technical monitoring.

**d) Power monitoring.** A 48x A100 node consumes approximately 15-20 kW at full utilization. Operating 72 such nodes for 1.5 years consumes ~14-19 GWh of electricity — comparable to a small industrial facility. Power utilities could be required to report sustained high-power draws from facilities known to contain AI hardware. However, 15-20 kW per site is well within the range of normal commercial and residential power consumption, making this ineffective for individual sub-CCC nodes.

**Assessment:** Enhanced chip tracking and inspections are **moderately effective** and represent the lowest-collateral-damage enforcement approach. Mandatory utilization reporting combined with enhanced whistleblower programs directly increases the cost and risk of prolonged distributed training. The main limitation is that they are reactive rather than preventive, and their effectiveness depends on implementation quality and political will.

### 10.9 Synthesis: Recommended Treaty Modifications

No single countermeasure closes the distributed training loophole. The analysis reveals that effectiveness depends critically on the **actor type**:

**Against non-state actors ($3M-$360M, 4-500 nodes):**

| Countermeasure | Effectiveness | Registration Burden | Recommendation |
|:--|:--|:--|:--|
| Lower CCC threshold (to 4-8) | Low-Medium | Moderate (catches 8-GPU systems) | **Consider** — increases inspection surface |
| Memory threshold (1-2 TB) | Low-Medium | Low | **Recommended** — closes A100 exploit |
| Bandwidth caps | None | Moderate | Not recommended (technically ineffective) |
| Traffic monitoring at GPU sites | Low-Medium | Low | **Recommended** — supplementary signal |
| TEE/Remote attestation | **High** | Medium | **Strongly recommended** (medium-term) |
| Orchestration regulation | None | Low | Not recommended (unenforceable) |
| Model possession redefinition | Medium | None | **Recommended** — legal clarification |
| Enhanced chip tracking | **Medium-High** | Low | **Strongly recommended** |
| Whistleblower programs | **Medium-High** | None | **Strongly recommended** |

**Against state actors ($1B+, 2,000+ nodes):**

| Countermeasure | Effectiveness | Notes |
|:--|:--|:--|
| Lower CCC threshold | None | Same cost, more nodes |
| Memory threshold | None | Same cost, more nodes |
| TEE/Remote attestation | Low | State actors can manufacture unattested chips |
| Enhanced chip tracking | Low | Classified procurement, domestic fabs |
| Financial monitoring | **Medium** | Detectable above ~$1B for most states |
| Diplomatic/intelligence | **Medium-High** | The only effective lever at state scale |

**Recommended package of treaty modifications:**

1. **Add a 1 TB VRAM threshold to the CCC definition** (in addition to the compute threshold). This closes the specific 48x A100 exploit — no standard 8-GPU system exceeds 1 TB VRAM, so registration burden is minimal. The evader is forced to 12x A100 nodes (60B model) or H100 nodes (91B), reducing model quality without substantially increasing cost.

2. **Mandate TEE-based workload attestation** on all AI accelerators manufactured after treaty ratification. This is the single most impactful technical countermeasure for non-state actors, as it eliminates plausible deniability at the hardware level. Requires cooperation from NVIDIA, AMD, Intel, and other manufacturers.

3. **Redefine model possession** to include distributed possession and control of orchestration. This is a no-cost legal clarification that closes a definitional gap.

4. **Strengthen whistleblower programs** with financial bounties proportional to detected violation scale. Given the operational security challenges of sustained distributed training (Section 6.7), human intelligence is likely more effective than technical monitoring.

5. **Require utilization reporting** for all registered AI chips, with anomaly detection for sustained high-utilization patterns consistent with training (as opposed to inference or scientific computing, which have different utilization profiles).

6. **Consider lowering the CCC compute threshold to 4-8 H100-equivalents.** While this does not increase evasion cost, it forces the evader into more numerous, smaller nodes — each of which is a point of inspection exposure. It also constrains model size (60-125B vs 240B). The trade-off is that standard 8-GPU systems (DGX, cloud instances, university clusters) would require registration, but these are typically operated by entities with the resources to comply.

7. **Mandate traffic monitoring (not bandwidth caps) at registered GPU facilities.** Bandwidth caps are technically ineffective due to DiLoCo's bandwidth insensitivity. However, requiring facilities with registered AI chips to allow WAN traffic monitoring provides a supplementary detection signal that could corroborate other intelligence.

8. **Do not impose bandwidth caps.** DiLoCo's bandwidth insensitivity (Section 9) means that even a 10x bandwidth reduction produces only a 3-6% reduction in achievable compute — negligible security benefit relative to the operational burden.

**The hard truth:** Even with all recommended modifications, the distributed training loophole cannot be fully closed against a determined state actor. A nation-state with domestic chip manufacturing (unattested chips), classified procurement (invisible to financial monitoring), and sovereign territory (immune to inspections) can operate a distributed training run that is fundamentally undetectable by technical means. Against such an actor, the treaty's enforcement ultimately depends on diplomatic, intelligence, and economic instruments — the same tools used in nuclear nonproliferation, with the same fundamental limitations.
