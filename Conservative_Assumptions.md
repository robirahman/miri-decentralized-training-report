# Conservative Assumptions in the Decentralized Training Simulator

The simulator is designed to estimate decentralized training capabilities using techniques that are **proven at scale today**. This document catalogues the ways in which the simulator is systematically conservative — that is, where its assumptions underestimate what a well-resourced developer could achieve. These conservatisms fall into two categories:

1. **Modeling conservatisms**: Places where the simulator uses pessimistic formulas or parameters even for currently-validated techniques.
2. **Omitted techniques**: Cutting-edge methods demonstrated in recent research but not yet validated at the 100B+ parameter, 50+ node scale the simulator targets.

The distinction matters for policy: modeling conservatisms mean the simulator *already* underestimates capability with today's methods, while omitted techniques represent a trajectory of future improvement that would widen the gap further.

## 1. Modeling Conservatisms (Proven Techniques, Pessimistic Parameters)

These are cases where the simulator's formula or parameter choice is more pessimistic than what the literature supports, even restricting to well-validated techniques.

### 1.1 GPipe Bubble Overhead

The simulator uses the classic GPipe bubble formula: $(S-1)/(M+S-1)$ idle fraction, where $S$ is pipeline stages and $M$ is micro-batches. For $S=3, M=8$, this gives 22.2% bubble overhead.

**Why this is conservative:** The GPipe formula represents the *worst-case* scheduling. The 1F1B (one forward, one backward) schedule, which is standard practice and has been since [Narayanan et al. (2019)](https://arxiv.org/abs/1811.06965), reduces memory pressure and in practice achieves lower bubble than the GPipe formula predicts. The simulator does not account for this.

**Magnitude:** 5-15% overestimate of pipeline overhead depending on configuration.

### 1.2 Sync Interval Penalty ($\alpha$)

The efficiency penalty $\eta_H = 1 - \alpha \cdot \log_{10}(H)$ uses $\alpha \approx 0.08$ at 144B, derived from [Charles et al. (2025)](https://arxiv.org/abs/2503.09799) fits at 35M-10B. The extrapolation to 144B uses a power-law decay $\alpha(P) = 0.49 \cdot P^{-0.21}$, but the paper's own data shows that the penalty decreases with scale faster than this power law predicts at the largest tested sizes.

**Why this is conservative:** The $\alpha$ extrapolation was chosen to be a reasonable central estimate, but the confidence interval skews toward lower values at 100B+. A developer training a 144B model may experience $\alpha \approx 0.05$-0.06 rather than 0.08.

**Magnitude:** 2-4% efficiency underestimate at $H=200$.

### 1.3 Straggler Factor

The simulator models straggler overhead as $f(n) = 1 + 0.05 \cdot \log_2(n)$, calibrated to datacenter variance. Over WAN with heterogeneous consumer hardware, stragglers could be worse — but the model also does not account for **asynchronous DiLoCo**, where stragglers simply contribute to the next sync round rather than blocking the current one. Streaming DiLoCo ([Douillard et al., 2025](https://arxiv.org/abs/2501.18512)) explicitly enables this.

**Why this is conservative:** With async/streaming operation, the straggler factor should be significantly lower because no node waits for any other node.

**Magnitude:** 5-15% overestimate at 50+ nodes.

### 1.4 MFU Assumptions

The simulator uses 35-40% MFU for consumer hardware (RTX 4090s) and 45-50% for H100s. These are reasonable for dense models, but MoE architectures can achieve higher effective utilization because the per-expert computation is smaller and maps better to GPU tensor cores. DeepSeek-V3 achieved 60%+ MFU on its MoE architecture.

**Why this is conservative:** MoE scenarios may achieve 5-10 percentage points higher MFU than assumed, particularly with optimized expert parallelism kernels.

**Magnitude:** Scenario-dependent; 5-15% for MoE configurations.

### 1.5 Effective $H$ for Hierarchical DiLoCo

The hierarchical efficiency formula uses $H_{\text{eff}} = H_{\text{inner}} \cdot H_{\text{regional}}^{0.5}$ — a heuristic with no direct empirical calibration. The $\sqrt{H_R}$ exponent was chosen conservatively (a full multiplicative $H_{\text{inner}} \cdot H_{\text{regional}}$ would imply the regional sync is as good as the inner sync, which is unlikely; $\sqrt{}$ splits the difference).

**Why this is conservative:** The true exponent could be anywhere from 0.5 to 0.8. At $H_{\text{inner}}=100, H_{\text{regional}}=10$, the simulator predicts $H_{\text{eff}} \approx 316$. An exponent of 0.7 would give $H_{\text{eff}} \approx 501$ — still high, but with a lower $\alpha$-based penalty.

**Magnitude:** 1-3% efficiency difference.

## 2. Omitted Techniques (Cutting-Edge, Not Yet Validated at Scale)

These techniques have been demonstrated in published research but have **not been tested at the 100B+ parameter scale** the simulator targets. They are not incorporated because their effectiveness at scale is unknown. However, they represent active research directions that could mature if decentralized training becomes more common. Detailed per-technique analysis is in [Simulator Documentation, Appendix B](Simulator_Documentation.md).

### 2.1 Zero Bubble Pipeline Scheduling

[Qi et al. (2024)](https://arxiv.org/abs/2401.10241) splits the backward pass into input-gradient and weight-gradient phases, rescheduling the weight-gradient phase to fill pipeline bubble slots. ZB-2p achieves <1% bubble vs. the simulator's 11-27% (GPipe formula). This is a local computation reordering that does not change the communication pattern, so it is directly applicable to WAN pipeline parallelism.

- **Evidence scale:** 28.3B parameters, 8 pipeline stages
- **Potential improvement:** 10-30% on pipeline-parallel compute (Mode B and C)

### 2.2 Optimized Async DiLoCo (Delayed Nesterov + DyLU)

[Liu, Douillard et al. (2024)](https://arxiv.org/abs/2401.09135) demonstrate that a corrected Nesterov momentum schedule (Delayed Nesterov) combined with dynamic adjustment of local steps based on worker speed (DyLU) can fully close the gap between asynchronous and synchronous DiLoCo. At $H=50$ with 20M parameters, async DiLoCo with DN+DyLU *slightly beats* synchronous training.

- **Evidence scale:** 150M parameters, 16 workers
- **Potential improvement:** 5-15% (by reducing $\alpha$ toward zero and eliminating straggler penalties)

### 2.3 Hierarchical Momentum (HALoS)

[Kim et al. (2025)](https://arxiv.org/abs/2506.04531) uses separate hierarchical momentum terms at the local and global levels. HALoS matches synchronous SGD quality with a 7.5x wall-clock speedup — implying near-zero algorithmic penalty for hierarchical training.

- **Evidence scale:** 70M parameters only
- **Potential improvement:** 3-5% (by reducing the hierarchical $H_{\text{eff}}$ penalty)

### 2.4 Sparse Parameter Averaging (SPARTA)

[SPARTA (2025)](https://openreview.net/pdf?id=stFPf3gzq1) continuously exchanges 0.1-0.5% of parameters (not at sync boundaries). At $H=10{,}000$ with 0.1% exchange, it achieves 14.3% better perplexity than DiLoCo-alone at $H=10{,}000$, while using 1000x less communication. The improvement comes from sparse averaging acting as a regularizer against local model drift.

This is the most provocative result because it suggests the sync interval penalty could be largely decoupled from $H$ — a relationship the simulator treats as fundamental. However, the paper explicitly notes it "doesn't scale well beyond 16 nodes."

- **Evidence scale:** 124M parameters, 2-8 nodes
- **Potential improvement:** 25-35 percentage points at very high $H$ (highly speculative at scale)

### 2.5 Loss-Tolerant Training (UDP)

[Weintraub et al. (2025)](https://arxiv.org/abs/2507.07114) show that 10% packet loss causes only 0.8% perplexity degradation on LLaMA-2 7B. Switching from TCP to a loss-tolerant UDP protocol could dramatically reduce tail latencies caused by retransmissions, which are the primary source of WAN straggler overhead.

- **Evidence scale:** 7B parameters, 64 GPUs
- **Potential improvement:** 10-15% reduction in effective straggler overhead

## 3. Cumulative Conservative Gap

Estimating the total conservatism requires care because some factors compound multiplicatively while others overlap. A rough decomposition:

**For flat DiLoCo (Modes A/D):**

| Source | Estimated Gap | Confidence |
|:--|:--|:--|
| $\alpha$ pessimism (1.2) | 2-4% | Medium |
| Straggler model (1.3) | 5-10% | Medium |
| Async DiLoCo (2.2) | 5-10% | Low-medium |
| Loss-tolerant protocols (2.5) | 3-5% | Low-medium |
| **Cumulative (multiplicative)** | **14-26%** | |

**For pipeline-parallel DiLoCo (Modes B/C):**

| Source | Estimated Gap | Confidence |
|:--|:--|:--|
| GPipe bubble (1.1) | 10-20% | High |
| $\alpha$ pessimism (1.2) | 2-4% | Medium |
| Straggler model (1.3) | 5-10% | Medium |
| Zero Bubble PP (2.1) | 10-25% | Medium |
| Async DiLoCo (2.2) | 5-10% | Low-medium |
| **Cumulative (multiplicative)** | **28-52%** | |

Note: The GPipe bubble (1.1) and Zero Bubble PP (2.1) overlap — together they describe the gap between the simulator's GPipe formula and the state-of-the-art ZB-2p schedule. The combined effect is 10-30%, not 20-45%.

**Adjusted cumulative estimates:**

- **Flat DiLoCo:** The simulator underestimates effective compute by an estimated **15-25%** if all currently-researched techniques are successfully scaled. With only proven techniques and conservative parameter choices, the underestimate is **5-10%**.
- **Pipeline-parallel DiLoCo:** The simulator underestimates effective compute by an estimated **20-40%** due to the GPipe bubble being the largest single source of conservatism.

## 4. Chinchilla Amplification Uncertainty

The simulator converts loss degradation to FLOP-equivalents via the Chinchilla scaling law. This conversion amplifies small loss differences at large model sizes: at 250B, a 1.6% loss increase translates to a ~79% FLOP penalty because only 5% of total loss is improvable (see [Scaling_Law_Uncertainty.md](Scaling_Law_Uncertainty.md) for the full analysis).

This amplification is sensitive to the scaling exponents $\alpha$ and $\beta$, which are fit to models up to ~16B. Within the formal 95% CI on $\alpha$ alone ($\pm 0.02$), the FLOP penalty for the same raw loss ranges from ~52% to ~98%. Unlike the conservatisms in Sections 1–3, this uncertainty is **not directionally conservative** — it could go either way.

The direction matters for governance:

- If the true $\alpha$ is **smaller** (flatter curve): the amplification is less severe, $C_{\text{quality}}$ is higher, and distributed training produces **more capable** models than the simulator estimates. Evasion is easier.
- If the true $\alpha$ is **larger** (steeper curve): amplification is more severe, $C_{\text{quality}}$ is lower, and evasion produces less capable models.

The simulator's default $\alpha = 0.348$ is a mid-range estimate. The conclusions most affected are absolute $C_{\text{quality}}$ figures at 100B+ scale; hardware costs, node counts, and qualitative feasibility assessments are unaffected.

## 5. What This Means for Policy

The simulator's conservatism is deliberate: it models what is achievable with techniques proven at scale, not what might be achievable with cutting-edge research. This is the appropriate baseline for governance analysis because:

1. **Capability thresholds should be robust.** If the simulator shows a configuration exceeds a compute threshold, the conclusion holds even if the simulator is somewhat pessimistic. Overestimating capability would be worse for governance — it would cause false alarms.

2. **The conservatism is bounded.** The cumulative gap is 15-40%, not 2-10x. A configuration the simulator estimates at $5 \times 10^{24}$ FLOP effective compute might actually achieve $6$-$7 \times 10^{24}$ — a meaningful difference for precise threshold comparisons, but not enough to change the order-of-magnitude conclusions.

3. **The gap will likely narrow over time.** Decentralized training is an active research area. Techniques like zero-bubble scheduling and optimized async DiLoCo are straightforward engineering improvements that are likely to be validated at scale within 1-2 years. SPARTA-like sparse averaging is more speculative but could be transformative if it scales.

4. **A determined, technically sophisticated evader would outperform the simulator's predictions.** The simulator models a competent but not extraordinary implementation. A team that implemented zero-bubble PP, async DiLoCo with DN+DyLU, and loss-tolerant protocols could plausibly achieve 20-30% more effective compute than the simulator predicts, using hardware configurations the simulator already models.

Policymakers should interpret the simulator's estimates as a **lower bound** on what is achievable, with the understanding that the true capability is likely 15-25% higher for flat DiLoCo workloads and 20-40% higher for pipeline-parallel workloads. Governance thresholds should be set with this margin in mind.
