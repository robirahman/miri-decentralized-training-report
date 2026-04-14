# Scaling Law Uncertainty and the Chinchilla Amplification Effect

The simulator converts all quality degradation (replica penalty, compression loss, overtraining) into FLOP-equivalents via the Chinchilla scaling law. This document examines the uncertainty in that conversion and its implications for the analysis.

## 1. What the Simulator Assumes

The simulator uses the corrected Chinchilla loss function from [Besiroglu et al. (2024)](https://arxiv.org/abs/2404.10102):

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

| Parameter | Value | SE |
|:--|:--|:--|
| $E$ | 1.8172 | 0.03 |
| $A$ | 482.01 | 124.58 |
| $\alpha$ | 0.3478 | 0.02 |
| $B$ | 2085.43 | 1293.23 |
| $\beta$ | 0.3658 | 0.02 |

Two simulator quantities depend critically on this law:

- **$\eta_{\text{replica}}$** (§4.7): converts the replica loss multiplier into a FLOP penalty via binary search for the optimally-allocated compute that achieves the degraded loss.
- **$\chi$** (§4.9): converts overtraining into a FLOP penalty via the same mechanism.

Both use the scaling law's shape to translate small loss differences into compute-equivalent penalties.

## 2. The Chinchilla Amplification Effect

At large model sizes, the loss approaches the irreducible floor $E = 1.82$. The "improvable" portion of the loss — the part that more compute can reduce — shrinks:

| Model Size | Loss (Chinchilla-optimal) | Improvable ($L - E$) | % of total loss |
|:--|:--|:--|:--|
| 1B | 2.50 | 0.68 | 27% |
| 10B | 2.12 | 0.30 | 14% |
| 72B | 1.97 | 0.15 | 7.6% |
| 250B | 1.91 | 0.096 | 5.0% |
| 1000B | 1.88 | 0.058 | 3.1% |

When a quality penalty (e.g., from replica divergence) increases the loss by a small percentage of the *total* loss, it consumes a much larger percentage of the *improvable* portion. The scaling exponents then determine how much additional compute is needed to overcome that:

**Example: 1.6% total loss increase at 250B (M=72, H=200)**

- Absolute penalty: 0.030 nats
- As fraction of improvable: 31%
- Doubling compute reduces improvable loss by only ~11% (because $\alpha \approx 0.35$)
- Result: ~79% FLOP penalty ($\eta_{\text{replica}} \approx 0.22$)

This is a structural property of power-law scaling with small exponents, not a modeling error. The same mechanism amplifies any loss degradation — compression quality, activation compression, or overtraining — whenever the model operates near the loss floor.

## 3. How Well-Constrained Are $\alpha$ and $\beta$?

### 3.1 Published Estimates

| Source | $\alpha$ | $\beta$ | $E$ | Fit range | Notes |
|:--|:--|:--|:--|:--|:--|
| [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) Approach 1 | — | — | 1.69 | 70M–16B | $a \approx 0.50$, $b \approx 0.50$ |
| Hoffmann Approach 3 | 0.339 | 0.285 | 1.69 | 70M–16B | Inconsistent with Approaches 1–2 |
| [Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102) | 0.348 | 0.366 | 1.82 | 70M–16B | Corrected Approach 3; simulator default |
| [Ruan et al. 2025](https://arxiv.org/abs/2509.23963) | 0.20–0.48 | — | 1.57–1.90 | — | Sensitivity under data perturbations |

The compute-optimal allocation ratio ($a \approx 0.50$, $b \approx 0.50$, meaning parameters and data should scale equally) is robust across multiple independent papers including [DeepSeek (2024)](https://arxiv.org/abs/2401.02954) ($a=0.524$), [Porian et al. (2024)](https://arxiv.org/abs/2406.19146) ($a \approx 0.50$ after correcting all methodology issues), and [LLaMA 3 (2024)](https://arxiv.org/abs/2407.21783) ($a \approx 0.47$). The *individual* exponents $\alpha$ and $\beta$ in the loss function are less well-constrained: the formal SE of 0.02 gives a 95% CI of $\alpha \in [0.31, 0.39]$, but Ruan et al.'s perturbation analysis suggests a wider effective range of 0.20–0.48.

### 3.2 What $E$ Does and Does Not Affect

$E$ represents the entropy of natural language — the inherent unpredictability that no model can eliminate regardless of size or data. Published estimates range from 1.6 to 1.82. However, $E$ has only a modest effect on the FLOP conversion:

| $E$ | $\eta_{\text{replica}}$ (250B, M=72, H=200) | FLOP penalty |
|:--|:--|:--|
| 1.50 | 0.270 | 73% |
| 1.69 (Hoffmann) | 0.235 | 77% |
| 1.82 (Besiroglu) | 0.215 | 79% |
| 2.00 | 0.189 | 81% |

The ~5pp range across all published $E$ values is small relative to the exponent sensitivity.

### 3.3 The Exponents Drive the Amplification

Varying $\alpha$ has a dramatic effect:

| $\alpha$ | $\beta$ | Improvable at 250B | FLOP penalty (1.6% loss) |
|:--|:--|:--|:--|
| 0.25 | 0.27 | 44% | ~24% |
| 0.31 | 0.33 | 13% | ~52% |
| 0.348 | 0.366 (default) | 5.0% | ~79% |
| 0.40 | 0.42 | 1.2% | ~98% |

Within the 95% CI of Besiroglu's fit ($\alpha \in [0.31, 0.39]$), the FLOP penalty ranges from ~52% to ~98% for the same raw loss degradation. This is the dominant source of uncertainty in any C_quality calculation that involves Chinchilla amplification.

## 4. Can Published 70B+ Losses Constrain the Exponents?

Several large models have published approximate training losses (many read from figures, not tabulated):

| Model | Parameters | Tokens | Approx. Loss (CE, nats) | Quality | Tokenizer |
|:--|:--|:--|:--|:--|:--|
| Chinchilla-70B | 70B | 1.4T | ~1.94 | Table | 32K |
| OPT-175B | 175B | 300B | ~1.96–2.00 | Logs | 50K |
| BLOOM-176B | 176B | 366B | 1.939 | Model card (exact) | 250K |
| LLaMA-2-70B | 70B | 2T | ~1.52 | Curve | 32K |
| Falcon-180B | 180B | 3.5T | ~1.46–1.50 | Curve | ~65K |
| Qwen-72B | 72B | 3T | ~1.50–1.55 | Curve | ~152K |
| LLaMA-3-70B | 70B | 15T | ~1.26–1.28 | Curve | 128K |
| Nemotron-4-340B | 340B | 9T | ~1.20–1.25 | Curve | 256K |
| LLaMA-3.1-405B | 405B | 15.6T | ~1.08–1.10 | Curve | 128K |

**These data cannot reliably constrain $\alpha$ and $\beta$ at 70B+ scale**, for three reasons:

1. **Tokenizer incompatibility.** Cross-entropy loss depends on vocabulary size and tokenization. A 128K-token vocabulary (LLaMA-3) assigns different loss values than a 32K vocabulary (Chinchilla) for the same text quality. Absolute loss values are not comparable across tokenizers; even loss *differences* across model families are confounded.

2. **Overtraining confounds within families.** The one family with a consistent tokenizer and data distribution (LLaMA-3: 8B/70B/405B all trained on ~15T tokens) has wildly different overtraining ratios:
   - 8B at 15T: 73× Chinchilla-optimal
   - 70B at 15T: 8.4× Chinchilla-optimal
   - 405B at 15.6T: 1.5× Chinchilla-optimal

   The 8B model at 73× overtraining is in a regime where data repetition limits further improvement — the scaling law, which assumes IID tokens, does not model this. The observed loss improvement from 70B→405B (0.18) is *larger* than from 8B→70B (0.12), producing a pattern of accelerating returns that is inconsistent with $L = E + A/N^\alpha$ for any $\alpha > 0$. This is likely caused by the 8B model's loss being inflated by data repetition, not by a change in the scaling exponent.

3. **No controlled experiment at 70B+ scale.** Constraining $\alpha$ at 70B+ would require multiple model sizes (e.g., 70B, 140B, 280B, 405B) all trained at the same overtraining ratio with the same tokenizer and data. No lab has published such an experiment. Meta, Google, and DeepSeek likely have internal scaling curves at these sizes, but they do not publish the raw $(N, D, L)$ data needed for fitting.

The [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437) provides standardized Pile-test bits-per-byte (BPB) comparisons that normalize across tokenizers (LLaMA-3.1-405B: 0.542 BPB, Qwen2.5-72B: 0.638 BPB), but these span different architectures (dense vs. MoE) and token budgets, preventing a clean $\alpha$ fit.

## 5. Alternative Quality Metrics

The FLOP-equivalent metric ($\eta_{\text{replica}}$, $C_{\text{quality}}$) is one way to measure the impact of distributed training penalties. Two alternatives provide different perspectives:

| Metric | Definition | 250B, M=72, H=200 | Behavior with model size |
|:--|:--|:--|:--|
| **Raw loss multiplier** | $L_{\text{degraded}} / L_{\text{baseline}}$ | 1.016 (1.6% worse) | Penalty decreases |
| **FLOP efficiency** ($\eta_{\text{replica}}$) | Optimally-allocated $C'$ achieving same loss, divided by $C_{\text{actual}}$ | 0.22 (78% penalty) | Penalty increases |
| **Effective model size** | Chinchilla-optimal $N'$ achieving same loss | 116B (46% of 250B) | Ratio roughly stable (~44–55%) |

The raw loss multiplier captures the direct quality impact: how much worse is the model's prediction? The FLOP efficiency captures the compute-accounting perspective: how much of the compute budget was "wasted" relative to an optimal centralized setup? The effective model size captures the capability perspective: what size of well-trained centralized model matches this quality?

These metrics diverge at large scale precisely because of the Chinchilla amplification: a small raw loss penalty translates to a large FLOP penalty when operating near the loss floor.

## 6. Impact on Report Conclusions

| Conclusion | Depends on Chinchilla amplification? | Robust to $\alpha$ uncertainty? |
|:--|:--|:--|
| Evasion is feasible with 4+ sub-CCC nodes | No (raw FLOP throughput) | **Yes** |
| Cost of evasion ($3M–$50M) | No (hardware cost) | **Yes** |
| Memory threshold constrains model size | No (direct VRAM constraint) | **Yes** |
| PP-DiLoCo enables larger models at scale | Weakly (PP overhead is compute, not loss) | **Yes** |
| $C_{\text{quality}} = 1.55 \times 10^{25}$ at 72 nodes | **Yes** (η_replica, χ) | **Uncertain** |
| 10^26 $C_{\text{quality}}$ is infeasible | **Yes** (C_quality ceiling depends on amplification) | **Uncertain** |
| Bandwidth matters moderately (6–12% range) | Weakly (η_H is independent of Chinchilla) | **Mostly yes** |

The conclusions most sensitive to $\alpha$ uncertainty are the absolute $C_{\text{quality}}$ figures — the ones that convert raw compute throughput to quality-adjusted equivalents. Hardware cost estimates, node count requirements, and the qualitative feasibility assessments are robust because they depend on FLOP throughput, not on the loss-to-FLOP conversion.

## 7. Implications

The Chinchilla amplification introduces substantial uncertainty in $C_{\text{quality}}$ at 100B+ scale, driven primarily by the scaling exponents $\alpha$ and $\beta$ rather than the irreducible loss $E$. The formal 95% CI on $\alpha$ alone produces a ~50–90% range for the FLOP penalty from a 1.6% loss degradation at 250B. The effective range may be wider (Ruan et al.).

$C_{\text{quality}}$ and effective model size measure different aspects of the distributed training penalty. $C_{\text{quality}}$ answers "how much centralized compute would match this quality?" while effective model size answers "what centralized model matches this quality?" Reporting both provides a more complete picture, since the two metrics respond differently to scaling law uncertainty.

The direction of uncertainty matters for the governance analysis. If the true $\alpha$ is *smaller* than 0.348 (the loss curve is flatter than modeled), the amplification is less severe, meaning distributed training is *more* capable than the simulator's $C_{\text{quality}}$ figures suggest — evasion is easier, not harder. Conversely, if $\alpha$ is larger, the amplification is more severe and evasion produces less capable models. The simulator's default parameters represent a mid-range estimate, not a conservative one.
