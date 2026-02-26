# Compression Quality: Evidence, Extrapolation, and Open Questions

The efficiency model in the [Governance Analysis](Governance_Analysis.md) (Section 4.2) includes compression quality factors that reduce C_local by 2-10% depending on the compression ratio and scenario. This document presents the evidence underlying those estimates. The analysis distinguishes three types of compression:

1. **Pseudo-gradient compression** (DiLoCo sync): Weight deltas (pseudo-gradients) are quantized and sparsified before transmission across WAN every H inner steps. This is the primary compression type in all modes. 16x default, 100x aggressive.
2. **Activation compression** (PP-Group DiLoCo only): Hidden-state activation tensors are compressed between pipeline stages at every micro-batch. Occurs only when the model is sharded across multiple co-located nodes. 4x default.
3. **Precision reduction** (FP16 → FP8): Training compute precision. Affects throughput, memory, and the base communication volume, but is not a separate quality factor — FP8 training quality is well-validated and treated as lossless.

## 1. What the Simulator Assumes

The simulator's compression quality model applies multiplicative factors to efficiency:

$$\eta = \eta_H \times \eta_{\text{pg-compression}} \times \eta_{\text{replicas}} \times \eta_{\text{act-compression}}$$

where $\eta_{\text{act-compression}} = 1$ for flat DiLoCo (no pipeline stages). The sync interval penalty ($\eta_H$) is the dominant term and is well-calibrated against empirical data (DiLoCo Scaling Laws, 2503.09799). The pseudo-gradient compression quality ($\eta_{\text{pg-compression}}$), replica penalty ($\eta_{\text{replicas}}$), and activation compression quality ($\eta_{\text{act-compression}}$) are estimated from the literature with varying levels of confidence.

**What the simulator does NOT model:**
- Error feedback (the mechanism for accumulating compression residuals) — whether it is used or needed
- Interaction effects between compression, replica count, and H
- Compression-induced outlier accumulation over long training runs
- The specific choice of compressor (TopK vs random-K, linear vs statistical quantization)

## 2. Pseudo-Gradient Compression: Empirical Evidence

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

## 3. What Is Validated vs. Extrapolated

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

## 4. Key Risk Factors

**Error feedback:** At 2-bit compression and below, error feedback (accumulating the difference between the original and compressed value for the next sync) is critical for convergence. MuLoCo and SparseLoCo both find that error feedback is essential at aggressive compression ratios. However, the Streaming DiLoCo paper does **not** use error feedback, and its lossless results at FP4 may not extend to more aggressive compression without it. Error feedback adds memory overhead (one full model copy for the error buffer — ~3.8 TB additional for a 240B model) and is not included in the simulator's memory model.

**Compounding unknowns:** The scenarios in [Section 8 of the Governance Analysis](Governance_Analysis.md) combine multiple factors that have not been tested jointly: (a) large models (91-240B), (b) many replicas (2,000+), (c) high H values (200-2000), and (d) aggressive compression (100x). Each factor has been studied somewhat independently, but the **interaction** is unknown. Even if each factor individually causes <5% degradation, compounding could produce larger effects.

**Scale-dependent outliers:** Quantization during training is known to cause outlier accumulation over long training runs. At 100B+ scale with trillions of tokens, this could be worse than at 1B scale with billions of tokens. Recent work on FP4 training ([Quartet](https://arxiv.org/abs/2505.14669), 2025) shows that most loss gap arises from forward-pass quantization, not gradient quantization — but this applies to per-step gradients, not DiLoCo pseudo-gradients.

## 5. Impact on Analysis Conclusions

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

The core governance conclusions (Sections 6-7 of the Governance Analysis) are robust across all compression quality scenarios because they rely on 16x pseudo-gradient compression, which is well-validated. The 10^27 scenarios (Section 8) carry meaningful uncertainty, particularly Config F's 100x compression assumption, and should be interpreted as estimates with a +-10% error bar rather than precise predictions.

## 6. Activation Compression: Evidence for PP-Group DiLoCo

Activation compression (compressing hidden-state tensors between pipeline stages) is distinct from pseudo-gradient compression. The key difference: activation errors accumulate through the pipeline (each stage boundary introduces error in both forward and backward passes), while pseudo-gradient errors average across replicas.

| Paper | Scale | Compression | Quality Impact | Notes |
|:--|:--|:--|:--|:--|
| [COAT](https://arxiv.org/abs/2410.19313) (ICLR 2025) | GPT-2, LLaMA-7B | FP8 activations (2x) | **Near-lossless** | Online dynamic quantization for all linear layers |
| [SWARM](https://arxiv.org/abs/2301.11913) (ICML 2023) | 1-7B, distributed PP | FP8 activations (2x) | **Near-lossless** | Validated in heterogeneous distributed PP setting |
| [TAH-Quant](https://arxiv.org/abs/2506.06984) (2025) | GPT-2 XL, Qwen2.5-3B | 4-bit adaptive (4x) | **Near-lossless** | Token-aware heterogeneous quantization |
| [GACT](https://proceedings.mlr.press/v162/liu22v.html) (ICML 2022) | ResNet, BERT, GPT-2 | 4-bit (4x) → 8x memory | **<0.5% degradation** | Exact gradient computation with quantized activations |
| [Protocol Models](https://arxiv.org/abs/2504.01943) (2025) | 8B Transformer | 100x (subspace decomposition) | **Lossless** | Decomposes activations into low-rank subspaces; not validated at 100B+ |

**Safe defaults for WAN pipeline parallelism:**

- **FP8 (2x):** Universally validated, no measurable quality loss. Suitable as a baseline.
- **4-bit adaptive (4x):** Near-lossless per recent literature. The default in this analysis. Per-boundary quality: 0.995 (expected), compounding to 0.990 at 2 stages, 0.970 at 4 stages.
- **10-16x:** Requires structural methods (subspace decomposition, protocol learning). Validated only at 8B. High risk of accumulation errors at 100B+ with deep pipelines.

**Key risk: depth accumulation.** With $S$ pipeline stages, each activation tensor passes through $2(S-1)$ stage boundaries (forward and backward). At 4x compression with expected quality 0.995 per boundary: $\eta_{\text{act}} = 0.995^{2(S-1)}$. At $S=4$: $\eta_{\text{act}} = 0.97$ (3% penalty). At $S=8$: $\eta_{\text{act}} = 0.93$ (7% penalty). This compounds with all other efficiency factors. Deep pipelines ($S > 6$) carry significant activation compression risk.
