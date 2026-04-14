# Overtraining Ratios in Published Language Models

This document surveys the overtraining ratios used in published large language models, providing empirical context for the simulator's overtraining penalty (chi) and the model-size/token tradeoffs that shape realistic distributed training scenarios.

## 1. Definition

The **overtraining ratio** is the ratio of actual training tokens to Chinchilla-optimal tokens:

$$\text{OT} = \frac{D_{\text{actual}}}{D_{\text{Chinchilla}}} = \frac{D}{25.6 \times N}$$

where the denominator uses the corrected Chinchilla estimate from [Besiroglu et al. (2024)](https://arxiv.org/abs/2404.10102). For MoE models, $N$ is the number of **active** parameters (which determines per-token compute), not total parameters.

At OT = 1, the model is Chinchilla-optimal: training compute is allocated equally between model size and data. At OT > 1, the model is overtrained (more tokens than compute-optimal). At OT < 1, the model is undertrained (fewer tokens than compute-optimal).

## 2. Published Models by Overtraining Ratio

### 2.1 Extreme Overtraining (OT > 100x)

All entries in this range are sub-10B models. Labs train tiny models on massive token budgets because training cost (proportional to $N \times D$) is low while inference savings from a smaller $N$ are enormous.

| Model | Year | Params | Tokens | OT ratio | Source |
|:--|:--|:--|:--|:--|:--|
| Qwen 3 0.6B | 2025 | 0.6B | 36T | ~2,340x | [Qwen Team (2025)](https://arxiv.org/abs/2505.09388) |
| Qwen 2.5 0.5B | 2024 | 0.5B | 18T | ~1,406x | [Qwen Team (2024)](https://arxiv.org/abs/2412.15115) |
| Qwen 3 1.7B | 2025 | 1.7B | 36T | ~828x | [Qwen Team (2025)](https://arxiv.org/abs/2505.09388) |
| SmolLM2 135M | 2024 | 135M | 2T | ~579x | [Allal et al. (2025)](https://arxiv.org/abs/2502.02737) |
| SmolLM2 360M | 2024 | 360M | 4T | ~435x | [Allal et al. (2025)](https://arxiv.org/abs/2502.02737) |
| Qwen 3 4B | 2025 | 4B | 36T | ~352x | [Qwen Team (2025)](https://arxiv.org/abs/2505.09388) |
| SmolLM2 1.7B | 2024 | 1.7B | 11T | ~253x | [Allal et al. (2025)](https://arxiv.org/abs/2502.02737) |
| Qwen 3 8B | 2025 | 8B | 36T | ~176x | [Qwen Team (2025)](https://arxiv.org/abs/2505.09388) |
| SmolLM3 3B | 2025 | 3B | 11.2T | ~146x | [Hugging Face (2025)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| Qwen 2.5 7B | 2024 | 7B | 18T | ~100x | [Qwen Team (2024)](https://arxiv.org/abs/2412.15115) |
| Qwen 3 14B | 2025 | 14B | 36T | ~100x | [Qwen Team (2025)](https://arxiv.org/abs/2505.09388) |

### 2.2 Heavy Overtraining (OT 10--100x)

The current norm for 7B--70B models trained in 2024--2025.

| Model | Year | Params (active) | Tokens | OT ratio | Source |
|:--|:--|:--|:--|:--|:--|
| LLaMA 4 Scout | 2025 | 17B active | 40T | ~92x | [Meta (2025)](https://arxiv.org/abs/2504.16227) |
| Gemma 3 1B | 2025 | 1B | 2T | ~78x | [Google (2025)](https://arxiv.org/abs/2503.19786) |
| LLaMA 3 8B | 2024 | 8B | 15T | ~73x | [Meta (2024)](https://arxiv.org/abs/2407.21783) |
| LLaMA 4 Maverick | 2025 | 17B active | 22T | ~51x | [Meta (2025)](https://arxiv.org/abs/2504.16227) |
| Gemma 2 9B | 2024 | 9B | 8T | ~35x | [Google (2024)](https://arxiv.org/abs/2408.00118) |
| Gemma 1 2B | 2024 | 2B | 2T | ~39x | [Google (2024)](https://arxiv.org/abs/2403.08295) |
| Gemma 1 7B | 2024 | 7B | 6T | ~33x | [Google (2024)](https://arxiv.org/abs/2403.08295) |
| Phi-4 14B | 2024 | 14B | 9.8T | ~27x | [Microsoft (2024)](https://arxiv.org/abs/2412.08905) |
| OLMo 2 7B | 2024 | 7B | 4T | ~22x | [AI2 (2024)](https://arxiv.org/abs/2501.00656) |
| Phi-2 2.7B | 2023 | 2.7B | 1.4T | ~20x | [Microsoft (2023)](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) |
| Yi-6B | 2023 | 6B | 3.1T | ~20x | [01.AI (2024)](https://arxiv.org/abs/2403.04652) |
| Gemma 2 27B | 2024 | 27B | 13T | ~19x | [Google (2024)](https://arxiv.org/abs/2408.00118) |
| DeepSeek-V3 | 2024 | 37B active | 14.8T | ~16x | [DeepSeek (2024)](https://arxiv.org/abs/2412.19437) |
| DeepSeek-V2 | 2024 | 21B active | 8.1T | ~15x | [DeepSeek (2024)](https://arxiv.org/abs/2405.04434) |
| OLMo 2 13B | 2024 | 13B | 5T | ~15x | [AI2 (2024)](https://arxiv.org/abs/2501.00656) |
| Phi-3 medium 14B | 2024 | 14B | 4.8T | ~13x | [Microsoft (2024)](https://arxiv.org/abs/2404.14219) |
| LLaMA 2 7B | 2023 | 7B | 2T | ~11x | [Touvron et al. (2023)](https://arxiv.org/abs/2307.09288) |

### 2.3 Moderate Overtraining (OT 1--10x)

| Model | Year | Params (active) | Tokens | OT ratio | Source |
|:--|:--|:--|:--|:--|:--|
| Qwen 2.5 72B | 2024 | 72B | 18T | ~9.8x | [Qwen Team (2024)](https://arxiv.org/abs/2412.15115) |
| LLaMA 3 70B | 2024 | 70B | 15T | ~8.4x | [Meta (2024)](https://arxiv.org/abs/2407.21783) |
| OLMo 2 32B | 2025 | 32B | 6T | ~7.3x | [AI2 (2025)](https://arxiv.org/abs/2501.00656) |
| LLaMA 2 13B | 2023 | 13B | 2T | ~6.0x | [Touvron et al. (2023)](https://arxiv.org/abs/2307.09288) |
| LLaMA 1 7B | 2023 | 7B | 1T | ~5.6x | [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971) |
| Qwen 2 72B | 2024 | 72B | 7T | ~3.8x | [Qwen Team (2024)](https://arxiv.org/abs/2407.10671) |
| Yi-34B | 2023 | 34B | 3.1T | ~3.6x | [01.AI (2024)](https://arxiv.org/abs/2403.04652) |
| LLaMA 1 13B | 2023 | 13B | 1T | ~3.0x | [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971) |
| LLaMA 1 33B | 2023 | 33B | 1.4T | ~1.7x | [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971) |
| Qwen 1 72B | 2023 | 72B | 3T | ~1.6x | [Bai et al. (2023)](https://arxiv.org/abs/2309.16609) |
| LLaMA 3 405B | 2024 | 405B | 15T | ~1.4x | [Meta (2024)](https://arxiv.org/abs/2407.21783) |
| DeepSeek-V1 67B | 2024 | 67B | 2T | ~1.2x | [DeepSeek (2024)](https://arxiv.org/abs/2401.02954) |
| LLaMA 2 70B | 2023 | 70B | 2T | ~1.1x | [Touvron et al. (2023)](https://arxiv.org/abs/2307.09288) |

### 2.4 Near or Below Chinchilla-Optimal (OT <= 1x)

Pre-Chinchilla large models were severely undertrained by modern standards.

| Model | Year | Params | Tokens | OT ratio | Source |
|:--|:--|:--|:--|:--|:--|
| Pythia 6.9B | 2023 | 6.9B | 300B | ~1.7x | [Biderman et al. (2023)](https://arxiv.org/abs/2304.01373) |
| Pythia 12B | 2023 | 12B | 300B | ~0.98x | [Biderman et al. (2023)](https://arxiv.org/abs/2304.01373) |
| LLaMA 1 65B | 2023 | 65B | 1.4T | ~0.84x | [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971) |
| Chinchilla 70B | 2022 | 70B | 1.4T | ~0.78x | [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) |
| Falcon-180B | 2023 | 180B | 3.5T | ~0.76x | [Almazrouei et al. (2023)](https://arxiv.org/abs/2311.16867) |
| BLOOM 176B | 2022 | 176B | 366B | ~0.081x | [BigScience (2022)](https://arxiv.org/abs/2211.05100) |
| GPT-3 175B | 2020 | 175B | 300B | ~0.067x | [Brown et al. (2020)](https://arxiv.org/abs/2005.14165) |
| PaLM 540B | 2022 | 540B | 780B | ~0.056x | [Chowdhery et al. (2022)](https://arxiv.org/abs/2204.02311) |
| Gopher 280B | 2021 | 280B | 300B | ~0.042x | [Rae et al. (2021)](https://arxiv.org/abs/2112.11446) |

## 3. Trends

### 3.1 Historical Shift

The industry underwent a dramatic reversal between 2020 and 2025:

| Era | Typical OT for large models | Typical OT for small models | Governing assumption |
|:--|:--|:--|:--|
| 2020--2022 | 0.04--0.08x (undertrained) | 0.5--1x | "Bigger model = better" |
| 2022--2023 | 0.8--1.1x (Chinchilla-optimal) | 3--6x | "Scale model and data equally" |
| 2024--2025 | 1.4--10x | 30--2,340x | "Overtrain for inference efficiency" |

[Epoch AI (2025)](https://epoch.ai/data-insights/training-tokens-per-parameter/) documented that the median tokens-per-active-parameter ratio for open-weight models grew from ~10 in 2022 to ~250--300 in 2025, a compound growth rate of ~3.1x per year.

### 3.2 Overtraining Scales Inversely with Model Size

Within a model family, the smallest variants receive the most extreme overtraining:

| Family | Smallest variant | OT | Largest variant | OT |
|:--|:--|:--|:--|:--|
| LLaMA 3 | 8B | 73x | 405B | 1.4x |
| Qwen 3 | 0.6B | 2,340x | 235B (22B active) | 64x |
| Qwen 2.5 | 0.5B | 1,406x | 72B | 9.8x |
| Gemma 2 | 2B | 39x | 27B | 19x |
| LLaMA 2 | 7B | 11x | 70B | 1.1x |

This pattern is economically rational. Training cost is $O(N \times D)$, while inference cost per token is $O(N)$. For a model that will serve many inference requests, spending more on training ($D$) to reduce inference cost ($N$) is cost-effective. The tradeoff favors extreme overtraining of small models because the marginal training cost of each additional token is low when $N$ is small.

### 3.3 Maximum Published Overtraining Ratios by Model Scale

| Scale | Highest published OT | Model |
|:--|:--|:--|
| < 1B | ~2,340x | Qwen 3 0.6B (36T tokens) |
| 1--10B | ~828x | Qwen 3 1.7B (36T tokens) |
| 10--70B | ~100x | Qwen 3 14B (36T tokens) |
| 70B+ (dense) | ~10x | Qwen 2.5 72B (18T tokens) |
| 70B+ (MoE active) | ~92x | LLaMA 4 Scout (17B active, 40T tokens) |

## 4. Diminishing Returns and Practical Ceilings

### 4.1 Returns Are Real but Diminishing

[Meta (2024)](https://arxiv.org/abs/2407.21783) reported that LLaMA 3 8B performance continued to improve log-linearly up to 15T tokens (73x Chinchilla-optimal), and the 70B model was "still learning" at 15T. This is the strongest industrial evidence that overtraining yields substantial gains far beyond Chinchilla-optimal.

[Sardana et al. (2024)](https://arxiv.org/abs/2401.00448) ("Beyond Chinchilla-Optimal", ICML 2024) trained 47 models and found that quality continues to improve up to 10,000x overtraining in controlled experiments, though the marginal benefit per additional token decreases. They showed that the compute-optimal overtraining ratio depends on expected inference demand: models serving billions of queries justify much higher overtraining ratios.

### 4.2 Catastrophic Overtraining

[Springer et al. (2025)](https://arxiv.org/abs/2503.19206) ("Overtrained Language Models Are Harder to Fine-Tune") identified a practical ceiling: for OLMo 1B, fine-tuning effectiveness degraded beyond ~2.5T tokens (~1,000x Chinchilla-optimal). The mechanism is progressive parameter sensitivity: overtrained weights become brittle and respond poorly to SFT, RLHF, and other post-training methods.

This suggests a distinction between:
- **Pre-training loss**: continues to decrease at extreme overtraining ratios
- **Post-training utility**: may peak and then degrade if the base model is too overtrained to fine-tune effectively

### 4.3 Data Quality at Extreme Ratios

At extreme overtraining ratios, data quality becomes a binding constraint. The Phi series (Microsoft) demonstrated that curated and synthetic data can achieve strong performance at moderate OT ratios (20--30x), while the Qwen 3 and SmolLM series show that combining quality with quantity at extreme scale (100--2000x) yields state-of-the-art results for small models.

## 5. Implications for the Simulator

The simulator's chi ($\chi$) parameter penalizes overtraining by converting the gap between actual and Chinchilla-optimal token counts into a FLOP-equivalent penalty via the scaling law. The empirical data shows:

1. **OT ratios of 1--10x are standard for 70B+ models and should not be treated as unusual.** The simulator's chi penalty at these ratios is modest (chi > 0.8 typically).

2. **OT ratios of 10--100x are standard for 7B--14B models.** If the simulator is used to evaluate distributed training of small models, high overtraining ratios are realistic, not pathological.

3. **OT > 100x is common for sub-3B models but rare above 14B.** At these ratios, chi drops substantially, reflecting the diminishing returns that are real but accepted by the industry for inference-cost reasons. The chi penalty accurately captures the compute inefficiency, but labs accept this inefficiency because it produces cheaper-to-serve models.

4. **For the governance analysis**, the relevant question is whether a distributed training cluster can achieve enough C_local to train a model that poses capability concerns. The typical frontier model in 2025 (70B--400B dense, or 17B--37B active MoE) is trained at 1--16x overtraining. The simulator's scenarios at these scales produce chi > 0.7, meaning overtraining is a modest efficiency penalty, not a showstopper.
