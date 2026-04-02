# Covenant-72B vs. MIRI Simulator Predictions: Comparative Analysis

This document compares the actual training setup and results of **Covenant-72B** ([arXiv:2603.08163](https://arxiv.org/abs/2603.08163)) against the predictions of the MIRI Decentralized Training Simulator, which was developed before Covenant-72B was released.

---

## 1. Setup Comparison

| Parameter | MIRI Simulator (10^25 config) | Covenant-72B (actual) |
|:--|:--|:--|
| **Model size** | 144B params (dense) | 72.7B params (dense) |
| **Architecture** | Generic transformer | 80-layer decoder, GQA (64 query / 8 KV heads), RoPE, d=8192 |
| **Training tokens** | 12T | ~1.1T |
| **Total compute** | ~1.0×10²⁵ FLOP | ~4.8×10²³ FLOP (6 × 72B × 1.1T) |
| **Nodes (peers)** | 72 | 20 contributing (avg 16.9 active; ~70 unique over full run) |
| **GPUs per node** | 16× GH200 | 8× B200 |
| **Node PFLOPS** | ~15.8 PFLOPS (FP16) | ~18 PFLOPS (est. for 8× B200 FP16) |
| **WAN bandwidth** | 100 Mbps symmetric | 500 Mbps down / 110 Mbps up (asymmetric) |
| **Precision** | FP16/BF16 | BF16 mixed precision |
| **Algorithm** | DiLoCo (Streaming) | SparseLoCo (DiLoCo + Top-k sparsification + 2-bit quant + error feedback) |
| **Inner steps (H)** | 128 | 30 |
| **Compression ratio** | 16× | >146× |
| **Outer optimizer** | Not specified (assumed Nesterov) | SGD with constant α=1 (later 0.65) |
| **Local batch** | 131,072 tokens (32 seqs × 4096) | 393,216 tokens (192 seqs × 2048) |
| **Sequence length** | 4096 | 2048 |

### Key differences in setup

1. **Scale gap**: Covenant is ~20× smaller in total compute than the simulator's reference scenario. The simulator targeted 10²⁵ FLOP; Covenant used ~4.8×10²³ FLOP.
2. **Fewer, larger inner batches, fewer inner steps**: Covenant used H=30 (vs. 128) but much larger local batch sizes per step (393K vs. 131K tokens). This means each outer round processes ~12M tokens per peer rather than the simulator's ~17M tokens — a broadly similar amount per synchronization cycle.
3. **Much higher compression**: Covenant achieved >146× compression vs. the simulator's 16× baseline. This is the most dramatic difference.
4. **Asymmetric bandwidth**: Covenant had asymmetric bandwidth (110 Mbps up / 500 Mbps down), vs. the simulator's assumption of 100 Mbps symmetric. The uplink constraint (~110 Mbps) is close to the simulator's 100 Mbps assumption, validating the simulator's choice of bottleneck bandwidth.

---

## 2. Predictions That Were Validated

### 2.1 Communication is NOT the bottleneck (confirmed)

The simulator's central finding — that DiLoCo-style training over commodity internet is **compute-bound, not communication-bound** — is strongly validated.

- **Simulator prediction**: At 100 Mbps with 16× compression, the bottleneck is "compute" for all node counts tested (see Part 4A of simulator output). Even at 10 Mbps bandwidth, the regime remains compute-bound.
- **Covenant actual**: ~94.5% compute utilization, with only 70 seconds of communication overhead per 20-minute compute round. Communication is ~5.5% of wall-clock time.

This is the simulator's most important claim and it holds convincingly.

### 2.2 Streaming/overlapped synchronization works at scale (confirmed)

The simulator models streaming DiLoCo as `T_outer = max(compute, comm)` rather than `compute + comm`. Covenant confirms this model: by using Cloudflare R2 as an intermediary and overlapping uploads with the compute window, communication overhead is almost fully hidden.

### 2.3 Aggressive compression is feasible without major quality loss (confirmed)

The simulator's compression quality model (§4.6) assigned:
- 16× compression: 2% expected quality penalty
- 100× compression: 5% expected quality penalty

Covenant achieved **>146× compression** using Top-k sparsification (chunk size 4096, k=64) + 2-bit quantization + error feedback (β=0.95). The resulting model performs competitively with centralized baselines (LLaMA-2-70B, K2-65B) on standard benchmarks. The paper does not report ablations isolating compression effects, but the overall competitive results suggest the quality penalty is within the simulator's "expected" to "optimistic" range.

This validates — and arguably exceeds — the simulator's optimistic scenario at 100× compression (η_compression = 0.99).

### 2.4 Straggler handling via threshold aggregation works (confirmed)

The simulator models a threshold strategy where the system proceeds with the fastest ~90% of nodes. Covenant implemented exactly this: "slightly more active participants than aggregated contributors" for rapid replacement upon dropout, with an average of 24.4 active submissions per step but only 20 aggregated. This matches the simulator's "backup workers" or threshold strategy.

### 2.5 Hardware utilization in the expected range (confirmed)

The simulator assumes 40% MFU as a baseline, noting that INTELLECT-1 achieved 37-43% MFU. While Covenant reports "94.5% compute utilization," this metric measures the fraction of wall time spent computing (vs. waiting for communication) — it is NOT the same as MFU. The actual MFU of each B200 peer is unreported but is implicitly folded into throughput. The ~94.5% figure validates the streaming DiLoCo assumption that communication can be hidden behind compute.

---

## 3. Predictions That Need Adjustment

### 3.1 Compression ratio assumptions are too conservative

**Simulator assumption**: 16× compression as the default, with 100× as an aggressive high-end scenario requiring "FP4 + 25× sparse" or "2-bit + TopK."

**Covenant reality**: >146× achieved with relatively simple techniques (chunk-wise Top-k + 2-bit quantization + error feedback) at 72B scale, producing competitive results.

**Recommended adjustment**: The simulator should update its default compression ratio from 16× to at least 100×, and model 150-200× as the high-end scenario. The compression quality penalty at 100-150× should be revised from the current "expected 5%" to "expected 2-3%," since Covenant demonstrates competitive quality at 146×. The SparseLoCo paper's error-feedback mechanism appears to largely solve the quality degradation that motivated the conservative assumptions.

### 3.2 Inner step count (H) assumptions may need recalibration

**Simulator assumption**: H=128 as the reference, with the quality penalty modeled as η_H = max(0.4, (1 - α·log₁₀(H))).

**Covenant chose**: H=30, which is dramatically lower than the simulator's H=128.

Why did Covenant use H=30 rather than a higher value?
- Their much higher compression ratio (146×) meant communication time was already very low (~70 seconds), so there was little pressure to increase H further to amortize communication.
- At H=30, the simulator's α formula gives a very small penalty (~1-2% for a 72B model).
- The Charles et al. (2025) scaling laws, which the simulator cites, used H=30 as their reference configuration.

**Implication**: The simulator's use of H=128 is conservative for high-compression scenarios. When compression exceeds ~100×, the optimal H shifts much lower (toward 20-50), because the communication time is so small that there's no benefit to additional inner steps, and lower H gives better convergence. The simulator should model H as a function of compression ratio and bandwidth, not as a fixed parameter. At 146× compression over 100 Mbps, the sync time for a 72B model is:

```
72B × 2 bytes × 8 bits/byte / 146 / 100 Mbps ≈ 79 seconds (upload)
```

This is close to Covenant's measured 70 seconds, confirming the simulator's communication time model is accurate. But with only 70 seconds of sync time, H=30 with ~20 minutes of compute already gives 94.5% utilization. There is no reason to push H higher.

### 3.3 The simulator should model asymmetric bandwidth

**Simulator assumption**: Symmetric bandwidth (100 Mbps up and down).

**Covenant reality**: 500 Mbps down / 110 Mbps up. The uplink is the bottleneck for uploading pseudo-gradients; the downlink bottleneck matters for downloading aggregated updates. With high compression, the asymmetry matters less (both directions are fast enough), but the simulator should still model separate upload and download bandwidths, since consumer/business internet connections are typically asymmetric.

**Recommended change**: Add `bandwidth_up_bps` and `bandwidth_down_bps` parameters. The sync time becomes:
```
T_sync = (V_bits / BW_up) + (V_bits / BW_down) + Latency
```
instead of `2 × V_bits / BW`.

### 3.4 The replica count penalty may be less severe than modeled

**Simulator model**: At 20 replicas with a 72B model, the simulator's formula gives:
- β(72B) = 1.0923 × (72×10⁹)^(-0.2342) ≈ 0.0053
- L_replicas(20, 72B) = 20^0.0053 ≈ 1.016 (1.6% loss increase)
- After Chinchilla amplification, this becomes a moderate FLOP penalty.

**Covenant reality**: With ~17-20 contributing peers (averaged), Covenant-72B achieves:
- ARC-Challenge: 56.8% (vs. LLaMA-2-70B: 57.4%) — within 0.6pp
- MMLU: 67.1% (vs. LLaMA-2-70B: 65.6%) — Covenant is *better*
- HellaSwag: 80.6% (vs. 84.3%) — 3.7pp gap
- WinoGrande: 75.9% (vs. 80.4%) — 4.5pp gap

The mixed results make it hard to isolate the replica penalty from other confounders (different training data, different model size, different token count — Covenant used 1.1T tokens vs. LLaMA-2's 2T). However, the overall competitive performance at roughly half the training tokens suggests the combined penalties from distributed training (compression + replica divergence + fewer tokens) are modest.

### 3.5 Object storage as aggregation topology is not modeled

The simulator assumes a direct upload-to-aggregator-and-download model. Covenant used **Cloudflare R2 object storage** as an intermediary: peers upload compressed pseudo-gradients to R2, a coordinator reads them, aggregates, and writes back to R2, and peers download the result.

This pattern has different latency/throughput characteristics than direct peer-to-aggregator communication:
- **Advantage**: Commodity cloud storage is cheap, globally distributed, and handles burst traffic well.
- **Disadvantage**: Added latency from two storage hops (upload → R2 → aggregator → R2 → download).

The simulator's sync time model should probably add a constant overhead (10-30 seconds) for object-storage-based aggregation.

### 3.6 Dynamic participation is not modeled

The simulator assumes a fixed number of nodes throughout training. Covenant had **dynamic participation**: 70+ unique peers over the course of training, but only ~17-20 contributing at any given time. Peers joined and left freely, managed by a blockchain-based coordination layer.

This has implications for:
- **Straggler model**: The straggler factor should account for the ability to drop and replace slow peers, which Covenant effectively demonstrated.
- **Replica count**: The effective replica count varies over time, which the simulator's fixed-N model doesn't capture.
- **Robustness**: Dynamic participation means the training can continue even as individual peers fail, which the simulator doesn't model.

---

## 4. What Covenant Tells Us About Large-Scale Predictions

### 4.1 The simulator's 10²⁵ FLOP scenario remains plausible

Covenant at ~4.8×10²³ FLOP is roughly 20× smaller than the simulator's reference 10²⁵ scenario. Scaling from Covenant's setup:
- Increasing from 20 to 72 nodes: The simulator's straggler model (f(72) = 1.31) is plausible given Covenant's experience with dynamic participation.
- Increasing model size from 72B to 144B: Requires more VRAM per node (Covenant used 8× B200; the simulator assumed 16× GH200 with 2304 GB). The model fitting constraint is hardware-specific.
- Increasing tokens from 1.1T to 12T: Linear scaling of training time. No fundamental obstacle.

The key question is whether the efficiency observed at 20 nodes and 72B maintains at 72 nodes and 144B. The simulator predicts η ≈ 0.86 for this configuration with 16× compression. With Covenant-level compression (146×), the communication overhead would be even lower, suggesting η could be higher than the simulator predicts.

### 4.2 Higher compression enables more aggressive scaling

The most significant implication of Covenant is that **146× compression works at 72B scale**. If this holds at 144B+ scale (which is plausible given that compression quality generally improves with model size — larger models have more redundant pseudo-gradients), then:

1. The required H for compute-bound operation drops dramatically (from H=200+ to H=30-50).
2. Lower H means less replica drift and better convergence.
3. The effective efficiency η should be **higher** than the simulator predicts with 16× compression.

For the simulator's 72-node, 144B scenario:
- With 16× compression (simulator default): H_min=200, η=0.858
- With 146× compression (Covenant-validated): H_min≈22 (computed from sync time), η would improve to ~0.90-0.92 due to the lower H penalty.

### 4.3 The 10²⁷ FLOP scenarios need updated compression assumptions

The simulator's 10²⁷ configurations (Part 2 of output) show various configurations achieving η of 0.83-0.89. These were computed with 16× compression. If 146× compression is used instead:

- The bandwidth sensitivity analysis (Part 4A) shows that going from 16× to higher compression primarily reduces H_min, which reduces the H-penalty. At 100× compression with hierarchical mode, η reaches 0.89.
- With 146×+ compression, even flat (non-hierarchical) DiLoCo at thousands of nodes could achieve η ≈ 0.87-0.90, since the communication overhead becomes negligible.
- The "Hierarchical + 100x compression" configurations were already the simulator's most optimistic scenario. Covenant's results suggest this optimistic scenario is, if anything, conservative.

---

## 5. Recommended Simulator Updates

Based on the Covenant-72B results, the following adjustments are recommended:

### 5.1 Update compression defaults
- **Default compression ratio**: 16× → 100×
- **High-end compression ratio**: 100× → 200×
- **Compression quality at 100×**: "expected" from 0.95 → 0.97
- **Compression quality at 200×**: new row, "expected" 0.95

### 5.2 Model H as a derived parameter
Instead of fixing H=128, compute H_optimal as:
```
H_optimal = max(H_min_for_compute_bound, H_min_for_quality)
```
Where H_min_for_quality ≈ 20-50 (based on DiLoCo literature), and H_min_for_compute_bound is the minimum H such that compute time ≥ sync time. Covenant demonstrates that H=30 works well when compression is high enough.

### 5.3 Add asymmetric bandwidth support
Model upload and download bandwidths separately. Use the lower (typically upload) bandwidth for the dominant direction.

### 5.4 Add a "Covenant-validated" configuration
Create a new config file matching Covenant's actual setup for direct comparison:
```json
{
  "model": {"parameters": 72700000000, "precision_bytes": 2},
  "hardware": {
    "num_nodes": 20,
    "flop_per_node": 18000000000000000,
    "memory_per_node": 1440000000000,
    "bandwidth_bps": 110000000,
    "mfu": 0.40
  },
  "algorithm": {
    "name": "DiLoCo",
    "inner_steps": 30,
    "compression_ratio": 146.0,
    "streaming": true
  },
  "training": {
    "total_tokens": 1100000000000,
    "local_batch_tokens": 393216
  }
}
```

### 5.5 Revise the compression quality table (§4.6)

Current table anchored at experiments up to 15B scale. Covenant provides a new anchor point:

| Compression Ratio | Old Expected | New Expected (with Covenant anchor) |
|:--|:--|:--|
| 16× | 0.98 | 0.98 (unchanged) |
| 100× | 0.95 | 0.97 |
| 146× | (not modeled) | 0.96-0.97 |
| 200× | (not modeled) | 0.95 |

### 5.6 Consider SparseLoCo as a distinct algorithm

The simulator currently models "DiLoCo" as a single algorithm. SparseLoCo (Top-k sparsification with error feedback) is a qualitatively different approach that achieves higher compression than the quantization-only methods the simulator was designed around. The error-feedback mechanism (β=0.95) ensures that information lost to sparsification is eventually transmitted, which is why quality holds at 146×. The simulator could model this explicitly as an algorithm variant with its own compression-quality curve.

---

## 6. Summary

| Aspect | Simulator Prediction | Covenant Reality | Assessment |
|:--|:--|:--|:--|
| Compute-bound regime | ✅ Yes | ✅ Yes (94.5% utilization) | **Validated** |
| Streaming overlap works | ✅ Yes | ✅ Yes | **Validated** |
| 16× compression feasible | ✅ Yes | Exceeded (146×) | **Validated and exceeded** |
| Quality competitive with centralized | ✅ At modest penalty | ✅ Competitive with LLaMA-2-70B | **Validated** |
| Straggler mitigation | ✅ Threshold works | ✅ Dynamic participation | **Validated** |
| H=128 optimal | ⚠️ Assumed | H=30 used (lower is better when compression is high) | **Needs adjustment** |
| 100 Mbps symmetric | ⚠️ Assumed | 110 up / 500 down (asymmetric) | **Needs adjustment** |
| 100× compression quality = 5% penalty | ⚠️ Assumed | Appears ≤2-3% at 146× | **Too conservative** |
| Fixed node count | ⚠️ Assumed | Dynamic participation (10-24 active) | **Needs modeling** |

**Bottom line**: Covenant-72B validates the simulator's core thesis — that decentralized training over commodity internet is feasible, compute-bound, and produces competitive models. The main surprise is that compression works even better than the simulator's "expected" case, enabling lower synchronization intervals (H=30) and higher effective efficiency than predicted. The simulator's predictions for larger-scale scenarios (10²⁵-10²⁷ FLOP) are, if anything, conservative in light of these results.
