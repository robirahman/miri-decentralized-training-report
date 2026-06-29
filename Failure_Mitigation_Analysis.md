# Hardware Failures, Stragglers, and Their Mitigation in Decentralized Training

This document analyzes how random hardware failures and slow ("straggler") nodes affect the
decentralized-training evasion scenarios in *Does Distributed Training Undermine Compute
Governance?*, and how the available mitigation techniques change the picture. It is backed by the
simulator's `reliability_model()` (see [Simulator_Documentation.md §5](Simulator_Documentation.md))
and the literature review in [straggler-mitigation-notes.md](straggler-mitigation-notes.md).

## 1. The model in one paragraph

For a given node configuration and mitigation strategy, the simulator computes four knobs:
a **sync-tail factor** `f_tail` (slowest-worker wait per sync), a **goodput** `u ∈ (0,1]`
(fraction of GPU-hours doing useful work after failures, downtime, lost work and checkpoint
overhead), a **quality penalty** `eta_mit` (staleness / dropped tokens), and a **cost multiplier**
`cost_mult` (extra hardware for redundancy). Failures arrive at a per-GPU rate `λ`, so the
cluster-wide failure rate `R = G·λ` grows **linearly with total GPU count `G`** — the central reason
failures matter more at scale. Checkpoint/recovery overhead follows the Young/Daly optimum
`L(t) = c/t + R(t/2 + t_recover)`, minimized at `t* = √(2c/R)`.

The default failure rate is **λ = 2×10⁻⁵ per GPU-hour (one failure per 50,000 GPU-hours)**, which
converges across Meta's Llama 3 405B run (419 interruptions in 54 days on 16K GPUs), Epoch AI's
analysis, and this paper's Appendix F.

## 2. Effect on the paper's scenarios

**The paper's feasibility conclusions hold — and the reason is now explicit.** Appendix F assumes
synchronized workers and argues that realistic failure rates do not overturn the governance
conclusions. The reliability model confirms this, but makes clear *why*: it holds **only because the
DiLoCo family the paper relies on is inherently failure-tolerant** — replicas synchronize only every
`H` steps, so a node can drop out and rejoin between syncs without stalling the cluster. An evader
using these methods (which they would, since they are also what makes low-bandwidth training viable)
pays almost nothing for failures.

At the default `relay` strategy, the headline compute numbers are unchanged by failures:

| Scenario | Total GPUs | Cluster MTBF | C_local (no failures) | C_local (λ=2×10⁻⁵, relay) |
|---|---|---|---|---|
| ~GPT-4 scale / 10²⁵ (N=72) | 3,600 | 13.9 h | 7.06×10²⁴ | **7.06×10²⁴ (−0.0%)** |
| ~Llama-405B compute (N=625) | 31,250 | 1.6 h | 2.76×10²⁵ | **2.76×10²⁵ (−0.0%)** |
| frontier 10²⁶⁺ (N=4000) | 200,000 | 0.2 h | 9.25×10²⁵ | **9.25×10²⁵ (−0.0%)** |

(Config: `50× A100 80GB` sub-threshold nodes, 100 Mbps WAN, 1.5-year window. "No failures" =
λ set to 0; the small `relay` sync-tail is present in both columns.)

## 3. How much does the 2×10⁻⁵ failure rate hurt?

**It depends entirely on the training method — this is the key finding.** The table below shows
**goodput** (the pure multiplicative throughput hit from failures) at λ = 2×10⁻⁵:

| Method | GPT-4 scale (3.6k GPU) | Llama-405B (31k GPU) | Frontier (200k GPU) |
|---|---|---|---|
| **none** — sync, no checkpoint | 0.2% | 0.02% | 0.004% |
| **synchronous** — checkpoint + full-cluster restart | 95.5% | 80.0% | **9.0%** |
| **checkpoint_elastic** — checkpoint + elastic rejoin | 96.7% | 90.4% | 75.7% |
| **relay / threshold / backup_workers** | ≥99.9% | ≥99.9% | ≥99.9% |
| **async** (Decoupled DiLoCo) | 99.9% | 98.8% | 94.2% |

Reading this:

- **For an evader using elastic / decentralized methods, 2×10⁻⁵ is negligible (<1%)** — even at
  200,000 GPUs. This is the realistic case, and it is why failures do not rescue compute governance.
- **Naive whole-cluster-stall synchronous training** is where the failure rate bites: it loses ~5%
  at GPT-4 scale, **20%** at Llama-405B scale, and a crippling **91%** at frontier scale, because the
  cluster MTBF there falls to ~12 minutes and every failure halts all 200,000 GPUs for recovery.
- **With no checkpointing at all, training is simply impossible** at these scales (goodput → 0): a
  run that must restart from scratch on every failure never finishes.

## 4. How much do the mitigation techniques help?

Measured as training-time inflation (1/goodput) relative to the unmitigated `none` baseline at
λ = 2×10⁻⁵:

| Scale | `none` time inflation | best strategy | speed-up from mitigation |
|---|---|---|---|
| GPT-4 (3.6k GPU) | 474× | ~1.0× | **~470×** |
| Llama-405B (31k GPU) | 4,110× | ~1.0× | **~4,100×** |
| Frontier (200k GPU) | 26,299× | ~1.0× | **~26,000×** |

Concretely, in increasing order of sophistication:

1. **Checkpointing + elastic recovery** is the difference between *infeasible* and *feasible*: it
   bounds lost work to one checkpoint interval instead of the whole run. Asynchronous checkpointing
   hides ~98% of the write cost (synchronous checkpointing can otherwise slow training up to 43%).
2. **Moving from synchronous to a non-blocking elastic method** (relay / async) recovers the
   remaining 5–91% — failed nodes drop out and rejoin instead of stalling the cluster.
3. **Backup / redundant workers** buy near-perfect goodput for a flat **+5% hardware cost**
   (overprovision ~5%, proceed on the first N gradients — Chen et al.).
4. **Threshold / quorum aggregation** costs ~10–15% in training quality (dropped-token staleness)
   but zero time and zero extra hardware.
5. **Async / Decoupled DiLoCo** is the current state of the art for the failure dimension: it holds
   ~88% goodput even under high failure rates, at the price of a small (~3%) staleness penalty.

## 5. Governance takeaway

Realistic hardware failures **do not close the evasion loophole** — a determined evader adopts
DiLoCo-style elasticity essentially for free, and the headline feasibility numbers are unchanged.
What failures do is **raise the operational sophistication required** (checkpointing, elastic
recovery, fail-slow handling) and create one governance-relevant asymmetry worth highlighting:

> The cheapest known fast-recovery technique, **GPU-memory checkpointing** (storing optimizer state
> on `M` peer GPUs, per Epoch AI), needs large spare HBM per node. The simulator disables it
> automatically when the model already fills VRAM — which is exactly the regime the paper's proposed
> **HBM threshold** forces an evader into. So failures impose a *structurally larger* tax on a
> sub-threshold operator than on a frontier datacenter: **the memory cap and the failure-tolerance
> burden reinforce each other.**

In short: failures are a tax on sophistication, not a barrier to evasion — but the proposed memory
threshold makes that tax bite harder, which is a (modest) point in its favor.

## 6. Reproducing these numbers

```bash
# Full default report ends with a mitigation-comparison section:
python3 evasion_calculator.py

# Just the comparison for a given config and node count:
python3 evasion_calculator.py --mitigation-table "50x A100 80GB" 625

# Sweep the failure rate or pin a strategy:
python3 evasion_calculator.py --failure-rate 1e-4 --mitigation synchronous
```

The interactive [web simulator](simulator-web/) exposes the same controls (mitigation strategy,
failure rate, recovery time, checkpoint mode, fail-slow fraction/severity, backup overprovision) and
is verified to produce identical reliability-model outputs to the Python engine.
