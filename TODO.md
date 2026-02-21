# Project TODOs

## Simulator Extensions
- [x] **Pipeline Parallelism (PP) Logic:** Implement timing models for sharding models across nodes (required for models > 1T parameters).
- [ ] **Network Latency:** Add `latency_ms` parameters to model the "bubble" overhead in Pipeline Parallelism.
- [ ] **Heterogeneity:** Model clusters with mismatched GPU types (e.g., some A100s mixed with GH200s).

## Analysis & Policy
- [ ] **Feasibility at 10^26/10^27 FLOPs:** Identify the exact point where WAN-based training becomes physically impossible due to synchronization overhead.
- [ ] **Policy Lever Testing:**
    *   Effect of 10 Mbps vs 100 Mbps caps.
    *   Effect of lowering the "Covered Compute Cluster" threshold from 16 GPUs to 4 or 8.
    *   Effect of memory-based reporting requirements (e.g., total aggregate VRAM).

## Reporting
- [ ] Draft formal memo for MIRI Technical Governance Team on technical feasibility of treaty evasion.
