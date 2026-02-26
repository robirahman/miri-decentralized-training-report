# Project TODOs

## Simulator Extensions
- [x] **Pipeline Parallelism (PP) Logic:** Implement timing models for sharding models across nodes (required for models > 1T parameters).
- [ ] **Network Latency:** Add `latency_ms` parameters to model the "bubble" overhead in Pipeline Parallelism.
- [ ] **Heterogeneity:** Model clusters with mismatched GPU types (e.g., some A100s mixed with GH200s).

## Analysis & Policy
- [x] **Feasibility at 10^26/10^27 FLOPs:** 10^26 achievable under all network conditions; 10^27 achievable with hierarchical+100x even at 10 Mbps (Section 9).
- [x] **Policy Lever Testing:**
    *   Effect of 10 Mbps vs 100 Mbps caps: 3-6% reduction in C_local; ineffective (Section 9.2).
    *   Effect of lowering the CCC threshold from 16 to 4 or 8: cost unchanged (~$3M for 10^24); model quality degrades; high collateral (Section 10.1).
    *   Effect of memory-based reporting requirements: closes 48x A100 exploit but cost unchanged; 1 TB recommended (Section 10.2).
- [x] **Treaty modification analysis:** Full evaluation of 9 countermeasures in Governance_Analysis.md Section 10.

## Reporting
- [ ] Draft formal memo for MIRI Technical Governance Team on technical feasibility of treaty evasion.
