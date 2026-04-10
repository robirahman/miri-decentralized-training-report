# Preventing Illicit Decentralized Training

A technical feasibility study for the [MIRI technical governance team](https://arxiv.org/abs/2511.10783), analyzing whether actors could evade an international AI compute treaty by distributing training across small, unregistered nodes connected over the internet.

## Background

MIRI's proposed international treaty requires all compute clusters above 16 H100-equivalents to be registered and monitored, and prohibits training runs above 10^24 FLOP. However, an evader could attempt to use many sub-threshold nodes running decentralized training protocols (e.g., DiLoCo) over WAN connections, where each individual node has plausible deniability.

This project investigates:

1. **Technical feasibility** of decentralized training at 10^25 FLOP and beyond using sub-threshold hardware
2. **Simulation** of training time, bandwidth requirements, and bottlenecks under realistic network constraints
3. **Governance countermeasures** — treaty amendments and enforcement mechanisms to close the loophole

## Contents

- **[Governance Analysis](Governance_Analysis.md)** — End-to-end analysis of the evasion scenario: hardware selection, training duration, achievable model scale, and detection strategies
- **[Decentralized Training Simulator](Decentralized%20training%20simulator/)** — Python simulator estimating time-to-train for LLMs over bandwidth-constrained networks, comparing DiLoCo, DDP, and pipeline parallelism
- **[Web Simulator](simulator-web/)** — Interactive browser-based version of the simulator
- **[Compression Quality](Compression_Quality.md)** — Analysis of gradient compression techniques and their impact on training quality
- **[Scaling Law Uncertainty](Scaling_Law_Uncertainty.md)** — Analysis of how Chinchilla scaling law parameter uncertainty affects the FLOP-equivalence conversion
- **[Traffic Fingerprinting Analysis](Traffic_Fingerprinting_Analysis.md)** — How network traffic analysis could detect illicit distributed training
- **[CCC VRAM Amendment](CCC_VRAM_Amendment.md)** — Proposed treaty amendment adding a VRAM threshold to the covered compute cluster definition

## Key Findings

An evader operating 50 A100-80GB nodes (just below the 16 H100-equivalent threshold) connected at 100 Mbps could use DiLoCo to train models up to ~250B parameters, achieving over 10^25 FLOP in roughly 1.5 years. Detection is difficult because each node is individually indistinguishable from legitimate sub-threshold use. The analysis proposes additional monitoring based on VRAM thresholds, network traffic fingerprinting, and hardware procurement patterns.
