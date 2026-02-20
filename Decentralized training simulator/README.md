# Decentralized Training Simulator

This simulator estimates the feasibility and time-to-train for large language models using decentralized training techniques over heterogeneous and bandwidth-constrained networks (e.g., the internet).

## Goals
1.  **Feasibility Analysis:** Determine if a specific model size and training configuration can be executed within a target timeframe given hardware and network constraints.
2.  **Bottleneck Identification:** Identify whether compute, memory, or network bandwidth is the limiting factor.
3.  **Technique Comparison:** Compare different distributed training algorithms (e.g., DiLoCo, SWARM, Pipeline Parallelism) and compression techniques (Quantization, Sparsification).

## Supported Algorithms
*   **DiLoCo (Distributed Low-Communication):** Data parallelism with infrequent synchronization.
*   **Standard Data Parallelism (DDP):** Frequent synchronization (baseline).
*   **Pipeline Parallelism (PP):** Sharding layers across nodes (future work).

## Usage

```bash
python simulator.py --config configs/10e25_flop.yaml
```

## Configuration
Configurations are defined in YAML files. Key parameters include:
*   **Model:** Parameter count, precision (FP16, FP8, INT4).
*   **Hardware:** Node compute (FLOP/s), Memory (VRAM), Inter-node Bandwidth.
*   **Algorithm:** Type (DiLoCo, DDP), Inner steps, Compression ratio.

## References
Based on research by Epoch AI, Prime Intellect, and MIRI.
