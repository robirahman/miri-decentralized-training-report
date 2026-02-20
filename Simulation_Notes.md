# Simulation Notes: 10^25 FLOP Baseline

## Configuration
*   **Target:** 1.02e25 FLOPs (12T tokens, 144B parameter model)
*   **Hardware:** 72 nodes, each with 16x GH200 (estimated 32 PFLOPS/node at 40% MFU)
*   **Network:** 100 Mbps symmetric WAN bandwidth per node
*   **Algorithm:** Streaming DiLoCo
    *   Inner steps: 128
    *   Compression: 16x (e.g., 4-bit quantization + sparsification)
    *   Local Batch: 131k tokens

## Results
*   **Estimated Training Time:** 331.14 days
*   **Bottleneck:** Communication (Network Bandwidth)
*   **Duty Cycle Analysis:**
    *   Compute Block (128 steps): ~1,132 seconds (~19 mins)
    *   Sync Block (Upload/Download): ~2,880 seconds (~48 mins)
    *   Even with infrequent synchronization, the 100 Mbps link keeps the GPUs idle for significant portions of the run if synchronization is blocking.

## Conclusions
The $10^{25}$ FLOP run is technically feasible within a one-year window, but it is highly sensitive to inter-node bandwidth. Any reduction in WAN speed or increase in model size (requiring more frequent sync) quickly pushes the project into "Impractical" territory (>1 year).
