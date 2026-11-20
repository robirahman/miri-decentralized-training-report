Latency sensitivity: minimum-cost hardware to reach 10^25 FLOP C_local in 740 days, using actual per-node VRAM. All connections at 100 Mbps with the listed one-way latency. MFU = 40%, compression = 150x, overtraining constrained to 1x--100x. Hardware cost applies Cottier et al. (2024) multipliers: 1.64x chip-to-server, 1.23x server-to-cluster.

| Latency | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |
|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 ms | 16x H100 FP8 | 36 | Flat | 91B | 19 | 0.3495 | 1.0e25 | 0.3541 | 3.6e24 | 23x | $29.0M |
| 30 ms | 16x H100 FP8 | 36 | Flat | 91B | 19 | 0.3495 | 1.0e25 | 0.3541 | 3.6e24 | 23x | $29.0M |
| 100 ms | 16x H100 FP8 | 36 | Flat | 91B | 19 | 0.3495 | 1.0e25 | 0.3541 | 3.6e24 | 23x | $29.0M |
| 300 ms | 16x H100 FP8 | 36 | Hier (8x4) | 91B | 19 | 0.3495 | 1.0e25 | 0.3541 | 3.6e24 | 23x | $29.0M |
