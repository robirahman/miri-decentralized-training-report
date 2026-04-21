Latency sensitivity: minimum-cost hardware to reach 10^25 FLOP C_local in 740 days, using actual per-node VRAM. All connections at 100 Mbps with the listed one-way latency. MFU = 40%, compression = 150x, overtraining constrained to 1x--100x. Hardware cost applies Cottier et al. (2024) multipliers: 1.64x chip-to-server, 1.23x server-to-cluster.

| Latency | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |
|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 ms | 16x GH200 FP8 | 34 | Hier (8x4) | 160B | 19 | 0.3698 | 1.0e25 | 0.6250 | 6.4e24 | 7.0x | $30.7M |
| 30 ms | 16x GH200 FP8 | 34 | Hier (8x4) | 160B | 19 | 0.3698 | 1.0e25 | 0.6250 | 6.4e24 | 7.0x | $30.7M |
| 100 ms | 16x GH200 FP8 | 34 | Hier (8x4) | 160B | 19 | 0.3698 | 1.0e25 | 0.6250 | 6.4e24 | 7.0x | $30.7M |
| 300 ms | 16x GH200 FP8 | 34 | Hier (8x4) | 160B | 19 | 0.3698 | 1.0e25 | 0.6250 | 6.4e24 | 7.0x | $30.7M |
