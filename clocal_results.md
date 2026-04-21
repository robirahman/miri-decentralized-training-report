Minimum-cost hardware to reach C_local targets in 740 days, using actual per-node VRAM. All network connections at 100 Mbps with 100 ms latency. MFU = 40%, compression = 150x, overtraining constrained to 1x--100x. Hardware cost applies Cottier et al. (2024) multipliers: 1.64x chip-to-server, 1.23x server-to-cluster.

| Target | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |
|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 10^24 | 16x H100 FP8 | 2 | Flat | 91B | 18 | 0.7957 | 1.3e24 | 0.9796 | 1.3e24 | 1.3x | $1.6M |
| 3.3 x 10^24 | 16x GH200 FP8 | 7 | Flat | 160B | 19 | 0.5973 | 3.4e24 | 0.9635 | 3.3e24 | 1.4x | $6.3M |
| 10^25 | 16x GH200 FP8 | 34 | Hier (8x4) | 160B | 19 | 0.3698 | 1.0e25 | 0.6250 | 6.4e24 | 7.0x | $30.7M |
| 2.1 x 10^25 | 16x GH200 FP8 | 101 | Hier (8x12) | 160B | 19 | 0.2580 | 2.1e25 | 0.3689 | 7.8e24 | 21x | $91.3M |
| 3.8 x 10^25 | 50x A100 80GB | 625 | Hier (10x62) | 250B | 19 | 0.1524 | 3.8e25 | 0.3214 | 1.2e25 | 26x | $441.3M |
| 6.6 x 10^25 | 16x H100 FP8 | 2,880 | PP (2x1440) | 180B | 3 | 0.4244 | 6.6e25 | 0.2901 | 1.9e25 | 31x | $2.32B |
| 10^26 | 16x H100 FP8 | 4,706 | PP (2x2353) | 180B | 3 | 0.3934 | 1.0e26 | 0.2123 | 2.1e25 | 51x | $3.80B |
