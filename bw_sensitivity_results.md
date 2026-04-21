Bandwidth sensitivity: minimum-cost hardware to reach 10^25 FLOP C_local in 740 days, using actual per-node VRAM. Symmetric rows set WAN, regional, and PP channels to the listed bandwidth with 100 ms latency (0.1 ms at 1000 Gbps to model intra-datacenter InfiniBand/NVLink conditions). Asymmetric rows (US avg, China avg) use the listed down/up values for WAN; regional and PP channels use the upload rate (peer-to-peer bottleneck). MFU = 40%, compression = 150x, overtraining constrained to 1x--100x. Hardware cost applies Cottier et al. (2024) multipliers: 1.64x chip-to-server, 1.23x server-to-cluster. Rows sorted by cost, descending.

| BW | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |
|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 Mbps | 16x GH200 FP8 | 3,046 | PP (2x1523) | 315B | 4 | 0.4250 | 1.0e25 | 0.9518 | 9.5e24 | 1.5x | $2.75B |
| 30 Mbps | 50x A100 80GB | 168 | Hier (10x16) | 250B | 61 | 0.1493 | 1.0e25 | 0.6213 | 6.2e24 | 7.0x | $118.6M |
| China avg (207 / 47 Mbps) | 16x GH200 FP8 | 38 | Flat | 160B | 25 | 0.3266 | 1.0e25 | 0.5969 | 6.0e24 | 7.8x | $34.3M |
| US avg (310 / 57 Mbps) | 16x GH200 FP8 | 34 | Flat | 160B | 20 | 0.3639 | 1.0e25 | 0.6250 | 6.3e24 | 7.0x | $30.7M |
| 100 Mbps | 16x GH200 FP8 | 34 | Hier (8x4) | 160B | 19 | 0.3698 | 1.0e25 | 0.6250 | 6.4e24 | 7.0x | $30.7M |
| 300 Mbps | 16x H100 FP8 | 23 | Flat | 91B | 7 | 0.5391 | 1.0e25 | 0.4507 | 4.5e24 | 15x | $18.6M |
| 1 Gbps | 16x H100 FP8 | 16 | Flat | 91B | 2 | 0.8160 | 1.1e25 | 0.5370 | 5.7e24 | 10x | $12.9M |
| 1000 Gbps | 16x H100 FP8 | 13 | Flat | 91B | 1 | 0.9900 | 1.0e25 | 0.5886 | 6.1e24 | 8.3x | $10.5M |
