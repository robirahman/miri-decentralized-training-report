Bandwidth sensitivity: minimum-cost hardware to reach 10^25 FLOP C_local in 740 days, using actual per-node VRAM. Symmetric rows set WAN, regional, and PP channels to the listed bandwidth with 100 ms latency (0.1 ms at 1000 Gbps to model intra-datacenter InfiniBand/NVLink conditions). Asymmetric rows (US avg, China avg) use the listed down/up values for WAN; regional and PP channels use the upload rate (peer-to-peer bottleneck). MFU = 40%, compression = 150x, overtraining constrained to 1x--100x. Hardware cost applies Cottier et al. (2024) multipliers: 1.64x chip-to-server, 1.23x server-to-cluster. Rows sorted by cost, descending.

| BW | Config | Nodes | Mode | Model | H | eta | C_local | chi | C_quality | OT | Cost |
|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 Mbps | 16x H100 FP8 | 141 | Hier (12x11) | 91B | 185 | 0.0878 | 1.0e25 | 0.1469 | 1.5e24 | 90x | $113.8M |
| 30 Mbps | 16x H100 FP8 | 66 | Hier (8x8) | 91B | 61 | 0.1872 | 1.0e25 | 0.2456 | 2.5e24 | 42x | $53.3M |
| China avg (207 / 47 Mbps) | 16x H100 FP8 | 41 | Flat | 91B | 25 | 0.3058 | 1.0e25 | 0.3286 | 3.3e24 | 26x | $33.1M |
| US avg (310 / 57 Mbps) | 16x H100 FP8 | 36 | Flat | 91B | 20 | 0.3437 | 1.0e25 | 0.3541 | 3.5e24 | 23x | $29.0M |
| 100 Mbps | 16x H100 FP8 | 36 | Flat | 91B | 19 | 0.3495 | 1.0e25 | 0.3541 | 3.6e24 | 23x | $29.0M |
| 300 Mbps | 16x H100 FP8 | 23 | Flat | 91B | 7 | 0.5391 | 1.0e25 | 0.4507 | 4.5e24 | 15x | $18.6M |
| 1 Gbps | 16x H100 FP8 | 16 | Flat | 91B | 2 | 0.8160 | 1.1e25 | 0.5370 | 5.7e24 | 10x | $12.9M |
| 1000 Gbps | 16x H100 FP8 | 13 | Flat | 91B | 1 | 0.9900 | 1.0e25 | 0.5886 | 6.1e24 | 8.3x | $10.5M |
