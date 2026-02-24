**To:** mryabinin0@gmail.com
**Subject:** Validating Pipeline Parallelism bottlenecks in high-latency WAN environments

Hi Max,

I’m working on an open-source simulator for decentralized LLM training and have been heavily referencing your work on SWARM Parallelism and Petals.

We are currently refining our "Mode B" logic, which handles Pipeline Parallelism for models where M_req > Node_VRAM. Given your experience with heterogeneous and P2P clusters, I would love to get your take on how we are modeling the "Pipeline Bubble" in high-latency environments. Specifically, I'm curious if our latency penalty for micro-batch handovers is too optimistic for real-world internet jitter.

Do you have any availability for a brief chat to ensure our simulator captures the "reality" of P2P training bottlenecks accurately?

Best,

[Your Name]