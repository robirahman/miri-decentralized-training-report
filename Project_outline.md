# MIRI Technical Governance Team project: Preventing Illicit Distributed Training



### Background and problem statement

MIRI has proposed an international treaty to pause dangerous AI development so that no one anywhere on Earth creates a misaligned superintelligence (which would lead to loss of control and human extinction).

The treaty requires that all compute clusters larger than 16 H100-equivalents must be reported and monitored to ensure they are not being used for prohibited AI development activities.

However, corporations or countries can attempt to violate the treaty by doing decentralized training: using nodes smaller than the monitoring limit and transferring data through WAN connections (possibly through VPNs) to collectively train models larger than the prohibited compute threshold.

* Note that the treaty already prohibits this - clusters individually smaller than the unmonitored limit are considered to be a larger, single cluster that is required to be reported and monitored if they are connected by networking - but it would be possible to evade reporting requirements on any individual component of the network.
* If inspectors discovered any individual small node within this distributed training operation, the operator would have plausible deniability, claiming that it is below the threshold where hardware is required to be reported, not connected to any other hardware, and not being used for any restricted purposes.
* It would be very difficult to catch such an operation red-handed: you would need to discover multiple nodes, and prove they are communicating with each other for an ML training job (which is obfuscated by other network traffic).



### Project goals

1. Investigate the technical feasibility of decentralized training to execute ML pre-training workloads larger than 10^25 FLOPs.

* 10^24 FLOP is the banned threshold in MIRI's proposed treaty. 10^25 FLOP of decentralized training has been concluded feasible by past research. Fact-check this work and then investigate even larger illicit operations. 10^26 FLOP decentralized jobs seem likely feasible based on past work. What about 10^27 or even larger?

2. Create a decentralized training simulator, building on the Epoch AI distributed training simulator but accounting for network latency and limited bandwidth between nodes.
3. Determine the best way to close the loophole.

* Currently, hardware reporting requirements for covered compute clusters only specify FLOP/s thresholds. Should the treaty be amended to impose additional restrictions, such as maximum memory or bandwidth thresholds that trigger reporting requirements?
* What additional enforcement mechanisms would help uncover illicit decentralized training operations?
