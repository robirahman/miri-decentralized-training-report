# Research Ideas: Detection, Economics, and Enforcement (Feb 21, 2026)

Following the initial technical feasibility study of $10^{25}$ FLOP decentralized training, the following areas are proposed for deeper investigation by the MIRI Technical Governance Team.

---

## 1. Detection and Traffic Fingerprinting
Decentralized training protocols (DiLoCo, Pipeline Parallelism) create distinct network "heartbeats" that may be detectable even through encrypted VPN tunnels.

*   **Heartbeat Analysis:** DiLoCo synchronizations happen at highly regular, multi-minute intervals with massive, sustained bandwidth bursts. Pipeline Parallelism has a per-step micro-burst pattern.
*   **Research Question:** Can we develop a traffic classifier that identifies illicit training runs based on packet metadata (size, frequency, destination mesh)? 
*   **Governance Outcome:** Propose that the treaty mandate ISPs monitor and report "bursty, high-bandwidth encrypted meshes" that match these fingerprints.

## 2. The Economics of Evasion
We need to understand the cost-benefit ratio for a state or corporation choosing decentralized evasion over a secret data center.

*   **Algorithmic Waste vs. Satellite Detection:** Decentralized training is ~20–40% less compute-efficient due to synchronization overhead and algorithmic decay. However, it is much harder to detect via satellite (no large thermal footprint or single massive power draw).
*   **Orchestration Regulation:** Many decentralized runs use permissionless crypto-incentives (e.g., Bittensor). 
*   **Governance Outcome:** Evaluate whether the treaty should regulate the "Orchestration Layer"—making the distribution of software optimized for WAN-based scaling a violation.

## 3. Hardware-Level Enforcement (TEEs)
The "Plausible Deniability" problem is that an operator can claim their sub-threshold node is used for legitimate purposes.

*   **Remote Attestation:** Modern chips (NVIDIA H100/Blackwell) feature "Confidential Computing." A governance mechanism could require chips to "attest" their kernels to an international monitoring body.
*   **The Revocation "Kill Switch":** If a decentralized run is detected, can we revoke the cryptographic keys of the involved chips to stop the run remotely?
*   **Research Task:** Study NVIDIA’s "Confidential Computing" whitepapers to see if it can be repurposed for treaty enforcement.

## 4. Redefining "Model Possession"
Current treaty language likely assumes a model is a file stored on a disk.

*   **Distributed Weights:** In "Protocol Models," no single node ever holds the full model weights; they are sharded and encrypted across the network.
*   **Legal Loophole:** If no one person "possesses" the model, the treaty may be unenforceable.
*   **Governance Outcome:** Refine legal language to include "Distributed Possession" or "Control of Orchestration" as the threshold for violation.

---

## Next Steps for the Simulator
- [ ] **Traffic Simulation:** Export a "pcap" style log from the simulator to see if it matches known ISP detection patterns.
- [ ] **Cost Modeling:** Add a "USD Cost" component that accounts for the 40% algorithmic waste and the price of WAN data transfer.
