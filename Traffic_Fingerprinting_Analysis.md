# Traffic Fingerprinting and ISP Monitoring as Treaty Enforcement

## 1. The Proposal

A compute governance treaty could require party states to mandate ISP-level monitoring for network traffic signatures of decentralized AI training. The motivation: DiLoCo-style distributed training allows actors to train large AI models using many small GPU clusters that individually fall below the treaty's registration threshold (the "Covered Chip Cluster" or CCC). Hardware-level countermeasures like TEE attestation and chip activation gates require cooperation from chip manufacturers and take years to deploy. Network monitoring could, in principle, detect illegal training using existing infrastructure — including on legacy chips that predate any hardware enforcement mechanism.

This document analyzes whether such monitoring would be effective, distinguishing between two variants: **targeted monitoring** at known GPU facilities, and **broad ISP-level monitoring** across all network traffic.

## 2. The DiLoCo Traffic Signature

DiLoCo (Distributed Low-Communication training) creates a distinctive network pattern. In the primary evasion scenario analyzed in the companion report — 72 nodes each containing 50x A100 GPUs training a 250B parameter model with 16x pseudo-gradient compression:

- **Sync payload:** ~31 GB per direction per synchronization round (250B params x 16 bytes / 16x compression / 8 bits per byte)
- **Streaming:** With streaming DiLoCo, each sync is spread over the full inner-step period (~100 minutes), producing a sustained ~39 Mbps average upload and ~39 Mbps download (~78 Mbps total bidirectional)
- **Cadence:** Syncs repeat at regular intervals (~100 minutes) continuously for months
- **Topology:** All N nodes communicate with a central coordinator (parameter server) or in a ring/mesh pattern. In hierarchical DiLoCo, nodes form regional groups with additional intra-group communication
- **Encryption:** All traffic travels over standard VPN or TLS connections
- **Duration:** Training runs last 3-12 months at sustained near-100% utilization

The potentially distinctive features are: (1) the regular cadence with fixed payload sizes, (2) the sustained duration over months, (3) the bidirectional symmetry, and (4) the multi-node topology where N geographically distributed endpoints all communicate with the same coordinator on the same schedule.

## 3. Two Distinct Proposals

These are often conflated but have very different cost-benefit profiles:

**Targeted monitoring at registered GPU facilities.** The treaty requires facilities known to house AI accelerators (data centers, cloud providers, research labs) to permit network metadata collection — traffic volumes, destination IPs, and timing patterns — reported to the international verification agency. This is analogous to IAEA monitoring at declared nuclear facilities.

**Broad ISP-level monitoring.** The treaty requires all ISPs in party states to deploy traffic analysis systems that scan for signatures of distributed training across *all* customer traffic. This is analogous to requiring telecommunications companies to scan all phone calls for specific conversation patterns.

## 4. Arguments For ISP Monitoring

**No hardware manufacturer cooperation required.** Unlike TEE attestation and chip activation gates — which need NVIDIA, AMD, and other manufacturers to embed enforcement firmware in their chips — ISP monitoring works with existing network infrastructure. It doesn't require any changes to AI hardware and could be deployed immediately.

**Covers legacy and pre-existing chips.** Any hardware-based enforcement mechanism (TEE, activation gates, serialization) only applies to newly manufactured chips. The existing global stock of AI accelerators — millions of GPUs already deployed — lacks enforcement firmware. ISP monitoring covers training on these legacy chips, closing a multi-year window that hardware measures leave open.

**Covers both owned and rented hardware.** The analysis in the companion report assumes evaders own their hardware. But an evader could also rent GPUs from cloud providers across multiple accounts. ISP monitoring detects the network signature regardless of hardware ownership, complementing cloud-provider KYC requirements.

**Mesh topology is more distinctive than individual links.** While a single node's traffic (sustained 50-100 Mbps bidirectional VPN) is unremarkable, the *correlated* pattern — N endpoints all communicating with the same coordinator at synchronized intervals — is more distinctive. An ISP or international monitoring system that could correlate traffic across endpoints would see a pattern that few legitimate workloads produce at that scale.

**Deterrent effect.** Even imperfect monitoring raises the operational security burden on evaders. Every countermeasure the evader must deploy (traffic shaping, VPN routing, cover traffic) increases complexity, cost, and the chance of operational mistakes that lead to detection through other channels.

**Extends existing ISP capabilities.** ISPs already perform deep packet inspection and traffic analysis for network management, abuse detection, and lawful intercept obligations. The technical infrastructure for traffic analysis exists; the proposal extends it to a new detection target.

**Detection of unregistered training.** This is the key scenario where ISP monitoring adds value that no other countermeasure provides: detecting training on *unregistered* hardware at *unregistered* facilities. Hardware tracking, chip serialization, and facility monitoring all assume the chips and facilities are known. An evader who acquires GPUs through informal channels and operates from an unmarked facility is invisible to hardware-focused measures but might be visible through network traffic.

## 5. Arguments Against ISP Monitoring

**Easy software-only evasion.** The DiLoCo traffic signature can be disguised through traffic shaping at near-zero cost to the evader:
- **Jitter injection:** Randomize the timing of sync transmissions within each inner-step period, breaking the regular cadence
- **Padding and cover traffic:** Add random-length dummy packets to vary the payload size; generate cover traffic during non-sync periods to eliminate the on/off pattern
- **Variable-rate streaming:** Instead of streaming at a constant ~39 Mbps, vary the rate randomly while maintaining the same total data transferred
- **Multi-hop routing:** Route traffic through commercial VPN services, Tor, or multiple intermediate servers to break the direct node-to-coordinator correlation
- **Protocol mimicry:** Shape traffic to match the statistical profile of video streaming, cloud backup, or other high-bandwidth applications

These countermeasures are purely software — they require no hardware procurement, no physical infrastructure, and can be implemented in a few hundred lines of code. This asymmetry is fundamental: the monitoring system requires expensive, permanent infrastructure deployed across all ISPs; the evasion requires a software wrapper that can be written in a day. This contrasts sharply with hardware countermeasures like TEE attestation, where evasion requires manufacturing custom chips — a billion-dollar capability limited to state actors.

**Encryption blocks payload inspection.** All DiLoCo traffic uses standard VPN or TLS encryption. ISPs can observe traffic volumes, timing, and destination IPs, but cannot inspect the content. They cannot distinguish pseudo-gradient synchronization from video streaming, cloud backup, file transfer, or any other encrypted bulk data transfer. The proposal is limited to *metadata analysis* — which, as described above, is easily defeated by traffic shaping.

**Massive false positive rate.** The traffic profile of a DiLoCo node — sustained 50-100 Mbps bidirectional encrypted traffic to a remote server, running 24/7 for months — describes millions of legitimate services worldwide:
- Cloud computing instances running inference, rendering, or simulation
- Content delivery networks and video streaming infrastructure
- Enterprise VPN connections and remote desktop services
- Backup and disaster recovery replication
- Scientific computing and legitimate federated learning
- Multiplayer game servers and real-time communication platforms

Even the more distinctive mesh correlation signal (N endpoints synchronized) has legitimate analogues: distributed rendering farms, federated learning deployments, CDN cache synchronization, and distributed database replication. The false positive rate at ISP scale — millions of customers per major ISP — would generate an unmanageable volume of alerts, each requiring investigation to distinguish from legitimate traffic.

**Mass surveillance infrastructure.** Requiring ISPs to deploy traffic pattern analysis across all customer traffic is mass surveillance, even if limited to metadata. This would face strong opposition from:
- **Privacy advocates and civil liberties organizations:** who have successfully opposed similar proposals for encryption backdoors and communications metadata retention in democratic countries
- **The technology industry:** which depends on user trust and would resist government-mandated traffic analysis
- **Some governments:** that view such requirements as incompatible with their constitutional privacy protections

Historical precedent is instructive: the Clipper Chip proposal (1993), which would have required all encryption to include a government backdoor, was defeated by a coalition of industry and civil liberties advocates. The EU's proposed Chat Control regulation has faced sustained opposition. Mandatory ISP monitoring for AI training patterns would encounter similar resistance, potentially undermining broader support for the compute governance treaty.

**Sovereignty friction.** Different countries have different legal frameworks governing ISP monitoring, data retention, and law enforcement access to communications metadata. Requiring uniform ISP monitoring across all party states means:
- Countries with strong privacy protections (e.g., EU GDPR framework) would need to modify domestic law
- Countries with existing surveillance infrastructure might resist international oversight of their monitoring systems
- The data-sharing requirements create tensions around intelligence sharing, sovereignty over communications data, and mutual trust

**Cross-ISP and cross-border correlation is the hard problem.** The most technically distinctive signal — N distributed nodes all communicating with the same coordinator at regular intervals — requires correlating traffic metadata *across multiple ISPs and national borders in near-real-time*. This requires:
- A standardized metadata format and reporting protocol across all ISPs in all party states
- An international data fusion center that receives and correlates these feeds
- Legal frameworks for real-time cross-border sharing of communications metadata
- Sufficient computational resources to perform correlation analysis across billions of flows

No such international data-sharing framework exists today. Building one would take years, cost billions, and create a powerful surveillance tool with uses far beyond AI training detection — raising additional governance and abuse concerns.

**State actors route around civilian ISP monitoring.** The most capable and dangerous evaders — state actors with intelligence and military resources — have access to:
- Military and diplomatic communications networks invisible to civilian ISPs
- Dedicated leased fiber connections that bypass ISP infrastructure entirely
- Satellite communications links
- Sovereign territory where civilian ISP monitoring requirements do not apply

ISP monitoring is inherently limited to the civilian internet infrastructure. The actors most likely to operate at dangerous scale (10^27+ FLOP) are precisely those with the resources to avoid civilian networks entirely.

**Individual nodes are unremarkable.** In a 72-node distributed training run, each individual node generates ~78 Mbps of encrypted traffic. This is indistinguishable from a busy cloud server, a video encoding workstation, or an enterprise VPN gateway. The distinctive signal exists only in the *correlation* across nodes — which, as discussed above, requires the cross-ISP fusion infrastructure that doesn't exist.

**Asymmetric cost structure.** The fundamental problem with ISP monitoring as an enforcement mechanism is the cost asymmetry:
- **Cost to deploy monitoring:** Billions of dollars in infrastructure across all ISPs in all party states, plus ongoing operational costs, legal frameworks, and international coordination
- **Cost to evade monitoring:** Near-zero — a software traffic shaping wrapper that any competent engineer can implement

This asymmetry means the monitoring system is perpetually outmatched. Every dollar spent on more sophisticated traffic analysis can be defeated by a few lines of code. This contrasts with hardware-based countermeasures, where the cost of evasion (manufacturing custom chips) is orders of magnitude higher than the cost of enforcement (adding firmware to existing fab processes).

## 6. Comparison: Targeted vs. Broad Monitoring

| Dimension | Targeted (GPU facilities) | Broad (ISP-level) |
|:--|:--|:--|
| **Privacy impact** | Low — only registered facilities | High — all ISP customers |
| **Implementation cost** | Low — network taps at known sites | High — ISP-wide infrastructure |
| **Evasion difficulty** | Medium — must conceal training at a known facility with registered chips | Low — software traffic shaping |
| **Coverage** | Only registered facilities with known chips | All network traffic |
| **False positive rate** | Low — baseline is known (registered GPU site) | Very high — millions of legitimate services |
| **Sovereignty issues** | Minimal — facility operators consent as condition of chip registration | Significant — requires domestic ISP regulation |
| **Legal precedent** | Similar to IAEA facility monitoring | Similar to defeated encryption backdoor proposals |
| **Detection of unregistered training** | No — only sees registered facilities | Theoretically yes, but easily evaded |
| **Manufacturer cooperation** | Not needed | Not needed |
| **Timeline to deploy** | Months (piggybacks on chip registration) | Years (requires legal frameworks + infrastructure) |

## 7. Assessment and Recommendation

**Targeted monitoring at registered GPU facilities** is recommended as a supplementary detection signal. It has low privacy impact (monitoring occurs at facilities that have already consented to oversight as a condition of chip registration), low implementation cost (network taps or metadata reporting at known sites), and provides useful corroborating evidence for investigations triggered by other means (chip procurement intelligence, financial monitoring, whistleblower reports). It cannot detect training at *unregistered* facilities, but it raises the cost and risk of conducting illegal training at *registered* ones.

**Broad ISP-level monitoring** is not recommended. The cost-benefit analysis is decisively unfavorable:

1. The evasion is trivially cheap (software traffic shaping), while the monitoring infrastructure is enormously expensive
2. The false positive rate at ISP scale would overwhelm investigative capacity
3. The privacy and sovereignty costs would generate political opposition that could undermine the broader treaty
4. The most capable and dangerous actors (state-level) have the easiest path to avoid civilian ISP monitoring entirely
5. The one technically distinctive signal (multi-node correlation) requires an international real-time data-sharing framework that doesn't exist and would take years to build
6. Historical precedent (Clipper Chip, Chat Control) suggests that democratic societies reject mass communications surveillance mandates

The fundamental asymmetry — cheap, software-only evasion versus expensive, permanent surveillance infrastructure — means that broad ISP monitoring would impose large costs on society while providing minimal enforcement value. The treaty's enforcement resources are better directed toward hardware-level mechanisms (TEE attestation, chip activation gates, foundry serialization) where the cost of evasion is orders of magnitude higher, and toward human intelligence mechanisms (whistleblower programs, procurement monitoring) where the detection does not depend on network observability.
