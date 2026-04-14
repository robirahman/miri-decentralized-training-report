# Rename η_Chinchilla to χ

## Motivation

η (eta) currently serves double duty: its sub-components (η_H, η_compression, η_replica) represent distributed training inefficiencies, while η_Chinchilla represents the overtraining penalty from the Chinchilla scaling law. These are conceptually distinct — one is about communication overhead, the other about model sizing — so they should use different symbols.

## Recommendation

Replace η_Chinchilla with **χ** (chi).

- **Mnemonic:** χ for **Ch**inchilla — easy to remember.
- **Visually distinct from η:** No risk of confusion when scanning equations.
- **No conflicts:** α and β are already taken in this paper; χ is unused.

## Affected equations

Before:
$$C_{\text{quality}} = C_{\text{local}} \times \eta_{\text{Chinchilla}}$$

After:
$$C_{\text{quality}} = C_{\text{local}} \times \chi$$

η remains purely the distributed training efficiency:
$$\eta = \eta_H \times \eta_{\text{compression}} \times \eta_{\text{replica}}$$

## Files to update

- Paper_Outline_TAIGR.md
- Simulator_Documentation.md
- Scaling_Law_Uncertainty.md
- Any simulator code that uses `eta_chinchilla` as a variable name
