# Hardware Node Configurations

Node definitions from `evasion_calculator.py`. All configurations are sized to sit just under the CCC threshold of 16 H100-equivalents (15.84 PFLOPS FP16).

## CONFIGS (FP16)

FP16 mixed-precision training. Pseudo-gradients are 16-bit before compression; model state is 16 bytes per parameter.

| Node | GPUs | PFLOPS FP16 | VRAM (GB) | Cost per GPU (USD) | H100-equivalent |
|:--|--:|--:|--:|--:|--:|
| 50× A100 80GB    | 50 | 15.600 | 4,000 | $7,000  | 15.76 |
| 16× GH200        | 16 | 15.840 | 2,304 | $28,000 | 16.00 |
| 16× H100 SXM     | 16 | 15.840 | 1,280 | $25,000 | 16.00 |
| 49× Ascend 910B  | 49 | 15.680 | 3,136 | $16,000 | 15.84 |
| 26× Ascend 910C  | 26 | 15.600 | 3,328 | $26,000 | 15.76 |
| 57× TPU v4       | 57 | 15.675 | 1,824 | $12,000 | 15.83 |
| 80× TPU v5e      | 80 | 15.760 | 1,280 | $6,000  | 15.92 |
| 34× TPU v5p      | 34 | 15.606 | 3,230 | $20,000 | 15.76 |
| 17× TPU v6e      | 17 | 15.606 | 544   | $25,000 | 15.76 |

**Per-GPU throughput assumptions:**
- A100 80GB: 312 TFLOPS FP16
- GH200 / H100 SXM: 990 TFLOPS FP16
- Ascend 910B: 320 TFLOPS FP16; Ascend 910C: 600 TFLOPS FP16 (est.)
- TPU v4: 275 TFLOPS BF16; v5e: 197; v5p: 459; v6e: 918 (BF16 treated as FP16-equivalent)

**Notes:**
- Ascend 910B/C are only available domestically in China; not export-available.
- TPU costs are capital-equivalent estimates derived from cloud rental rates.
- A100 cost reflects the early-2026 secondary market.

## CONFIGS_FP8

FP8 variants of the same physical nodes. CCC threshold is computed from FP16 capacity (`pflops_fp16`), while `pflops` reflects FP8 throughput (~2× FP16). Model state is 14 bytes per parameter (1+1+4+4+4 for optimizer + weights + grads), and pseudo-gradients are 8 bits before compression.

| Node | GPUs | PFLOPS FP8 | PFLOPS FP16 | VRAM (GB) | Cost per GPU (USD) |
|:--|--:|--:|--:|--:|--:|
| 16× H100 FP8         | 16 | 31.68 | 15.84 | 1,280 | $25,000 |
| 16× GH200 FP8        | 16 | 31.68 | 15.84 | 2,304 | $28,000 |
| 49× Ascend 910B FP8  | 49 | 31.36 | 15.68 | 3,136 | $16,000 |
| 26× Ascend 910C FP8  | 26 | 31.20 | 15.60 | 3,328 | $26,000 |
| 17× TPU v6e FP8      | 17 | 31.21 | 15.61 | 544   | $25,000 |

**Per-GPU FP8 throughput assumptions:**
- H100 / GH200: 1,980 TFLOPS FP8
- Ascend 910B: 640 TFLOPS FP8; Ascend 910C: 1,200 TFLOPS FP8 (est.)
- TPU v6e: 1,836 TFLOPS FP8

**Shared FP8 parameters:**
- `bytes_per_param`: 14
- `bits_per_pseudo_grad`: 8

Source: `evasion_calculator.py` lines 14–80 (CONFIGS) and 122–175 (CONFIGS_FP8).

## Pod / Rack Presets (Web Simulator Only)

In addition to the sub-CCC nodes above, the web simulator (`simulator-web/src/App.tsx`) offers manufacturer-defined scale-up units as presets. These represent a single high-bandwidth interconnect domain (NVLink for NVIDIA, ICI for TPUs, UB for Huawei) or a named reference architecture (DGX SuperPOD). They exceed the CCC threshold and model scenarios where each "node" in the distributed-training topology is a full pod/rack.

### NVIDIA pods

| Preset | Chips | PFLOPS FP16 | VRAM (GB) | Interconnect |
|:--|--:|--:|--:|:--|
| GH200 NVL32             | 32    | 31.68   | 4,608    | NVLink (1 rack) |
| GB200 NVL72             | 72    | 162.0   | 13,824   | NVLink (1 rack) |
| DGX H100 SuperPOD (1 SU) | 256  | 253.44  | 20,480   | InfiniBand over 32 DGX H100 |
| DGX A100 SuperPOD (Selene) | 1,120 | 349.44 | 89,600 | InfiniBand over 140 DGX A100 |

Per-GPU: B200 = 2.25 PFLOPS FP16 (dense), 192 GB HBM3e. NVL72 has 72 B200s (36 GB200 superchips × 2). H100 SuperPOD "1 SU" = 32 DGX nodes × 8 H100; A100 SuperPOD reference is 140 DGX nodes × 8 A100 80GB.

### Chinese pods

| Preset | Chips | PFLOPS FP16 | VRAM (GB) | Interconnect |
|:--|--:|--:|--:|:--|
| CloudMatrix 384 (Ascend 910C) | 384 | 230.4 | 49,152 | Huawei UB (1 rack system) |

Huawei CloudMatrix 384 places 384 Ascend 910C chips in a single UB (Unified Bus) scale-up domain across 16 racks.

### Google TPU pods

| Preset | Chips | PFLOPS BF16 | VRAM (GB) | Interconnect |
|:--|--:|--:|--:|:--|
| TPU v4 pod   | 4,096 | 1,126.4  | 131,072 | 3D torus ICI |
| TPU v5e pod  | 256   | 50.43    | 4,096   | 2D torus ICI |
| TPU v5p pod  | 8,960 | 4,112.64 | 851,200 | 3D torus ICI |
| TPU v6e pod  | 256   | 235.01   | 8,192   | 2D torus ICI |

All TPU pod sizes are the maximum single-ICI domain published by Google Cloud. BF16 is treated as FP16-equivalent for simulator purposes.

Source: `simulator-web/src/App.tsx` `HARDWARE_PRESETS`.
