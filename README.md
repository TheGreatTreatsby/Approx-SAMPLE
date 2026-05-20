# Approx SAMPLE: Towards Non-Cooperative  SAR-ATR via Controllable Approximate Simulation
<div align=center>
<img src="pipeline.png" width="1500"/>
</div>
## 📖 Introduction

**Approx SAMPLE** is a benchmark dataset designed explicitly for **Non-Cooperative SAR Automatic Target Recognition (SAR-ATR)**.  
While existing datasets like SAMPLE rely on precisely-aligned synthetic and measured SAR image pairs (exact 3D models, precise registration), such ideal assumptions are unrealistic in non-cooperative scenarios where target geometry, material, and background are uncertain.

We propose a **controllable approximate simulation framework** to systematically introduce:
- **Limited geometric fidelity** (open-source 3D models, mesh decimation, approximate scaling)
- **Material approximations** (PEC with local dielectric corrections)
- **Randomized target placements** (breaking pixel-level registration)
- **Complex background clutter** (real SAR image-driven clutter synthesis)

The resulting dataset retains physical electromagnetic consistency while explicitly featuring the misalignment and uncertainty typical of non-cooperative settings, enabling more robust evaluation of cross-domain SAR-ATR algorithms.

### 🔑 Key Features
- **Progressive difficulty**: Three configurations (`Center`, `Random`, `Complex`)
- **Real-image-driven clutter**: Backgrounds synthesized from real SAR imagery
- **Multi-level annotations**: SAR images, quarter-power reflectivity maps, target shadow masks, and orthographic projection masks
- **Class-aligned with SAMPLE**: 10 target classes (2S1, BMP2, BTR70, M1, M2, M35, M548, M60, T72, ZSU23)
- **Open-source & white-box**: Full construction pipeline, desktop-grade simulation (CST), no HPC require
> **🗂️ Note**: The full dataset will be publicly released upon paper publication. For now, please refer to the [example](./example) folder for sample data.
> <div align=center>
  <img src="example/Approx SAMPLE-Complex/bmp2_synth_A_elevDeg_016_azCenter_6.004928912572714_2difft.png" width="100%"/>
  <img src="example/Approx SAMPLE-Complex/m2_synth_A_elevDeg_014_azCenter_18.985293283579836_2difft.png" width="100%"/>
  <img src="example/Approx SAMPLE-Complex/m60_synth_A_elevDeg_015_azCenter_29.01557484299443_2difft.png" width="100%"/>
</div>
