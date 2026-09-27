# STORMFUSION Satellite Architecture & Data-Contract Compatibility Report

**SIH Problem Statement:** SIH26070 — Multi-Modal AI for North Indian Ocean Tropical Cyclone Intelligence  
**Evaluation Scope:** Satellite Input Contract, Multi-Channel Radiance Encoding vs. Level-2 Cloud Mask, Fusion Invariance, and Training Safety Gates  
**Status Date:** September 8, 2026  
**Auditor:** Antigravity (Advanced Agentic AI Pair Programmer)

---

## Executive Summary

An independent, rigorous architectural audit was executed across the **STORMFUSION** multi-modal deep learning pipeline. This audit evaluates the feasibility of adapting the system from its intended 3-channel spectral radiance contract to the currently downloaded 1-channel EUMETSAT Meteosat-8 Cloud Mask (`CLM-IODC`), contrasted with waiting for Level 1.5 spectral radiances (`HRSEVIRI-IODC`).

### Key Findings:
1. **Contract Definition**: The spatial input contract is strictly defined as `[B, 8, 3, 500, 500]` across model, dataset, and provider configurations.
2. **Channel Physics**: The 3 expected channels (`TIR1` 10.8 µm, `WV` 6.2 µm, `TIR2` 12.0 µm) supply critical physical signals (brightness temperature, upper-tropospheric moisture, split-window moisture correction) that cannot be replicated by categorical cloud flags.
3. **Fusion Invariance**: The [AvailabilityAwareGatedFusion](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/gated_fusion.py) module operates strictly on 1D embeddings (`sat_dim=128`, `track_dim=64`, `era5_dim=64`) and is completely independent of the satellite input channel count.
4. **Checkpoint Incompatibility**: Existing integration checkpoints contain weights for a 3-channel initial convolution (`[64, 3, 7, 7]`). Switching the primary backbone to 1-channel breaks checkpoint state loading.
5. **Verdict**: **Option A is strongly recommended**. Preserve the 3-channel architecture and unblock Level 1.5 `HRSEVIRI-IODC` via EUMETSAT General License acceptance. Duplicating or tinting cloud mask flags to create fake spectral channels is scientifically invalid and strictly prohibited.

---

## 1. Codebase Inspection & Data Contract Audit

```mermaid
graph TD
    RawSat[Raw Satellite Observation] --> Loader[Satellite Preprocessing Provider]
    Loader -->|Standardized Tensor [B, 8, 3, 500, 500]| ResNet[ResNet18 Spatial Encoder]
    ResNet -->|Spatial Maps [B*8, 512, H/16, W/16]| ConvLSTM[ConvLSTM Temporal Encoder]
    ConvLSTM -->|Global Average Pooling + Projection| SatEmb[Satellite Embedding [B, 128]]
    
    TrackHist[Track History [B, 8, 8]] --> TrackGRU[Track GRU Encoder]
    TrackGRU --> TrackEmb[Track Embedding [B, 64]]
    
    ERA5Grid[ERA5 Reanalysis [B, 8, 4, 32, 32]] --> ERA5CNN[ERA5 Lightweight CNN]
    ERA5CNN --> ERA5Emb[ERA5 Embedding [B, 64]]
    
    SatEmb --> Fusion[Availability-Aware Gated Fusion]
    TrackEmb --> Fusion
    ERA5Emb --> Fusion
    
    Fusion -->|Fused Representation [B, 64]| MultiTask[StormFusion Multi-Task Trunk]
    MultiTask --> Heads[Detection / Pattern / Intensity / Pressure / TrackDelta]
```

### Detailed Inspection Responses

| # | Inspection Item | Status / Findings | Primary Code Locations |
| :--- | :--- | :--- | :--- |
| **1** | **Where `[B, 3, 500, 500]` is defined** | Configured globally and enforced across data loading, preprocessing, and model tensor reshaping. | • [`configs/model_config.yaml#L6-L22`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/configs/model_config.yaml#L6-L22)<br>• [`configs/dataset_config.yaml#L21-L25`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/configs/dataset_config.yaml#L21-L25)<br>• [`configs/eumetsat_config.yaml#L40-L77`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/configs/eumetsat_config.yaml#L40-L77)<br>• [`models/baseline_satellite_model.py#L67,77-80`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/baseline_satellite_model.py#L67-L80) |
| **2** | **Which three channels are expected** | 1. `TIR1` (Clean Thermal IR ~10.8 µm)<br>2. `WV` (Water Vapor ~6.2 µm)<br>3. `TIR2` (Split-Window IR ~12.0 µm) | • [`configs/eumetsat_config.yaml#L41-L74`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/configs/eumetsat_config.yaml#L41-L74)<br>• [`preprocessing/eumetsat_iodc_provider.py#L343`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/preprocessing/eumetsat_iodc_provider.py#L343) |
| **3** | **Configurable `in_channels` in Encoder** | **Supported**: `ResNet18SpatialEncoder` dynamically replaces `backbone.conv1` if `in_channels != 3`. | • [`models/satellite_encoder.py#L34-L45`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/satellite_encoder.py#L34-L45)<br>• [`models/baseline_satellite_model.py#L26,41`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/baseline_satellite_model.py#L26-L41) |
| **4** | **Single-channel support in preprocessing** | **Unsupported**: All existing providers (`eumetsat_iodc_provider.py`, `noaa_satellite_provider.py`, `satellite_loader.py`) hardcode 3-channel output stacking. | • [`preprocessing/eumetsat_iodc_provider.py#L402`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/preprocessing/eumetsat_iodc_provider.py#L402)<br>• [`preprocessing/noaa_satellite_provider.py#L145`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/preprocessing/noaa_satellite_provider.py#L145) |
| **5** | **Gated Fusion independence from channel count** | **Fully Independent**: Gated Fusion projects `satellite_embedding` `[B, 128]` into `fusion_dim=64`. Zero dependency on input spatial channels. | • [`models/gated_fusion.py#L28-L46`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/gated_fusion.py#L28-L46) |
| **6** | **ResNet18 conv1 instantiation** | Defaults to `in_channels=3`. Can be instantiated with `in_channels=1` via explicit constructor parameter. | • [`models/satellite_encoder.py#L17`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/satellite_encoder.py#L17)<br>• [`models/stormfusion.py#L59`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/stormfusion.py#L59) |
| **7** | **Checkpoint dependency on 3-channel structure** | **Dependent**: Existing integration checkpoints have `conv1.weight` shape `[64, 3, 7, 7]`. Loading into a 1-channel model causes PyTorch shape mismatch. | • `checkpoints/step16_integration_test.pt`<br>• `checkpoints/dev_training/best.pt` |
| **8** | **Grad-CAM and MC-Dropout compatibility** | **Compatible**: Grad-CAM hooks `layer4` (`512` features) and MC-Dropout targets `nn.Dropout`; both are agnostic to input channel count. | • [`explainability/gradcam.py#L56-L73`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/explainability/gradcam.py#L56-L73)<br>• [`uncertainty/mc_dropout.py#L18-L23`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/uncertainty/mc_dropout.py#L18-L23) |
| **9** | **Missing-modality semantics for cloud mask** | **Unsupported**: The missing-modality system is ternary (`[sat, track, era5]`). No separate semantic gate exists for cloud mask presence. | • [`models/gated_fusion.py#L158`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/gated_fusion.py#L158)<br>• [`models/stormfusion.py#L230-L245`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/models/stormfusion.py#L230-L245) |
| **10** | **Manifest product and channel tracking** | **Not Recorded**: `sequence_manifest.csv` records timesteps (`sat_t0..sat_t7`) and binary valid flags, but not channel counts or product types. | • [`data/processed/sequences/sequence_manifest.csv`](file:///c:/Users/Niveditha%20B%20N/OneDrive/Documents/SIH%20-%202026/STORMFUSION/data/processed/sequences/sequence_manifest.csv) |

---

## 2. Product Comparison

### CURRENT CONTRACT
* **Satellite product**: Multi-spectral Level 1.5 geolocated/calibrated radiances (`HRSEVIRI-IODC` / `GOES-16 ABI L1b`)
* **Expected channels**: 3 channels (`TIR1` 10.8 µm, `WV` 6.2 µm, `TIR2` 12.0 µm) in Kelvin ($160\text{ K} - 330\text{ K}$)
* **Expected shape**: `[B, 8, 3, 500, 500]`
* **Encoder**: `ResNet18SpatialEncoder` (conv1: `[64, 3, 7, 7]`) + `ConvLSTM` ($64$ hidden channels) $\to$ `[B, 128]`
* **Fusion dependency**: Independent of input channel count; depends strictly on the 128-dimensional output embedding `[B, 128]`
* **Checkpoint dependency**: Dependent on 3-channel first layer (`conv1.weight: [64, 3, 7, 7]`)

### CLM-IODC
* **Real observation**: **YES** (Authentic EUMETSAT MPEF Level 2.0 Cloud Mask GRIB2 data on disk)
* **Channel count**: 1 channel (categorical/binary cloud mask)
* **Can current model consume it**: **NO** (Shape mismatch: `[1, 500, 500]` vs `[3, 500, 500]`, and numerical mismatch: binary/categorical mask vs continuous radiometric Kelvin temperatures)
* **Required architectural change**: Either change `in_channels=1` in `satellite_encoder` (breaking checkpoint compatibility) or implement an independent `CloudMaskEncoder` modality branch with a 4th fusion gate.
* **Scientific risks**:
  1. Complete loss of cloud-top brightness temperature gradients required to detect cyclone eye boundaries and convective eyewall cooling.
  2. Total absence of water vapor absorption dynamics (6.2 µm), blinding the model to environmental vertical shear and dry-air entrainment.
  3. Artificial replication of 1 channel into 3 fake channels would introduce false correlations and violate SIH scientific integrity standards.

### HRSEVIRI-IODC
* **Access status**: `ACCESS_BLOCKED` (Collection requires acceptance of standard EUMETSAT Data Store "General License" at `https://user.eumetsat.int`)
* **Expected channel/product characteristics**: Level 1.5 radiometrically calibrated and geolocated SEVIRI radiances containing Band 9 (`IR_108`), Band 5 (`WV_062`), and Band 10 (`IR_120`) over the North Indian Ocean at 15-minute intervals.
* **Current model compatibility**: **100% COMPATIBLE** (Direct drop-in match for `[8, 3, 500, 500]` contract).
* **Required change**: None to model architecture, fusion, or loss functions. Only requires unblocking user portal license terms.

---

## 3. Architecture Recommendation

### **RECOMMENDED: OPTION A**
**Keep current 3-channel architecture and wait for HRSEVIRI-IODC.**

#### Justification:
- **Preserves Cyclone Physics**: Tropical cyclone intensity estimation (Dvorak technique and modern deep learning models such as DeepTC) fundamentally requires physical brightness temperatures. Categorical cloud masks lack the resolution of convective intensity necessary to predict central pressure and maximum sustained winds.
- **Architectural Stability**: Zero changes are required to `StormFusionModel`, `BaselineSatelliteModel`, `ConvLSTM`, `GatedFusion`, or the multi-task prediction heads.
- **Unblocking Pathway**: Accessing `EO:EUM:DAT:MSG:HRSEVIRI-IODC` does not require paying fees or institutional clearance—it is an automated click-through license agreement on the official EUMETSAT user portal.

> [!NOTE]
> If a dual-modality pathway (Option B) is desired in the future, it should be designed as an **auxiliary, independent `CloudMaskEncoder` branch** that maps `[B, 8, 1, 500, 500] -> [B, 64]` alongside a 4th gate in `AvailabilityAwareGatedFusion`, rather than duplicating fake spectral channels.

---

## 4. Scientific Safety & Training Readiness Gate

```text
======================================================================
STORMFUSION SCIENTIFIC READINESS GATE
======================================================================
  MOSDAC / INSAT Satellite Data    : REMOVED FROM CURRENT IMPLEMENTATION
  Real NOAA Satellite Data         : VERIFIED_REAL (Development Only; NIO Compatibility = False)
  Real EUMETSAT CLM-IODC           : VERIFIED_REAL_METEOROLOGICAL_DERIVED (Cloud Mask Only)
  Real EUMETSAT HRSEVIRI-IODC      : UNAVAILABLE / ACCESS_BLOCKED (Awaiting General License)
  ERA5 Atmospheric Data            : VERIFIED_REAL (1 verified file)
  IBTrACS Historical Tracks        : VERIFIED_REAL (Cleaned NIO catalog)

  TRAINING ALLOWED                 : NO
  TRAINING READINESS               : NOT_READY
======================================================================
```

**Conclusion**: The system remains strictly gated at `training_readiness: NOT_READY` and `training_allowed: False`. No model training will be initiated until authentic Level 1.5 multi-spectral observations are acquired and verified.
