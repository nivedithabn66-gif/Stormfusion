# STORMFUSION

**Production-Oriented AI/ML Multimodal System for Tropical Cyclone Identification, Classification, and Prediction in the North Indian Ocean**

---

## 1. STORMFUSION

**STORMFUSION** is an advanced operational AI/ML framework engineered specifically for tropical cyclone analysis across the North Indian Ocean (NIO) region, encompassing both the Bay of Bengal and the Arabian Sea.

---

## 2. SIH26070 Problem Statement

* **Statement ID:** SIH26070 (Smart India Hackathon 2026)
* **Title:** To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data.
* **Domain:** Satellite Meteorology, Remote Sensing, AI/ML, Climate Analytics.

---

## 3. Problem Overview

The North Indian Ocean (NIO) accounts for a significant proportion of global cyclone-induced devastation due to high coastal population densities, shallow bathymetry (especially in the Bay of Bengal), and complex atmospheric-oceanic interactions. Accurate identification, pattern classification, intensity estimation, and track forecasting are essential for early warning systems.

Conventional forecasting relies heavily on manual Dvorak techniques and traditional numerical weather prediction (NWP) models, which can be computationally intensive and slower to update. Integrating multi-source geostationary satellite observations with historical tracks, reanalysis fields, and sea-surface temperature data using modern deep learning presents a major opportunity for operational meteorology.

---

## 4. Project Objective

STORMFUSION aims to develop an end-to-end, multi-task, multi-modal AI framework to perform:

1. **Tropical Cyclone Detection / Identification** (Binary/Multi-class presence detection)
2. **Cyclone Pattern Classification** (Structural and cloud pattern recognition e.g., Eye, Curved Band, Shear, Central Dense Overcast)
3. **Intensity Estimation** (Maximum Sustained Wind speed in knots/m/s)
4. **Central Pressure Estimation** (Minimum Sea Level Pressure in hPa)
5. **Future Track Prediction** (Latitude/Longitude trajectory forecaster over standard forecast lead times)
6. **Uncertainty Estimation** (Epistemic and aleatoric confidence metrics for predictions)
7. **Model Explainability** (Visual attribution maps detailing key meteorological features influencing model outputs)
8. **Missing-Modality Robustness** (Graceful degradation of inference performance when one or more input data streams are missing or delayed)

---

## 5. Proposed Solution

STORMFUSION addresses the operational challenges of NIO cyclone forecasting by fusing spatial, temporal, environmental, and oceanic data streams within a unified multi-task deep learning architecture. Rather than relying on a single data source, STORMFUSION dynamically weights inputs based on data availability and quality, ensuring high reliability under real-world sensor outages or transmission delays.

---

## 6. Planned Architecture

The planned AI system architecture integrates domain-tailored neural network encoders for each data modality:

```text
Satellite imagery (IR_108, WV_062, IR_120)
       │
       ▼
   ResNet18 (Spatial feature extractor)
       │
       ▼
    ConvLSTM (Spatio-temporal sequence encoder)
       │
       ▼
  [Satellite Embedding]
       │
       ├────────────────────────┐
       │                        │
ERA5 Reanalysis             Historical Storm Track
       │                        │
       ▼                        ▼
Lightweight Env Encoder      Track GRU
       │                        │
       ▼                        ▼
[Environmental Embedding]   [Track Embedding]
       │                        │
       └───────────┬────────────┘
                   │
                   ▼
    Availability-Aware Gated Fusion
                   │
                   ▼
    Shared Multi-Task Representation
                   │
         ┌─────────┼─────────┬─────────┬─────────┐
         ▼         ▼         ▼         ▼         ▼
     Detection  Pattern  Intensity Pressure  Track
       Head      Head      Head      Head     Head
```

> **Important Conceptual Note:** Individual components such as ResNet, ConvLSTM, GRU, Gated Fusion, MC-Dropout, and Grad-CAM are established machine learning blocks and are NOT claimed as individually novel. The primary engineering contribution of STORMFUSION lies in the **NIO-focused operational integration, multimodal fusion schema, missing-modality robustness mechanisms, epistemic/aleatoric uncertainty quantification, visual explainability pipeline, and rigorous storm-disjoint evaluation methodology**.

---

## 7. Dataset & Satellite Provider Ecosystem

STORMFUSION uses **EUMETSAT HRSEVIRI-IODC as its sole satellite data provider**:

```text
                    Satellite Data Provider
                               │
                               ▼
                     EUMETSAT HRSEVIRI-IODC
                               │
                               ▼
                       EUMETSAT Ingestion
                               │
                               ▼
                     STORMFUSION Preprocessing
```

* **EUMETSAT HRSEVIRI-IODC (Meteosat-9/8 at 45.5°E):** The operational satellite provider for the North Indian Ocean (NIO), providing High Rate SEVIRI Level 1.5 multi-spectral imagery (`IR_108`, `WV_062`, `IR_120`) via the official EUMETSAT Data Store (`EO:EUM:DAT:MSG:HRSEVIRI-IODC`).
* **NOAA GOES:** Open-satellite data source used strictly for preliminary pipeline development and offline benchmarking.
* **MOSDAC / INSAT:** MOSDAC/INSAT integration has been completely removed from the current implementation to maintain a single, consistent, legally sound satellite-data pipeline.
* **IBTrACS (International Best Track Archive for Climate Stewardship):** Global tropical cyclone track database maintained by **NOAA NCEI**, filtered specifically for North Indian Ocean storm tracks.
* **IMD / RSMC New Delhi Reference Information:** Official best-track data and meteorological bulletins from the **India Meteorological Department (IMD)** for ground-truth validation.
* **ERA5 Reanalysis Data:** Atmospheric environmental fields (wind vectors at 850/200 hPa, relative humidity, geopotential height, sea level pressure) from **Copernicus CDS / ECMWF**.
* **NOAA OISST (Optimum Interpolation Sea Surface Temperature):** High-resolution daily sea-surface temperature data from **NOAA**.

> **Terminology Clarification:**
> * **EUMETSAT HRSEVIRI-IODC:** Primary North Indian Ocean satellite data provider (`EO:EUM:DAT:MSG:HRSEVIRI-IODC`).
> * **STORMFUSION:** Multi-modal AI tropical cyclone intelligence system.
> * **MOSDAC / INSAT:** Removed from current implementation.

---

## 8. ML Components

* **Satellite Encoder:** ResNet18 spatial feature extractor paired with ConvLSTM to capture dynamic cloud pattern evolution across consecutive satellite frames.
* **Track Encoder:** Gated Recurrent Unit (GRU) to process temporal sequences of past cyclone coordinates, movement speed, heading, and intensity history.
* **Environmental Encoder:** Lightweight Convolutional/Dense Encoder to process multi-level atmospheric variables from ERA5/NWP fields.
* **Fusion Module:** Availability-Aware Gated Fusion network that dynamically adjusts modal weights when specific inputs (e.g., satellite imagery or atmospheric fields) are unavailable.
* **Multi-Task Prediction Heads:** Specialized task heads sharing a joint latent representation for concurrent multi-objective inference.

---

## 9. Robustness Strategy

Operational satellite and environmental data feeds often suffer from missing frames, transmission lag, or missing channels. STORMFUSION incorporates:

* **Availability-Aware Gated Fusion:** Learns dynamic masking and gating weights to rely on remaining available modalities (e.g., track-only or satellite+track) when ERA5 or satellite streams drop out.
* **Modality Dropout Training:** Randomly masks input modalities during training to force the shared representation to remain informative under sub-optimal inputs.

---

## 10. Uncertainty and Explainability

* **Uncertainty Quantification:** Monte Carlo Dropout (MC-Dropout) and ensemble variance estimation provide confidence intervals for intensity, pressure, and track forecast outputs.
* **Visual Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) generates spatial heatmap attributions over satellite imagery, displaying the exact cloud regions (e.g., eyewall convection, feeder bands) driving pattern classification and intensity estimation.

---

## 11. Evaluation Strategy

To prevent data leakage caused by strong autocorrelation between adjacent satellite images of the same storm:

* **Storm-Disjoint Dataset Splitting:** Training, validation, and test splits are strictly partitioned by whole cyclone events rather than random time-step frames. No storm present in the training set will ever appear in the validation or test sets.

---

## 12. Development Roadmap

- [x] **STEP 1** — Project foundation
- [x] **STEP 2** — IBTrACS acquisition & NIO storm catalog processing
- [x] **STEP 3** — Satellite data pipeline & provenance verification
- [x] **STEP 4** — ERA5 pipeline architecture (CDS auth verified)
- [x] **STEP 5** — Dataset alignment, 100% deterministic sequence generation & storm-disjoint splitting
- [x] **STEP 6 / 6A** — EUMETSAT HRSEVIRI-IODC Level 1.5 data provider & OAuth client
- [x] **STEP 7 / 7.1** — PyTorch Dataset, DataLoader pipeline & storm coverage audit
- [x] **STEP 8A** — Baseline ResNet18 + ConvLSTM Satellite Encoder (COMPLETE & UNIT-TESTED)
- [x] **STEP 8B** — Track GRU & ERA5 Environmental Encoders (COMPLETE & UNIT-TESTED)
- [x] **STEP 8C** — Availability-Aware Gated Fusion (COMPLETE & UNIT-TESTED)
- [x] **STEP 9** — Shared Multi-Task Prediction Trunk & Heads (COMPLETE & UNIT-TESTED)
- [x] **STEP 10A** — Reproducible Training Infrastructure (COMPLETE & UNIT-TESTED — NO SCIENTIFIC TRAINING)
- [x] **STEP 10B-PREP** — Real Data Verification & Provenance Gate (COMPLETE & UNIT-TESTED — NO SCIENTIFIC TRAINING)
- [x] **STEP 11** — Real EUMETSAT IODC Acquisition & Provenance Architecture (COMPLETE & UNIT-TESTED)
- [x] **STEP 12** — Real ERA5 Acquisition, Provenance Verification & Alignment Unlock (COMPLETE & UNIT-TESTED — Cyclone FANI 2019 controlled ERA5 workflow)
- [x] **STEP 13** — Missing-Modality Robustness & Graceful-Degradation Framework (COMPLETE & UNIT-TESTED — 8-scenario degradation suite)
- [x] **STEP 14** — Satellite Provider Consolidation: EUMETSAT HRSEVIRI-IODC as sole operational provider (MOSDAC REMOVED)

---

## 13. Project Structure

```text
STORMFUSION/
│
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── service.py
│
├── data/
│   ├── raw/
│   │   ├── eumetsat/
│   │   ├── ibtracs/
│   │   ├── imd/
│   │   ├── era5/
│   │   └── oisst/
│   │
│   ├── processed/
│   │   ├── satellite/
│   │   ├── atmosphere/
│   │   ├── ocean/
│   │   ├── tracks/
│   │   ├── eumetsat_provenance.json
│   │   ├── era5_provenance.json
│   │   └── sequences/
│   │
│   └── splits/
│
├── preprocessing/
│   ├── storm_dataset.py
│   ├── satellite_provider.py
│   ├── eumetsat_iodc_provider.py
│   ├── era5_loader.py
│   ├── track_loader.py
│   ├── sequence_qc.py
│   ├── eumetsat_provenance.py
│   ├── era5_provenance.py
│   ├── alignment_qc.py
│   └── compute_normalization.py
│
├── models/
│   ├── __init__.py
│   ├── convlstm.py
│   ├── satellite_encoder.py
│   ├── baseline_satellite_model.py
│   ├── track_encoder.py
│   ├── era5_encoder.py
│   ├── gated_fusion.py
│   ├── multitask_model.py
│   ├── losses.py
│   └── heads/
│       ├── __init__.py
│       ├── detection_head.py
│       ├── pattern_head.py
│       ├── intensity_head.py
│       ├── pressure_head.py
│       └── track_prediction_head.py
│
├── training/
│   ├── __init__.py
│   ├── reproducibility.py
│   ├── device.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── logger.py
│   ├── readiness.py
│   └── trainer.py
│
├── evaluation/
├── robustness/
├── explainability/
├── configs/
│   ├── dataset_config.yaml
│   ├── eumetsat_config.yaml
│   └── model_config.yaml
│
├── scripts/
│   ├── test_step10b_prep.py
│   ├── test_step10a.py
│   ├── test_step9.py
│   ├── test_step8c.py
│   ├── test_step8b.py
│   ├── test_step11.py
│   ├── test_baseline_model.py
│   ├── test_dataloader.py
│   ├── verify_real_era5.py
│   ├── build_sequence_manifest.py
│   └── verify_historical_hrseviri_readiness.py
│
├── checkpoints/
├── logs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 14. Data Access Notes

* **Satellite Data:** Sourced exclusively via the official EUMETSAT Data Store (`EO:EUM:DAT:MSG:HRSEVIRI-IODC`). Access to historical Level 1.5 collections requires acceptance of the free standard General License at `https://user.eumetsat.int`.
* **ERA5 Reanalysis Note:** ERA5 atmospheric fields are used primarily for historical model training and research evaluation due to publication latency (~5 days). The operational deployment design allows replacing ERA5 with real-time NWP forecasts (e.g., GFS, NCUM, or IMD GFS) without changing the downstream network topology.
* **Credentials Management:** Environment keys, API tokens, and CDS/EUMETSAT credentials must be managed via local `.env` files and never committed to source repositories.

---

## 15. Operational Deployment Considerations

* Fast, asynchronous FastAPI backend designed to serve multi-task predictions.
* Modular model design enabling real-time fallback when multi-spectral satellite imagery or environmental fields are missing.
* Native support for batch processing and single-event operational forecast evaluation.

---

## 16. Current Development Status

* **Status:** EUMETSAT HRSEVIRI-IODC is the sole operational satellite provider for the North Indian Ocean. MOSDAC/INSAT integration has been completely removed.
* **Architecture:**
  * **Sole Operational Satellite Provider:** EUMETSAT HRSEVIRI-IODC (`EO:EUM:DAT:MSG:HRSEVIRI-IODC` at 45.5°E)
  * **MOSDAC / INSAT:** REMOVED FROM CURRENT IMPLEMENTATION (0 operational dependencies)
  * **Model Input Contract:** Strictly frozen at `[B, 8, 3, 500, 500]`, `float32`, channels `WV_062`, `IR_108`, `IR_120`, 8 unique satellite timestamps.
* **Real Data Availability:**
  * **EUMETSAT HRSEVIRI-IODC:** 1 genuine 2026 MSG2 observation (`REAL_DATA_VALIDATION_ONLY`). Historical collection access blocked by `GeneralLicense`.
  * **Verified Real Training Sequences:** 0
  * **ERA5 Reanalysis:** 1 verified real file (FANI 2019 controlled)
  * **IBTrACS Historical Tracks:** Verified real NIO catalog (1842–2025)
* **Training Performed:** `NO` (Scientific training gate remains strictly locked: `TRAINING_BLOCKED = TRUE`, `TRAINING_READY = FALSE`).
* **Normalization Status:** `NOT_COMPUTED (NO TRAINING DATA)` (Enforced train-split isolation; 2026 validation file strictly excluded).





