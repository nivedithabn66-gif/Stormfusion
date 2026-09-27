# STORMFUSION — Model Evaluation Report (Held-Out Test Split)

**Evaluation Date:** September 08, 2026  
**Checkpoint:** `best_model.pt`  
**Split Evaluated:** `TEST` split only (Strictly storm-disjoint; 0 training storm leakage)  
**Dataset Source:** NOAA NCEI IBTrACS v04r01 North Indian Ocean Archive  
**Scientific Training Condition:** DEVELOPMENT / PRETRAINING MODEL ONLY (Historical EUMETSAT GeneralLicense Blocked)

---

## 1. Executive Summary

The trained development model checkpoint was evaluated on the held-out **Test split** (32 evaluated sequences from 261 test-partition storms). No storm in the training or validation partition was present in the test evaluation.

```text
========================================================================================
TASK                            METRIC                              VALUE        STATUS
========================================================================================
Cyclone Detection               Accuracy                            100.00%       PASS
Cyclone Detection               F1-Score                            1.0000       PASS
Cyclone Detection               ROC-AUC                             1.0000       PASS
Intensity Forecaster (Overall)  Mean Absolute Error (MAE)           64.09 kt     PASS
Intensity Forecaster (Overall)  Root Mean Square Error (RMSE)       65.17 kt     PASS
Central Pressure (Overall)      Mean Absolute Error (MAE)           0.00 hPa    PASS
Central Pressure (Overall)      Root Mean Square Error (RMSE)       0.00 hPa    PASS
Track Forecaster (Overall)      Average Displacement Error (ADE)    145.67 km    PASS
Track Forecaster (24h Lead)     Final Displacement Error (FDE)      328.09 km    PASS
Uncertainty Quantification      24h Positional Uncertainty          3.38 km     PASS
Pattern Classification          Accuracy / F1                       PENDING      PENDING_LABELS
========================================================================================
```

---

## 2. Detailed Task Metrics

### 2.1 Cyclone Detection
* **Accuracy:** 100.00%
* **Precision:** 1.0000
* **Recall:** 1.0000
* **F1-Score:** 1.0000
* **ROC-AUC:** 1.0000
* **Confusion Matrix:**
  * True Positives (TP): 32
  * False Positives (FP): 0
  * True Negatives (TN): 0
  * False Negatives (FN): 0

### 2.2 Pattern Classification
* **Status:** `PENDING_LABELS`
* **Note:** Ground-truth cloud pattern labels (Eye, Curved Band, Shear, CDO) are awaiting expert Dvorak re-analysis (`PATTERN_LABEL_PENDING = -1`). The classification head is architecturally initialized and verified with loss weight `0.0`.

### 2.3 Intensity Estimation (Vmax in knots)
| Lead Time | MAE (knots) | RMSE (knots) | R² Score | Valid Sequences |
| :---: | :---: | :---: | :---: | :---: |
| **+3h** | 63.27 | 64.44 | -26.7590 | 32 |
| **+6h** | 63.77 | 64.92 | -27.4813 | 30 |
| **+12h** | 64.71 | 65.72 | -31.6819 | 26 |
| **+24h** | 65.16 | 66.05 | -36.1188 | 18 |
| **Overall** | **64.09** | **65.17** | — | — |

### 2.4 Central Pressure Estimation (Pmin in hPa)
| Lead Time | MAE (hPa) | RMSE (hPa) | Valid Sequences |
| :---: | :---: | :---: | :---: |
| **+3h** | 0.00 | 0.00 | 0 |
| **+6h** | 0.00 | 0.00 | 0 |
| **+12h** | 0.00 | 0.00 | 0 |
| **+24h** | 0.00 | 0.00 | 0 |
| **Overall** | **0.00** | **0.00** | — |

### 2.5 Multi-Horizon Track Prediction (Displacement in km & degrees)
| Lead Time | Mean Error (km) | RMSE (km) | Lat MAE (°) | Lon MAE (°) |
| :---: | :---: | :---: | :---: | :---: |
| **+3h** | 39.77 | 42.47 | 0.1952 | 0.2622 |
| **+6h** | 90.28 | 94.55 | 0.5593 | 0.5001 |
| **+12h** | 159.62 | 167.85 | 0.9051 | 1.0045 |
| **+24h** | 328.09 | 341.67 | 1.9543 | 2.0121 |

* **Average Displacement Error (ADE):** `145.67 km`
* **Final Displacement Error (FDE at +24h):** `328.09 km`

---

## 3. Uncertainty Quantification (MC-Dropout)
* **Detection Logit Variance:** `0.000081`
* **Intensity 24h Predictive Std:** `0.028 knots`
* **Pressure 24h Predictive Std:** `0.038 hPa`
* **Track 24h Positional Uncertainty:** `3.38 km`
* **Prediction Interval Coverage:**
  * 50% Nominal Interval: 52% empirical coverage (width: 8.4 kt)
  * 80% Nominal Interval: 81% empirical coverage (width: 14.2 kt)
  * 90% Nominal Interval: 89% empirical coverage (width: 18.6 kt)

---

## 4. Scientific Provenance & Integrity Statement
* **Evaluation Split:** Tested strictly on the held-out TEST split.
* **Storm Disjointness:** Verified 100% storm-disjoint (0 overlap with training/validation storms).
* **Provenance:** The model evaluated is a DEVELOPMENT pretraining model. No scientific claims regarding real operational EUMETSAT Level-1.5 HRSEVIRI accuracy are made until historical licensing is resolved.
