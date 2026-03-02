# eeg-asd-detection-idw-xai-

# EEG-Based ASD Detection with IDW Interpolation, Ensemble ML, and Explainable AI

This repository contains the implementation (and supporting materials) for the paper:

**“An Ensemble Machine Learning Approach for ASD Detection: Handling Missing Channels with IDW Interpolation and Unveiling Key EEG Features through Explainable AI.”**

We propose an EEG-based Autism Spectrum Disorder (ASD) screening framework that:
- **Recovers missing/corrupted EEG channels** using **Inverse Distance Weighting (IDW)** interpolation (preserving electrode topography),
- Performs **EEG preprocessing + structured feature extraction**,
- Trains an **ensemble** of gradient-boosting models (**XGBoost, LightGBM, CatBoost**) combined via a **Voting Classifier**,
- Provides **interpretable global feature importance** by averaging native tree-based importances across models.

**Best reported performance:** **95.9% accuracy** using **5-fold cross-validation** on the Sheffield University Autism EEG dataset (ASD vs. non-ASD).


## Highlights
- ✅ Handles **missing EEG channels** using **IDW interpolation**  
- ✅ Extracts **26 features per channel × 64 channels = 1664 features/subject**  
- ✅ Uses **Voting Ensemble** (XGBoost + LightGBM + CatBoost)  
- ✅ Uses **5-fold CV** to reduce overfitting risk on a small cohort  
- ✅ Provides **global feature ranking** via averaged tree importances  

---

## Method Overview
**Pipeline**
1. **Input EEG recordings** (64-channel montage; downsampled if originally recorded with 128 channels)
2. **Missing channel recovery** using **Inverse Distance Weighting (IDW)** based on electrode geometry
3. **Filtering / noise reduction**
4. **Feature extraction** (per channel → concatenate across channels)
5. **Training & evaluation** with **5-fold cross-validation**
6. **Explainable AI**: compute model-wise feature importances and **average across ensemble** for stable global ranking

---

## Dataset
This work uses the **Sheffield University Autism Dataset**, consisting of EEG signals from **56 participants** (28 ASD, 28 non-ASD), aged 18–68 years, recorded using a **BioSemi ActiveTwo** montage.

> ⚠️ Note: Due to dataset licensing/permissions, the raw data may not be included in this repository.  
You may need to request access from the dataset owner/provider and place it locally in the expected folder structure.

---

## Features
We use a structured feature set extracted from each EEG channel, including statistical, time-domain, and non-linear / complexity measures.

- **Total features per subject:** `64 channels × 26 features = 1664 features`

Example feature types (typical categories used in the paper):
- Time-domain: waveform length, zero-crossing, etc.
- Statistical: mean, variance, skewness, kurtosis
- Entropy / complexity: sample entropy, permutation entropy
- Fractal measures: Hurst exponent, DFA, correlation dimension

---

## Models
Base learners (tree-based gradient boosting):
- **XGBoost**
- **LightGBM**
- **CatBoost**

Ensemble:
- **Voting Classifier** combining the three models

Evaluation:
- **5-fold cross-validation**
- Metrics typically reported: accuracy, precision, recall, F1-score, confusion matrix (depending on your scripts)

---

## Explainable AI (XAI)
Instead of using model-agnostic methods (e.g., permutation, SHAP, LIME), we use a **model-specific global importance** approach:

1. Train each model inside the ensemble
2. Extract each model’s **native tree-based feature importance**
3. **Average importances across models** to stabilize rankings and obtain a consistent global importance list

This produces electrode/feature-aware insight into what drives ASD vs. non-ASD classification.

---

## Results
- **Best cross-validated accuracy:** **95.9%** (5-fold CV) using the **Voting Ensemble**
- The most influential feature reported in the paper includes:
  - **Waveform length** at a **centro-parietal region** channel (lead 19), contributing ~**1.48%** of total ensemble importance (global ranking).

> Exact results may vary slightly depending on preprocessing, random seeds, and library versions.

