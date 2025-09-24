# MC-H-Geo
MC-H-Geo: A multi-scale contextual hierarchical framework for fine-grained lithological classification of outcrop TLS point clouds, integrating feature engineering, gated expert classification, and geological post-processing.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)]()

---

## 🔍 Introduction
**MC-H-Geo** is a novel framework for **fine-grained lithological classification of terrestrial laser scanning (TLS) outcrop point clouds**.  
It integrates **multi-scale contextual feature engineering**, a **Gated Expert Classifier**, and a **two-step geological post-processing** strategy to achieve geologically consistent lithological mapping at the microfacies scale.

This repository provides the source code, demo data, and scripts to reproduce the core results presented in our article.

---


**Workflow:**
1. **Anchor Point Construction** – voxelize the outcrop into 10 mm grids and generate anchor points.  
2. **Multi-scale Contextual Feature Engine** – extract spectral, geometric, textural, frequency-domain, and cross-scale differential features at 10 cm and 30 cm scales.  
3. **Gated Expert Classifier** –  
   - Stage 1: Random Forest for vegetation–rock separation.  
   - Stage 2: Stacking ensemble (RF + XGBoost + MLP, fused by logistic regression) for mudstone, sandstone, and siltstone discrimination.  
4. **Two-step Geological Post-processing** – stratigraphic Z-axis correction + neighborhood smoothing for geologically coherent outputs.  

---

## ⚙️ Requirements
- Python 3.9+  
- Core dependencies (see `environment.yml`):
  - python=3.9
  - numpy>=1.21
  - pandas>=1.3
  - scikit-learn=1.7.1
  - xgboost>=1.6
  - catboost>=1.1
  - lightgbm>=3.3
  - imbalanced-learn>=0.9
  - shap>=0.41
  - matplotlib>=3.5
  - seaborn>=0.11
  - tqdm>=4.64
  - numba>=0.55
  - llvmlite>=0.38
  - joblib=1.5.1
  - scikit-image=0.20.0
  - pip
  - pip:
      - jinja2>=3.0
      - reportlab>=3.6  
