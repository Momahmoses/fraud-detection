# Fraud Detection Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end fraud detection pipeline, from synthetic data generation and exploratory data analysis through to model training, evaluation, and real-time prediction, using classical ML on financial transaction data.

---

## Problem Statement

Financial fraud costs the global economy hundreds of billions annually. This project demonstrates a complete ML pipeline for detecting fraudulent transactions: feature engineering on behavioural signals, model training with imbalanced-class handling, and a prediction API.

---

## Features

| Feature | Description |
|---------|-------------|
| Synthetic Data Generation | Realistic transaction dataset with configurable fraud rate |
| Exploratory Data Analysis | Distribution plots, correlation heatmaps, fraud pattern analysis |
| ML Pipeline | Logistic Regression, Random Forest, XGBoost comparison |
| Imbalanced Class Handling | SMOTE oversampling and class-weight tuning |
| Model Evaluation | AUC-ROC, Precision-Recall, Confusion Matrix |
| Prediction Script | Single-transaction fraud probability scoring |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn, XGBoost |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Imbalanced Learning | imbalanced-learn (SMOTE) |

---

## Project Structure

```
fraud-detection/
├── data/
│   └── fraud_dataset.csv     # Synthetic transaction dataset
├── models/                   # Saved model artifacts
├── plots/                    # EDA and evaluation charts
├── data_generator.py         # Synthetic fraud dataset generator
├── eda.py                    # Exploratory data analysis
├── train.py                  # Model training and evaluation
└── predict.py                # Single-transaction prediction
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt  # if present, else: pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn

# Generate data, train, and predict
python data_generator.py
python eda.py
python train.py
python predict.py
```

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
