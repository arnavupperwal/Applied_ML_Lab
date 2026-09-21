# Applied Machine Learning Lab (Applied_ML_Lab)

Welcome to the **Applied Machine Learning Lab** repository! This repository contains structured laboratory modules covering core to advanced concepts in machine learning and deep learning, aligned with the course syllabus and experiment outline.

---

## 📚 Laboratory Projects Index

| Exp # | Experiment Title | Description | Preprocessing & Dataset Status | Directory Link |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Predicting Housing Prices | Regression model based on features like location, size, and amenities. | Preprocessed & Split (381 train / 55 val / 109 test, 14 features) | [EXP_1_Predicting_Housing_Prices](./EXP_1_Predicting_Housing_Prices/README.md) |
| **Exp 2** | Iris Flower Classification | Classification model predicting species of iris flowers. | Preprocessed & Split (102 train / 15 val / 30 test, cleaned 3 dups) | [EXP_2_Iris_Flower_Classification](./EXP_2_Iris_Flower_Classification/README.md) |
| **Exp 3** | Handwritten Digit Recognition | Classification using MNIST dataset with PCA feature reduction. | Preprocessed & Split (29.4k train / 4.2k val / 8.4k test, 154 PCA features) | [EXP_3_Handwritten_Digit_Recognition](./EXP_3_Handwritten_Digit_Recognition/README.md) |
| **Exp 4** | Breast Cancer Diagnosis | Binary classification model using medical diagnostic data. | Preprocessed & Split (398 train / 57 val / 114 test, 10 PCA features) | [EXP_4_Breast_Cancer_Diagnosis](./EXP_4_Breast_Cancer_Diagnosis/README.md) |
| **Exp 5** | Sentiment Analysis | Sentiment classification of text reviews/tweets using TF-IDF. | Preprocessed & Split (275 train / 40 val / 79 test, cleaned 105 dups, 50 TF-IDF) | [EXP_5_Sentiment_Analysis](./EXP_5_Sentiment_Analysis/README.md) |
| **Exp 6** | Spam Detection | Email/SMS spam classification using NLP techniques. | Workspace Initialized | [EXP_6_Spam_Detection](./EXP_6_Spam_Detection/README.md) |
| **Exp 7** | Predicting Stock Prices | Time series prediction model for stock forecasting. | Workspace Initialized | [EXP_7_Predicting_Stock_Prices](./EXP_7_Predicting_Stock_Prices/README.md) |
| **Exp 8** | Credit Risk Assessment | Credit scoring model using historical financial data. | Workspace Initialized | [EXP_8_Credit_Risk_Assessment](./EXP_8_Credit_Risk_Assessment/README.md) |
| **Exp 9** | Recommendation System | Recommendation system (collaborative/content-based). | Workspace Initialized | [EXP_9_Recommendation_System](./EXP_9_Recommendation_System/README.md) |
| **Exp 10** | Anomaly Detection | Outlier detection in activity/transaction data. | Workspace Initialized | [EXP_10_Anomaly_Detection](./EXP_10_Anomaly_Detection/README.md) |
| **Exp 11** | Customer Churn | Churn prediction in subscription business. | Workspace Initialized | [EXP_11_Customer_Churn](./EXP_11_Customer_Churn/README.md) |
| **Exp 12** | Fake News Detection | Article classification using NLP techniques. | Workspace Initialized | [EXP_12_Fake_News_Detection](./EXP_12_Fake_News_Detection/README.md) |
| **Exp 13** | Disease Diagnosis | Medical diagnostic dataset disease prediction. | Workspace Initialized | [EXP_13_Disease_Diagnosis](./EXP_13_Disease_Diagnosis/README.md) |

---

## 📖 General Guides & Resources

- [DATA_PREPROCESSING_GUIDE.md](./DATA_PREPROCESSING_GUIDE.md) - Comprehensive guide for dataset acquisition, auto-detecting preprocessing pipeline, and dataset splitting.
- [preprocessing_checklist.md](./preprocessing_checklist.md) - Checklist of essential preprocessing, cleaning, wrangling, scaling, and feature reduction techniques.
- [datasets_index.md](./datasets_index.md) - Index of available datasets across the workspace.
- [experiment_template.md](./experiment_template.md) - Template for experiment documentation and reporting.
- [scripts/preprocess_template.py](./scripts/preprocess_template.py) - Reusable preprocessing framework supporting 5 key techniques (Sampling, Cleaning, Wrangling, Normalization, Reduction) with auto-detection.

---

## 🛠 Preprocessing Pipeline Features (`preprocess_template.py`)

The centralized preprocessing pipeline implements 5 core preprocessing stages:
1. **Data Sampling**: Fractional subsampling and class balancing / downsampling.
2. **Data Cleaning**: Duplicate row removal, missing value imputation, and empty/ID column pruning.
3. **Data Wrangling**: Target label encoding, text normalization & TF-IDF extraction, and categorical One-Hot Encoding.
4. **Data Normalization**: Numerical feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
5. **Data Reduction**: Low-variance column pruning and PCA dimensionality reduction.
6. **Auto-Detection (`auto_preprocess`)**: Conditionally executes **ONLY** the required techniques based on dataset inspection.

---

## 📁 Standard Experiment Subfolder Structure

Each experiment folder is organized according to the required lab workflow:

```text
EXP_X_<Experiment_Name>/
├── README.md                  # Experiment documentation & dataset notes
├── Dataset/                   # Data directory
│   ├── <raw_dataset>.csv      # Raw dataset (if placed)
│   ├── Training/              # Training set (train.csv)
│   ├── Validation/            # Validation set (val.csv)
│   └── Test/                  # Test set (test.csv)
└── scripts/                   # Reusable preprocessing & model scripts
    └── preprocess.py          # Data preprocessing pipeline wrapper
```

---

## 🛠 Getting Started

### Global Setup
```bash
# Clone repository and enter directory
cd Applied_ML_Lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install common ML stack
pip install -r requirements.txt
```
