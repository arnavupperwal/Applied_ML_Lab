# Applied Machine Learning Lab (Applied_ML_Lab)

Welcome to the **Applied Machine Learning Lab** repository! This repository contains structured laboratory modules covering core to advanced concepts in machine learning and deep learning, aligned with the course syllabus and experiment outline.

---

## 📚 Laboratory Projects Index

| Exp # | Experiment Title | Description | Directory Link |
| :--- | :--- | :--- | :--- |
| **Exp 1** | Predicting Housing Prices | Develop a regression model based on features like location, size, and amenities. | [EXP_1_Predicting_Housing_Prices](./EXP_1_Predicting_Housing_Prices/README.md) |
| **Exp 2** | Iris Flower Classification | Build a classification model predicting species of iris flowers. | [EXP_2_Iris_Flower_Classification](./EXP_2_Iris_Flower_Classification/README.md) |
| **Exp 3** | Handwritten Digit Recognition | Implement using MNIST dataset and neural network. | [EXP_3_Handwritten_Digit_Recognition](./EXP_3_Handwritten_Digit_Recognition/README.md) |
| **Exp 4** | Breast Cancer Diagnosis | Classification model using medical imaging data (mammograms). | [EXP_4_Breast_Cancer_Diagnosis](./EXP_4_Breast_Cancer_Diagnosis/README.md) |
| **Exp 5** | Sentiment Analysis | Tool classifying reviews using NLP techniques. | [EXP_5_Sentiment_Analysis](./EXP_5_Sentiment_Analysis/README.md) |
| **Exp 6** | Spam Detection | Tool classifying emails using NLP techniques. | [EXP_6_Spam_Detection](./EXP_6_Spam_Detection/README.md) |
| **Exp 7** | Predicting Stock Prices | Time series prediction model for stock forecasting. | [EXP_7_Predicting_Stock_Prices](./EXP_7_Predicting_Stock_Prices/README.md) |
| **Exp 8** | Credit Risk Assessment | Credit scoring model using historical financial data. | [EXP_8_Credit_Risk_Assessment](./EXP_8_Credit_Risk_Assessment/README.md) |
| **Exp 9** | Recommendation System | Movie/book recommendation system (collaborative/content-based). | [EXP_9_Recommendation_System](./EXP_9_Recommendation_System/README.md) |
| **Exp 10** | Anomaly Detection | Outlier detection in subscription business. | [EXP_10_Anomaly_Detection](./EXP_10_Anomaly_Detection/README.md) |
| **Exp 11** | Customer Churn | Churn prediction in subscription business. | [EXP_11_Customer_Churn](./EXP_11_Customer_Churn/README.md) |
| **Exp 12** | Fake News Detection | Article classification using NLP. | [EXP_12_Fake_News_Detection](./EXP_12_Fake_News_Detection/README.md) |
| **Exp 13** | Disease Diagnosis | Medical image disease diagnosis. | [EXP_13_Disease_Diagnosis](./EXP_13_Disease_Diagnosis/README.md) |

---

## 📁 Standard Experiment Subfolder Structure

Each experiment folder is organized according to the required lab workflow:

```text
EXP_X_<Experiment_Name>/
├── README.md                  # Experiment documentation & dataset notes
├── Dataset/                   # Data directory
│   ├── Training/              # Training set
│   ├── Validation/            # Validation set
│   └── Test/                  # Test set
└── scripts/                   # Reusable preprocessing scripts
    └── preprocess.py          # Data preprocessing pipeline
```

---

## 🎯 Lab Workflow & Tasks

1. **Organization & Dataset Setup**:
   - Download or place raw datasets into `Dataset/`.
   - Update `README.md` with column descriptions and metadata.
2. **Preprocessing**:
   - Perform data sampling, cleaning, wrangling, normalization, and reduction.
3. **Dataset Splitting**:
   - Split preprocessed dataset into `Training`, `Validation`, and `Test` subdirectories.
4. **Coding Task**:
   - Implement data preprocessing in `scripts/preprocess.py` and model code in notebooks or scripts.

---

## 🛠 Getting Started

### Global Setup
```bash
# Clone repository and enter directory
cd Applied_ML_Lab

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install common ML stack
pip install -r requirements.txt
```
