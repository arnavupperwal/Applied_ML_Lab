# Exp 5: Sentiment Analysis

## 🎯 Objective & Overview
Build a Natural Language Processing (NLP) classification model to analyze text reviews/tweets and categorize their sentiment (Positive, Negative, Neutral).

---

## 📁 Current Directory Structure & Status
```text
EXP_5_Sentiment_Analysis/
├── README.md                          # Experiment documentation
├── Dataset/                           # Data directory
│   ├── sentiment_analysis.csv         # Raw dataset (499 rows, 7 columns)
│   ├── Training/train.csv             # Preprocessed Training split (275 samples, 60 columns)
│   ├── Validation/val.csv             # Preprocessed Validation split (40 samples, 60 columns)
│   └── Test/test.csv                  # Preprocessed Test split (79 samples, 60 columns)
└── scripts/                           # Scripts directory
    └── preprocess.py                  # Preprocessing wrapper script
```

---

## 📊 Dataset Information
- **Raw File**: `Dataset/sentiment_analysis.csv` (499 records, 7 attributes)
- **Target Variable**: `sentiment` (`negative` -> 0, `neutral` -> 1, `positive` -> 2)
- **Features**: `Year`, `Month`, `Day`, `Time of Tweet`, `text` (Input text), `Platform`.
- **Preprocessed Splits (70% / 10% / 20%)**:
  - **Cleaned Dataset**: 394 unique records (105 duplicate rows automatically detected and removed).
  - **Training set**: `Dataset/Training/train.csv` (275 records, 60 columns)
  - **Validation set**: `Dataset/Validation/val.csv` (40 records, 60 columns)
  - **Test set**: `Dataset/Test/test.csv` (79 records, 60 columns)

---

## 🛠 Preprocessing & Scripts Workflow
- **Preprocessing Pipeline (`scripts/preprocess.py`)**:
  - Calls `auto_preprocess` from `../../scripts/preprocess_template.py`.
  - **Data Cleaning**: Automatically detected and removed 105 duplicate rows.
  - **Data Wrangling**: Encoded target `sentiment` using `LabelEncoder`, performed text cleaning and extracted 50 TF-IDF numerical features from `text`, and One-Hot encoded categorical metadata (`Time of Tweet`, `Platform`).
  - **Data Normalization**: `StandardScaler` applied to all feature columns.
  - **Splitting**: Stratified 70/10/20 train/val/test export.
