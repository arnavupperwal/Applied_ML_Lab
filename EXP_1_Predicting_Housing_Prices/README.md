# Exp 1: Predicting Housing Prices

## 🎯 Objective & Overview
Develop a regression model based on features like location, size, and amenities to predict housing prices.

---

## 📁 Current Directory Structure & Status
```text
EXP_1_Predicting_Housing_Prices/
├── README.md                          # Experiment documentation
├── Dataset/                           # Data directory
│   ├── Housing.csv                    # Raw dataset (545 rows, 13 columns)
│   ├── Training/train.csv             # Preprocessed Training split (381 samples, 14 columns)
│   ├── Validation/val.csv             # Preprocessed Validation split (55 samples, 14 columns)
│   └── Test/test.csv                  # Preprocessed Test split (109 samples, 14 columns)
└── scripts/                           # Scripts directory
    ├── preprocess.py                  # Preprocessing wrapper script
    └── House_Prediction_Modal.py      # House price prediction model script
```

---

## 📊 Dataset Information
- **Raw File**: `Dataset/Housing.csv` (545 records, 13 attributes)
- **Target Variable**: `price`
- **Features**: `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`.
- **Preprocessed Splits (70% / 10% / 20%)**:
  - **Training set**: `Dataset/Training/train.csv` (381 records, 14 numeric features)
  - **Validation set**: `Dataset/Validation/val.csv` (55 records, 14 numeric features)
  - **Test set**: `Dataset/Test/test.csv` (109 records, 14 numeric features)

---

## 🛠 Preprocessing & Scripts Workflow
1. **Preprocessing Pipeline (`scripts/preprocess.py`)**:
   - Calls `auto_preprocess` from `../../scripts/preprocess_template.py`.
   - **Data Cleaning**: Verified zero nulls/duplicates.
   - **Data Wrangling**: One-Hot encoded binary and categorical features (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`).
   - **Data Normalization**: `StandardScaler` applied to continuous numerical feature columns.
   - **Splitting**: Stratified 70/10/20 train/val/test export.
2. **Model Script**:
   - `scripts/House_Prediction_Modal.py` is reserved for model training and evaluation (Linear Regression, Decision Tree, Random Forest).
