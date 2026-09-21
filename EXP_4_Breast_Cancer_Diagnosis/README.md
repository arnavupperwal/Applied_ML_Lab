# Exp 4: Breast Cancer Diagnosis

## 🎯 Objective & Overview
Build a binary classification model to diagnose breast cancer (Malignant vs. Benign) using medical feature measurements from digitized fine needle aspirate (FNA) images.

---

## 📁 Current Directory Structure & Status
```text
EXP_4_Breast_Cancer_Diagnosis/
├── README.md                          # Experiment documentation
├── Dataset/                           # Data directory
│   ├── data.csv                       # Raw dataset (569 rows, 33 columns)
│   ├── Training/train.csv             # Preprocessed Training split (398 samples, 11 columns)
│   ├── Validation/val.csv             # Preprocessed Validation split (57 samples, 11 columns)
│   └── Test/test.csv                  # Preprocessed Test split (114 samples, 11 columns)
└── scripts/                           # Scripts directory
    └── preprocess.py                  # Preprocessing wrapper script
```

---

## 📊 Dataset Information
- **Raw File**: `Dataset/data.csv` (569 records, 33 attributes)
- **Target Variable**: `diagnosis` (`M` = Malignant -> 1, `B` = Benign -> 0)
- **Features**: Cell nucleus features (`radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, etc.).
- **Preprocessed Splits (70% / 10% / 20%)**:
  - **Training set**: `Dataset/Training/train.csv` (398 records, 11 columns)
  - **Validation set**: `Dataset/Validation/val.csv` (57 records, 11 columns)
  - **Test set**: `Dataset/Test/test.csv` (114 records, 11 columns)

---

## 🛠 Preprocessing & Scripts Workflow
- **Preprocessing Pipeline (`scripts/preprocess.py`)**:
  - Calls `auto_preprocess` from `../../scripts/preprocess_template.py`.
  - **Data Cleaning**: Automatically dropped useless/identifier columns (`id` and empty trailing column `Unnamed: 32`).
  - **Data Wrangling**: Encoded target `diagnosis` ('M' -> 1, 'B' -> 0).
  - **Data Normalization**: `StandardScaler` applied across continuous medical feature measurements.
  - **Data Reduction**: `PCA` dimensionality reduction applied to 30 continuous features, retaining 10 principal components (95.16% explained variance).
  - **Splitting**: Stratified 70/10/20 train/val/test export.
