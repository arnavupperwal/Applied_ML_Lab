# Exp 2: Iris Flower Classification

## 🎯 Objective & Overview
Build a classification model predicting the species of iris flowers based on sepal and petal measurements.

---

## 📁 Current Directory Structure & Status
```text
EXP_2_Iris_Flower_Classification/
├── README.md                          # Experiment documentation
├── Dataset/                           # Data directory
│   ├── IRIS.csv                       # Raw dataset (150 rows, 5 columns)
│   ├── Training/train.csv             # Preprocessed Training split (102 samples, 5 columns)
│   ├── Validation/val.csv             # Preprocessed Validation split (15 samples, 5 columns)
│   └── Test/test.csv                  # Preprocessed Test split (30 samples, 5 columns)
└── scripts/                           # Scripts directory
    └── preprocess.py                  # Preprocessing wrapper script
```

---

## 📊 Dataset Information
- **Raw File**: `Dataset/IRIS.csv` (150 records, 5 attributes)
- **Features**: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Target Variable**: `species` (`Iris-setosa` -> 0, `Iris-versicolor` -> 1, `Iris-virginica` -> 2)
- **Preprocessed Splits (70% / 10% / 20%)**:
  - **Cleaned Dataset**: 147 unique records (3 duplicate rows automatically detected and removed).
  - **Training set**: `Dataset/Training/train.csv` (102 records)
  - **Validation set**: `Dataset/Validation/val.csv` (15 records)
  - **Test set**: `Dataset/Test/test.csv` (30 records)

---

## 🛠 Preprocessing & Scripts Workflow
- **Preprocessing Pipeline (`scripts/preprocess.py`)**:
  - Calls `auto_preprocess` from `../../scripts/preprocess_template.py`.
  - **Data Cleaning**: Automatically detected and dropped 3 duplicate rows.
  - **Data Wrangling**: Encoded target `species` using `LabelEncoder`.
  - **Data Normalization**: Scaled 4 numerical flower measurements using `StandardScaler`.
  - **Splitting**: Stratified 70/10/20 train/val/test export.
