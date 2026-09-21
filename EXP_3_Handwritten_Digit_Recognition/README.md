# Exp 3: Handwritten Digit Recognition

## 🎯 Objective & Overview
Implement a classification model (e.g., Neural Network / CNN / Random Forest / SVM) using the MNIST dataset to recognize handwritten digits (0-9).

---

## 📁 Current Directory Structure & Status
```text
EXP_3_Handwritten_Digit_Recognition/
├── README.md                          # Experiment documentation
├── Dataset/                           # Data directory
│   ├── Training/train.csv             # Preprocessed Training split (29,400 samples, 155 columns)
│   ├── Validation/val.csv             # Preprocessed Validation split (4,200 samples, 155 columns)
│   └── Test/test.csv                  # Preprocessed Test split (8,400 samples, 155 columns)
└── scripts/                           # Scripts directory
    └── preprocess.py                  # Preprocessing wrapper script
```

---

## 📊 Dataset Information
- **Raw Input**: MNIST `train.csv` (42,000 records, 785 columns: `label` + 784 pixels).
- **Features**: `label` (Target digit 0-9), `pixel0` through `pixel783` (784 pixel features).
- **Preprocessed Splits (70% / 10% / 20%)**:
  - **Training set**: `Dataset/Training/train.csv` (29,400 records, 155 columns)
  - **Validation set**: `Dataset/Validation/val.csv` (4,200 records, 155 columns)
  - **Test set**: `Dataset/Test/test.csv` (8,400 records, 155 columns)

---

## 🛠 Preprocessing & Scripts Workflow
- **Preprocessing Pipeline (`scripts/preprocess.py`)**:
  - Calls `auto_preprocess` from `../../scripts/preprocess_template.py`.
  - **Data Normalization**: `MinMaxScaler` applied to scale pixel intensities [0, 255] -> [0, 1].
  - **Data Reduction**: `PCA` dimensionality reduction applied to 708 active pixel features, retaining 154 principal components (95.04% explained variance).
  - **Splitting**: Stratified 70/10/20 train/val/test export.
