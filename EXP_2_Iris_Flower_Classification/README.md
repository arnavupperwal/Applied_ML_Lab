# Exp 2: Iris Flower Classification

## 🎯 Objective & Overview
Build a classification model predicting species of iris flowers.

---

## 📁 Directory Structure
```text
EXP_2_Iris_Flower_Classification/
├── README.md                  # Experiment documentation
├── Dataset/                   # Experiment datasets split by set
│   ├── Training/              # Training dataset split
│   ├── Validation/            # Validation dataset split
│   └── Test/                  # Test dataset split
└── scripts/                   # Reusable preprocessing & model scripts
    └── preprocess.py          # Data preprocessing pipeline script
```

---

## 📋 Task & Preprocessing Workflow (Whiteboard Guidelines)

### 1. Dataset Preparation & Readme Update
- Download / place the raw dataset into the `Dataset/` directory.
- Document dataset schema, source, and column descriptions in this file (`README.md`).

### 2. Preprocessing Steps
Perform required preprocessing steps on raw data:
- **Data Sampling**: Filter, sample, or balance dataset if needed.
- **Cleaning & Wrangling**: Handle missing values, outliers, duplicate records, and invalid data types.
- **Normalization / Scaling**: Apply StandardScaler, MinMaxScaler, or text vectorization.
- **Dimensionality Reduction**: Apply feature selection, PCA, or relevant feature extraction.

### 3. Train / Validation / Test Splitting
- Partition dataset into **Training**, **Validation**, and **Test** sets.
- Save output split datasets in `Dataset/Training/`, `Dataset/Validation/`, and `Dataset/Test/`.
- Document final split ratios (e.g., 80% train, 10% validation, 10% test).

---

## 💻 Coding Task
- Implement preprocessing logic in `scripts/preprocess.py`.
- Develop training, evaluation, and visualization models in standard notebooks or Python scripts within this directory.
