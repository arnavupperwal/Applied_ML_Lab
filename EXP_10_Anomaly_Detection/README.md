# Exp 10: Anomaly Detection

## 🎯 Objective & Overview
Build an anomaly detection model (e.g., Isolation Forest, One-Class SVM, LOF) to identify outliers and unusual patterns in subscription or transaction data.

---

## 📁 Current Directory Structure & Status
```text
EXP_10_Anomaly_Detection/
├── README.md                          # Experiment documentation
├── Dataset/                           # Data directory
│   ├── Training/                      # Training dataset split directory
│   ├── Validation/                    # Validation dataset split directory
│   └── Test/                          # Test dataset split directory
└── scripts/                           # Scripts directory
    └── preprocess.py                  # Preprocessing wrapper script
```

---

## 📊 Dataset & Workflow Information
- **Status**: Workspace initialized with required directory layout.
- **Target Task**: Outlier & Anomaly Detection (Unsupervised / Semi-supervised).
- **Scripts**: `scripts/preprocess.py` points to `../../scripts/preprocess_template.py` for feature scaling, outlier filtering, and dataset splitting.
