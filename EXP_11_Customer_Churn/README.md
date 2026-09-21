# Exp 11: Customer Churn

## 🎯 Objective & Overview
Build a binary classification model to predict customer churn in a subscription business based on usage patterns, contract details, and account metadata.

---

## 📁 Current Directory Structure & Status
```text
EXP_11_Customer_Churn/
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
- **Target Task**: Binary Classification (Churn vs. Retained).
- **Scripts**: `scripts/preprocess.py` uses `../../scripts/preprocess_template.py` for categorical encoding, feature scaling, imbalance handling, and dataset splitting.
