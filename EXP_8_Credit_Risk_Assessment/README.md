# Exp 8: Credit Risk Assessment

## 🎯 Objective & Overview
Build a credit scoring classification model to assess credit risk and predict default probability using historical financial applicant data.

---

## 📁 Current Directory Structure & Status
```text
EXP_8_Credit_Risk_Assessment/
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
- **Status**: Workspace initialized with required directory layout. Credit scoring dataset (`bank_credit_scoring.csv` available in `archive (3).`) is ready to be placed in `Dataset/`.
- **Target Task**: Binary Classification & Risk Scoring (Default vs. Non-Default).
- **Scripts**: `scripts/preprocess.py` points to `../../scripts/preprocess_template.py` for handling missing values, encoding categorical variables, scaling numerical features, and train/val/test splitting.
