# Exp 7: Predicting Stock Prices

## 🎯 Objective & Overview
Develop a time series prediction and regression model for stock price forecasting using historical stock market data.

---

## 📁 Current Directory Structure & Status
```text
EXP_7_Predicting_Stock_Prices/
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
- **Status**: Workspace initialized with required directory layout. Historical stock dataset (`nifty_500.csv` available in `archive (2).`) is ready to be placed in `Dataset/`.
- **Target Task**: Financial Time Series Forecasting / Regression.
- **Scripts**: `scripts/preprocess.py` references `../../scripts/preprocess_template.py` for date indexing, feature scaling, window sequence creation, and train/val/test splitting.
