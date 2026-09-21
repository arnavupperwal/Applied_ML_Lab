# Exp 12: Fake News Detection

## 🎯 Objective & Overview
Implement an NLP classification model (e.g., PassiveAggressive, Naive Bayes, Logistic Regression) to classify news articles as authentic or fake news.

---

## 📁 Current Directory Structure & Status
```text
EXP_12_Fake_News_Detection/
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
- **Target Task**: Binary NLP Text Classification (Real vs. Fake News).
- **Scripts**: `scripts/preprocess.py` references `../../scripts/preprocess_template.py` for text cleaning, TF-IDF vectorization, and dataset splitting.
