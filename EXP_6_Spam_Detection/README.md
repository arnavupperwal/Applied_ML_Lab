# Exp 6: Spam Detection

## 🎯 Objective & Overview
Develop a text classification model using NLP techniques to classify messages/emails as Spam or Ham (legitimate).

---

## 📁 Current Directory Structure & Status
```text
EXP_6_Spam_Detection/
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
- **Status**: Workspace initialized with required directory layout. Raw spam dataset (`spam.csv` available in `archive (1).`) is ready to be placed in `Dataset/`.
- **Target Task**: Binary NLP Text Classification (Spam vs. Ham).
- **Scripts**: `scripts/preprocess.py` points to `../../scripts/preprocess_template.py` for text tokenization, TF-IDF vectorization, and train/val/test splitting.
