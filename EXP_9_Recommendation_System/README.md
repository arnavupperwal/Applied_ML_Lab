# Exp 9: Recommendation System

## 🎯 Objective & Overview
Build a recommendation engine (collaborative filtering or content-based filtering) to suggest items (e.g., movies/books) to users based on historical interaction data.

---

## 📁 Current Directory Structure & Status
```text
EXP_9_Recommendation_System/
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
- **Target Task**: Collaborative Filtering / Content-Based Filtering / Matrix Factorization.
- **Scripts**: `scripts/preprocess.py` references `../../scripts/preprocess_template.py` for user-item matrix creation, rating normalization, and dataset partitioning.
