# Comprehensive Data Preprocessing Guide: Using `preprocess_template.py`

This guide explains step-by-step how to download data from the internet, organize it into the proper folder structure, understand how `preprocess_template.py` works under the hood, and run the preprocessing pipeline yourself without requiring automated data processing assistance.

---

## Table of Contents
1. [Overview & Requirements](#1-overview--requirements)
2. [Step 1: Downloading Raw Data from the Internet](#step-1-downloading-raw-data-from-the-internet)
3. [Step 2: Directory Setup & File Placement](#step-2-directory-setup--file-placement)
4. [Step 3: How `preprocess_template.py` Works (Code Breakdown)](#step-3-how-preprocess_templatepy-works-code-breakdown)
5. [Step 4: Setting Up Your Python Environment](#step-4-setting-up-your-python-environment)
6. [Step 5: Step-by-Step Guide to Running Preprocessing Yourself](#step-5-step-by-step-guide-to-running-preprocessing-yourself)
7. [Step 6: Organizing Processed Datasets (Train / Val / Test)](#step-6-organizing-processed-datasets-train--val--test)
8. [Step 7: Customizing Preprocessing for Different Datasets](#step-7-customizing-preprocessing-for-different-datasets)
9. [Step 8: Pre-Training Verification Checklist](#step-8-pre-training-verification-checklist)

---

## 1. Overview & Requirements

Data preprocessing is a crucial phase in machine learning workflows. It transforms raw, dirty, or unstructured dataset files into cleaned, standardized, and split matrices (Train/Validation/Test) ready for model training.

In this repository (`Applied_ML_Lab`), the standardized script template is located at:
```text
Applied_ML_Lab/scripts/preprocess_template.py
```

### Pre-requisites
- **Python 3.8+**
- Packages: `pandas`, `scikit-learn`

---

## Step 1: Downloading Raw Data from the Internet

When working on a Machine Learning experiment (e.g., Kaggle dataset, UCI Repository, GitHub raw CSV), follow these steps to obtain your data:

### Method A: Direct Browser Download
1. Open your web browser and navigate to your dataset source (e.g., [Kaggle](https://www.kaggle.com), [UCI ML Repository](https://archive.ics.uci.edu), or GitHub).
2. Click **Download** to save the file (typically a `.csv` or `.zip` file) into your local `Downloads` folder.
3. If downloaded as a ZIP file, extract it:
   ```bash
   unzip raw_dataset.zip -d ~/Downloads/extracted_data/
   ```

### Method B: Terminal Download (`curl` / `wget`)
If you have a direct URL to a raw `.csv` file:
```bash
# Example using curl
curl -L -o my_dataset.csv "https://example.com/path/to/dataset.csv"

# Example using wget
wget -O my_dataset.csv "https://example.com/path/to/dataset.csv"
```

---

## Step 2: Directory Setup & File Placement

Place raw files in designated project folders so that scripts can reliably find them.

### Option 1: Global Lab Data Directory (Recommended for shared datasets)
Place raw files in `Applied_ML_Lab/data/raw/`:
```text
Applied_ML_Lab/
└── data/
    ├── raw/
    │   └── my_dataset.csv          <-- PLACE DOWNLOADED RAW CSV HERE
    └── processed/                  <-- SCRIPT OUTPUTS GO HERE
```

### Option 2: Experiment-Specific Subfolder Directory
Place raw files inside the specific experiment's `Dataset/` folder:
```text
Applied_ML_Lab/
└── EXP_1_Predicting_Housing_Prices/
    ├── Dataset/
    │   ├── Housing.csv             <-- PLACE DOWNLOADED RAW CSV HERE
    │   ├── Training/
    │   ├── Validation/
    │   └── Test/
    └── scripts/
        └── preprocess.py
```

---

## Step 3: How `preprocess_template.py` Works (Code Breakdown)

The `preprocess_template.py` script standardizes 5 major pipeline stages with conditional auto-detection:

```
[Raw CSV] ──► 1. Sample Data ──► 2. Clean Data ──► 3. Wrangle & Encode ──► 4. Scale / Normalize ──► 5. Reduce Dimensions ──► 6. Split & Save ──► [Train/Val/Test CSVs]
```

### 1. Data Sampling (`sample_data`)
- Supports fractional row sampling (`sample_frac`), fixed size sampling (`sample_n`), and class balancing / downsampling for imbalanced target classes.

### 2. Data Cleaning (`clean_data`)
- Detects and removes duplicate rows (`df.drop_duplicates()`).
- Auto-imputes missing values for numeric columns (median/mean) and categorical columns (most frequent).
- Drops 100% empty/null columns, constant columns, and useless identifier columns (e.g. `id`, `patient_id`, `Unnamed: 0`).

### 3. Data Wrangling & Feature Engineering (`wrangle_data`)
- **Target Label Encoding**: Encodes non-numeric target variables using `LabelEncoder`.
- **Text Wrangling**: Cleans raw text (lowercasing, url/punctuation removal) and extracts TF-IDF numerical vector features (`TfidfVectorizer`).
- **Categorical Feature Encoding**: Automatically converts categorical/string attributes to numeric columns using One-Hot Encoding (`pd.get_dummies`).

### 4. Data Scaling & Normalization (`normalize_data`)
- Applies feature scaling (`StandardScaler`, `MinMaxScaler`, or `RobustScaler`) to continuous numerical attributes while preserving target columns.

### 5. Data Reduction (`reduce_data`)
- **Variance Thresholding**: Drops low-variance feature columns near zero variance.
- **PCA Dimensionality Reduction**: Reduces high-dimensional feature spaces (e.g. 784 pixels in MNIST or 30 medical measurements in Breast Cancer) to principal components retaining desired variance (e.g. 95% explained variance).

### 6. Auto-Preprocessing Pipeline (`auto_preprocess`)
- Evaluates dataset characteristics dynamically and executes **ONLY** the techniques required for that specific dataset.

### 7. Dataset Splitting & Export (`split_save`)
- Performs a two-stage stratified split into **Train set** (70%), **Validation set** (10%), and **Test set** (20%).
- Saves 3 output CSV files into designated output folders.

---

## Step 4: Setting Up Your Python Environment

Open your terminal and make sure your Python environment is activated:

```bash
# 1. Navigate to the lab folder
cd ~/Desktop/Applied_ML_Lab

# 2. Activate virtual environment (if not already active)
source venv/bin/activate

# 3. Ensure required packages are installed
pip install -r requirements.txt
```

---

## Step 5: Step-by-Step Guide to Running Preprocessing Yourself

Here is how you execute preprocessing manually via the Command Line Interface (CLI):

### Example 1: Running for a General Dataset in `data/raw/`

```bash
python scripts/preprocess_template.py \
  --input data/raw/my_dataset.csv \
  --target target_class \
  --output-prefix data/processed/my_dataset \
  --test-size 0.2 \
  --val-size 0.1
```

### Example 2: Running for an Experiment (e.g., Housing Prices)

```bash
# Navigate to experiment directory or run from root:
python EXP_1_Predicting_Housing_Prices/scripts/preprocess.py \
  --input EXP_1_Predicting_Housing_Prices/Dataset/Housing.csv \
  --target price \
  --output-prefix EXP_1_Predicting_Housing_Prices/Dataset/housing
```

### Command Line Arguments Reference
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--input` | *(Required)* | Relative or absolute path to the raw input CSV file. |
| `--output-prefix` | `processed` | Prefix (path + filename base) for output CSV files. |
| `--target` | `None` | Name of target/label column (enables stratified split). |
| `--test-size` | `0.2` | Fraction of dataset reserved for testing (e.g., 0.2 = 20%). |
| `--val-size` | `0.1` | Fraction of dataset reserved for validation (e.g., 0.1 = 10%). |

---

## Step 6: Organizing Processed Datasets (Train / Val / Test)

After running `preprocess_template.py` or `preprocess.py`, 3 CSV files will be generated in your designated output directory:
- `my_dataset_train.csv`
- `my_dataset_val.csv`
- `my_dataset_test.csv`

Move these into the standard folder structure if required by your experiment:

```bash
# Example for EXP_1
mv EXP_1_Predicting_Housing_Prices/Dataset/housing_train.csv EXP_1_Predicting_Housing_Prices/Dataset/Training/
mv EXP_1_Predicting_Housing_Prices/Dataset/housing_val.csv EXP_1_Predicting_Housing_Prices/Dataset/Validation/
mv EXP_1_Predicting_Housing_Prices/Dataset/housing_test.csv EXP_1_Predicting_Housing_Prices/Dataset/Test/
```

The resulting experiment directory will look like:
```text
EXP_1_Predicting_Housing_Prices/
├── Dataset/
│   ├── Housing.csv (Raw)
│   ├── Training/
│   │   └── housing_train.csv
│   ├── Validation/
│   │   └── housing_val.csv
│   └── Test/
│       └── housing_test.csv
├── README.md
└── scripts/
    └── preprocess.py
```

---

## Step 7: Customizing Preprocessing for Different Datasets

`preprocess_template.py` handles general numerical data out-of-the-box. Depending on your dataset, you may need minor custom modifications:

### 1. Handling Categorical (Text/String) Features
If your dataset contains categorical text features (e.g., `Gender`, `City`, `Yes/No`), perform One-Hot Encoding before scaling:

```python
# Add this function to your preprocess.py:
def encode_categorical(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
```

### 2. Custom Imputation Strategies
- For skewed numerical distributions, use `strategy='median'`.
- For categorical columns, impute with the most frequent value:
  ```python
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(strategy='most_frequent')
  df[cat_cols] = imputer.fit_transform(df[cat_cols])
  ```

---

## Step 8: Pre-Training Verification Checklist

Before running model training, verify your preprocessed files:

- [ ] Raw CSV is placed in `data/raw/` or `EXP_X/Dataset/`.
- [ ] Virtual environment is active (`source venv/bin/activate`).
- [ ] Preprocessing command was executed without errors.
- [ ] Output CSV files exist (`_train.csv`, `_val.csv`, `_test.csv`).
- [ ] No remaining null/missing values in processed files.
- [ ] Feature columns are appropriately scaled and encoded.
- [ ] Splitting preserves target class proportions (if target was specified).

Now you can build, train, and evaluate your ML models using your processed dataset splits!
