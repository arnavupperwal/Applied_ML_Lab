"""Preprocessing utilities template with intelligent auto-detection.

Supports 5 key data preprocessing techniques:
1. Data Sampling (fractional sampling, class balancing / stratified sampling)
2. Data Cleaning (duplicate removal, missing value imputation, empty/ID column pruning, outlier handling)
3. Data Wrangling (categorical encoding, text normalization & TF-IDF extraction, datetime wrangling)
4. Data Normalization (StandardScaler, MinMaxScaler, RobustScaler)
5. Data Reduction (PCA dimensionality reduction, variance thresholding)

Conditional execution ensures ONLY techniques required for a given dataset are performed.

Usage (CLI example):
python preprocess_template.py --input raw_data.csv --target label --output-prefix processed_data
"""
import argparse
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold


# Helper to identify non-numeric feature/target columns reliably across pandas versions
def is_non_numeric(series):
    return not pd.api.types.is_numeric_dtype(series)


def get_numeric_cols(df):
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def get_non_numeric_cols(df):
    return [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]


# ==========================================
# 1. DATA SAMPLING
# ==========================================
def sample_data(df, target_col=None, sample_frac=None, sample_n=None, balance_classes=False, seed=42):
    """Perform random or stratified row sampling / class balancing if required."""
    executed = False
    original_shape = df.shape
    
    # Subsampling by fraction or fixed n
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        executed = True
        print(f"[Sampling] Subsampled fraction {sample_frac}: {original_shape} -> {df.shape}")
    elif sample_n is not None and 0 < sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        executed = True
        print(f"[Sampling] Subsampled n={sample_n}: {original_shape} -> {df.shape}")
        
    # Class balancing (downsample majority classes to match smallest class size)
    if balance_classes and target_col and target_col in df.columns:
        counts = df[target_col].value_counts()
        if len(counts) > 1 and counts.min() < counts.max():
            min_count = counts.min()
            sampled_dfs = [group.sample(n=min_count, random_state=seed) 
                           for _, group in df.groupby(target_col)]
            df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            executed = True
            print(f"[Sampling] Balanced classes on '{target_col}' (n={min_count} per class): {original_shape} -> {df.shape}")

    if not executed:
        print("[Sampling] Skipped (No sampling requested or required).")
    return df


# ==========================================
# 2. DATA CLEANING
# ==========================================
def clean_data(df, target_col=None, drop_duplicates=True, drop_useless=True, impute=True, 
               impute_numeric_strategy='median', impute_categorical_strategy='most_frequent', 
               id_cols=None):
    """Clean dataset: remove duplicates, drop empty/useless/ID columns, impute missing values."""
    df = df.copy()
    actions = []

    # A. Remove duplicate rows
    if drop_duplicates:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            actions.append(f"Removed {dup_count} duplicate rows")

    # B. Drop useless columns (100% null, constant values, or specified ID columns)
    if drop_useless:
        cols_to_drop = []
        # Columns with 100% missing values
        null_cols = [col for col in df.columns if df[col].isnull().all()]
        cols_to_drop.extend(null_cols)
        
        # Columns with 1 unique constant value
        const_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1 and col not in null_cols]
        cols_to_drop.extend(const_cols)

        # Common ID column auto-detection
        if id_cols is None:
            id_cols = [col for col in df.columns if col.lower() in ['id', 'patient_id', 'user_id', 'unnamed: 0'] and col != target_col]
        cols_to_drop.extend([col for col in id_cols if col in df.columns and col not in cols_to_drop])

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            actions.append(f"Dropped useless/ID columns: {cols_to_drop}")

    # C. Impute missing values
    if impute and df.isnull().sum().sum() > 0:
        # Numeric columns
        num_cols = get_numeric_cols(df)
        if target_col and target_col in num_cols:
            num_cols = [c for c in num_cols if c != target_col]
            
        num_nulls = df[num_cols].isnull().sum().sum() if num_cols else 0
        if num_nulls > 0 and len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy=impute_numeric_strategy)
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
            actions.append(f"Imputed missing numeric values ({num_nulls} cells, strategy='{impute_numeric_strategy}')")

        # Categorical / string columns
        cat_cols = get_non_numeric_cols(df)
        if target_col and target_col in cat_cols:
            cat_cols = [c for c in cat_cols if c != target_col]
            
        cat_nulls = df[cat_cols].isnull().sum().sum() if cat_cols else 0
        if cat_nulls > 0 and len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy=impute_categorical_strategy)
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            actions.append(f"Imputed missing categorical values ({cat_nulls} cells, strategy='{impute_categorical_strategy}')")

    if actions:
        print(f"[Cleaning] Performed: {'; '.join(actions)}")
    else:
        print("[Cleaning] Skipped (Dataset is already clean, no nulls/duplicates/useless columns found).")
    return df


# ==========================================
# 3. DATA WRANGLING
# ==========================================
def _clean_text_string(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def wrangle_data(df, target_col=None, encode_categorical=True, text_cols=None, 
                 max_tfidf_features=100, seed=42):
    """Perform data wrangling: encode target, clean text & extract TF-IDF features, encode categorical columns."""
    df = df.copy()
    actions = []

    # A. Encode Target Column if string/categorical
    if target_col and target_col in df.columns:
        if is_non_numeric(df[target_col]):
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            actions.append(f"Encoded target '{target_col}' using LabelEncoder (classes: {list(le.classes_)})")

    # B. Text Wrangling & TF-IDF Vectorization
    if text_cols:
        for tcol in text_cols:
            if tcol in df.columns:
                cleaned_series = df[tcol].astype(str).apply(_clean_text_string)
                tfidf = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english')
                tfidf_mat = tfidf.fit_transform(cleaned_series).toarray()
                feature_names = [f"{tcol}_tfidf_{name}" for name in tfidf.get_feature_names_out()]
                tfidf_df = pd.DataFrame(tfidf_mat, columns=feature_names, index=df.index)
                
                df = df.drop(columns=[tcol])
                df = pd.concat([df, tfidf_df], axis=1)
                actions.append(f"Extracted {len(feature_names)} TF-IDF features from text column '{tcol}'")

    # C. Categorical Feature Encoding (One-Hot Encoding for non-target object/category columns)
    if encode_categorical:
        cat_cols = get_non_numeric_cols(df)
        if target_col and target_col in cat_cols:
            cat_cols = [c for c in cat_cols if c != target_col]

        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            actions.append(f"One-hot encoded categorical columns: {list(cat_cols)}")

    if actions:
        print(f"[Wrangling] Performed: {'; '.join(actions)}")
    else:
        print("[Wrangling] Skipped (No categorical encoding or text wrangling required).")
    return df


# ==========================================
# 4. DATA NORMALIZATION & SCALING
# ==========================================
def normalize_data(df, target_col=None, method='standard', cols_to_scale=None):
    """Normalize / scale numerical feature columns using StandardScaler, MinMaxScaler, or RobustScaler."""
    if method is None or str(method).lower() == 'none':
        print("[Normalization] Skipped (method=None).")
        return df

    df = df.copy()
    method = method.lower()

    if cols_to_scale is None:
        num_cols = get_numeric_cols(df)
        if target_col and target_col in num_cols:
            num_cols.remove(target_col)
        cols_to_scale = num_cols

    if len(cols_to_scale) == 0:
        print("[Normalization] Skipped (No numerical feature columns to scale).")
        return df

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"[Normalization] Scaled {len(cols_to_scale)} numerical features using {scaler.__class__.__name__}.")
    return df


# ==========================================
# 5. DATA REDUCTION
# ==========================================
def reduce_data(df, target_col=None, method='pca', n_components=None, variance_threshold=None, seed=42):
    """Perform feature reduction via low-variance thresholding or PCA dimensionality reduction."""
    df = df.copy()
    actions = []

    num_cols = get_numeric_cols(df)
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    if len(num_cols) == 0:
        print("[Reduction] Skipped (No feature columns available for reduction).")
        return df

    # A. Low-variance threshold filtering
    if variance_threshold is not None and variance_threshold > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        reduced_mat = selector.fit_transform(df[num_cols])
        kept_indices = selector.get_support(indices=True)
        kept_cols = [num_cols[i] for i in kept_indices]
        dropped_cols = list(set(num_cols) - set(kept_cols))

        if dropped_cols:
            df = df.drop(columns=dropped_cols)
            num_cols = kept_cols
            actions.append(f"VarianceThreshold ({variance_threshold}) dropped {len(dropped_cols)} low-variance columns")

    # B. PCA Dimensionality Reduction
    if method == 'pca' and n_components is not None:
        original_n_features = len(num_cols)
        pca = PCA(n_components=n_components, random_state=seed)
        pca_mat = pca.fit_transform(df[num_cols])
        
        n_kept = pca_mat.shape[1]
        pca_cols = [f"pca_{i}" for i in range(n_kept)]
        pca_df = pd.DataFrame(pca_mat, columns=pca_cols, index=df.index)

        df = df.drop(columns=num_cols)
        
        # Place target column at the end if present
        if target_col and target_col in df.columns:
            target_series = df.pop(target_col)
            df = pd.concat([pca_df, df, target_series], axis=1)
        else:
            df = pd.concat([pca_df, df], axis=1)

        exp_var = sum(pca.explained_variance_ratio_) * 100
        actions.append(f"PCA reduced {original_n_features} features -> {n_kept} principal components ({exp_var:.2f}% variance retained)")

    if actions:
        print(f"[Reduction] Performed: {'; '.join(actions)}")
    else:
        print("[Reduction] Skipped (No dimensionality reduction requested or required).")
    return df


# ==========================================
# AUTO PREPROCESSING PIPELINE
# ==========================================
def auto_preprocess(df, target_col=None, sample_frac=None, sample_n=None, balance_classes=False,
                    drop_duplicates=True, drop_useless=True, impute=True,
                    encode_categorical=True, text_cols=None, max_tfidf_features=100,
                    normalize_method='standard', reduce_method=None, pca_components=None,
                    variance_threshold=None, seed=42):
    """Conditional Auto-Preprocessing execution. Evaluates dataset requirements and runs ONLY required steps."""
    print("=" * 60)
    print(f"Starting Preprocessing Pipeline (Initial Shape: {df.shape})")
    print("=" * 60)

    # 1. SAMPLING: Run if explicitly requested or class balancing needed
    if sample_frac is not None or sample_n is not None or balance_classes:
        df = sample_data(df, target_col=target_col, sample_frac=sample_frac, sample_n=sample_n, 
                         balance_classes=balance_classes, seed=seed)
    else:
        print("[Sampling] Skipped (Not requested).")

    # 2. CLEANING: Run if dataset has duplicates, nulls, or useless/id columns
    has_duplicates = df.duplicated().sum() > 0
    has_nulls = df.isnull().sum().sum() > 0
    has_useless = any(df[c].isnull().all() or df[c].nunique(dropna=True) <= 1 or c.lower() in ['id', 'patient_id', 'unnamed: 0'] for c in df.columns if c != target_col)

    if has_duplicates or has_nulls or has_useless or drop_duplicates or drop_useless or impute:
        df = clean_data(df, target_col=target_col, drop_duplicates=drop_duplicates, drop_useless=drop_useless, 
                        impute=impute)
    else:
        print("[Cleaning] Skipped (Dataset is clean).")

    # 3. WRANGLING: Run if non-numeric/string/categorical columns or text columns present
    has_cat = any(is_non_numeric(df[c]) for c in df.columns)
    if has_cat or text_cols:
        df = wrangle_data(df, target_col=target_col, encode_categorical=encode_categorical, 
                          text_cols=text_cols, max_tfidf_features=max_tfidf_features, seed=seed)
    else:
        print("[Wrangling] Skipped (No categorical or text columns found).")

    # 4. NORMALIZATION: Run if numeric features exist and scaling requested
    num_cols = get_numeric_cols(df)
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    
    if len(num_cols) > 0 and normalize_method:
        df = normalize_data(df, target_col=target_col, method=normalize_method, cols_to_scale=num_cols)
    else:
        print("[Normalization] Skipped (No numerical features to scale).")

    # 5. REDUCTION: Run if high feature count (>20) or explicitly requested
    if (len(num_cols) > 20 and reduce_method is not None) or pca_components is not None or variance_threshold is not None:
        df = reduce_data(df, target_col=target_col, method=reduce_method or 'pca', 
                         n_components=pca_components, variance_threshold=variance_threshold, seed=seed)
    else:
        print(f"[Reduction] Skipped (Feature count = {len(num_cols)} <= 20 and no reduction explicitly requested).")

    print("=" * 60)
    print(f"Preprocessing Complete (Final Shape: {df.shape})")
    print("=" * 60)
    return df


# ==========================================
# DATASET SPLITTING & SAVE
# ==========================================
def split_save(df, target_col=None, out_prefix='processed', test_size=0.2, val_size=0.1, seed=42):
    """Split dataset into Train (70%), Validation (10%), Test (20%) and save CSVs."""
    out_dir = Path(out_prefix).parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    stratify = df[target_col] if (target_col and target_col in df.columns and df[target_col].nunique() < 50) else None
    
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    val_relative = val_size / (1.0 - test_size)
    stratify_tv = train_val[target_col] if (target_col and target_col in train_val.columns and train_val[target_col].nunique() < 50) else None
    
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed, stratify=stratify_tv)

    train_path = f"{out_prefix}_train.csv"
    val_path = f"{out_prefix}_val.csv"
    test_path = f"{out_prefix}_test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Splits Saved Successfully:")
    print(f" - Train: {train_path} {train.shape}")
    print(f" - Val:   {val_path} {val.shape}")
    print(f" - Test:  {test_path} {test.shape}")


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing CLI Utility")
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output-prefix', default='processed', help='Output filename prefix')
    parser.add_argument('--target', default=None, help='Name of target column')
    parser.add_argument('--sample-frac', type=float, default=None, help='Fraction of data to sample (0.0-1.0)')
    parser.add_argument('--balance-classes', action='store_true', help='Balance target class distributions')
    parser.add_argument('--normalize', default='standard', choices=['standard', 'minmax', 'robust', 'none'], help='Normalization method')
    parser.add_argument('--text-cols', nargs='+', default=None, help='List of text feature column names')
    parser.add_argument('--pca-components', type=float, default=None, help='PCA components (int n or float variance e.g. 0.95)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    pca_comp = args.pca_components
    if pca_comp is not None and pca_comp > 1.0:
        pca_comp = int(pca_comp)

    norm_method = None if args.normalize == 'none' else args.normalize

    df_proc = auto_preprocess(
        df, 
        target_col=args.target, 
        sample_frac=args.sample_frac,
        balance_classes=args.balance_classes,
        text_cols=args.text_cols,
        normalize_method=norm_method,
        reduce_method='pca' if pca_comp is not None else None,
        pca_components=pca_comp
    )

    split_save(df_proc, target_col=args.target, out_prefix=args.output_prefix, test_size=args.test_size, val_size=args.val_size)


if __name__ == '__main__':
    main()
