"""Preprocessing utilities template.

This file provides a small set of helper functions and a CLI stub that labs
can import or copy into `scripts/preprocess.py` in each lab folder.

Usage (example):
python preprocess_template.py --input ../data/raw/mydata.csv --output ../data/processed/mydata_processed.csv
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_csv(path, nrows=None):
    return pd.read_csv(path, nrows=nrows)


def report_basic(df):
    print("Shape:", df.shape)
    print(df.dtypes)
    print(df.head())
    print(df.describe(include='all'))


def impute_numeric(df, strategy='median'):
    num_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy=strategy)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df


def scale_numeric(df):
    num_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def split_save(df, target_col=None, out_prefix='processed', test_size=0.2, val_size=0.1, seed=42):
    # If target_col provided, stratify by it when possible
    stratify = df[target_col] if target_col in df.columns else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    val_relative = val_size / (1 - test_size)
    stratify_tv = train_val[target_col] if (target_col in df.columns) else None
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed, stratify=stratify_tv)
    train.to_csv(f"{out_prefix}_train.csv", index=False)
    val.to_csv(f"{out_prefix}_val.csv", index=False)
    test.to_csv(f"{out_prefix}_test.csv", index=False)
    print(f"Saved: {out_prefix}_train.csv, {out_prefix}_val.csv, {out_prefix}_test.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output-prefix', default='processed', help='Output filename prefix')
    parser.add_argument('--target', default=None, help='Name of target column')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    args = parser.parse_args()

    df = load_csv(args.input)
    report_basic(df)
    df = impute_numeric(df)
    df = scale_numeric(df)
    split_save(df, target_col=args.target, out_prefix=args.output_prefix, test_size=args.test_size, val_size=args.val_size)


if __name__ == '__main__':
    main()
