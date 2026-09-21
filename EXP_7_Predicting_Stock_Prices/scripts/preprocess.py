"""Preprocessing script for Experiment 7: Stock Price Prediction."""
import sys
from pathlib import Path
import pandas as pd

# Add root scripts directory to Python path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "scripts"))

from preprocess_template import auto_preprocess, split_save


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    input_csv = exp_dir / "Dataset" / "nifty_500.csv"

    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Auto-preprocessing pipeline:
    # 1. Cleaning: drops useless/duplicate rows and nulls
    # 2. Wrangling: One-Hot Encodes categorical features ('Industry', 'Series')
    # 3. Normalization: StandardScaler on financial metrics ('Open', 'High', 'Low', 'Volume')
    df_processed = auto_preprocess(
        df,
        target_col='Last Traded Price',
        normalize_method='standard',
        encode_categorical=True
    )

    # Save to Dataset/ directory
    training_prefix = exp_dir / "Dataset" / "nifty"
    
    split_save(
        df_processed,
        target_col='Last Traded Price',
        out_prefix=str(training_prefix),
        test_size=0.2,
        val_size=0.1
    )

    # Move split CSVs into Training/, Validation/, and Test/ folders
    dataset_dir = exp_dir / "Dataset"
    (dataset_dir / "Training").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Validation").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Test").mkdir(parents=True, exist_ok=True)

    (dataset_dir / "nifty_train.csv").replace(dataset_dir / "Training" / "train.csv")
    (dataset_dir / "nifty_val.csv").replace(dataset_dir / "Validation" / "val.csv")
    (dataset_dir / "nifty_test.csv").replace(dataset_dir / "Test" / "test.csv")

    print("\nPreprocessed dataset successfully saved:")
    print(" - Training:   Dataset/Training/train.csv")
    print(" - Validation: Dataset/Validation/val.csv")
    print(" - Test:       Dataset/Test/test.csv")


if __name__ == '__main__':
    main()
