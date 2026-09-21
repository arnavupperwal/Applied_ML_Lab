"""Preprocessing script for Experiment 1: Predicting Housing Prices."""
import sys
from pathlib import Path
import pandas as pd

# Add root scripts directory to Python path to import preprocess_template
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "scripts"))

from preprocess_template import auto_preprocess, split_save


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    input_csv = exp_dir / "Dataset" / "Housing.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_csv}")

    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Run Auto-Preprocessing Pipeline:
    # 1. Cleaning: inspects duplicates/nulls
    # 2. Wrangling: One-Hot Encodes categorical features ('mainroad', 'guestroom', etc.)
    # 3. Normalization: StandardScaler for continuous numeric features ('area', 'bedrooms', etc.)
    df_processed = auto_preprocess(
        df,
        target_col='price',
        normalize_method='standard',
        encode_categorical=True
    )

    # Save to standard experiment subfolders
    training_prefix = exp_dir / "Dataset" / "housing"
    
    # Perform stratified/balanced split into Train (70%), Val (10%), Test (20%)
    split_save(
        df_processed,
        target_col='price',
        out_prefix=str(training_prefix),
        test_size=0.2,
        val_size=0.1
    )

    # Re-organize / move outputs into Training/, Validation/, Test/ folders
    dataset_dir = exp_dir / "Dataset"
    (dataset_dir / "Training").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Validation").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Test").mkdir(parents=True, exist_ok=True)

    (dataset_dir / "housing_train.csv").replace(dataset_dir / "Training" / "train.csv")
    (dataset_dir / "housing_val.csv").replace(dataset_dir / "Validation" / "val.csv")
    (dataset_dir / "housing_test.csv").replace(dataset_dir / "Test" / "test.csv")

    print("\nSuccessfully updated EXP_1 split datasets:")
    print(f" - Training:   {dataset_dir / 'Training' / 'train.csv'}")
    print(f" - Validation: {dataset_dir / 'Validation' / 'val.csv'}")
    print(f" - Test:       {dataset_dir / 'Test' / 'test.csv'}")


if __name__ == '__main__':
    main()
