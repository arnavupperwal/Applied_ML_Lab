"""Preprocessing script for Experiment 3: Handwritten Digit Recognition."""
import sys
from pathlib import Path
import pandas as pd
import shutil

# Add root scripts directory to Python path to import preprocess_template
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "scripts"))

from preprocess_template import auto_preprocess, split_save


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = Path("/home/bleh/Desktop/Handwritten Digit Recognition.")
    input_csv = raw_data_dir / "train.csv"

    if not input_csv.exists():
        # Fallback check inside experiment Dataset folder
        input_csv = exp_dir / "Dataset" / "train.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_csv}")

    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Run Auto-Preprocessing Pipeline:
    # 1. Normalization: MinMaxScaler to scale 784 pixel intensities [0, 255] -> [0, 1]
    # 2. Data Reduction: PCA retaining 95% explained variance across 784 pixel dimensions
    df_processed = auto_preprocess(
        df,
        target_col='label',
        normalize_method='minmax',
        reduce_method='pca',
        pca_components=0.95,
        encode_categorical=False
    )

    # Save to standard experiment subfolders
    training_prefix = exp_dir / "Dataset" / "mnist"
    
    split_save(
        df_processed,
        target_col='label',
        out_prefix=str(training_prefix),
        test_size=0.2,
        val_size=0.1
    )

    # Re-organize / move outputs into Training/, Validation/, Test/ folders
    dataset_dir = exp_dir / "Dataset"
    (dataset_dir / "Training").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Validation").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Test").mkdir(parents=True, exist_ok=True)

    (dataset_dir / "mnist_train.csv").replace(dataset_dir / "Training" / "train.csv")
    (dataset_dir / "mnist_val.csv").replace(dataset_dir / "Validation" / "val.csv")
    (dataset_dir / "mnist_test.csv").replace(dataset_dir / "Test" / "test.csv")

    print("\nSuccessfully updated EXP_3 split datasets:")
    print(f" - Training:   {dataset_dir / 'Training' / 'train.csv'}")
    print(f" - Validation: {dataset_dir / 'Validation' / 'val.csv'}")
    print(f" - Test:       {dataset_dir / 'Test' / 'test.csv'}")


if __name__ == '__main__':
    main()
