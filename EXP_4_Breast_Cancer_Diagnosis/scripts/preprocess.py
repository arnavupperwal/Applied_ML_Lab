"""Preprocessing script for Experiment 4: Breast Cancer Diagnosis."""
import sys
from pathlib import Path
import pandas as pd

# Add root scripts directory to Python path to import preprocess_template
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "scripts"))

from preprocess_template import auto_preprocess, split_save


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    input_csv = exp_dir / "Dataset" / "data.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_csv}")

    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Run Auto-Preprocessing Pipeline:
    # 1. Cleaning: drops useless 'id' and 100% null trailing column 'Unnamed: 32'
    # 2. Wrangling: Label encodes target 'diagnosis' ('M' -> 1, 'B' -> 0)
    # 3. Normalization: StandardScaler on 30 feature measurements
    # 4. Reduction: PCA retaining 95% explained variance across multicollinear features (>20 features)
    df_processed = auto_preprocess(
        df,
        target_col='diagnosis',
        drop_useless=True,
        normalize_method='standard',
        reduce_method='pca',
        pca_components=0.95,
        encode_categorical=True
    )

    # Save to standard experiment subfolders
    training_prefix = exp_dir / "Dataset" / "breast_cancer"
    
    split_save(
        df_processed,
        target_col='diagnosis',
        out_prefix=str(training_prefix),
        test_size=0.2,
        val_size=0.1
    )

    # Re-organize / move outputs into Training/, Validation/, Test/ folders
    dataset_dir = exp_dir / "Dataset"
    (dataset_dir / "Training").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Validation").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Test").mkdir(parents=True, exist_ok=True)

    (dataset_dir / "breast_cancer_train.csv").replace(dataset_dir / "Training" / "train.csv")
    (dataset_dir / "breast_cancer_val.csv").replace(dataset_dir / "Validation" / "val.csv")
    (dataset_dir / "breast_cancer_test.csv").replace(dataset_dir / "Test" / "test.csv")

    print("\nSuccessfully updated EXP_4 split datasets:")
    print(f" - Training:   {dataset_dir / 'Training' / 'train.csv'}")
    print(f" - Validation: {dataset_dir / 'Validation' / 'val.csv'}")
    print(f" - Test:       {dataset_dir / 'Test' / 'test.csv'}")


if __name__ == '__main__':
    main()
