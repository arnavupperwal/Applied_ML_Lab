"""Preprocessing script for Experiment 5: Sentiment Analysis."""
import sys
from pathlib import Path
import pandas as pd

# Add root scripts directory to Python path to import preprocess_template
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "scripts"))

from preprocess_template import auto_preprocess, split_save


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    input_csv = exp_dir / "Dataset" / "sentiment_analysis.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_csv}")

    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Run Auto-Preprocessing Pipeline:
    # 1. Cleaning: auto-detects and removes 105 duplicate rows
    # 2. Wrangling: Text cleaning + TF-IDF feature extraction on 'text' column, One-Hot encoding 'Platform', LabelEncoding 'sentiment' target
    # 3. Normalization: StandardScaler on numerical and TF-IDF features
    df_processed = auto_preprocess(
        df,
        target_col='sentiment',
        text_cols=['text'],
        max_tfidf_features=50,
        normalize_method='standard',
        encode_categorical=True
    )

    # Save to standard experiment subfolders
    training_prefix = exp_dir / "Dataset" / "sentiment"
    
    split_save(
        df_processed,
        target_col='sentiment',
        out_prefix=str(training_prefix),
        test_size=0.2,
        val_size=0.1
    )

    # Re-organize / move outputs into Training/, Validation/, Test/ folders
    dataset_dir = exp_dir / "Dataset"
    (dataset_dir / "Training").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Validation").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Test").mkdir(parents=True, exist_ok=True)

    (dataset_dir / "sentiment_train.csv").replace(dataset_dir / "Training" / "train.csv")
    (dataset_dir / "sentiment_val.csv").replace(dataset_dir / "Validation" / "val.csv")
    (dataset_dir / "sentiment_test.csv").replace(dataset_dir / "Test" / "test.csv")

    print("\nSuccessfully updated EXP_5 split datasets:")
    print(f" - Training:   {dataset_dir / 'Training' / 'train.csv'}")
    print(f" - Validation: {dataset_dir / 'Validation' / 'val.csv'}")
    print(f" - Test:       {dataset_dir / 'Test' / 'test.csv'}")


if __name__ == '__main__':
    main()
