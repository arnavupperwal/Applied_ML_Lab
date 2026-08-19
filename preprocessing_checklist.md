# Preprocessing Checklist

Use this checklist when preparing datasets for experiments.

- [ ] Data sampling: confirm sampling strategy and random seed
- [ ] Inspect head, dtypes, and basic statistics (`.describe()`)
- [ ] Missing values: locate and decide (drop / impute). Document method.
- [ ] Outliers: detect and handle (winsorize / remove / transform)
- [ ] Data types: convert to appropriate dtypes (categorical, numeric, datetime)
- [ ] Encoding: One-hot / Ordinal / Target encoding as appropriate
- [ ] Scaling / Normalization: StandardScaler / MinMax / RobustScaler
- [ ] Feature engineering: create useful features and document rationale
- [ ] Dimensionality reduction: PCA / feature selection (if needed)
- [ ] Class imbalance handling: resampling, class weights, or focal loss
- [ ] Split into train/val/test with stratification where needed
- [ ] Save preprocessed dataset to `data/` with a clear filename
- [ ] Record preprocessing steps in the experiment README or notebook

Recommended file locations:
- Raw: `Applied_ML_Lab/data/raw/`
- Processed: `Applied_ML_Lab/data/processed/`

