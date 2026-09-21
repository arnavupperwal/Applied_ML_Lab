"""Simple Linear Regression Model for House Price Prediction (From Scratch).

Formula: Y = beta_0 + beta_1 * x
  Where:
    Y = Dependent variable (price)
    x = Independent variable (area)
    beta_1 = Covariance(x, y) / Variance(x)  (Slope)
    beta_0 = y_mean - beta_1 * x_mean       (Intercept)

Usage:
  python House_Prediction_Modal.py
  python House_Prediction_Modal.py --predict 2500 4200 6800 12000
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


class SimpleLinearRegressionScratch:
    def __init__(self):
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Slope

    def fit(self, x, y):
        """Fit Simple Linear Regression using closed-form Ordinary Least Squares equations."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("Variance of x is zero. Cannot compute slope beta_1.")

        self.beta_1 = numerator / denominator
        self.beta_0 = y_mean - self.beta_1 * x_mean
        return self

    def predict(self, x):
        """Predict target variable Y = beta_0 + beta_1 * x."""
        x = np.asarray(x, dtype=np.float64)
        if self.beta_0 is None or self.beta_1 is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.beta_0 + self.beta_1 * x


def evaluate_metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAE, and R^2 Score from scratch."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_true - y_pred

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum(errors ** 2)

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2_Score": r2
    }


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction (Simple Linear Regression from Scratch)")
    parser.add_argument('--predict', nargs='+', type=float, default=None, help='List of custom area values (sq ft) to predict prices for')
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parents[1]
    dataset_path = exp_dir / "Dataset" / "Housing.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print("=" * 60)
    print("      House Price Prediction - Simple Linear Regression")
    print("=" * 60)
    print(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Feature (x) and Target (y)
    x = df['area'].values
    y = df['price'].values

    print(f"Dataset shape: {df.shape}")
    print(f"Feature (x): 'area' (min: {x.min()}, max: {x.max()}, mean: {x.mean():.2f})")
    print(f"Target  (Y): 'price' (min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f})")

    # Fit Simple Linear Regression Model from Scratch
    model = SimpleLinearRegressionScratch()
    model.fit(x, y)

    print("\n" + "-" * 50)
    print("Learned Model Equation:")
    print(f"  Y = beta_0 + beta_1 * x")
    print(f"  price = {model.beta_0:,.2f} + ({model.beta_1:,.2f} * area)")
    print("-" * 50)
    print(f"  Intercept (beta_0): {model.beta_0:,.4f}")
    print(f"  Slope     (beta_1): {model.beta_1:,.4f}")
    print("-" * 50)

    # Predict on full dataset
    y_pred = model.predict(x)

    # Calculate Evaluation Metrics from Scratch
    metrics = evaluate_metrics(y, y_pred)

    print("\nModel Evaluation Metrics (Calculated from Scratch):")
    print(f"  R^2 Score:                 {metrics['R2_Score']:.4f}")
    print(f"  Mean Squared Error (MSE):  {metrics['MSE']:,.2f}")
    print(f"  Root Mean Sq Error (RMSE): {metrics['RMSE']:,.2f}")
    print(f"  Mean Absolute Error (MAE): {metrics['MAE']:,.2f}")

    # Custom or Default Predictions
    print("\n" + "=" * 60)
    if args.predict:
        print("Custom User Input Predictions:")
        sample_areas = args.predict
    else:
        print("Sample Predictions (Pass --predict <area1> <area2> ... for custom values):")
        sample_areas = [2000, 3500, 5000, 7500, 10000]

    for area in sample_areas:
        predicted_price = model.predict(area)
        print(f"  Area: {area:>8,.2f} sq ft  -->  Predicted Price: ${predicted_price:,.2f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
