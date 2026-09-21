"""Simple Linear Regression Model for Stock Price Prediction (From Scratch).

Formula: Y = beta_0 + beta_1 * x
  Where:
    Y = Dependent variable (Last Traded Price)
    x = Independent variable (Open price)
    beta_1 = Covariance(x, y) / Variance(x)  (Slope)
    beta_0 = y_mean - beta_1 * x_mean       (Intercept)

Usage:
  python Stock_Prediction_Modal.py
  python Stock_Prediction_Modal.py --predict 750 1800 4500 12000
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
    parser = argparse.ArgumentParser(description="Stock Price Prediction (Simple Linear Regression from Scratch)")
    parser.add_argument('--predict', nargs='+', type=float, default=None, help='List of custom opening prices (₹) to predict last traded prices for')
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parents[1]
    dataset_path = exp_dir / "Dataset" / "nifty_500.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print("=" * 60)
    print("      Stock Price Prediction - Simple Linear Regression")
    print("=" * 60)
    print(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Clean numeric inputs if needed
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['Last Traded Price'] = pd.to_numeric(df['Last Traded Price'], errors='coerce')

    # Drop any nulls if present
    df = df.dropna(subset=['Open', 'Last Traded Price']).reset_index(drop=True)

    # Feature (x) and Target (y)
    x = df['Open'].values
    y = df['Last Traded Price'].values

    print(f"Dataset shape: {df.shape}")
    print(f"Feature (x): 'Open' (min: {x.min():,.2f}, max: {x.max():,.2f}, mean: {x.mean():,.2f})")
    print(f"Target  (Y): 'Last Traded Price' (min: {y.min():,.2f}, max: {y.max():,.2f}, mean: {y.mean():,.2f})")

    # Fit Simple Linear Regression Model from Scratch
    model = SimpleLinearRegressionScratch()
    model.fit(x, y)

    print("\n" + "-" * 50)
    print("Learned Model Equation:")
    print(f"  Y = beta_0 + beta_1 * x")
    print(f"  Last_Traded_Price = {model.beta_0:,.2f} + ({model.beta_1:,.4f} * Open_Price)")
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
        sample_open_prices = args.predict
    else:
        print("Sample Predictions (Pass --predict <open1> <open2> ... for custom values):")
        sample_open_prices = [500, 1500, 3000, 10000, 25000]

    for open_price in sample_open_prices:
        predicted_ltp = model.predict(open_price)
        print(f"  Opening Price: ₹{open_price:>8,.2f}  -->  Predicted Last Traded Price: ₹{predicted_ltp:,.2f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
