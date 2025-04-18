import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Load data
dtype_spec = {
    'carpark_id': 'string',
    'area': 'category',
    'development': 'category',
    'location': 'string',
    'available_lots': 'int',
    'lot_type': 'category',
    'agency': 'category',
    'source': 'category',
    'update_datetime': 'string',
}

# load carparkavail data
df = pd.read_csv(r"C:\Users\user\Documents\Capstone\data\raw_carpark_avail_020325_290325.csv",
                            dtype=dtype_spec, parse_dates=['timestamp'])

# Sort for proper time alignment
carpark_avail.sort_values(["carpark_id", "timestamp"], inplace=True)

# Assume hourly frequency → 7 days * 24 = 168
LAG_HOURS = 168

df["y_true"] = df["available_lots"]
df["y_pred"] = df.groupby("carpark_id")["available_lots"].shift(LAG_HOURS)

# Drop any rows where prediction or target is NaN
df = df.dropna(subset=["y_true", "y_pred"])

# Global 80:20 time-based split
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Evaluate on test set
rmse = np.sqrt(mean_squared_error(test_df["y_true"], test_df["y_pred"]))
r2 = r2_score(test_df["y_true"], test_df["y_pred"])

print(f"Baseline RMSE (test set): {rmse:.2f}")
print(f"Baseline R² (test set): {r2:.4f}")