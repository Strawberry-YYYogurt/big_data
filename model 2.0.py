import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# -----------------------------
# 1) Data Loading and Merging
# -----------------------------

# Specify the data types
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

# Load carpark availability data
carpark_avail = pd.read_csv(
    r"C:\Users\user\Documents\Semester 3\raw_carpark_avail_020325_290325.csv",
    dtype=dtype_spec,
    parse_dates=['timestamp']
)

# Filter only 'C' lot_type
carpark_avail = carpark_avail[carpark_avail['lot_type'] == 'C']

# Drop total_lots and area from the raw availability data (we will re-add area from the info DataFrame)
carpark_avail = carpark_avail.drop(columns=['total_lots', 'area'])

# Drop duplicates: remove rows where 'agency' == 'HDB' and 'source' == 'lta'
carpark_avail = carpark_avail[~((carpark_avail['agency'] == 'HDB') & (carpark_avail['source'] == 'lta'))]

# Drop 'source' column
carpark_avail = carpark_avail.drop(columns=['source'])

# Extract time-based features
carpark_avail['hour'] = carpark_avail['timestamp'].dt.hour
carpark_avail['day_of_week'] = carpark_avail['timestamp'].dt.dayofweek
carpark_avail['is_weekend'] = carpark_avail['day_of_week'].isin([5, 6]).astype(int)

# Drop rows with missing target
carpark_avail = carpark_avail.dropna(subset=['available_lots'])

# Drop records with invalid available_lots (<0 or >5000)
carpark_avail = carpark_avail[
    (carpark_avail['available_lots'] >= 0) & (carpark_avail['available_lots'] <= 5000)
    ]

# Load carpark information
carpark_info = pd.read_csv(r"C:\Users\user\Documents\Semester 3\carpark_information.csv")
carpark_info['carpark_id'] = carpark_info['carpark_id'].astype(str)
carpark_info = carpark_info.dropna(subset=['area'])

# Merge availability data with carpark info
df = pd.merge(
    carpark_avail,
    carpark_info[['carpark_id', 'area', 'total_lots']],
    on='carpark_id',
    how='inner'
)

print(df.head())

# Check overall time range in the dataset
start_time = df['timestamp'].min()
end_time = df['timestamp'].max()
print(f"Time range in dataset: {start_time} → {end_time}")


# ------------------------------------
# 2) Add Lag Feature (24 hours only)
# ------------------------------------

def add_lag_features_per_carpark(df, target_col='available_lots', lags=[24]):
    """
    Sort by carpark_id & timestamp, then create lag features.
    Removes rows where lag_24 is NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'carpark_id', 'timestamp', and the target_col.
    target_col : str, default='available_lots'
        The column we want to lag.
    lags : list, default=[24]
        A list of lag offsets in 'rows'. Typically, 24 means 24 rows prior
        for hourly data.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new lag_{lag} column for each lag in `lags`.
        Rows with NaN lag values are dropped.
    """
    df = df.sort_values(['carpark_id', 'timestamp']).copy()

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('carpark_id')[target_col].shift(lag)

    # Drop rows where lag_24 is NaN
    df = df[df['lag_24'].notna()]
    return df


df = add_lag_features_per_carpark(df, target_col='available_lots', lags=[24])

# --------------------------------------------
# 3) Encode Categorical Features
# --------------------------------------------

le_area = LabelEncoder()
le_agency = LabelEncoder()

df['area_encoded'] = le_area.fit_transform(df['area'])
df['agency_encoded'] = le_agency.fit_transform(df['agency'])

# One-hot encode carpark_id
carpark_dummies = pd.get_dummies(df['carpark_id'], prefix='carpark')

# Concatenate dummy columns
df = pd.concat([df, carpark_dummies], axis=1)

# Ensure DataFrame is sorted by time (just for consistency)
df = df.sort_values(by='timestamp').reset_index(drop=True)

# --------------------------------------------
# 4) Define Feature Columns & Split Data
# --------------------------------------------

feature_columns = [
                      'hour',
                      'day_of_week',
                      'is_weekend',
                      'total_lots',
                      'area_encoded',
                      'agency_encoded',
                      'lag_24'
                  ] + list(carpark_dummies.columns)

X = df[feature_columns]
y = df['available_lots']

split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# --------------------------------------------
# 5) Train the XGBoost Model
# --------------------------------------------

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=163,
    max_depth=10,
    learning_rate=0.1030,
    subsample=0.9682,
    colsample_bytree=0.9914,
    random_state=42
)
model.fit(X_train, y_train)

# --------------------------------------------
# 6) Evaluate Performance
# --------------------------------------------

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Model RMSE: {rmse:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"Model R²: {r2:.4f}")

# --------------------------------------------
# 7) Feature Importance
# --------------------------------------------

importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Display the top 20 most important features
print(importance_df.head(20))

# --------------------------------------------
# 8) Plot Predicted vs. Actual
# --------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2
)
plt.xlabel("Actual Available Lots")
plt.ylabel("Predicted Available Lots")
plt.title("Predicted vs. Actual Available Lots")
plt.grid(True)
plt.tight_layout()
plt.show()


