import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import holidays

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

# add public holiday as feature
sg_holidays = holidays.Singapore()

carpark_avail['date'] = carpark_avail['timestamp'].dt.date
carpark_avail['is_holiday'] = carpark_avail['date'].apply(lambda d: int(d in sg_holidays))

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

df = df.sort_values(['carpark_id', 'timestamp']).copy()

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
                      'is_holiday',
                      'total_lots',
                      'area_encoded',
                      'agency_encoded'
                  ] + list(carpark_dummies.columns)

X = df[feature_columns]
y = df['available_lots']

split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# top 20 important feature
print(importance_df.head(20))
# Evaluate performance
y_pred =rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Model RMSE: {rmse:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"Model R²:", r2)