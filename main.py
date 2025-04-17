import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from predict import predict_available_lots

if __name__ == '__main__':
    # specify the data type
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
    carpark_avail = pd.read_csv(r"C:\Users\user\Documents\Capstone\data\raw_carpark_avail_020325_160325.csv",
                                dtype=dtype_spec, parse_dates=['timestamp'])

    # drop total_lots and area
    carpark_avail = carpark_avail.drop(columns=['total_lots'])
    carpark_avail = carpark_avail.drop(columns=['area'])

    # drop duplicates
    carpark_avail = carpark_avail[~((carpark_avail['agency'] == 'HDB') & (carpark_avail['source'] == 'lta'))]
    carpark_avail = carpark_avail.drop(columns=['source'])

    carpark_avail['hour'] = carpark_avail['timestamp'].dt.hour
    carpark_avail['day_of_week'] = carpark_avail['timestamp'].dt.dayofweek
    carpark_avail['is_weekend'] = carpark_avail['day_of_week'].isin([5, 6]).astype(int)

    # Drop rows with missing target
    carpark_avail = carpark_avail.dropna(subset=['available_lots'])

    # load carparkinfo data
    carpark_info = pd.read_csv(r"C:\Users\user\Documents\Capstone\data\carpark_information.csv")
    carpark_info['carpark_id'] = carpark_info['carpark_id'].astype(str)
    carpark_info = carpark_info.dropna(subset=['area'])

    df = pd.merge(carpark_avail, carpark_info[['carpark_id', 'area', 'total_lots']],
                  on='carpark_id', how='inner')

    # Label encode 'area','agency','lot_type'
    le = LabelEncoder()
    df['area_encoded'] = le.fit_transform(df['area'])
    df['agency_encoded'] = le.fit_transform(df['agency'])
    df['lot_type_encoded'] = le.fit_transform(df['lot_type'])

    # === Step 6: One-Hot Encode 'carpark_id' ===
    carpark_dummies = pd.get_dummies(df['carpark_id'], prefix='carpark')

    # set X, y
    X = pd.concat([
        df[['hour', 'day_of_week', 'is_weekend', 'area_encoded', 'agency_encoded', 'lot_type_encoded']],
        carpark_dummies
    ], axis=1)
    y = df['available_lots']

    # check time range
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()

    print(f"Time range in dataset: {start_time} → {end_time}")

    # === Step 5: Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Step 7: Train XGBoost model ===
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # === Step 8: Evaluate ===
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model RMSE: {rmse:.2f}")

    # === Plot predicted vs actual values ===
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Available Lots")
    plt.ylabel("Predicted Available Lots")
    plt.title("🔍 Predicted vs Actual Available Lots")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # test
    carpark_id_input = '1'
    timestamp_input = '2025-03-06 14:00:00'

    predicted_lots = predict_available_lots(
        carpark_id=carpark_id_input,
        timestamp_str=timestamp_input,
        carpark_info=carpark_info,
        model=model,
        le_area=le,
        le_agency=le,
        le_lot_type=le,
        feature_columns=X.columns
    )

    print(f"🔮 Predicted available lots for carpark {carpark_id_input} at {timestamp_input}: {predicted_lots:.2f}")
