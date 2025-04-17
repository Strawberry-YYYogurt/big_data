import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

#############################
# 1) Suppose these are loaded/defined
#############################

# Your trained XGBoost model
model = ...  # type: xgb.XGBRegressor

# Label encoders
le_area = ...
le_agency = ...

# List of carpark IDs from training (to build one-hot vectors)
training_carpark_ids = [...]

# DataFrame with historical availability for quick look-up
# Must have at least:
#    - carpark_id (str)
#    - timestamp (datetime)
#    - available_lots (float/int)
df_avail = ...

# Carpark info DataFrame with 'carpark_id', 'area', 'agency', 'total_lots'
carpark_info_df = ...


#############################
# 2) Helper Functions
#############################

def check_df_for_availability(carpark_id: str, timestamp: datetime) -> float:
    """
    Looks up 'available_lots' in df_avail for a given (carpark_id, timestamp).
    Returns None if no exact match is found.
    """
    # Filter df for the exact (carpark_id, timestamp)
    row = df_avail[
        (df_avail['carpark_id'] == carpark_id) &
        (df_avail['timestamp'] == timestamp)
        ]
    if not row.empty:
        return float(row.iloc[0]['available_lots'])
    else:
        return None


def get_carpark_info(carpark_id: str, carpark_info: pd.DataFrame):
    """
    Returns (area, agency, total_lots) from the carpark_info_df,
    or raises ValueError if not found.
    """
    row = carpark_info[carpark_info['carpark_id'] == carpark_id]
    if row.empty:
        raise ValueError(f"No info found for carpark_id={carpark_id}")

    area = row.iloc[0]['area']
    agency = row.iloc[0]['agency']
    total_lots = row.iloc[0]['total_lots']
    return area, agency, total_lots


def build_feature_vector(
        carpark_id: str,
        timestamp: datetime,
        lag_24: float,
        rolling_mean_3: float,
        rolling_std_3: float
) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame of features to pass into model.predict(),
    matching the columns used during training.
    """
    # Basic time features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # Monday=0, Sunday=6
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Lookup area, agency, total_lots
    area, agency, total_lots = get_carpark_info(carpark_id, carpark_info_df)
    area_encoded = le_area.transform([area])[0]
    agency_encoded = le_agency.transform([agency])[0]

    # Create dictionary for one row
    row_dict = {
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'total_lots': total_lots,
        'area_encoded': area_encoded,
        'agency_encoded': agency_encoded,
        'lag_24': lag_24,
        'rolling_mean_3': rolling_mean_3,
        'rolling_std_3': rolling_std_3
    }

    # One-hot for carpark_id
    for cp_id in training_carpark_ids:
        row_dict[f'carpark_{cp_id}'] = 1 if cp_id == carpark_id else 0

    # Same feature order as training
    feature_columns = [
                          'hour',
                          'day_of_week',
                          'is_weekend',
                          'total_lots',
                          'area_encoded',
                          'agency_encoded',
                          'lag_24',
                          'rolling_mean_3',
                          'rolling_std_3'
                      ] + [f'carpark_{cp_id}' for cp_id in training_carpark_ids]

    # Create a 1-row DataFrame
    return pd.DataFrame([row_dict], columns=feature_columns)


def get_lag_24_value(
        carpark_id: str,
        timestamp: datetime,
        recursion_depth=0,
        max_depth=5
) -> float:
    """
    Gets the 24-hr-lag availability from df_avail.
    If not found, recursively predicts for (timestamp - 24h).
    """
    if recursion_depth > max_depth:
        # Fallback if we can’t keep recursing
        return 0.0

    # 24 hours earlier
    t_past = timestamp - timedelta(hours=24)

    # Check if we have the real data in df_avail
    val = check_df_for_availability(carpark_id, t_past)
    if val is not None:
        return val

    # Otherwise, recursively predict
    return predict_availability(
        carpark_id,
        t_past,
        recursion_depth=recursion_depth + 1,
        max_depth=max_depth
    )


def get_rolling_features_3(
        carpark_id: str,
        timestamp: datetime,
        recursion_depth=0,
        max_depth=5
):
    """
    Computes rolling_mean_3 and rolling_std_3 from the past 3 timesteps
    (assuming each step is 1 hour), pulling actual data from df_avail or
    recursively predicting if missing.
    """
    avails = []
    for lag_hour in [1, 2, 3]:
        t_past = timestamp - timedelta(hours=lag_hour)
        val = check_df_for_availability(carpark_id, t_past)

        if val is None and recursion_depth <= max_depth:
            # Predict if not found
            val = predict_availability(
                carpark_id,
                t_past,
                recursion_depth=recursion_depth + 1,
                max_depth=max_depth
            )
        elif val is None:
            # Fallback if we hit recursion limit
            val = 0.0

        avails.append(val)

    rolling_mean_3 = float(np.mean(avails))
    rolling_std_3 = float(np.std(avails, ddof=1))  # sample std
    return rolling_mean_3, rolling_std_3


#############################
# 3) The predict function
#############################

def predict_availability(
        carpark_id: str,
        timestamp: datetime,
        recursion_depth=0,
        max_depth=5
) -> float:
    """
    Predicts the availability for (carpark_id, timestamp).

    Steps:
    1) Get lag_24 from df_avail or recursively predict.
    2) Get rolling mean/std from df_avail or recursively predict.
    3) Build a feature vector.
    4) model.predict(...)
    """
    # 1) lag_24
    lag_24_val = get_lag_24_value(
        carpark_id,
        timestamp,
        recursion_depth=recursion_depth,
        max_depth=max_depth
    )

    # 2) rolling mean/std
    rolling_mean_3, rolling_std_3 = get_rolling_features_3(
        carpark_id,
        timestamp,
        recursion_depth=recursion_depth,
        max_depth=max_depth
    )

    # 3) Build feature vector
    X_row = build_feature_vector(
        carpark_id,
        timestamp,
        lag_24_val,
        rolling_mean_3,
        rolling_std_3
    )

    # 4) Predict using the loaded model
    y_pred = model.predict(X_row)
    return float(y_pred[0])


#############################
# 4) Test it out
#############################

if __name__ == "__main__":
    test_carpark_id = "HE12"
    test_timestamp = datetime(2025, 4, 1, 9, 30)

    prediction = predict_availability(test_carpark_id, test_timestamp)
    print(f"Predicted availability for {test_carpark_id} at {test_timestamp}: {prediction:.2f}")
