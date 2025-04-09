import pandas as pd
import numpy as np
from datetime import timedelta
from geopy.distance import geodesic

def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Очистка и обработка
    df = df.drop(columns=['seg_id', 'depth', 'latitude', 'longitude', 'time', 'index', 'mlotst_glor', 'mlotst_oras', 'siconc_glor', 'siconc_oras', 'sithick_glor', 'sithick_oras', 'so_glor', 'so_oras', 'thetao_glor', 'thetao_oras', 'uo_glor', 'uo_oras', 'vo_glor', 'vo_oras', 'so_glor', 'so_oras', 'zos_glor', 'zos_oras'], errors='ignore')

    df = df.rename(columns={
        'mlotst_cglo': 'mlotst',
        'siconc_cglo': 'siconc',
        'sithick_cglo': 'sithick',
        'so_cglo': 'so',
        'thetao_cglo': 'thetao',
        'uo_cglo': 'uo',
        'vo_cglo': 'vo',
        'zos_cglo': 'zos'
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].map(lambda x: (x % 12 + 3) // 3)

    df["lat_diff"] = df["lat"].diff().fillna(0)
    df["lon_diff"] = df["lon"].diff().fillna(0)
    df["course_diff"] = df["course"].diff().fillna(0)

    df["distance_to_destination"] = df.apply(
        lambda row: geodesic((row["lat"], row["lon"]), (row["lat_destination"], row["lon_destination"])).km,
        axis=1
    )
    df["log_distance"] = np.log1p(df["distance_to_destination"])
    df["speed_diff"] = df["speed"].diff().fillna(0)
    df["acceleration"] = df["speed_diff"].diff().fillna(0)
    df["bearing_change"] = df["course"].diff().fillna(0)
    df["moving"] = (df["speed"] > 0).astype(int)

    df = df[df["speed"] < df["speed"].quantile(0.99)]
    df.fillna(0, inplace=True)

    return df


def apply_feature_scalers(df: pd.DataFrame, num_cols, meteo_cols, num_scaler, meteo_scaler):
    df[num_cols] = num_scaler.transform(df[num_cols])
    df[meteo_cols] = meteo_scaler.transform(df[meteo_cols])
    return df


def predict_eta_from_sequence(df_new, seq_length, scaler, lstm_feature_extractor,
                               xgb_model, feature_cols, label_scaler):
    df_new_sorted = df_new.sort_values(by="timestamp")

    if len(df_new_sorted) < seq_length:
        raise ValueError(f"Недостаточно данных: нужно минимум {seq_length}, получено {len(df_new_sorted)}")

    seq_df = df_new_sorted[feature_cols].tail(seq_length)
    scaled_seq = scaler.transform(seq_df.values)
    input_data = scaled_seq.reshape(1, seq_length, len(feature_cols))

    lstm_features = lstm_feature_extractor.predict(input_data, verbose=0)
    pred_scaled = xgb_model.predict(lstm_features.reshape(1, -1))

    eta_diff_seconds = float(label_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0])
    eta_timestamp = df_new_sorted["timestamp"].iloc[-1] + timedelta(seconds=eta_diff_seconds)

    return {
        "eta_diff_seconds": eta_diff_seconds,
        "eta": eta_timestamp
    }
