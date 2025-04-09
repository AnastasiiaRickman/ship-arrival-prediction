import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def fit_feature_scalers(df: pd.DataFrame, save_path: str = "models"):
    """
    Фит и сохранение скейлеров для числовых и метео-признаков.
    """

    os.makedirs(save_path, exist_ok=True)

    # Признаки
    num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff",
                "log_distance", "speed_diff", "acceleration", "bearing_change"]
    meteo_cols = [col for col in df.columns if any(x in col for x in
                    ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
    feature_cols = ["lat", "lon"] + num_cols + ["moving"] + meteo_cols

    # Скейлеры
    num_scaler = MinMaxScaler()
    meteo_scaler = MinMaxScaler()
    seq_scaler = MinMaxScaler()

    # Фит
    num_scaler.fit(df[num_cols])
    meteo_scaler.fit(df[meteo_cols])
    seq_scaler.fit(df[feature_cols])

    # Сохраняем
    joblib.dump(num_scaler, os.path.join(save_path, "num_scaler.pkl"))
    joblib.dump(meteo_scaler, os.path.join(save_path, "meteo_scaler.pkl"))
    joblib.dump(seq_scaler, os.path.join(save_path, "sequence_scaler.pkl"))

    print("✅ Скейлеры обучены и сохранены")

    return num_scaler, meteo_scaler, seq_scaler, num_cols, meteo_cols

