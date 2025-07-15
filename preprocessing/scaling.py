import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
def fit_feature_scalers(df: pd.DataFrame, num_cols: list, save_path: str = "models"):

    os.makedirs(save_path, exist_ok=True)
    df_scaled = df.copy()
    num_scaler = RobustScaler(quantile_range=(5, 95))

    df_scaled[num_cols] = num_scaler.fit_transform(df_scaled[num_cols])

    joblib.dump(num_scaler, os.path.join(save_path, "num_scaler.pkl"))

    print("✅ Скейлеры обучены, применены к данным и сохранены")

    return df_scaled, num_scaler


def apply_feature_scalers_from_saved(df: pd.DataFrame, num_cols: list, model_dir='models') -> pd.DataFrame:
    df_scaled = df.copy()

    # Загружаем скейлеры из отдельных файлов
    try:
        num_scaler = joblib.load(os.path.join(model_dir, 'num_scaler.pkl'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Не удалось найти файлы с скейлерами в директории {model_dir}.")

    if not num_cols:
        raise ValueError("num_cols список пуст.")
    
    missing_num_cols = [col for col in num_cols if col not in df.columns]

    # Применяем скейлеры
    df_scaled[num_cols] = num_scaler.transform(df[num_cols])

    return df_scaled