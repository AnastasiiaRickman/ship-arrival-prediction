import pandas as pd
import numpy as np
from datetime import timedelta
from geopy.distance import geodesic
from preprocessing.scaling import apply_feature_scalers_from_saved
from api.get_meteo import get_meteo_data

def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    #df = get_meteo_data(df)
    #df = df.drop(columns=['depth', 'latitude', 'longitude', 'time', 'index'])
    print("==> Исходные колонки:", df.columns)
    df = df.rename(columns={
        'mlotst_cglo': 'mlotst',
        'siconc_cglo': 'siconc',
        'sithick_cglo': 'sithick',
        'so_cglo': 'so',
        'thetao_cglo': 'thetao',
        'uo_cglo': 'uo',
        'vo_cglo': 'vo',
        'so_cglo': 'so',
        'zos_cglo': 'zos'
    })
    #df.fillna(0, inplace=True)  # Изменяет исходный датафрейм
    for column in df.select_dtypes(include=['float64', 'int64']).columns: 
        df[column] = df[column].fillna(df[column].mean())

    # Преобразуем timestamp в datetime без временной зоны
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    #df["season"] = df["month"].map(lambda x: (x % 12 + 3) // 3)
    print("==> После добавления season:", df.columns)

    #  === ГЕОГРАФИЧЕСКИЕ ПРИЗНАКИ ===
    def haversine(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).km

    df["lat_diff"] = df["lat"].diff().fillna(0)
    df["lon_diff"] = df["lon"].diff().fillna(0)
    df["course_diff"] = df["course"].diff().fillna(0)
    df["distance_to_destination"] = df.apply(
    lambda row: haversine(row["lat"], row["lon"], row["lat_destination"], row["lon_destination"]), axis=1)
    df["log_distance"] = np.log1p(df["distance_to_destination"])  # Логарифмирование

    # === ДИНАМИЧЕСКИЕ ПРИЗНАКИ ===
    df["speed_diff"] = df["speed"].diff().fillna(0)
    df["acceleration"] = df["speed_diff"].diff().fillna(0)
    df["bearing_change"] = df["course"].diff().fillna(0)
    df["moving"] = (df["speed"] > 0).astype(int)

    # === ОБРАБОТКА ВЫБРОСОВ ===
    #df = df[df["speed"] < df["speed"].quantile(0.99)]

    # === НОРМАЛИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ ===
    num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff", "log_distance", "speed_diff", "acceleration", "bearing_change"]
    # === МЕТЕОПРИЗНАКИ ===
    meteo_cols = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
    df = apply_feature_scalers_from_saved(df, num_cols, meteo_cols)
    print("==> После масштабирования:", df.columns)

    return df

import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from datetime import timedelta

def predict_eta_from_new_data(df_new, seq_length=10, model_dir='models'):
    """
    Прогноз ETA_diff и итогового ETA по последней строке новых данных судна.

    Все модели и скейлеры загружаются из директории model_dir.

    Возвращает:
    - ETA_diff (в секундах)
    - ETA (datetime: timestamp последней строки + ETA_diff)
    """
    print("==> Колонки на входе в predict:", df_new.columns)
    if 'timestamp' not in df_new.columns:
        raise ValueError("В DataFrame отсутствует колонка 'timestamp'.")

    # Загружаем feature_cols
    #feature_cols = joblib.load("models/feature_cols.pkl")
    #print("==> Feature columns из модели:", feature_cols)
    feature_cols = ['lat', 'lon', 'speed', 'course', 'lat_diff', 'lon_diff', 'course_diff', 'log_distance', 'speed_diff', 'acceleration', 'bearing_change', 'moving', 'mlotst', 'siconc', 'sithick', 'so', 'thetao', 'uo', 'vo', 'zos']
    # Загружаем скейлеры
    scaler = joblib.load("models/X_scaler.pkl")
    label_scaler = joblib.load("models/label_scaler.pkl")

    # Загружаем модели
    lstm_feature_extractor = tf.keras.models.load_model("models/lstm_feature_extractor.keras")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("models/xgb_model.json")

    # Сортировка по времени
    df_sorted = df_new.sort_values(by="timestamp")

    # Последние seq_length строк
    import streamlit as st
    print("🧠 df_sorted columns:", df_sorted.columns.tolist())
    st.text(f"🧠 df_sorted columns:\n{df_sorted.columns.tolist()}")
    print("📌 feature_cols from model:", feature_cols)
    st.text(f"📌 feature_cols from model:\n{feature_cols}")

    seq_df = df_sorted[feature_cols].tail(seq_length)
    print("==> Колонки перед моделью:", seq_df.columns)

    if len(seq_df) < seq_length:
        raise ValueError(f"Недостаточно данных: нужно минимум {seq_length}, а получено {len(seq_df)}")

    # Масштабирование
    scaled_seq = scaler.transform(seq_df.values)
    input_data = scaled_seq.reshape(1, seq_length, len(feature_cols))

    # Извлечение признаков через LSTM
    lstm_features = lstm_feature_extractor.predict(input_data, verbose=0)

    # Предсказание ETA_diff (в секундах)
    pred_scaled = xgb_model.predict(lstm_features.reshape(1, -1))
    eta_diff_seconds = float(label_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0])

    # ETA на основе времени последней строки
    last_time = df_sorted["timestamp"].iloc[-1]
    eta = last_time + timedelta(seconds=eta_diff_seconds)

    return {
        "eta_diff_seconds": eta_diff_seconds,
        "eta": eta,
        "base_time": last_time
    }
