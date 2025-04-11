import numpy as np
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic

def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Удаляем неиспользуемые колонки, если есть
    df = df.drop(columns=['seg_id', 'depth', 'latitude', 'longitude', 'time', 'index', 'mlotst_glor', 'mlotst_oras', 'siconc_glor', 'siconc_oras', 'sithick_glor', 'sithick_oras', 'so_glor', 'so_oras', 'thetao_glor', 'thetao_oras', 'uo_glor', 'uo_oras', 'vo_glor', 'vo_oras', 'so_glor', 'so_oras', 'zos_glor', 'zos_oras'])
    # Удаляем строки с пропусками
    df.dropna(inplace=True)

    # Переименовываем метео-колонки
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

    # Обработка времени
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["timestamp_unix"] = df["timestamp"].astype(np.int64) // 10**9
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].map(lambda x: (x % 12 + 3) // 3)

    #  === ГЕОГРАФИЧЕСКИЕ ПРИЗНАКИ ===
    def haversine(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).km

    df["lat_diff"] = df["lat"].diff().fillna(0)
    df["lon_diff"] = df["lon"].diff().fillna(0)
    df["course_diff"] = df["course"].diff().fillna(0)
    df["distance_to_destination"] = df.apply(
    lambda row: haversine(row["lat"], row["lon"], row["lat_destination"], row["lon_destination"]), axis=1)
    df["log_distance"] = np.log1p(df["distance_to_destination"])  # Логарифмирование

    # ETA на основе средней скорости
    avg_speed_kmh = df["speed"].mean() if df["speed"].mean() > 1 else 15
    avg_speed_mps = avg_speed_kmh / 3.6

    df["ETA_seconds"] = df["distance_to_destination"] * 1000 / avg_speed_mps
    df["timestamp_destination"] = df["timestamp"] + pd.to_timedelta(df["ETA_seconds"], unit="s")
    df["timestamp_destination_unix"] = df["timestamp_destination"].astype("int64") // 10**9
    df["ETA_diff"] = df["timestamp_destination_unix"] - df["timestamp_unix"]

    # Доп. признаки
    df["speed_diff"] = df["speed"].diff().fillna(0)
    df["acceleration"] = df["speed_diff"].diff().fillna(0)
    df["bearing_change"] = df["course"].diff().fillna(0)
    df["moving"] = (df["speed"] > 0).astype(int)

    # Удалим аномально высокие скорости
    df = df[df["speed"] < df["speed"].quantile(0.95)]

    return df
