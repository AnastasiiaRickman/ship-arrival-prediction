import numpy as np
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic

def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Удаляем неиспользуемые колонки, если есть
    df = df.drop(columns=['seg_id'])
    # Удаляем строки с пропусками
    df.dropna(inplace=True)

    # Обработка времени
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["timestamp_destination"] = pd.to_datetime(df["timestamp_destination"]).dt.tz_localize(None)

    df["timestamp_unix"] = df["timestamp"].astype(np.int64) // 10**9
    df["timestamp_destination_unix"] = df["timestamp_destination"].astype(np.int64) // 10**9

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

    df["ETA_diff"] = df["timestamp_destination_unix"] - df["timestamp_unix"]
    df = df[(df["ETA_diff"] > 60) & (df["ETA_diff"] < 86400 * 30)]  # От 1 минуты до 30 дней

    # Доп. признаки
    df["speed_diff"] = df["speed"].diff().fillna(0)
    df["acceleration"] = df["speed_diff"].diff().fillna(0)
    df["bearing_change"] = df["course"].diff().fillna(0)
    df["moving"] = (df["speed"] > 0).astype(int)

    df["log_distance"] = np.log1p(df["distance_to_destination"])  # Логарифмирование

    # Удалим аномально высокие скорости
    df = df[df["speed"] < df["speed"].quantile(0.95)]
    df = df[df["ETA_diff"] < df["ETA_diff"].quantile(0.95)]

    return df
