import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
def fit_feature_scalers(df: pd.DataFrame, num_cols: list, save_path: str = "models"):

    os.makedirs(save_path, exist_ok=True)
    df_scaled = df.copy()
    num_scaler = RobustScaler(quantile_range=(5, 95))
    #meteo_scaler = MinMaxScaler()

    df_scaled[num_cols] = num_scaler.fit_transform(df_scaled[num_cols])
    #df_scaled[meteo_cols] = meteo_scaler.fit_transform(df_scaled[meteo_cols])

    joblib.dump(num_scaler, os.path.join(save_path, "num_scaler.pkl"))
    #joblib.dump(meteo_scaler, os.path.join(save_path, "meteo_scaler.pkl"))

    print("✅ Скейлеры обучены, применены к данным и сохранены")

    return df_scaled, num_scaler

def fit_feature_scalers_new(df: pd.DataFrame, num_cols: list, log_cols: list = None, 
                        cyclical_cols: list = None, save_path: str = "models"):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.makedirs(save_path, exist_ok=True)
    df_scaled = df.copy()

    # Стандартизация
    num_scaler = StandardScaler()
    df_scaled[num_cols] = num_scaler.fit_transform(df_scaled[num_cols])

    # Робастная стандартизация
    robust_scaler = RobustScaler(quantile_range=(5, 95))
    df_scaled[["acceleration", "bearing_change"]] = robust_scaler.fit_transform(
        df_scaled[["acceleration", "bearing_change"]]
    )

    # Логарифмирование
    if log_cols:
        for col in log_cols:
            df_scaled[col] = np.log1p(np.maximum(df_scaled[col], 0))  # защита от отрицательных значений

    # Циклическое кодирование
    if cyclical_cols:
        period_map = {
            "hour": 24,
            "dayofweek": 7,
            "month": 12,
            "season": 4,
        }
        for col in cyclical_cols:
            if col in period_map:
                max_value = period_map[col]
                df_scaled[f"{col}_sin"] = np.sin(2 * np.pi * df_scaled[col] / max_value)
                df_scaled[f"{col}_cos"] = np.cos(2 * np.pi * df_scaled[col] / max_value)
            else:
                print(f"⚠️ Предупреждение: не задан период для признака {col}")

    # Сохранение скейлеров
    joblib.dump(num_scaler, os.path.join(save_path, "num_scaler.pkl"))
    joblib.dump(robust_scaler, os.path.join(save_path, "robust_scaler.pkl"))

    print("✅ Скейлеры обучены, применены к данным и сохранены")

    return df_scaled, num_scaler, robust_scaler

import joblib
import os

import joblib
import os

def apply_feature_scalers_from_saved(df: pd.DataFrame, num_cols: list, model_dir='models') -> pd.DataFrame:
    df_scaled = df.copy()

    # Загружаем скейлеры из отдельных файлов
    try:
        num_scaler = joblib.load(os.path.join(model_dir, 'num_scaler.pkl'))
        #meteo_scaler = joblib.load(os.path.join(model_dir, 'meteo_scaler.pkl'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Не удалось найти файлы с скейлерами в директории {model_dir}.")

    if not num_cols:
        raise ValueError("num_cols список пуст.")
    #if not meteo_cols:
    #    raise ValueError("meteo_cols список пуст.")

    missing_num_cols = [col for col in num_cols if col not in df.columns]
    #missing_meteo_cols = [col for col in meteo_cols if col not in df.columns]


    # Применяем скейлеры
    df_scaled[num_cols] = num_scaler.transform(df[num_cols])
    #df_scaled[meteo_cols] = meteo_scaler.transform(df[meteo_cols])

    return df_scaled

def apply_feature_scalers_from_saved_new(df: pd.DataFrame, num_cols: list, 
                                     temporal_cols: list = None, model_dir='models') -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_scaled = df.copy()

    # Загрузка скейлеров
    try:
        num_scaler = joblib.load(os.path.join(model_dir, 'num_scaler.pkl'))
        robust_scaler = joblib.load(os.path.join(model_dir, 'robust_scaler.pkl'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Не удалось найти файлы скейлеров в директории {model_dir}.")

    # Проверка на пропущенные колонки
    missing_num_cols = [col for col in num_cols if col not in df.columns]
    if missing_num_cols:
        raise ValueError(f"Входной DataFrame не содержит следующие числовые колонки: {missing_num_cols}")

    # Применение скейлеров
    df_scaled[num_cols] = num_scaler.transform(df[num_cols])
    df_scaled[["acceleration", "bearing_change"]] = robust_scaler.transform(
        df[["acceleration", "bearing_change"]]
    )

    # Циклическое кодирование временных признаков
    if temporal_cols:
        period_map = {
            "hour": 24,
            "dayofweek": 7,
            "month": 12,
            "season": 4,
        }
        for col in temporal_cols:
            if col in df.columns and col in period_map:
                max_value = period_map[col]
                df_scaled[f"{col}_sin"] = np.sin(2 * np.pi * df_scaled[col] / max_value)
                df_scaled[f"{col}_cos"] = np.cos(2 * np.pi * df_scaled[col] / max_value)
            else:
                print(f"⚠️ Пропущено циклическое кодирование для {col}: отсутствует в DataFrame или нет в period_map.")

    print("✅ Признаки масштабированы и закодированы.")

    return df_scaled

