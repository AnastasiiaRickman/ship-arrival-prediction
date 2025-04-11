import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def fit_feature_scalers(df: pd.DataFrame, num_cols: list, meteo_cols: list, save_path: str = "models"):
    """
    Обучает и сохраняет два MinMaxScaler-а: для числовых и метео-признаков.

    Аргументы:
        df (pd.DataFrame): исходные данные
        num_cols (list): список числовых признаков
        meteo_cols (list): список метео-признаков
        save_path (str): директория для сохранения скейлеров

    Возвращает:
        df_scaled (pd.DataFrame): копия df с нормализованными признаками
        num_scaler (MinMaxScaler): скейлер для num_cols
        meteo_scaler (MinMaxScaler): скейлер для meteo_cols
    """
    os.makedirs(save_path, exist_ok=True)
    df_scaled = df.copy()
    num_scaler = MinMaxScaler()
    meteo_scaler = MinMaxScaler()

    df_scaled[num_cols] = num_scaler.fit_transform(df_scaled[num_cols])
    df_scaled[meteo_cols] = meteo_scaler.fit_transform(df_scaled[meteo_cols])

    joblib.dump(num_scaler, os.path.join(save_path, "num_scaler.pkl"))
    joblib.dump(meteo_scaler, os.path.join(save_path, "meteo_scaler.pkl"))

    print("✅ Скейлеры обучены, применены к данным и сохранены")

    return df_scaled, num_scaler, meteo_scaler

import joblib
import os

import joblib
import os

def apply_feature_scalers_from_saved(df: pd.DataFrame, num_cols: list, meteo_cols: list, model_dir='models') -> pd.DataFrame:
    """
    Применяет ранее обученные скейлеры из сохранённых файлов к новому DataFrame.

    Аргументы:
        df (pd.DataFrame): новые данные
        num_cols (list): числовые признаки
        meteo_cols (list): метео-признаки
        model_dir (str): путь к папке, где хранятся скейлеры (по умолчанию 'models')

    Возвращает:
        pd.DataFrame: нормализованная копия входного DataFrame
    """
    df_scaled = df.copy()

    # Загружаем скейлеры из отдельных файлов
    try:
        num_scaler = joblib.load(os.path.join(model_dir, 'num_scaler.pkl'))
        meteo_scaler = joblib.load(os.path.join(model_dir, 'meteo_scaler.pkl'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Не удалось найти файлы с скейлерами в директории {model_dir}.")

    if not num_cols:
        raise ValueError("num_cols список пуст.")
    if not meteo_cols:
        raise ValueError("meteo_cols список пуст.")

    missing_num_cols = [col for col in num_cols if col not in df.columns]
    missing_meteo_cols = [col for col in meteo_cols if col not in df.columns]

    if missing_num_cols or missing_meteo_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_num_cols + missing_meteo_cols}")

    # Применяем скейлеры
    df_scaled[num_cols] = num_scaler.transform(df[num_cols])
    df_scaled[meteo_cols] = meteo_scaler.transform(df[meteo_cols])

    return df_scaled
