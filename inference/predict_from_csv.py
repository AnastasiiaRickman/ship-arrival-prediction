import pandas as pd
from preprocessing.predict_helpers import (
    preprocess_input_data,
    apply_feature_scalers,
    predict_eta_from_sequence
)

def predict_from_csv(csv_path: str,
                     seq_length: int,
                     scaler,
                     num_scaler,
                     meteo_scaler,
                     lstm_feature_extractor,
                     xgb_model,
                     label_scaler):
    """
    Полный процесс предсказания ETA из .csv
    """
    df = pd.read_csv(csv_path)
    df = preprocess_input_data(df)

    num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff",
                "log_distance", "speed_diff", "acceleration", "bearing_change"]
    meteo_cols = [col for col in df.columns if any(x in col for x in
                    ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
    feature_cols = ["lat", "lon"] + num_cols + ["moving"] + meteo_cols

    df = apply_feature_scalers(df, num_cols, meteo_cols, num_scaler, meteo_scaler)

    return predict_eta_from_sequence(
        df_new=df,
        seq_length=seq_length,
        scaler=scaler,
        lstm_feature_extractor=lstm_feature_extractor,
        xgb_model=xgb_model,
        feature_cols=feature_cols,
        label_scaler=label_scaler
    )

