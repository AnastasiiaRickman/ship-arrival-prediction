import os
import argparse
from inference.predict_from_csv import predict_from_csv
from utils.save_load import load_models

def main():
    parser = argparse.ArgumentParser(description="ETA Prediction from AIS track")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    model_dir = "models"
    scaler_path = os.path.join(model_dir, "scalers.pkl")
    lstm_path = os.path.join(model_dir, "lstm_model.keras")
    xgb_path = os.path.join(model_dir, "xgb_model.json")

    # === Загрузка моделей и скейлеров ===
    lstm_model, xgb_model, num_scaler, meteo_scaler, scaler, label_scaler = load_models(model_dir)

    # === Предсказание ===
    result = predict_from_csv(
        csv_path=args.csv,
        seq_length=10,
        scaler=scaler,
        num_scaler=num_scaler,
        meteo_scaler=meteo_scaler,
        lstm_feature_extractor=lstm_model,
        xgb_model=xgb_model,
        label_scaler=label_scaler
    )

    print(f"\n⏱ ETA_diff (сек): {result['eta_diff_seconds']:.2f}")
    print(f"📍 Предсказанный ETA: {result['eta']}\n")

if __name__ == "__main__":
    main()
