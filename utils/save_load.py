import os
import joblib
import tensorflow as tf
import xgboost as xgb


def save_models(lstm_model, xgb_model, num_scaler, meteo_scaler, label_scaler=None, scaler=None, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # Сохраняем модели
    lstm_model.save(os.path.join(model_dir, 'lstm_model.keras'))
    xgb_model.save_model(os.path.join(model_dir, 'xgb_model.json'))

    # Сохраняем скейлеры
    scalers = {
        'num_scaler': num_scaler,
        'meteo_scaler': meteo_scaler,
        'label_scaler': label_scaler,
        'scaler': scaler
    }
    joblib.dump(scalers, os.path.join(model_dir, 'scalers.pkl'))

    print(f"✅ Модели и скейлеры сохранены в {model_dir}/")


def load_models(model_dir='models'):
    # Загрузка моделей и скейлеров
    lstm_model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_model.keras'))
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, 'xgb_model.json'))

    scalers = joblib.load(os.path.join(model_dir, 'scalers.pkl'))
    return lstm_model, xgb_model, scalers['num_scaler'], scalers['meteo_scaler']
