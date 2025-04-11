import os
import joblib
import tensorflow as tf
import xgboost as xgb


def save_models(lstm_model, xgb_model, num_scaler, meteo_scaler, label_scaler=None, scaler=None, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # Сохраняем модели
    lstm_model.save(os.path.join(model_dir, 'lstm_model.keras'))
    inputs = tf.keras.Input(shape=lstm_model.input_shape[1:])
    x = lstm_model.layers[0](inputs)
    x = lstm_model.layers[1](x)
    x = lstm_model.layers[2](x)
    x = lstm_model.layers[3](x)
    x = lstm_model.layers[4](x)
    x = lstm_model.layers[5](x)
    feature_output = lstm_model.layers[6](x)
    lstm_feature_extractor = tf.keras.Model(inputs=inputs, outputs=feature_output)
    lstm_feature_extractor.save(os.path.join(model_dir, 'lstm_feature_extractor.keras'))

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
    # Загружаем модели
    lstm_model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_model.keras'))
    lstm_feature_extractor = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_feature_extractor.keras'))

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, 'xgb_model.json'))

    # Загружаем скейлеры
    scalers = joblib.load(os.path.join(model_dir, 'scalers.pkl'))

    return lstm_model, lstm_feature_extractor, xgb_model, scalers['num_scaler'], scalers['meteo_scaler'], scalers['scaler'], scalers['label_scaler']


