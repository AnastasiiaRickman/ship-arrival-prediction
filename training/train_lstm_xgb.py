import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib
import os

from preprocessing.scaling import fit_feature_scalers
from training.sequence_builder import create_sequences
from utils.save_load import save_models
from preprocessing.load_and_clean import load_and_prepare_data

# === 1. Загрузка и подготовка данных ===
import requests
import os


API_URL = "http://127.0.0.1:8000"
LOCAL_DATA_DIR = "data"  # Эта папка будет заполнена загруженными CSV

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# # Получаем список файлов с API
# response = requests.get(f"{API_URL}/files")
# csv_files = response.json()

# # Скачиваем каждый CSV
# for filename in csv_files:
#     file_url = f"{API_URL}/files/{filename}"
#     file_path = os.path.join(LOCAL_DATA_DIR, filename)

#     r = requests.get(file_url)
#     if r.status_code == 200:
#         with open(file_path, 'wb') as f:
#             f.write(r.content)
#         print(f"✅ Файл {filename} загружен.")
#     else:
#         print(f"❌ Ошибка при загрузке {filename}")

# После загрузки можно передать в пайплайн
from preprocessing.load_and_clean import load_and_prepare_data

df = load_and_prepare_data(LOCAL_DATA_DIR)

# === 2. Масштабирование ===
num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff",
            "log_distance", "speed_diff", "acceleration", "bearing_change"]

meteo_cols = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]

df, num_scaler, meteo_scaler = fit_feature_scalers(df, num_cols, meteo_cols)

# === 3. Построение последовательностей ===
feature_cols = ["lat", "lon"] + num_cols + ["moving"] + meteo_cols
joblib.dump(feature_cols, 'models/feature_cols.pkl')
dataset, labels = create_sequences(df[feature_cols].values, df["ETA_diff"].values, seq_length=10)

# === 4. Делим данные ===
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# === 5. Нормализация ===
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
joblib.dump(X_scaler, 'models/X_scaler.pkl')
label_scaler = StandardScaler()
y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()
joblib.dump(label_scaler, 'models/label_scaler.pkl')

# === 6. LSTM модель ===

lstm_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),  # 👈 теперь input здесь
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')
def build_lstm_model(input_shape):
    lstm_model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),

        # LSTM слои (без bidirectional, как в лучшем эксперименте)
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(8),

        # Полносвязные слои
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Финальный output
    ])

    lstm_model.compile(optimizer='adam', loss='mae')
    return lstm_model

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]
# === 6. LSTM Feature Extractor ===

# lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
print("🧠 Обучение LSTM...")
lstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,  # Можете попробовать уменьшить до 16
    callbacks=callbacks,
    verbose=1  # Для подробных логов
)

# Создаем Feature Extractor на основе обученной модели
inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = lstm_model.layers[0](inputs)  # Bidirectional LSTM
x = lstm_model.layers[1](x)       # Dropout
x = lstm_model.layers[2](x)       # LSTM
x = lstm_model.layers[3](x)       # Dropout
x = lstm_model.layers[4](x)       # Dense(32)
x = lstm_model.layers[5](x)       # Dense(16)
feature_output = lstm_model.layers[6](x)  # Dense(8) — наш слой признаков

# # СОЗДАНИЕ FEATURE EXTRACTOR (на основе обученной модели)
feature_extractor = tf.keras.Model(inputs=inputs, outputs=feature_output)

print("📐 Извлечение признаков...")
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

#=== 8. XGBoost ===
xgb_model = xgb.XGBRegressor(
    max_depth=10, 
    learning_rate=0.18907365925970332,
    n_estimators=362, 
    subsample=0.9183945866232028, 
    colsample_bytree=0.9069336171226844,
    gamma=0.7367508381925425, 
    reg_alpha=0.8747927173857223,
    reg_lambda=0.3767252935537126,
    objective='reg:squarederror'
)

print("🚀 Обучение XGBoost...")
xgb_model.fit(X_train_features, y_train)

y_pred = xgb_model.predict(X_test_features)
y_pred_original = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

print("📊 MAE (scaled):", mean_absolute_error(y_test, y_pred))
print("📊 MAE (original):", mean_absolute_error(y_test_original, y_pred_original))

# === 9. Сохранение ===
print("💾 Сохранение моделей...")
os.makedirs("models", exist_ok=True)
feature_extractor.save("models/lstm_feature_extractor.keras")
xgb_model.save_model("models/xgb_model.json")

print("✅ Всё готово!")
print("💾 Сохраняем извлечённые данные для экспериментов...")

os.makedirs("artifacts", exist_ok=True)

np.save("artifacts/X_train_features.npy", X_train_features)
np.save("artifacts/X_test_features.npy", X_test_features)
np.save("artifacts/y_train.npy", y_train)
np.save("artifacts/y_test.npy", y_test)
joblib.dump(label_scaler, "artifacts/label_scaler.pkl")

