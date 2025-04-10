import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import regularizers
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

num_scaler, meteo_scaler, seq_scaler, num_cols, meteo_cols = fit_feature_scalers(df)

# === 3. Построение последовательностей ===
feature_cols = ["lat", "lon"] + num_cols + ["moving"] + meteo_cols
dataset, labels = create_sequences(df[feature_cols].values, df["ETA_diff"].values, seq_length=10)

# === 4. Делим данные ===
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# === 5. Нормализация ===
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

label_scaler = MinMaxScaler()
y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()

# === 6. LSTM модель ===
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("🧠 Обучение LSTM...")
lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stop])

inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))


# === 7. Feature extractor ===
inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = lstm_model.layers[0](inputs)
x = lstm_model.layers[1](x)
x = lstm_model.layers[2](x)
x = lstm_model.layers[3](x)
x = lstm_model.layers[4](x)
x = lstm_model.layers[5](x)
feature_output = lstm_model.layers[6](x)

feature_extractor = tf.keras.Model(inputs=inputs, outputs=feature_output)

print("📐 Извлечение признаков...")
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# === 8. XGBoost ===
xgb_model = xgb.XGBRegressor(
    max_depth=7,
    learning_rate=0.18,
    n_estimators=362,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.008,
    reg_alpha=0.23,
    reg_lambda=0.37,
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
save_models(
    lstm_model=lstm_model,
    xgb_model=xgb_model,
    num_scaler=num_scaler,
    meteo_scaler=meteo_scaler,
    label_scaler=label_scaler,
    scaler=X_scaler
)

print("✅ Всё готово!")
print("💾 Сохраняем извлечённые данные для экспериментов...")

os.makedirs("artifacts", exist_ok=True)

np.save("artifacts/X_train_features.npy", X_train_features)
np.save("artifacts/X_test_features.npy", X_test_features)
np.save("artifacts/y_train.npy", y_train)
np.save("artifacts/y_test.npy", y_test)
joblib.dump(label_scaler, "artifacts/label_scaler.pkl")

