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
from preprocessing.load_and_clean import load_and_prepare_data

# === 1. Загрузка и подготовка данных ===
import requests
import os
import matplotlib.pyplot as plt
def plot_training_history(history):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График MAE (если используется)
    if 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('MAE over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('artifacts/training_history.png')  # Сохраняем график
    plt.show()

# API_URL = "http://127.0.0.1:8000"
LOCAL_DATA_DIR = "data"  # Эта папка будет заполнена загруженными CSV

# os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

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

# # После загрузки можно передать в пайплайн

df = load_and_prepare_data(LOCAL_DATA_DIR)

# === 2. Масштабирование ===
num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff",
            "log_distance", "speed_diff", "acceleration", "bearing_change"]

meteo_cols = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]

df, num_scaler, meteo_scaler = fit_feature_scalers(df, num_cols, meteo_cols)

# === 3. Построение последовательностей ===
baseline = ["speed", "lat", "lon", "distance_to_destination"]
geo = ["lat_diff", "lon_diff", "course_diff", "log_distance"]
temporal = ["hour", "dayofweek", "month", "season"]
dynamic = ["speed_diff", "acceleration", "bearing_change", "moving"]
meteo = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
synthetic = ["log_distance"]

feature_cols = baseline + geo + temporal + dynamic + synthetic
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

# lstm_model = tf.keras.Sequential([
#     tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),  # 👈 теперь input здесь
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(64, return_sequences=True),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(32),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
lstm_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),  # 👈 теперь input здесь
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)  # Feature слой
])

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')

# callbacks = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
# ]
# === 6. LSTM Feature Extractor ===

# lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
print("🧠 Обучение LSTM...")
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,  # Можете попробовать уменьшить до 16
    # callbacks=callbacks,
    verbose=1  # Для подробных логов
)

# После обучения LSTM
plot_training_history(history)

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
# xgb_model = xgb.XGBRegressor(
#     max_depth=10, 
#     learning_rate=0.18907365925970332,
#     n_estimators=362, 
#     subsample=0.9183945866232028, 
#     colsample_bytree=0.9069336171226844,
#     gamma=0.7367508381925425, 
#     reg_alpha=0.8747927173857223,
#     reg_lambda=0.3767252935537126,
#     objective='reg:squarederror'
# )

xgb_model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.3,
    reg_lambda=1.0
)

print("Обучение XGBoost...")
xgb_model.fit(X_train_features, y_train)
# Визуализация важности признаков XGBoost
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('artifacts/xgb_feature_importance.png')
plt.show()

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
print("Feature columns из модели:", feature_cols)
print("💾 Сохраняем извлечённые данные для экспериментов...")

os.makedirs("artifacts", exist_ok=True)

np.save("artifacts/X_train_features.npy", X_train_features)
np.save("artifacts/X_test_features.npy", X_test_features)
np.save("artifacts/y_train.npy", y_train)
np.save("artifacts/y_test.npy", y_test)
joblib.dump(label_scaler, "artifacts/label_scaler.pkl")