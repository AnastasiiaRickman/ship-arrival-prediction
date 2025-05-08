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

df, num_scaler = fit_feature_scalers(df, num_cols)

# === 3. Построение последовательностей ===
baseline = ["speed", "lat", "lon", "distance_to_destination"]
geo = ["lat_diff", "lon_diff", "course_diff", "log_distance"]
temporal = ["hour", "dayofweek", "month", "season"]
dynamic = ["speed_diff", "acceleration", "bearing_change", "moving"]
synthetic = ["log_distance"]

#df, num_scaler, robust_scaler = fit_feature_scalers(df, num_cols, synthetic, temporal)

feature_cols = baseline + geo + temporal + dynamic + synthetic
joblib.dump(feature_cols, 'models/feature_cols.pkl')
# тут проверить метку!!!
dataset, labels = create_sequences(df[feature_cols].values, df["ETA_diff"].values, seq_length=10)
_, distances = create_sequences(
    df[feature_cols].values,
    df["distance_to_destination"].values,
    seq_length=10
)

# === 4. Делим данные ===

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# === 5. Нормализация ===

from sklearn.preprocessing import RobustScaler
# X_scaler = RobustScaler(quantile_range=(5, 95))
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
joblib.dump(X_scaler, 'models/X_scaler.pkl')
label_scaler = StandardScaler()
y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()
joblib.dump(label_scaler, 'models/label_scaler.pkl')

from tensorflow.keras import layers, regularizers

def build_feature_lstm_model(input_shape, num_heads=4, key_dim=64, dropout_rate=0.25):
    inputs = layers.Input(shape=input_shape)
    
    # Входная проекция
    x = layers.Dense(64, activation='swish')(inputs)
    
    # Первый BiLSTM + Residual
    lstm1 = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))
    )(x)
    lstm1 = layers.LayerNormalization()(lstm1)
    
    # Второй BiLSTM
    lstm2 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(lstm1)
    lstm2 = layers.LayerNormalization()(lstm2)

    # Attention
    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate
    )(query=lstm2, value=lstm2)

    # Residual connection
    residual = layers.Add()([lstm2, attention])

    # Комбинирование
    x = layers.GlobalAveragePooling1D()(residual)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Признаки для XGBoost
    features = layers.Dense(64, activation='swish', name='features_for_xgb')(x)

    # 🎯 Финальный выход для регрессии
    output = layers.Dense(1, name='regression_output')(features)
    
    return tf.keras.Model(inputs=inputs, outputs=output)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_loss'
    )
]

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,  # Фиксированное значение
    weight_decay=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=True
)

def hybrid_loss(y_true, y_pred):
    # MSE компонента
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # MAE компонента с весами
    abs_error = tf.abs(y_true - y_pred)
    weights = tf.where(y_true < 1.0, 3.0, 1.0)
    mae_loss = tf.reduce_mean(weights * abs_error)
    
    return 0.7 * mse_loss + 0.3 * mae_loss

# Инициализация модели

# Построим модель
lstm_model = build_feature_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Компилируем модель
lstm_model.compile(
    optimizer=optimizer,
    loss=hybrid_loss,
    metrics=['mae']
)

# Обучаем
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# После обучения LSTM
plot_training_history(history)

# Модель, извлекающая признаки
feature_extractor = tf.keras.Model(
    inputs=lstm_model.input,  # Входной слой модели
    outputs=lstm_model.get_layer('features_for_xgb').output  # Промежуточный слой для извлечения признаков
)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)


# === 8. XGBoost ===
xgb_model = xgb.XGBRegressor(
    max_depth=6, 
    learning_rate=0.08654969092291069,
    n_estimators=496, 
    subsample=0.8648324782679198, 
    colsample_bytree=0.7177686830730101,
    gamma=0.005168380231533848, 
    reg_alpha=0.49457023434877395,
    reg_lambda=0.30214603419143765,
    objective='reg:squarederror'
)

# xgb_model = xgb.XGBRegressor(
#     max_depth=6,
#     learning_rate=0.05,
#     n_estimators=500,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=1,
#     reg_alpha=0.3,
#     reg_lambda=1.0
# )

# Обучение XGBoost
print("Обучение XGBoost...")
xgb_model.fit(X_train_features, y_train)

def weighted_mae(y_true, y_pred, eps=1e-6):
    weights = 1 / (np.abs(y_true) + eps)  # Основной вес
    weights = np.where(y_true < 3600, 1.0, weights)  # Для рейсов <1 часа фиксируем вес=1
    return np.mean(weights * np.abs(y_pred - y_true))

def relative_mae(y_true, y_pred, eps=1e-6):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + eps))


# После получения предсказаний
y_pred = xgb_model.predict(X_test_features)
y_pred_original = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Оценка в оригинальном масштабе (дни)
print("📊 MAE (original):", mean_absolute_error(y_test_original, y_pred_original))
print("📊 Weighted MAE (original):", weighted_mae(y_test_original, y_pred_original))
print("📊 Relative MAE (original):", relative_mae(y_test_original, y_pred_original) * 100, "%")

# Оценка в scaled-масштабе (если нужно)
print("📊 MAE (scaled):", mean_absolute_error(y_test, y_pred))
print("📊 Weighted MAE (scaled):", weighted_mae(y_test, y_pred))
print("📊 Relative MAE (scaled):", relative_mae(y_test, y_pred) * 100, "%")


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