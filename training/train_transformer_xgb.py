# === 1. Загрузка и подготовка данных ===
import requests
import os
import matplotlib.pyplot as plt  # 🔥 Добавлено для графиков

# API_URL = "http://127.0.0.1:8000"
LOCAL_DATA_DIR = "data"  # Эта папка будет заполнена загруженными CSV

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
import matplotlib.pyplot as plt
def plot_transformer_training(history, model_name="Transformer"):
    """Визуализация процесса обучения Transformer"""
    plt.figure(figsize=(15, 6))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # График MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} MAE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'artifacts/{model_name.lower()}_training_history.png')
    plt.show()

from preprocessing.load_and_clean import load_and_prepare_data

df = load_and_prepare_data(LOCAL_DATA_DIR)

# === 2. Масштабирование ===
num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff",
            "log_distance", "speed_diff", "acceleration", "bearing_change"]

meteo_cols = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]

from preprocessing.scaling import fit_feature_scalers
df, num_scaler, meteo_scaler = fit_feature_scalers(df, num_cols, meteo_cols)

# === 3. Построение последовательностей ===
baseline = ["speed", "lat", "lon", "distance_to_destination"]
geo = ["lat_diff", "lon_diff", "course_diff", "log_distance"]
temporal = ["hour", "dayofweek", "month", "season"]
dynamic = ["speed_diff", "acceleration", "bearing_change", "moving"]
meteo = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
synthetic = ["log_distance"]

feature_cols = baseline + geo + temporal + dynamic + synthetic

from training.sequence_builder import create_sequences
dataset, labels = create_sequences(df[feature_cols].values, df["ETA_diff"].values, seq_length=10)

# === 4. Делим данные ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# === 5. Нормализация ===
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

label_scaler = RobustScaler()
y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()

# === 6. Модель ===
from keras import regularizers, layers, models
import tensorflow as tf

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(inputs.shape[-1], kernel_regularizer=regularizers.l2(1e-4))(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

def build_stronger_transformer(input_shape,
                                head_size=64,
                                num_heads=4,
                                ff_dim=128,
                                num_blocks=2,
                                mlp_units=[128, 64],
                                dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalMaxPooling1D()(x)
    for units in mlp_units:
        x = layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1)(x)
    model = models.Model(inputs, output)
    return model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("Обучение Transformer...")
model = build_stronger_transformer(input_shape=(X_train.shape[1], X_train.shape[2]))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)
plot_transformer_training(history, "Stronger Transformer")

# Предсказания:
from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
y_pred_original = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

print("📊 MAE (scaled):", mean_absolute_error(y_test, y_pred))
print("📊 MAE (original):", mean_absolute_error(y_test_original, y_pred_original))

# Сохраняем
print("💾 Сохранение моделей...")
import os
os.makedirs("models", exist_ok=True)

print("✅ Всё готово!")
