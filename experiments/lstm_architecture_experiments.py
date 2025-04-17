import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing.load_and_clean import load_and_prepare_data
from preprocessing.scaling import fit_feature_scalers
from training.sequence_builder import create_sequences

def build_lstm_model(input_shape, hidden_units_1=64, hidden_units_2=32, dropout_1=0.3, dropout_2=0.2, use_bidirectional=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    
    if use_bidirectional:
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units_1, return_sequences=True)))
    else:
        model.add(tf.keras.layers.LSTM(hidden_units_1, return_sequences=True))
    model.add(tf.keras.layers.Dropout(dropout_1))

    model.add(tf.keras.layers.LSTM(hidden_units_2, return_sequences=False))
    model.add(tf.keras.layers.Dropout(dropout_2))

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def plot_training_history(histories, config_names):
    plt.figure(figsize=(12, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, config_names):
        plt.plot(history.history['loss'], label=f'{name} Train')
        plt.plot(history.history['val_loss'], '--', label=f'{name} Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График MAE
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, config_names):
        plt.plot(history.history['mae'], label=f'{name} Train')
        plt.plot(history.history['val_mae'], '--', label=f'{name} Val')
    plt.title('MAE Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('artifacts/architecture_comparison.png')
    plt.show()

def run_experiment(config, X_train, y_train, X_test, y_test):
    # Создаем модель только с нужными параметрами
    model_params = config.copy()
    model_params.pop('name', None)  # Удаляем 'name' если есть
    
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        **model_params
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = model.predict(X_test).flatten()
    mae = model.evaluate(X_test, y_test, verbose=0)[1]
    return history, mae

if __name__ == "__main__":
    # Загрузка и подготовка данных один раз
    print("🔄 Загрузка данных...")
    df = load_and_prepare_data("data")
    num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff", 
               "log_distance", "speed_diff", "acceleration", "bearing_change"]
    meteo_cols = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
    
    df, num_scaler, meteo_scaler = fit_feature_scalers(df, num_cols, meteo_cols)
    baseline = ["speed", "lat", "lon", "distance_to_destination"]
    geo = ["lat_diff", "lon_diff", "course_diff", "log_distance"]
    temporal = ["hour", "dayofweek", "month", "season"]
    dynamic = ["speed_diff", "acceleration", "bearing_change", "moving"]
    meteo = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
    synthetic = ["log_distance"]

    feature_cols = baseline + geo + temporal + dynamic + synthetic
    dataset, labels = create_sequences(df[feature_cols].values, df["ETA_diff"].values, seq_length=10)
    
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    label_scaler = StandardScaler()
    y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Конфигурации для экспериментов
    configs = [
        {"hidden_units_1": 64, "hidden_units_2": 32, "dropout_1": 0.3, 
         "dropout_2": 0.2, "use_bidirectional": True, "name": "Bidirectional LSTM"},
        {"hidden_units_1": 128, "hidden_units_2": 64, "dropout_1": 0.4, 
         "dropout_2": 0.3, "use_bidirectional": False, "name": "Deep LSTM"},
        {"hidden_units_1": 32, "hidden_units_2": 16, "dropout_1": 0.2, 
         "dropout_2": 0.1, "use_bidirectional": False, "name": "Simple LSTM"}
    ]

    results = []
    histories = []
    
    for config in configs:
        print(f"\n🚀 Запуск конфигурации: {config['name']}")
        history, mae = run_experiment(config, X_train, y_train, X_test, y_test)
        results.append((config['name'], mae))
        histories.append(history)
        print(f"📊 {config['name']} - MAE: {mae:.4f}")

    # Визуализация результатов
    plot_training_history(histories, [c['name'] for c in configs])
    
    # Таблица сравнения
    print("\n🏆 Сравнение архитектур:")
    for name, mae in sorted(results, key=lambda x: x[1]):
        print(f"{name:<20} | MAE: {mae:.4f}")