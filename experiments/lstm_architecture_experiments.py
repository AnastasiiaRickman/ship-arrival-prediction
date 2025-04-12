import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing.load_and_clean import load_and_prepare_data
from preprocessing.scaling import fit_feature_scalers
from training.sequence_builder import create_sequences


def build_lstm_model(input_shape, hidden_units_1=64, hidden_units_2=32, dropout_1=0.3, dropout_2=0.2, use_bidirectional=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))  # <-- Правильный способ задания входа

    if use_bidirectional:
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units_1, return_sequences=True)))
    else:
        model.add(tf.keras.layers.LSTM(hidden_units_1, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(dropout_1))

    # model.add(tf.keras.layers.LSTM(hidden_units_2, return_sequences=False))
    # model.add(tf.keras.layers.Dropout(dropout_2))

    # model.add(tf.keras.layers.Dense(32, activation='relu'))
    # model.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dense(8, activation='relu'))
    # model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mae')
    return model


def run_experiment(config):
    print("🔄 Загрузка данных...")
    # Загрузка данных
    df = load_and_prepare_data("data")

    # Определение колонок
    num_cols = ["speed", "course", "lat_diff", "lon_diff", "course_diff", "log_distance", "speed_diff", "acceleration", "bearing_change"]
    meteo_cols = [col for col in df.columns if any(x in col for x in ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo", "zos"])]
    print(f"Размер данных: {df.shape}")

    print("🔄 Масштабирование и создание последовательностей...")
    # Масштабирование и создание последовательностей
    df, num_scaler, meteo_scaler = fit_feature_scalers(df, num_cols, meteo_cols)
    feature_cols = ["lat", "lon"] + num_cols + ["moving"] + meteo_cols
    dataset, labels = create_sequences(df[feature_cols].values, df["ETA_diff"].values, seq_length=10)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    label_scaler = StandardScaler()
    y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Создание модели
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        hidden_units_1=config.get("hidden_units_1", 64),
        hidden_units_2=config.get("hidden_units_2", 32),
        dropout_1=config.get("dropout_1", 0.3),
        dropout_2=config.get("dropout_2", 0.2),
        use_bidirectional=config.get("use_bidirectional", True),
    )

    # Обучение модели
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test).flatten()
    y_pred_original = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

    return mae, rmse


if __name__ == "__main__":
    configs = [
        {"hidden_units_1": 64, "hidden_units_2": 32, "dropout_1": 0.3, "dropout_2": 0.2, "use_bidirectional": True},
        {"hidden_units_1": 32, "hidden_units_2": 16, "dropout_1": 0.2, "dropout_2": 0.1, "use_bidirectional": False},
        {"hidden_units_1": 128, "hidden_units_2": 64, "dropout_1": 0.4, "dropout_2": 0.3, "use_bidirectional": True},
    ]

    for i, config in enumerate(configs):
        print(f"\n🚀 Эксперимент {i+1}: {config}")
        mae, rmse = run_experiment(config)
        print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f}")

