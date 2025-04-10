import streamlit as st
import pandas as pd
from inference.predict_from_csv import predict_from_csv
from utils.save_load import load_models

# Загрузка моделей и скейлеров
model_dir = "models"
lstm_model, lstm_feature_extractor, xgb_model, num_scaler, meteo_scaler, scaler, label_scaler = load_models()
# Заголовок и описание
st.title('Предсказание ETA для судов')
st.write('Загрузите CSV файл с данными для предсказания ETA.')

# Форма загрузки файла
uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    # Сохранение файла на диск
    with open("temp_file.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Чтение файла
    df = pd.read_csv("temp_file.csv")
    st.write('Данные, загруженные из файла:')
    st.dataframe(df.head())
    st.write(f"Количество строк после загрузки: {len(df)}")

    # Предсказание ETA
    try:
        result = predict_from_csv(
            csv_path="temp_file.csv",
            seq_length=10,
            scaler=scaler,
            num_scaler=num_scaler,
            meteo_scaler=meteo_scaler,
            lstm_feature_extractor=lstm_model,
            xgb_model=xgb_model,
            label_scaler=label_scaler
        )
        st.subheader('Результаты предсказания:')
        st.write(f"⏱ ETA_diff (сек): {result['eta_diff_seconds']:.2f}")
        st.write(f"📍 Предсказанный ETA: {result['eta']}")
    except ValueError as e:
        st.error(f"Ошибка при предсказании: {e}")
