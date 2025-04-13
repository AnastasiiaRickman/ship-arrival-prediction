import streamlit as st
import pandas as pd
from preprocessing.predict_helpers import preprocess_input_data, predict_eta_from_new_data
def debug_df(df, label="DataFrame"):
    st.subheader(f"🔍 Debug — {label}")
    st.write("Shape:", df.shape)
    st.text(f"✔️ Колонки df_processed:\n{df.columns.tolist()}")
    st.dataframe(df.head())

# Заголовок и описание
st.title('⛵ Предсказание ETA для судов')
st.write('Загрузите CSV файл с данными для предсказания ETA.')

# Форма загрузки файла
uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    # Сохранение файла на диск
    with open("temp_file.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Чтение и отображение данных
    df_raw = pd.read_csv("temp_file.csv")
    st.write('📄 Загруженные данные:')
    st.dataframe(df_raw.head())
    st.write(f"🔢 Кол-во строк в файле: {len(df_raw)}")

    # Предсказание ETA
    try:
        debug_df(df_raw, "Исходный DF")
        df_processed = preprocess_input_data(df_raw)
        debug_df(df_processed, "DF после препроцессинга")
        result = predict_eta_from_new_data(df_processed)

        st.subheader('📊 Результаты предсказания:')
        st.write(f"⏱ ETA_diff (сек): {result['eta_diff_seconds']:.2f}")
        st.write(f"📍 Предсказанный ETA: {result['eta']}")
        st.write(f"🕓 Последнее время в треке: {result['base_time']}")
    except ValueError as e:
        st.error(f"❌ Ошибка при предсказании: {e}")
        st.write("Columns:", list(df_processed.columns))
