import logging
import pandas as pd
import argparse
from preprocessing.predict_helpers import preprocess_input_data, predict_eta_from_new_data

logging.basicConfig(level=logging.DEBUG)

def main(file_path):
    logging.debug("🔥 cli_predict.py выполняется")
    logging.debug(f"📂 Загрузка данных из файла: {file_path}")
    
    try:
        df_raw = pd.read_csv(file_path)
        logging.debug("✅ Файл загружен успешно.")
    except Exception as e:
        logging.error(f"❌ Ошибка при чтении CSV: {e}")
        return

    try:
        df_processed = preprocess_input_data(df_raw)
        logging.debug("✅ Данные обработаны успешно.")
        result = predict_eta_from_new_data(df_processed)
        logging.debug("✅ Предсказание выполнено успешно.")

        logging.info("\n✅ Предсказание успешно:")
        logging.info(f"⏱ ETA_diff (сек): {result['eta_diff_seconds']:.2f}")
        logging.info(f"📍 Предсказанный ETA: {result['eta']}")
        logging.info(f"🕓 Последняя отметка времени: {result['base_time']}")
    except Exception as e:
        logging.error(f"❌ Ошибка при предсказании: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🚢 Предсказание ETA для трека судна по CSV файлу")
    parser.add_argument('--file', type=str, required=True, help='Путь к CSV файлу с данными')

    args = parser.parse_args()
    main(args.file)
