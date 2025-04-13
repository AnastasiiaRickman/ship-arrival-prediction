import pandas as pd
import xarray as xr
import copernicusmarine as cm
from datetime import timezone
from concurrent.futures import ThreadPoolExecutor

def download_meteo_data(lat, lon, time):
    try:
        dataset_id = "cmems_mod_glo_phy-all_my_0.25deg_P1D-m"
        variables = [
            "mlotst_cglo", "siconc_cglo", "sithick_cglo", "so_cglo", 
            "thetao_cglo", "uo_cglo", "vo_cglo", "zos_cglo"
        ]

        username = "arickman"
        password = "Nastya1998"

        time = pd.to_datetime(time)
        formatted_time = time.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        ).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Формируем уникальное имя файла (временное)
        filename = f"cmems_data_{lat}_{lon}_{time.strftime('%Y%m%d')}.nc"
        output_path = f"/tmp/{filename}"

        # Загружаем файл
        cm.subset(
            username=username,
            password=password,
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=lon,
            maximum_longitude=lon,
            minimum_latitude=lat,
            maximum_latitude=lat,
            start_datetime=formatted_time,
            end_datetime=formatted_time,
            output_filename=filename,
            output_directory="/tmp"
        )

        # Загружаем в память
        ds = xr.open_dataset(output_path)
        df = ds.to_dataframe().reset_index()

        # Удаляем временный файл
        os.remove(output_path)

        return df.iloc[0:1]

    except Exception as e:
        print(f"Ошибка при обработке данных для координат ({lat}, {lon}) и времени {time}: {e}")
        return None

# Функция обработки строки (получение данных по координатам и времени)
def process_row(row):
    lat = row['lat']
    lon = row['lon']
    time = row['timestamp']
    meteo_df = download_meteo_data(lat, lon, time)
    if meteo_df is not None:
        meteo_df['index'] = row.name
        return meteo_df
    else:
        # Логируем ошибку для строки
        print(f"Не удалось загрузить данные для строки с индексом {row.name}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame вместо None


# Функция для обработки нескольких строк с использованием многозадачности
def get_meteo_data(df2):
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_row, [row for _, row in df2.iterrows()])

    # Исключаем None или пустые DataFrame из результатов
    valid_results = [result for result in results if not result.empty]

    if valid_results:
        meteo_data = pd.concat(valid_results, ignore_index=True)
        df_combined = pd.merge(df2, meteo_data, left_index=True, right_on='index', how='left')
        return df_combined
    else:
        print("Нет доступных метеорологических данных для объединения.")
        return None

