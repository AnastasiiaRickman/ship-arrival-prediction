import pandas as pd
import glob
from preprocessing.feature_engineering import preprocess_input_data

def load_and_prepare_data(path_glob: str) -> pd.DataFrame:
    files = glob.glob(f"{path_glob}/*.csv")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            df = preprocess_input_data(df)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка при чтении {f}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()
