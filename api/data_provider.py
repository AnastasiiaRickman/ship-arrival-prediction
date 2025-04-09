
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from typing import List

app = FastAPI()

DATA_DIR = "/Users/anastasiakrutcova/Desktop/ship-arrival-prediction-data"

@app.get("/files", response_model=List[str])
def list_csv_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

@app.get("/files/{filename}")
def get_csv_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Файл не найден"}
    return FileResponse(file_path, media_type="text/csv", filename=filename)

