# server.py
from flask import Flask, request, jsonify
import pandas as pd
import requests

app = Flask(__name__)

@app.route('/process-data', methods=['POST'])
def process_data():
    # Получаем данные из запроса
    data = request.json  # Это будет словарь с данными DataFrame
    df = pd.DataFrame(data)  # Преобразуем обратно в DataFrame

    # Отправляем в Google Colab для получения метео-данных
    colab_url = "https://<ngrok-url>/process-meteo-data"  # URL вашего Colab
    response = requests.post(colab_url, json=df.to_dict())  # Отправляем данные

    # Получаем результат и возвращаем его обратно
    result = response.json()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
