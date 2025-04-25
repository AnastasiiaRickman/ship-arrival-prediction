# Ship Arrival Prediction

Этот проект представляет собой модель предсказания времени прибытия судна, построенную с использованием алгоритмов машинного обучения. Приложение реализовано на Streamlit и упаковано в Docker для удобного локального развертывания.

## 🧠 Описание проекта

- **Цель**: Предсказать ожидаемое время прибытия судна (ETA) на основе входных параметров.
- **ML-модели**: Находятся в директории [`training`](./training), где содержатся обученные модели и скрипты обучения.
- **Интерфейс**: Веб-приложение, построенное с помощью `Streamlit` — файл [`app.py`](./app.py).
- **Docker**: Присутствует `Dockerfile` для удобного развертывания и запуска приложения в изолированной среде.

## 🚀 Как запустить

### 1. Клонируйте репозиторий:

```bash
git clone https://github.com/AnastasiiaRickman/ship-arrival-prediction.git
cd ship-arrival-prediction
```

### 2. Постройте и запустите Docker-контейнер:

```bash
docker build -t ship-arrival-app .
docker run -p 8501:8501 ship-arrival-app
```

### 3. Откройте приложение в браузере:

Перейдите по адресу [http://localhost:8501](http://localhost:8501)

## 📁 Структура проекта

├── app.py                     # Streamlit-приложение для предсказания ETA
├── Dockerfile                 # Docker-образ для развертывания
├── requirements.txt           # Зависимости проекта
│
├── api/                       # Работа с внешними данными
│   ├── data_provider.py
│   └── get_meteo.py
│
├── artifacts/                 # Папка для артефактов (может использоваться для сохранения моделей, логов и пр. для экспериментов)
│
├── data/                      # Данные проекта (содержит данные для обучения модели. Может быть пустой и заполняться через data_provider.py)
│
├── experiments/              # Скрипты экспериментов и обучения моделей
│   ├── lstm_architecture_experiments.py
│   └── train_xgb_with_optuna.py
│
├── models/                    # Обученные модели и связанные артефакты
│   ├── final_xgb_model.pkl
│   ├── lstm_feature_extractor.keras
│   ├── xgb_model.json
│   ├── *.pkl                  # Скалеры, метаданные и др.
│
├── preprocessing/            # Обработка и подготовка данных
│   ├── feature_engineering.py
│   ├── load_and_clean.py
│   ├── predict_helpers.py
│   └── scaling.py
│
└── training/                  # Дополнительные вспомогательные модули/настройки обучения


## 🛠 Используемые технологии

- Python
- Scikit-learn
- Streamlit
- Docker

## 📬 Обратная связь

Если у вас есть предложения, замечания или идеи для улучшения — пишите в Issues или открывайте Pull Request.
