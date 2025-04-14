import optuna
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === 1. Подготовка признаков ===
X_train_features = np.load("artifacts/X_train_features.npy")
X_test_features = np.load("artifacts/X_test_features.npy")
y_train = np.load("artifacts/y_train.npy")
y_test = np.load("artifacts/y_test.npy")
label_scaler = joblib.load("artifacts/label_scaler.pkl")

# === 2. Целевая функция для Optuna ===
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "objective": "reg:squarederror"
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_features, y_train)
    preds = model.predict(X_test_features)
    return mean_absolute_error(y_test, preds)

# === 3. Запуск Optuna ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

print("\nЛучшие параметры:")
print(study.best_params)

# === 4. Финальная модель ===
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train_features, y_train)

# === 5. Предсказание и метрики ===
y_pred = final_model.predict(X_test_features)
y_pred_original = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

print(f"\n📊 MAE: {mae:.2f}")
print(f"📊 RMSE: {rmse:.2f}")

# === 6. Графики ===
errors = y_test_original - y_pred_original
sns.histplot(errors, bins=30, kde=True)
plt.title("Распределение ошибок ETA")
plt.xlabel("Ошибка (факт - предсказание)")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([min(y_test_original), max(y_test_original)],
         [min(y_test_original), max(y_test_original)],
         "r--")
plt.xlabel("Фактическое ETA")
plt.ylabel("Предсказанное ETA")
plt.title("Факт vs Предсказание")
plt.grid(True)
plt.show()

# === 7. Сохраняем результаты ===
joblib.dump(final_model, "models/final_xgb_model.pkl")

results_df = pd.DataFrame(study.trials_dataframe())
results_df.to_csv("experiments/xgb_optuna_results.csv", index=False)

print("\n✅ Финальная модель и результаты сохранены.")
