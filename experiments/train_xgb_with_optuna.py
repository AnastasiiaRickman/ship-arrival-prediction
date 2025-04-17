import optuna
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Загрузка данных
X_train_features = np.load("artifacts/X_train_features.npy")
X_test_features = np.load("artifacts/X_test_features.npy")
y_train = np.load("artifacts/y_train.npy")
y_test = np.load("artifacts/y_test.npy")
label_scaler = joblib.load("artifacts/label_scaler.pkl")

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
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

# Оптимизация
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

# Лучшие параметры
print("\nЛучшие параметры:")
print(study.best_params)

# Финальная модель с историей обучения
final_model = xgb.XGBRegressor(
    **study.best_params,
    eval_metric="mae"
)

eval_set = [(X_train_features, y_train), (X_test_features, y_test)]
final_model.fit(
    X_train_features, 
    y_train,
    eval_set=eval_set,
    verbose=True
)

# Графики обучения
results = final_model.evals_result()
plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['mae'], label='Train')
plt.plot(results['validation_1']['mae'], label='Validation')
plt.title('XGBoost Learning Curve')
plt.xlabel('Boosting Rounds')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.savefig("artifacts/xgb_learning_curve.png")
plt.show()

# Предсказания и метрики
y_pred = final_model.predict(X_test_features)
y_pred_original = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

print(f"\n📊 MAE: {mae:.2f}")
print(f"📊 RMSE: {rmse:.2f}")

# Дополнительные графики
errors = y_test_original - y_pred_original
plt.figure(figsize=(10, 5))
sns.histplot(errors, bins=30, kde=True)
plt.title("Распределение ошибок")
plt.xlabel("Ошибка (факт - предсказание)")
plt.grid(True)
plt.savefig("artifacts/error_distribution.png")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([min(y_test_original), max(y_test_original)], 
         [min(y_test_original), max(y_test_original)], 
         'r--')
plt.xlabel("Фактические значения")
plt.ylabel("Предсказанные значения")
plt.title("Факт vs Предсказание")
plt.grid(True)
plt.savefig("artifacts/actual_vs_predicted.png")
plt.show()

# Сохранение модели
joblib.dump(final_model, "models/final_xgb_model.pkl")
print("\n✅ Финальная модель сохранена.")
import pandas as pd
import seaborn as sns

# Конвертация в DataFrame
trials_df = pd.DataFrame([t.params for t in study.trials])
trials_df['MAE'] = [t.value for t in study.trials]

# График зависимости MAE от параметров
sns.pairplot(trials_df, 
             vars=['max_depth', 'learning_rate', 'n_estimators', 'MAE'],
             diag_kind='kde')
plt.savefig('artifacts/parameters_impact.png')
trials_df.to_csv('optuna_results.csv', index=False)