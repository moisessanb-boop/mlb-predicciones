import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv('datos_mlb_features.csv')

features = [
    'Es_Local',
    'Prom_Carreras_Anotadas_L10',
    'Prom_Carreras_Permitidas_L10',
    'Pct_Victorias_L10',
    'Prom_Carreras_Anotadas_L10_Rival',
    'Prom_Carreras_Permitidas_L10_Rival',
    'Pct_Victorias_L10_Rival',
]

# Division cronologica: entrenamos con 2022-2023, probamos con 2024 (datos "nuevos" para el modelo)
train = df[df['Season'].isin([2022, 2023])]
test = df[df['Season'] == 2024]

print(f"Partidos de entrenamiento (2022-2023): {len(train)}")
print(f"Partidos de prueba (2024): {len(test)}")

X_train = train[features]
X_test = test[features]

# ========== MODELO 1: MONEYLINE (clasificacion: gana o pierde) ==========
y_train_ml = train['Gano']
y_test_ml = test['Gano']

modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_ml.fit(X_train, y_train_ml)

pred_ml = modelo_ml.predict(X_test)
acc = accuracy_score(y_test_ml, pred_ml)

# Baseline: si siempre dijeramos "gana el local"
baseline_local = (test['Es_Local'] == test['Gano']).mean()

print("\n--- MONEYLINE ---")
print(f"Precision del modelo en 2024: {acc:.3f} ({acc*100:.1f}%)")
print(f"Baseline (siempre predecir 'gana el local'): {baseline_local:.3f} ({baseline_local*100:.1f}%)")

# ========== MODELO 2: TOTAL DE CARRERAS (regresion) ==========
y_train_tot = train['Total_Carreras']
y_test_tot = test['Total_Carreras']

modelo_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_tot.fit(X_train, y_train_tot)

pred_tot = modelo_tot.predict(X_test)
mae = mean_absolute_error(y_test_tot, pred_tot)
rmse = np.sqrt(mean_squared_error(y_test_tot, pred_tot))

# Baseline: predecir siempre el promedio del set de entrenamiento
baseline_pred = np.full(len(y_test_tot), y_train_tot.mean())
mae_baseline = mean_absolute_error(y_test_tot, baseline_pred)

print("\n--- TOTAL DE CARRERAS ---")
print(f"Error promedio del modelo (MAE): {mae:.2f} carreras")
print(f"Error promedio del baseline (MAE): {mae_baseline:.2f} carreras")
print(f"RMSE del modelo: {rmse:.2f}")
print(f"Promedio real de carreras en 2024: {y_test_tot.mean():.2f}")

# ========== IMPORTANCIA DE VARIABLES ==========
print("\n--- VARIABLES MAS IMPORTANTES (Moneyline) ---")
importancias = pd.Series(modelo_ml.feature_importances_, index=features).sort_values(ascending=False)
print(importancias)

# ========== GUARDAR MODELOS ==========
joblib.dump(modelo_ml, 'modelo_moneyline.pkl')
joblib.dump(modelo_tot, 'modelo_total.pkl')
print("\nModelos guardados: modelo_moneyline.pkl, modelo_total.pkl")
