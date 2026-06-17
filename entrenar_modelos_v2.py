import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

df = pd.read_csv('datos_mlb_features_v2.csv')

base = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
        'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha']
features = ['Es_Local'] + base + [c + '_Rival' for c in base]

train = df[df['Season'].isin([2022, 2023])]
test = df[df['Season'] == 2024]

print(f"Entrenamiento: {len(train)} | Prueba (2024): {len(test)}\n")

X_train, X_test = train[features], test[features]

# ================= MONEYLINE =================
y_train_ml, y_test_ml = train['Gano'], test['Gano']
baseline_local = (test['Es_Local'] == test['Gano']).mean()

rf_ml = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
rf_ml.fit(X_train, y_train_ml)
acc_rf = accuracy_score(y_test_ml, rf_ml.predict(X_test))

hgb_ml = HistGradientBoostingClassifier(max_depth=4, random_state=42)
hgb_ml.fit(X_train, y_train_ml)
acc_hgb = accuracy_score(y_test_ml, hgb_ml.predict(X_test))

print("--- MONEYLINE ---")
print(f"Baseline (gana el local):        {baseline_local:.3f}")
print(f"Random Forest (v2 features):     {acc_rf:.3f}")
print(f"Gradient Boosting (v2 features): {acc_hgb:.3f}")

# ================= TOTAL DE CARRERAS =================
y_train_tot, y_test_tot = train['Total_Carreras'], test['Total_Carreras']
mae_baseline = mean_absolute_error(y_test_tot, np.full(len(y_test_tot), y_train_tot.mean()))

rf_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
rf_tot.fit(X_train, y_train_tot)
mae_rf = mean_absolute_error(y_test_tot, rf_tot.predict(X_test))

hgb_tot = HistGradientBoostingRegressor(max_depth=4, random_state=42)
hgb_tot.fit(X_train, y_train_tot)
mae_hgb = mean_absolute_error(y_test_tot, hgb_tot.predict(X_test))

print("\n--- TOTAL DE CARRERAS (MAE, menor es mejor) ---")
print(f"Baseline (promedio general):     {mae_baseline:.3f}")
print(f"Random Forest (v2 features):     {mae_rf:.3f}")
print(f"Gradient Boosting (v2 features): {mae_hgb:.3f}")

# ================= IMPORTANCIA DE VARIABLES =================
print("\n--- VARIABLES MAS IMPORTANTES (Moneyline, Random Forest) ---")
print(pd.Series(rf_ml.feature_importances_, index=features).sort_values(ascending=False).head(8))

# ================= GUARDAR =================
joblib.dump(rf_ml, 'modelo_moneyline_v2.pkl')
joblib.dump(rf_tot, 'modelo_total_v2.pkl')
print("\nModelos guardados: modelo_moneyline_v2.pkl, modelo_total_v2.pkl")
