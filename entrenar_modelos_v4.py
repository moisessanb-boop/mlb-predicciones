import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

df = pd.read_csv('datos_mlb_features_v4.csv')

base_v2 = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10',
           'Pct_Victorias_L10','Diff_Carreras_L10',
           'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp',
           'Pct_Victorias_Temp','Racha']

# Variables nuevas v4
vars_nuevas = ['PF_norm','SP_Rest_Local','SP_GS_Local','SP_Rest_Visit','SP_GS_Visit']

features = (['Es_Local'] + base_v2 + [c+'_Rival' for c in base_v2] + vars_nuevas)

train = df[df['Season'].isin([2022,2023,2024,2025])]
test  = df[df['Season'] == 2026]

print(f"Entrenamiento (2022-2025): {len(train)}")
print(f"Prueba (2026):             {len(test)}")
print(f"Variables totales:         {len(features)}\n")

X_train, X_test = train[features], test[features]

# ===== MONEYLINE =====
modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_ml.fit(X_train, train['Gano'])

acc      = accuracy_score(test['Gano'], modelo_ml.predict(X_test))
baseline = (test['Es_Local'] == test['Gano']).mean()

print("--- MONEYLINE ---")
print(f"Baseline (gana el local):  {baseline:.3f}")
print(f"Random Forest v4:          {acc:.3f}")
print(f"Referencia v2 (backtest):  0.509")

# ===== TOTAL DE CARRERAS =====
modelo_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_tot.fit(X_train, train['Total_Carreras'])

mae      = mean_absolute_error(test['Total_Carreras'], modelo_tot.predict(X_test))
mae_base = mean_absolute_error(test['Total_Carreras'],
                               np.full(len(test), train['Total_Carreras'].mean()))

print("\n--- TOTAL DE CARRERAS (MAE menor = mejor) ---")
print(f"Baseline MAE:              {mae_base:.3f}")
print(f"Random Forest v4 MAE:      {mae:.3f}")
print(f"Referencia v2 (backtest):  3.577")

# ===== IMPORTANCIA =====
print("\n--- TOP 10 VARIABLES (Moneyline) ---")
imp = pd.Series(modelo_ml.feature_importances_, index=features).sort_values(ascending=False)
print(imp.head(10))

print("\n--- TOP 10 VARIABLES (Total Carreras) ---")
imp_tot = pd.Series(modelo_tot.feature_importances_, index=features).sort_values(ascending=False)
print(imp_tot.head(10))

# ===== GUARDAR =====
joblib.dump(modelo_ml,  'modelo_moneyline_v2.pkl')
joblib.dump(modelo_tot, 'modelo_total_v2.pkl')
print("\nModelos guardados.")
