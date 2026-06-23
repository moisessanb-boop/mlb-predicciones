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
vars_nuevas = ['PF_norm','SP_Rest_Local','SP_GS_Local','SP_Rest_Visit','SP_GS_Visit']
features = ['Es_Local'] + base_v2 + [c+'_Rival' for c in base_v2] + vars_nuevas

# Margen absoluto de victoria
df['Margen_Abs'] = abs(df['R'] - df['RA'])

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
print(f"--- MONEYLINE ---")
print(f"Baseline: {baseline:.3f}  |  Modelo v4: {acc:.3f}")

# ===== TOTAL DE CARRERAS =====
modelo_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_tot.fit(X_train, train['Total_Carreras'])
mae_tot = mean_absolute_error(test['Total_Carreras'], modelo_tot.predict(X_test))
print(f"\n--- TOTAL DE CARRERAS ---")
print(f"MAE modelo v4: {mae_tot:.3f}")

# ===== MARGEN DE VICTORIA (nuevo) =====
modelo_margen = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
modelo_margen.fit(X_train, train['Margen_Abs'])
mae_margen = mean_absolute_error(test['Margen_Abs'], modelo_margen.predict(X_test))
print(f"\n--- MARGEN DE VICTORIA ---")
print(f"MAE margen: {mae_margen:.3f} carreras")
print(f"Margen promedio real (2026): {test['Margen_Abs'].mean():.2f}")

# ===== IMPORTANCIA =====
print("\n--- TOP 8 VARIABLES (Moneyline) ---")
imp = pd.Series(modelo_ml.feature_importances_, index=features).sort_values(ascending=False)
print(imp.head(8))

# ===== GUARDAR =====
joblib.dump(modelo_ml,     'modelo_moneyline_v2.pkl')
joblib.dump(modelo_tot,    'modelo_total_v2.pkl')
joblib.dump(modelo_margen, 'modelo_margen.pkl')
print("\nModelos guardados: modelo_moneyline_v2.pkl, modelo_total_v2.pkl, modelo_margen.pkl")
