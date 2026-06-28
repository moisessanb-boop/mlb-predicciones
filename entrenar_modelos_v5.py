import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

df = pd.read_csv('datos_mlb_features_v5.csv')

base_v2 = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10',
           'Pct_Victorias_L10','Diff_Carreras_L10',
           'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp',
           'Pct_Victorias_Temp','Racha']
vars_v4 = ['PF_norm','SP_Rest_Local','SP_GS_Local','SP_Rest_Visit','SP_GS_Visit']
vars_v5 = ['SP_ERA_Local','SP_WHIP_Local','SP_K9_Local',
           'SP_ERA_Visit','SP_WHIP_Visit','SP_K9_Visit']

features = (['Es_Local'] + base_v2 + [c+'_Rival' for c in base_v2] + vars_v4 + vars_v5)

df['Margen_Abs'] = abs(df['R'] - df['RA'])
train = df[df['Season'].isin([2022,2023,2024,2025])]
test  = df[df['Season'] == 2026]

print(f"Entrenamiento (2022-2025): {len(train)}")
print(f"Prueba (2026):             {len(test)}")
print(f"Variables totales:         {len(features)}  (+6 vs v4)\n")

X_train, X_test = train[features], test[features]

# ===== MONEYLINE =====
modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_ml.fit(X_train, train['Gano'])
acc      = accuracy_score(test['Gano'], modelo_ml.predict(X_test))
baseline = (test['Es_Local'] == test['Gano']).mean()
print(f"--- MONEYLINE ---")
print(f"Baseline (local):    {baseline:.3f}")
print(f"Random Forest v5:    {acc:.3f}")
print(f"Referencia v4:       0.521")

# ===== TOTAL =====
modelo_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_tot.fit(X_train, train['Total_Carreras'])
mae_tot = mean_absolute_error(test['Total_Carreras'], modelo_tot.predict(X_test))
print(f"\n--- TOTAL DE CARRERAS ---")
print(f"Random Forest v5:    {mae_tot:.3f}")
print(f"Referencia v4:       3.538")

# ===== MARGEN =====
modelo_margen = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
modelo_margen.fit(X_train, train['Margen_Abs'])
mae_margen = mean_absolute_error(test['Margen_Abs'], modelo_margen.predict(X_test))
print(f"\n--- MARGEN DE VICTORIA ---")
print(f"Random Forest v5:    {mae_margen:.3f}")
print(f"Referencia v4:       2.151")

# ===== IMPORTANCIA =====
print("\n--- TOP 12 VARIABLES (Moneyline) ---")
imp = pd.Series(modelo_ml.feature_importances_, index=features).sort_values(ascending=False)
print(imp.head(12))

# Ver dónde quedaron las variables del abridor
print("\n--- Posición de variables del abridor individual ---")
for v in vars_v5:
    pos = list(imp.index).index(v) + 1
    print(f"  {v}: #{pos} ({imp[v]:.4f})")

# ===== GUARDAR (solo si mejora) =====
print("\n¿Guardar modelos v5? Comparar con v4 primero.")
if acc >= 0.521:
    joblib.dump(modelo_ml,     'modelo_moneyline_v2.pkl')
    joblib.dump(modelo_tot,    'modelo_total_v2.pkl')
    joblib.dump(modelo_margen, 'modelo_margen.pkl')
    print(f"✓ v5 mejora o iguala a v4 ({acc:.3f} vs 0.521) - modelos guardados")
else:
    print(f"✗ v5 ({acc:.3f}) no supera v4 (0.521) - modelos NO guardados")
