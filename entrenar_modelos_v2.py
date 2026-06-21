import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

df = pd.read_csv('datos_mlb_features_v2.csv')

base = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
        'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha']
features = ['Es_Local'] + base + [c+'_Rival' for c in base]

# Entrenamos con TODO lo disponible (2022-2025)
# La prueba real es el desempeño diario en 2026
train = df[df['Season'].isin([2022, 2023, 2024, 2025])]

print(f"Partidos de entrenamiento (2022-2025): {len(train)}")
print(f"Distribucion por temporada:")
print(train['Season'].value_counts().sort_index())

X_train = train[features]

# ================= MONEYLINE =================
modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_ml.fit(X_train, train['Gano'])

# ================= TOTAL DE CARRERAS =================
modelo_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_tot.fit(X_train, train['Total_Carreras'])

# ================= IMPORTANCIA DE VARIABLES =================
print("\n--- TOP 8 VARIABLES (Moneyline) ---")
print(pd.Series(modelo_ml.feature_importances_, index=features).sort_values(ascending=False).head(8))

# ================= GUARDAR =================
joblib.dump(modelo_ml,  'modelo_moneyline_v2.pkl')
joblib.dump(modelo_tot, 'modelo_total_v2.pkl')
print("\nModelos guardados con entrenamiento 2022-2025.")
print("La prueba real es el desempeño diario en 2026.")
