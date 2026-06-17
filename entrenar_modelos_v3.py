import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

df = pd.read_csv('datos_mlb_features_v3.csv')

base_team = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
              'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha']
base_pitch = ['K9_L10','BB9_L10','HR9_L10','WHIP_L10','K9_Temp','BB9_Temp','HR9_Temp','WHIP_Temp']

features = (['Es_Local'] + base_team + [c+'_Rival' for c in base_team]
             + base_pitch + [c+'_Rival' for c in base_pitch])

train = df[df['Season'].isin([2022, 2023])]
test = df[df['Season'] == 2024]

print(f"Entrenamiento: {len(train)} | Prueba (2024): {len(test)} | Variables: {len(features)}\n")

X_train, X_test = train[features], test[features]

# ================= MONEYLINE =================
y_train_ml, y_test_ml = train['Gano'], test['Gano']
baseline_local = (test['Es_Local'] == test['Gano']).mean()

rf_ml = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
rf_ml.fit(X_train, y_train_ml)
acc_rf = accuracy_score(y_test_ml, rf_ml.predict(X_test))

hgb_ml = HistGradientBoostingClassifier(max_depth=4, random_state=42)
hgb_ml.fit(X_train, y_train_ml)
acc_hgb = accuracy_score(y_test_ml, hgb_ml.predict(X_test))

print("--- MONEYLINE ---")
print(f"Baseline (gana el local):        {baseline_local:.3f}")
print(f"Random Forest (v3, con pitcheo): {acc_rf:.3f}")
print(f"Gradient Boosting (v3):          {acc_hgb:.3f}")
print("(Referencia v2 sin pitcheo: RF = 0.557)")

# ================= TOTAL DE CARRERAS =================
y_train_tot, y_test_tot = train['Total_Carreras'], test['Total_Carreras']
mae_baseline = mean_absolute_error(y_test_tot, np.full(len(y_test_tot), y_train_tot.mean()))

rf_tot = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
rf_tot.fit(X_train, y_train_tot)
mae_rf = mean_absolute_error(y_test_tot, rf_tot.predict(X_test))

hgb_tot = HistGradientBoostingRegressor(max_depth=4, random_state=42)
hgb_tot.fit(X_train, y_train_tot)
mae_hgb = mean_absolute_error(y_test_tot, hgb_tot.predict(X_test))

print("\n--- TOTAL DE CARRERAS (MAE, menor es mejor) ---")
print(f"Baseline (promedio general):     {mae_baseline:.3f}")
print(f"Random Forest (v3, con pitcheo): {mae_rf:.3f}")
print(f"Gradient Boosting (v3):          {mae_hgb:.3f}")
print("(Referencia v2 sin pitcheo: RF = 3.424)")

# ================= IMPORTANCIA DE VARIABLES =================
print("\n--- TOP 10 VARIABLES (Moneyline, Random Forest) ---")
print(pd.Series(rf_ml.feature_importances_, index=features).sort_values(ascending=False).head(10))

print("\n--- TOP 10 VARIABLES (Total Carreras, Random Forest) ---")
print(pd.Series(rf_tot.feature_importances_, index=features).sort_values(ascending=False).head(10))

# ================= GUARDAR =================
joblib.dump(rf_ml, 'modelo_moneyline_v3.pkl')
joblib.dump(rf_tot, 'modelo_total_v3.pkl')
print("\nModelos guardados: modelo_moneyline_v3.pkl, modelo_total_v3.pkl")
