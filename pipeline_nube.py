import pybaseball as pyb
pyb.cache.disable()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib, time

EQUIPOS = [
    'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
    'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','ATH',
    'PHI','PIT','SDP','SEA','SFG','STL','TBR','TEX','TOR','WSN'
]
TEMPORADAS = [2024, 2025, 2026]

print("="*50)
print("1. DESCARGANDO DATOS MLB")
print("="*50)
filas = []
total = len(EQUIPOS) * len(TEMPORADAS)
contador = 0
for temporada in TEMPORADAS:
    for equipo in EQUIPOS:
        contador += 1
        print(f"[{contador}/{total}] {equipo} {temporada}...")
        try:
            datos = pyb.schedule_and_record(temporada, equipo)
            datos['Season'] = temporada
            datos['Team'] = equipo
            filas.append(datos)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)
df_crudo = pd.concat(filas, ignore_index=True)
print(f"Filas descargadas: {len(df_crudo)}")

print("\n2. LIMPIANDO DATOS")
df = df_crudo.dropna(subset=['R','RA']).copy()
df['Gano'] = df['W/L'].str.startswith('W').astype(int)
df['Es_Local'] = (df['Home_Away'] == 'Home').astype(int)
df['Total_Carreras'] = df['R'] + df['RA']
df['Fecha_texto'] = df['Date'].str.split(', ').str[1] + ' ' + df['Season'].astype(str)
df['Fecha_texto'] = df['Fecha_texto'].str.replace(r'\s*\(\d\)', '', regex=True)
df['Fecha'] = pd.to_datetime(df['Fecha_texto'], format='%b %d %Y')
df['Game_Num_Day'] = df['Date'].astype(str).str.extract(r'\((\d+)\)')
df['Game_Num_Day'] = df['Game_Num_Day'].fillna(0).astype(int)
df = df.sort_values(['Team','Fecha']).reset_index(drop=True)
df.to_csv('datos_mlb_limpio.csv', index=False)
print(f"Filas limpias: {len(df)}")

print("\n3. CREANDO VARIABLES")
base = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
        'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha']

df['Prom_Carreras_Anotadas_L10'] = df.groupby('Team')['R'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Prom_Carreras_Permitidas_L10'] = df.groupby('Team')['RA'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Pct_Victorias_L10'] = df.groupby('Team')['Gano'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Diff_Carreras_L10'] = df['Prom_Carreras_Anotadas_L10'] - df['Prom_Carreras_Permitidas_L10']
df['Prom_Carreras_Anotadas_Temp'] = df.groupby(['Team','Season'])['R'].transform(lambda x: x.shift(1).expanding().mean())
df['Prom_Carreras_Permitidas_Temp'] = df.groupby(['Team','Season'])['RA'].transform(lambda x: x.shift(1).expanding().mean())
df['Pct_Victorias_Temp'] = df.groupby(['Team','Season'])['Gano'].transform(lambda x: x.shift(1).expanding().mean())
df['Racha'] = df.groupby(['Team','Season'])['Streak'].transform(lambda x: x.shift(1))

perfil = df[['Team','Fecha','Game_Num_Day'] + base]
df = df.merge(perfil, left_on=['Opp','Fecha','Game_Num_Day'], right_on=['Team','Fecha','Game_Num_Day'], suffixes=('','_Rival'))
df = df.drop(columns=['Team_Rival'])
cols_clave = base + [c+'_Rival' for c in base]
df = df.dropna(subset=cols_clave)
print(f"Filas con features: {len(df)}")

print("\n4. ENTRENANDO MODELOS")
FEATURES = ['Es_Local'] + base + [c+'_Rival' for c in base]
train = df[df['Season'].isin([2024,2025])]
test  = df[df['Season'] == 2026]

X_train, X_test = train[FEATURES], test[FEATURES]

modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_ml.fit(X_train, train['Gano'])
if len(test) > 0:
    acc = accuracy_score(test['Gano'], modelo_ml.predict(X_test))
    print(f"Precision Moneyline: {acc:.3f}")

modelo_tot = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
modelo_tot.fit(X_train, train['Total_Carreras'])
if len(test) > 0:
    mae = mean_absolute_error(test['Total_Carreras'], modelo_tot.predict(X_test))
    print(f"MAE Total Carreras: {mae:.3f}")

joblib.dump(modelo_ml,  'modelo_moneyline_v2.pkl')
joblib.dump(modelo_tot, 'modelo_total_v2.pkl')
print("Modelos guardados.")
print("\nPipeline completado.")
