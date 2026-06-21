import pybaseball as pyb
pyb.cache.disable()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from pybaseball.datasources.bref import BRefSession
from io import StringIO
import joblib, time, re

EQUIPOS = [
    'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
    'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','ATH',
    'PHI','PIT','SDP','SEA','SFG','STL','TBR','TEX','TOR','WSN'
]
TEMPORADAS = [2024, 2025, 2026]

PARK_FACTORS = {
    'ARI': 100, 'ATL': 98,  'BAL': 103, 'BOS': 104, 'CHC': 101,
    'CHW': 99,  'CIN': 107, 'CLE': 100, 'COL': 118, 'DET': 99,
    'HOU': 100, 'KCR': 100, 'LAA': 97,  'LAD': 96,  'MIA': 91,
    'MIL': 99,  'MIN': 102, 'NYM': 100, 'NYY': 106, 'ATH': 97,
    'PHI': 103, 'PIT': 101, 'SDP': 93,  'SEA': 95,  'SFG': 95,
    'STL': 99,  'TBR': 100, 'TEX': 105, 'TOR': 102, 'WSN': 99,
}

base_v2 = [
    'Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10',
    'Pct_Victorias_L10','Diff_Carreras_L10',
    'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp',
    'Pct_Victorias_Temp','Racha'
]
vars_nuevas = ['PF_norm','SP_Rest_Local','SP_GS_Local','SP_Rest_Visit','SP_GS_Visit']
FEATURES = ['Es_Local'] + base_v2 + [c+'_Rival' for c in base_v2] + vars_nuevas

# ===================== 1. DESCARGA PARTIDOS =====================
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
            datos['Team']   = equipo
            filas.append(datos)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)
df_crudo = pd.concat(filas, ignore_index=True)
print(f"Filas descargadas: {len(df_crudo)}")

# ===================== 2. DESCARGA PITCHEO =====================
print("\n2. DESCARGANDO DATOS DE PITCHEO")
session = BRefSession()

def descargar_pitcheo_equipo(season, team):
    url = f"https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=p&year={season}"
    content = session.get(url).content.decode('utf-8')
    content_sc = content.replace('<!--', '').replace('-->', '')
    tablas = pd.read_html(StringIO(content_sc))
    df = tablas[1]
    df.columns = [c[1] for c in df.columns]
    df = df.rename(columns={'Unnamed: 3_level_1': 'Home_Marker'})
    df['Gtm'] = pd.to_numeric(df['Gtm'], errors='coerce')
    df = df.dropna(subset=['Gtm']).copy()
    for c in ['IP','H','R','ER','HR','BB','SO']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    def conv_ip(ip):
        e = int(ip); d = round(ip-e,1)
        return e + {0.0:0,0.1:1,0.2:2}.get(d,0)/3
    df['IP_dec'] = df['IP'].apply(conv_ip)
    df['K9']  = df['SO'] * 9 / df['IP_dec']
    df['BB9'] = df['BB'] * 9 / df['IP_dec']
    df['Season'] = season
    df['Team']   = team
    df = df.sort_values('Gtm').reset_index(drop=True)
    col_p = [c for c in df.columns if 'Pitchers' in str(c)]
    col_p = col_p[0] if col_p else None
    cols = ['Team','Season','Date','IP_dec','K9','BB9']
    if col_p:
        df = df.rename(columns={col_p: 'Pitchers_Used'})
        cols.append('Pitchers_Used')
    return df[cols]

filas_p = []
total_p = len(EQUIPOS) * len(TEMPORADAS)
cont_p  = 0
for temporada in TEMPORADAS:
    for equipo in EQUIPOS:
        cont_p += 1
        print(f"[{cont_p}/{total_p}] Pitcheo {equipo} {temporada}...")
        try:
            datos = descargar_pitcheo_equipo(temporada, equipo)
            filas_p.append(datos)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1.5)
df_pitch = pd.concat(filas_p, ignore_index=True)
print(f"Filas pitcheo: {len(df_pitch)}")

# ===================== 3. LIMPIEZA =====================
print("\n3. LIMPIANDO DATOS")
df = df_crudo.dropna(subset=['R','RA']).copy()
df['Gano']          = df['W/L'].str.startswith('W').astype(int)
df['Es_Local']      = (df['Home_Away'] == 'Home').astype(int)
df['Total_Carreras'] = df['R'] + df['RA']
df['Fecha_texto']   = df['Date'].str.split(', ').str[1] + ' ' + df['Season'].astype(str)
df['Fecha_texto']   = df['Fecha_texto'].str.replace(r'\s*\(\d\)', '', regex=True)
df['Fecha']         = pd.to_datetime(df['Fecha_texto'], format='%b %d %Y')
df['Game_Num_Day']  = df['Date'].astype(str).str.extract(r'\((\d+)\)')
df['Game_Num_Day']  = df['Game_Num_Day'].fillna(0).astype(int)
df = df.sort_values(['Team','Fecha']).reset_index(drop=True)
df.to_csv('datos_mlb_limpio.csv', index=False)
print(f"Filas limpias: {len(df)}")

# ===================== 4. FEATURES =====================
print("\n4. CREANDO VARIABLES")
df['Prom_Carreras_Anotadas_L10']   = df.groupby('Team')['R'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Prom_Carreras_Permitidas_L10'] = df.groupby('Team')['RA'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Pct_Victorias_L10']            = df.groupby('Team')['Gano'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Diff_Carreras_L10']            = df['Prom_Carreras_Anotadas_L10'] - df['Prom_Carreras_Permitidas_L10']
df['Prom_Carreras_Anotadas_Temp']   = df.groupby(['Team','Season'])['R'].transform(lambda x: x.shift(1).expanding().mean())
df['Prom_Carreras_Permitidas_Temp'] = df.groupby(['Team','Season'])['RA'].transform(lambda x: x.shift(1).expanding().mean())
df['Pct_Victorias_Temp']           = df.groupby(['Team','Season'])['Gano'].transform(lambda x: x.shift(1).expanding().mean())
df['Racha']                        = df.groupby(['Team','Season'])['Streak'].transform(lambda x: x.shift(1))

# Park factor
df['PF_norm'] = df.apply(
    lambda r: PARK_FACTORS.get(r['Team'] if r['Es_Local']==1 else r['Opp'], 100) / 100.0,
    axis=1)

# Pitcher abridor
def parse_starter(pu):
    if pd.isna(pu): return None, None, None
    first = str(pu).split(',')[0].strip()
    m = re.match(r'^(.+?)\s*\((\d+)(?:-([^)]+))?\)', first)
    if not m: return None, None, None
    name = m.group(1).strip()
    rest = min(int(m.group(2)), 10)
    gs   = None
    if m.group(3):
        for p in m.group(3).split('-'):
            if p.strip().isdigit(): gs = int(p.strip()); break
    return name, rest, gs

if 'Pitchers_Used' in df_pitch.columns:
    df_pitch['Fecha'] = pd.to_datetime(
        df_pitch['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True))
    parsed = df_pitch['Pitchers_Used'].apply(parse_starter)
    df_pitch[['SP_Name','SP_Rest','SP_GS']] = pd.DataFrame(parsed.tolist(), index=df_pitch.index)
    df_pitch = df_pitch.sort_values(['SP_Name','Fecha']).reset_index(drop=True)
    df_pitch['SP_GS_L5']   = df_pitch.groupby('SP_Name')['SP_GS'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df_pitch['SP_GS_Temp'] = df_pitch.groupby('SP_Name')['SP_GS'].transform(lambda x: x.shift(1).expanding().mean())
    df_pitch['SP_GS_F']    = df_pitch['SP_GS_L5'].fillna(df_pitch['SP_GS_Temp'])
    df_pitch['SP_Rest']    = df_pitch['SP_Rest'].fillna(5)
    sp_table = df_pitch[['Team','Fecha','SP_Rest','SP_GS_F']].rename(columns={'SP_GS_F':'SP_GS'})
    df_pitch['Game_Num_Day'] = df_pitch['Date'].astype(str).str.extract(r'\((\d+)\)')
    df_pitch['Game_Num_Day'] = df_pitch['Game_Num_Day'].fillna(0).astype(int)
    sp_table = df_pitch[['Team','Fecha','Game_Num_Day','SP_Rest','SP_GS_F']].rename(columns={'SP_GS_F':'SP_GS'})
    df = df.merge(sp_table, on=['Team','Fecha','Game_Num_Day'], how='left')
    df = df.rename(columns={'SP_Rest':'SP_Rest_Local','SP_GS':'SP_GS_Local'})
    sp_rival = sp_table.rename(columns={'Team':'Opp','SP_Rest':'SP_Rest_Visit','SP_GS':'SP_GS_Visit'})
    df = df.merge(sp_rival, on=['Opp','Fecha','Game_Num_Day'], how='left')
else:
    df['SP_Rest_Local'] = 5.0
    df['SP_GS_Local']   = 50.0
    df['SP_Rest_Visit'] = 5.0
    df['SP_GS_Visit']   = 50.0

df['SP_GS_Local']   = df['SP_GS_Local'].fillna(50.0)
df['SP_GS_Visit']   = df['SP_GS_Visit'].fillna(50.0)
df['SP_Rest_Local'] = df['SP_Rest_Local'].fillna(5.0)
df['SP_Rest_Visit'] = df['SP_Rest_Visit'].fillna(5.0)

# Cruce con rival
perfil = df[['Team','Fecha','Game_Num_Day'] + base_v2]
df = df.merge(perfil, left_on=['Opp','Fecha','Game_Num_Day'],
              right_on=['Team','Fecha','Game_Num_Day'], suffixes=('','_Rival'))
df = df.drop(columns=['Team_Rival'])
cols_clave = base_v2 + [c+'_Rival' for c in base_v2]
df = df.dropna(subset=cols_clave)
print(f"Filas con features: {len(df)}")

# ===================== 5. ENTRENAMIENTO =====================
print("\n5. ENTRENANDO MODELOS (2024-2025)")
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
    print(f"MAE Total Carreras:  {mae:.3f}")

joblib.dump(modelo_ml,  'modelo_moneyline_v2.pkl')
joblib.dump(modelo_tot, 'modelo_total_v2.pkl')
print("Modelos guardados.")
print("\nPipeline v4 completado.")
