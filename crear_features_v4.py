import pandas as pd
import numpy as np
import re

print("=== CREAR FEATURES V4 (Park Factor + Pitcher Abridor) ===\n")

# ============================================================
# PARTE 1: FACTOR DE PARQUE
# ============================================================
PARK_FACTORS = {
    'ARI': 100, 'ATL': 98,  'BAL': 103, 'BOS': 104, 'CHC': 101,
    'CHW': 99,  'CIN': 107, 'CLE': 100, 'COL': 118, 'DET': 99,
    'HOU': 100, 'KCR': 100, 'LAA': 97,  'LAD': 96,  'MIA': 91,
    'MIL': 99,  'MIN': 102, 'NYM': 100, 'NYY': 106, 'ATH': 97,
    'PHI': 103, 'PIT': 101, 'SDP': 93,  'SEA': 95,  'SFG': 95,
    'STL': 99,  'TBR': 100, 'TEX': 105, 'TOR': 102, 'WSN': 99,
}

# ============================================================
# PARTE 2: PITCHER ABRIDOR — parsear datos de pitcheo
# ============================================================
def parse_starter(pitchers_used):
    if pd.isna(pitchers_used) or str(pitchers_used) == 'nan':
        return None, None, None
    first = str(pitchers_used).split(',')[0].strip()
    match = re.match(r'^(.+?)\s*\((\d+)(?:-([^)]+))?\)', first)
    if not match:
        return None, None, None
    name  = match.group(1).strip()
    rest  = min(int(match.group(2)), 10)
    gamescore = None
    extra = match.group(3)
    if extra:
        for part in extra.split('-'):
            if part.strip().isdigit():
                gamescore = int(part.strip())
                break
    return name, rest, gamescore

print("Cargando datos de pitcheo...")
df_pitch = pd.read_csv('datos_pitcheo_crudo.csv')

# Extraer fechas limpias
df_pitch['Fecha'] = pd.to_datetime(
    df_pitch['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True))
df_pitch['Game_Num_Day'] = df_pitch['Date'].astype(str).str.extract(r'\((\d+)\)')
df_pitch['Game_Num_Day'] = df_pitch['Game_Num_Day'].fillna(0).astype(int)

# Parsear abridor
parsed = df_pitch['Pitchers_Used'].apply(parse_starter)
df_pitch[['SP_Name','SP_Rest','SP_GS']] = pd.DataFrame(parsed.tolist(), index=df_pitch.index)

# GameScore promedio últimas 5 salidas (sin data leak)
df_pitch = df_pitch.sort_values(['SP_Name','Fecha']).reset_index(drop=True)
df_pitch['SP_GS_L5']  = df_pitch.groupby('SP_Name')['SP_GS'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=2).mean())
df_pitch['SP_GS_Temp'] = df_pitch.groupby('SP_Name')['SP_GS'].transform(
    lambda x: x.shift(1).expanding().mean())
df_pitch['SP_GS_Final'] = df_pitch['SP_GS_L5'].fillna(df_pitch['SP_GS_Temp'])
df_pitch['SP_Rest']     = df_pitch['SP_Rest'].fillna(5)

# Tabla de pitcher features por equipo y fecha
sp_table = df_pitch[['Team','Fecha','Game_Num_Day',
                      'SP_Rest','SP_GS_Final']].copy()
sp_table.columns = ['Team','Fecha','Game_Num_Day','SP_Rest','SP_GS']

print(f"Partidos con info de pitcher: {sp_table['SP_GS'].notna().sum()}")

# ============================================================
# PARTE 3: CARGAR FEATURES V2 Y AGREGAR NUEVAS VARIABLES
# ============================================================
print("\nCargando features v2...")
df = pd.read_csv('datos_mlb_features_v2.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# --- Factor de parque (aplica al equipo LOCAL) ---
df['PF'] = df.apply(
    lambda r: PARK_FACTORS.get(r['Team'] if r['Es_Local']==1 else r['Opp'], 100),
    axis=1)
# Normalizado: 1.0 = neutro, >1 = hitter-friendly, <1 = pitcher-friendly
df['PF_norm'] = df['PF'] / 100.0
print(f"Factor de parque agregado. Rango: {df['PF'].min()}-{df['PF'].max()}")

# --- Pitcher abridor: merge propio ---
df = df.merge(sp_table, on=['Team','Fecha','Game_Num_Day'], how='left')
df = df.rename(columns={'SP_Rest':'SP_Rest_Local','SP_GS':'SP_GS_Local'})

# --- Pitcher abridor: merge rival ---
sp_rival = sp_table.rename(columns={
    'Team':'Opp','SP_Rest':'SP_Rest_Visit','SP_GS':'SP_GS_Visit'})
df = df.merge(sp_rival, on=['Opp','Fecha','Game_Num_Day'], how='left')

# Rellenar GameScore faltante con promedio general (50 = starter promedio)
df['SP_GS_Local']  = df['SP_GS_Local'].fillna(50.0)
df['SP_GS_Visit']  = df['SP_GS_Visit'].fillna(50.0)
df['SP_Rest_Local'] = df['SP_Rest_Local'].fillna(5.0)
df['SP_Rest_Visit'] = df['SP_Rest_Visit'].fillna(5.0)

# ============================================================
# PARTE 4: QUITAR FILAS SIN HISTORIAL
# ============================================================
base_v2 = ['Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10',
           'Pct_Victorias_L10','Diff_Carreras_L10',
           'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp',
           'Pct_Victorias_Temp','Racha']
cols_clave = base_v2 + [c+'_Rival' for c in base_v2]
antes = len(df)
df = df.dropna(subset=cols_clave)
print(f"\nFilas antes: {antes} | Filas finales: {len(df)}")
print(f"Duplicados: {df.duplicated(subset=['Team','Opp','Fecha','Game_Num_Day']).sum()}")

print("\nEjemplo de variables nuevas:")
print(df[['Team','Fecha','PF_norm','SP_Rest_Local','SP_GS_Local',
          'SP_Rest_Visit','SP_GS_Visit']].head(5).to_string(index=False))

df.to_csv('datos_mlb_features_v4.csv', index=False)
print("\nGuardado en datos_mlb_features_v4.csv")
