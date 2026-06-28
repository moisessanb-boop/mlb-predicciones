import pandas as pd
import numpy as np
import re

print("=== CREAR FEATURES V5 (Pitcher Abridor Individual) ===\n")

# ============================================================
# PARTE 1: Calcular stats de cada abridor HASTA cada fecha
# ============================================================
print("1. Procesando abridores históricos...")
df_pitch = pd.read_csv('datos_pitcheo_crudo.csv')
df_pitch['Fecha'] = pd.to_datetime(
    df_pitch['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True),
    errors='coerce')
df_pitch['Game_Num_Day'] = df_pitch['Date'].astype(str).str.extract(r'\((\d+)\)')
df_pitch['Game_Num_Day'] = df_pitch['Game_Num_Day'].fillna(0).astype(int)

def extraer_abridor(pu):
    if pd.isna(pu):
        return None
    first = str(pu).split(',')[0].strip()
    m = re.match(r'^(.+?)\s*\(', first)
    return m.group(1).strip() if m else first

df_pitch['SP'] = df_pitch['Pitchers_Used'].apply(extraer_abridor)
df_pitch = df_pitch.sort_values(['SP','Fecha']).reset_index(drop=True)

# Acumulados sin data leak (shift para excluir la salida actual)
df_pitch['cum_ER'] = df_pitch.groupby('SP')['ER'].transform(lambda x: x.shift(1).expanding().sum())
df_pitch['cum_IP'] = df_pitch.groupby('SP')['IP_dec'].transform(lambda x: x.shift(1).expanding().sum())
df_pitch['cum_H']  = df_pitch.groupby('SP')['H'].transform(lambda x: x.shift(1).expanding().sum())
df_pitch['cum_BB'] = df_pitch.groupby('SP')['BB'].transform(lambda x: x.shift(1).expanding().sum())
df_pitch['cum_SO'] = df_pitch.groupby('SP')['SO'].transform(lambda x: x.shift(1).expanding().sum())

# ERA, WHIP, K9 hasta la fecha (mínimo 10 IP para ser confiable)
mask_ip = df_pitch['cum_IP'] >= 10
df_pitch['SP_ERA']  = np.where(mask_ip, df_pitch['cum_ER'] * 9 / df_pitch['cum_IP'], np.nan)
df_pitch['SP_WHIP'] = np.where(mask_ip, (df_pitch['cum_H'] + df_pitch['cum_BB']) / df_pitch['cum_IP'], np.nan)
df_pitch['SP_K9']   = np.where(mask_ip, df_pitch['cum_SO'] * 9 / df_pitch['cum_IP'], np.nan)

# Rellenar con promedio de liga cuando no hay historial suficiente
df_pitch['SP_ERA']  = df_pitch['SP_ERA'].fillna(4.50)
df_pitch['SP_WHIP'] = df_pitch['SP_WHIP'].fillna(1.30)
df_pitch['SP_K9']   = df_pitch['SP_K9'].fillna(8.0)

sp_table = df_pitch[['Team','Fecha','Game_Num_Day','SP_ERA','SP_WHIP','SP_K9']].copy()
print(f"   Salidas procesadas: {len(sp_table)}")

# ============================================================
# PARTE 2: Cargar features v4 y agregar stats del abridor
# ============================================================
print("2. Cargando features v4...")
df = pd.read_csv('datos_mlb_features_v4.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Merge abridor LOCAL
df = df.merge(sp_table, on=['Team','Fecha','Game_Num_Day'], how='left')
df = df.rename(columns={'SP_ERA':'SP_ERA_Local','SP_WHIP':'SP_WHIP_Local','SP_K9':'SP_K9_Local'})

# Merge abridor VISITANTE
sp_rival = sp_table.rename(columns={
    'Team':'Opp','SP_ERA':'SP_ERA_Visit','SP_WHIP':'SP_WHIP_Visit','SP_K9':'SP_K9_Visit'})
df = df.merge(sp_rival, on=['Opp','Fecha','Game_Num_Day'], how='left')

# Rellenar faltantes
for col in ['SP_ERA_Local','SP_ERA_Visit']:
    df[col] = df[col].fillna(4.50)
for col in ['SP_WHIP_Local','SP_WHIP_Visit']:
    df[col] = df[col].fillna(1.30)
for col in ['SP_K9_Local','SP_K9_Visit']:
    df[col] = df[col].fillna(8.0)

print(f"   Filas finales: {len(df)}")
print(f"   Duplicados: {df.duplicated(subset=['Team','Opp','Fecha','Game_Num_Day']).sum()}")

print("\nEjemplo de stats de abridor individual:")
print(df[['Team','Opp','Fecha','SP_ERA_Local','SP_WHIP_Local','SP_K9_Local',
          'SP_ERA_Visit']].head(5).to_string(index=False))

df.to_csv('datos_mlb_features_v5.csv', index=False)
print("\nGuardado en datos_mlb_features_v5.csv")
