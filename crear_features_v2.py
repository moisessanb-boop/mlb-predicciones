import pandas as pd

df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(['Team', 'Fecha']).reset_index(drop=True)

# Numero de juego del dia (para dobles juegos)
df['Game_Num_Day'] = df['Date'].str.extract(r'\((\d+)\)')
df['Game_Num_Day'] = df['Game_Num_Day'].fillna(0).astype(int)

# ---------- Variables L10 (igual que antes) ----------
df['Prom_Carreras_Anotadas_L10'] = df.groupby('Team')['R'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Prom_Carreras_Permitidas_L10'] = df.groupby('Team')['RA'].transform(lambda x: x.shift(1).rolling(10).mean())
df['Pct_Victorias_L10'] = df.groupby('Team')['Gano'].transform(lambda x: x.shift(1).rolling(10).mean())

# ---------- NUEVO: Diferencial de carreras (forma reciente) ----------
df['Diff_Carreras_L10'] = df['Prom_Carreras_Anotadas_L10'] - df['Prom_Carreras_Permitidas_L10']

# ---------- NUEVO: Promedios de TEMPORADA completa (se reinician cada año) ----------
df['Prom_Carreras_Anotadas_Temp'] = df.groupby(['Team','Season'])['R'].transform(lambda x: x.shift(1).expanding().mean())
df['Prom_Carreras_Permitidas_Temp'] = df.groupby(['Team','Season'])['RA'].transform(lambda x: x.shift(1).expanding().mean())
df['Pct_Victorias_Temp'] = df.groupby(['Team','Season'])['Gano'].transform(lambda x: x.shift(1).expanding().mean())

# ---------- NUEVO: Racha (antes del partido actual, se reinicia cada temporada) ----------
df['Racha'] = df.groupby(['Team','Season'])['Streak'].transform(lambda x: x.shift(1))

# ---------- Tabla de perfil para cruzar con el rival ----------
cols_perfil = [
    'Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
    'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha'
]
perfil = df[['Team','Fecha','Game_Num_Day'] + cols_perfil]

df = df.merge(
    perfil,
    left_on=['Opp','Fecha','Game_Num_Day'],
    right_on=['Team','Fecha','Game_Num_Day'],
    suffixes=('','_Rival')
)
df = df.drop(columns=['Team_Rival'])

# ---------- Quitar filas sin historial suficiente ----------
cols_clave = cols_perfil + [c + '_Rival' for c in cols_perfil]
antes = len(df)
df = df.dropna(subset=cols_clave)
print(f"Filas antes de dropna: {antes}")
print(f"Filas finales: {len(df)}")

duplicados = df.duplicated(subset=['Team','Opp','Fecha','Game_Num_Day']).sum()
print(f"Duplicados: {duplicados}")

df.to_csv('datos_mlb_features_v2.csv', index=False)
print("Guardado en datos_mlb_features_v2.csv")
