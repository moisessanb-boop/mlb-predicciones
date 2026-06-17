import pandas as pd

# ---------- 1. Cargar y preparar datos de pitcheo ----------
df_pitch = pd.read_csv('datos_pitcheo_crudo.csv')

# Extraer numero de doble juego: "(1)"/"(2)" -> 1/2, si no hay -> 0
df_pitch['Game_Num_Day'] = df_pitch['Date'].astype(str).str.extract(r'\((\d+)\)')
df_pitch['Game_Num_Day'] = df_pitch['Game_Num_Day'].fillna(0).astype(int)

# Quitar el sufijo "(N)" y convertir a fecha
fecha_limpia = df_pitch['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True)
df_pitch['Fecha'] = pd.to_datetime(fecha_limpia)

df_pitch = df_pitch.sort_values(['Team', 'Season', 'Fecha', 'Game_Num_Day']).reset_index(drop=True)

# ---------- 2. Promedios moviles de pitcheo (L10 y temporada completa) ----------
metricas = ['K9', 'BB9', 'HR9', 'WHIP']
for m in metricas:
    df_pitch[f'{m}_L10'] = df_pitch.groupby('Team')[m].transform(lambda x: x.shift(1).rolling(10).mean())
    df_pitch[f'{m}_Temp'] = df_pitch.groupby(['Team', 'Season'])[m].transform(lambda x: x.shift(1).expanding().mean())

cols_pitcheo = [f'{m}_L10' for m in metricas] + [f'{m}_Temp' for m in metricas]
perfil_pitcheo = df_pitch[['Team', 'Fecha', 'Game_Num_Day'] + cols_pitcheo]

# ---------- 3. Cargar dataset principal (v2) ----------
df = pd.read_csv('datos_mlb_features_v2.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

print(f"Filas antes de cruzar pitcheo: {len(df)}")

# ---------- 4. Cruzar estadisticas de pitcheo PROPIAS ----------
df = df.merge(perfil_pitcheo, on=['Team', 'Fecha', 'Game_Num_Day'], how='left')

# ---------- 5. Cruzar estadisticas de pitcheo del RIVAL ----------
perfil_rival = perfil_pitcheo.rename(columns={'Team': 'Opp'})
df = df.merge(perfil_rival, on=['Opp', 'Fecha', 'Game_Num_Day'], how='left', suffixes=('', '_Rival'))

print(f"Filas despues de cruzar pitcheo: {len(df)}")

# ---------- 6. Quitar filas sin historial de pitcheo ----------
cols_clave_pitcheo = cols_pitcheo + [c + '_Rival' for c in cols_pitcheo]
antes = len(df)
df = df.dropna(subset=cols_clave_pitcheo)
print(f"Filas finales (con historial de pitcheo): {len(df)}")
print(f"Eliminadas: {antes - len(df)}")

# ---------- 7. Verificacion de duplicados ----------
dup = df.duplicated(subset=['Team','Opp','Fecha','Game_Num_Day']).sum()
print(f"Duplicados: {dup}")

print("\nEjemplo de columnas nuevas:")
print(df[['Team','Fecha','Opp','K9_L10','BB9_L10','HR9_L10','WHIP_L10',
          'K9_L10_Rival','BB9_L10_Rival','HR9_L10_Rival','WHIP_L10_Rival']].head())

df.to_csv('datos_mlb_features_v3.csv', index=False)
print("\nGuardado en datos_mlb_features_v3.csv")
