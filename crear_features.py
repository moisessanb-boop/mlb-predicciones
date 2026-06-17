import pandas as pd

df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Asegurar orden cronológico por equipo
df = df.sort_values(['Team', 'Fecha']).reset_index(drop=True)

# --- Identificar el número de juego del día (para dobles juegos) ---
# "Tuesday, Apr 19 (1)" -> 1, "Tuesday, Apr 19 (2)" -> 2, partidos normales -> 0
df['Game_Num_Day'] = df['Date'].str.extract(r'\((\d+)\)')
df['Game_Num_Day'] = df['Game_Num_Day'].fillna(0).astype(int)

# --- Promedios móviles de los ULTIMOS 10 partidos, sin incluir el actual ---
df['Prom_Carreras_Anotadas_L10'] = (
    df.groupby('Team')['R']
      .transform(lambda x: x.shift(1).rolling(10).mean())
)

df['Prom_Carreras_Permitidas_L10'] = (
    df.groupby('Team')['RA']
      .transform(lambda x: x.shift(1).rolling(10).mean())
)

df['Pct_Victorias_L10'] = (
    df.groupby('Team')['Gano']
      .transform(lambda x: x.shift(1).rolling(10).mean())
)

# --- Tabla de "perfil reciente" de cada equipo en cada fecha + juego del día ---
perfil = df[['Team', 'Fecha', 'Game_Num_Day', 'Prom_Carreras_Anotadas_L10',
              'Prom_Carreras_Permitidas_L10', 'Pct_Victorias_L10']]

# --- Pegar el perfil del RIVAL a cada partido (ahora con llave exacta) ---
df = df.merge(
    perfil,
    left_on=['Opp', 'Fecha', 'Game_Num_Day'],
    right_on=['Team', 'Fecha', 'Game_Num_Day'],
    suffixes=('', '_Rival')
)

df = df.drop(columns=['Team_Rival'])

# --- Quitar partidos sin suficiente historial (inicio de cada temporada) ---
columnas_clave = [
    'Prom_Carreras_Anotadas_L10', 'Prom_Carreras_Permitidas_L10', 'Pct_Victorias_L10',
    'Prom_Carreras_Anotadas_L10_Rival', 'Prom_Carreras_Permitidas_L10_Rival', 'Pct_Victorias_L10_Rival'
]
antes = len(df)
df = df.dropna(subset=columnas_clave)
despues = len(df)

print(f"Filas antes del merge: 14488 (esperado)")
print(f"Filas despues del merge, antes de dropna: {antes}")
print(f"Filas finales (con historial completo): {despues}")
print(f"Filas eliminadas (sin 10 partidos previos): {antes - despues}")

print("\nVerificando que NO haya duplicados:")
duplicados = df.duplicated(subset=['Team','Opp','Fecha','Game_Num_Day']).sum()
print(f"Filas duplicadas: {duplicados}")

print("\nEjemplo de datos finales:")
print(df[['Team','Fecha','Opp','Es_Local',
          'Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10',
          'Prom_Carreras_Anotadas_L10_Rival','Prom_Carreras_Permitidas_L10_Rival','Pct_Victorias_L10_Rival',
          'Gano','Total_Carreras']].head(10))

df.to_csv('datos_mlb_features.csv', index=False)
print("\nGuardado en datos_mlb_features.csv")
