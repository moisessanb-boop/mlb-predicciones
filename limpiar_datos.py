import pandas as pd

# Cargamos los datos crudos
df = pd.read_csv('datos_mlb_crudo.csv')

# 1. Convertir W/L a un número: 1 = ganó, 0 = perdió
df['Gano'] = df['W/L'].str.startswith('W').astype(int)

# 2. Convertir Home_Away a número: 1 = jugó en casa, 0 = jugó de visitante
df['Es_Local'] = (df['Home_Away'] == 'Home').astype(int)

# 3. Calcular el total de carreras del partido
df['Total_Carreras'] = df['R'] + df['RA']

# 4. Reconstruir la fecha completa
df['Fecha_texto'] = df['Date'].str.split(', ').str[1] + ' ' + df['Season'].astype(str)

# Quitar el indicador de doble juego, ej: "Apr 19 (1) 2022" -> "Apr 19 2022"
df['Fecha_texto'] = df['Fecha_texto'].str.replace(r'\s*\(\d\)', '', regex=True)

df['Fecha'] = pd.to_datetime(df['Fecha_texto'], format='%b %d %Y')

# 5. Ordenar los partidos cronológicamente por equipo
df = df.sort_values(['Team', 'Fecha']).reset_index(drop=True)

# Revisamos cómo quedó
print("Forma final:", df.shape)
print("\nColumnas nuevas creadas:")
print(df[['Team', 'Fecha', 'Opp', 'Es_Local', 'R', 'RA', 'Total_Carreras', 'Gano']].head(10))

# Guardamos el resultado limpio
df.to_csv('datos_mlb_limpio.csv', index=False)
print("\nGuardado en datos_mlb_limpio.csv")
