import pybaseball as pyb
pyb.cache.disable()  # siempre datos frescos de Baseball Reference

import pandas as pd
import time

equipos = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
]
temporadas_nuevas = [2025, 2026]

todas_las_filas = []
total = len(equipos) * len(temporadas_nuevas)
contador = 0

for temporada in temporadas_nuevas:
    for equipo in equipos:
        contador += 1
        print(f"[{contador}/{total}] Descargando {equipo} - {temporada}...")
        try:
            datos = pyb.schedule_and_record(temporada, equipo)
            datos['Season'] = temporada
            datos['Team'] = equipo
            todas_las_filas.append(datos)
        except Exception as e:
            print(f"  -> Error con {equipo} {temporada}: {e}")
        time.sleep(1)

df_nuevo = pd.concat(todas_las_filas, ignore_index=True)
print(f"\nFilas descargadas (2025-2026, antes de filtrar): {len(df_nuevo)}")

df_nuevo = df_nuevo.dropna(subset=['R', 'RA']).copy()
print(f"Filas con resultado (partidos ya jugados): {len(df_nuevo)}")

df_nuevo['Gano'] = df_nuevo['W/L'].str.startswith('W').astype(int)
df_nuevo['Es_Local'] = (df_nuevo['Home_Away'] == 'Home').astype(int)
df_nuevo['Total_Carreras'] = df_nuevo['R'] + df_nuevo['RA']
df_nuevo['Fecha_texto'] = df_nuevo['Date'].str.split(', ').str[1] + ' ' + df_nuevo['Season'].astype(str)
df_nuevo['Fecha_texto'] = df_nuevo['Fecha_texto'].str.replace(r'\s*\(\d\)', '', regex=True)
df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha_texto'], format='%b %d %Y')

df_anterior = pd.read_csv('datos_mlb_limpio.csv')
df_anterior['Fecha'] = pd.to_datetime(df_anterior['Fecha'])

df_total = pd.concat([df_anterior, df_nuevo], ignore_index=True)
df_total = df_total.sort_values(['Team', 'Fecha']).reset_index(drop=True)
df_total = df_total.drop_duplicates(subset=['Team', 'Fecha', 'Opp'], keep='last')

print(f"\nFilas totales (2022-2026): {len(df_total)}")
print(df_total['Season'].value_counts().sort_index())

df_total.to_csv('datos_mlb_limpio.csv', index=False)
print("\ndatos_mlb_limpio.csv actualizado con 2025-2026")
