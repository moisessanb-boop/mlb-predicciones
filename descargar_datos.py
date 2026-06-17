import pybaseball as pyb
import pandas as pd
import time

# Lista de los 30 equipos de MLB (abreviaciones usadas por Baseball Reference)
equipos = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
]

temporadas = [2022, 2023, 2024]

todas_las_filas = []
total_descargas = len(equipos) * len(temporadas)
contador = 0

for temporada in temporadas:
    for equipo in equipos:
        contador += 1
        print(f"[{contador}/{total_descargas}] Descargando {equipo} - {temporada}...")
        try:
            datos = pyb.schedule_and_record(temporada, equipo)
            datos['Season'] = temporada
            datos['Team'] = equipo
            todas_las_filas.append(datos)
        except Exception as e:
            print(f"  -> Error con {equipo} {temporada}: {e}")
        time.sleep(1)  # pausa de 1 segundo para no saturar el sitio web

# Juntamos todas las tablas en una sola
df_completo = pd.concat(todas_las_filas, ignore_index=True)

print(f"\nListo. Total de filas descargadas: {len(df_completo)}")

# Guardamos en un archivo CSV
df_completo.to_csv('datos_mlb_crudo.csv', index=False)
print("Guardado en datos_mlb_crudo.csv")
