import pybaseball as pyb
import pandas as pd
from datetime import datetime
import time

equipos = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
]

hoy = datetime.now()
temporada = hoy.year
fecha_busqueda = hoy.strftime('%b %-d')

print(f"Buscando partidos para: {hoy.date()} (formato BR: '{fecha_busqueda}')")

partidos = []
for equipo in equipos:
    print(f"Revisando calendario de {equipo}...")
    try:
        cal = pyb.schedule_and_record(temporada, equipo)
        cal['Date_simple'] = cal['Date'].str.split(', ').str[1].str.replace(r'\s*\(\d+\)', '', regex=True)
        encontrados = cal[cal['Date_simple'] == fecha_busqueda]
        for _, fila in encontrados.iterrows():
            if fila['Home_Away'] == 'Home':
                partidos.append({'Local': equipo, 'Visitante': fila['Opp'], 'Fecha': fila['Date']})
    except Exception as e:
        print(f"  -> Error con {equipo}: {e}")
    time.sleep(1)

print(f"\nPartidos encontrados para {hoy.date()}:")
for p in partidos:
    print(f"  {p['Visitante']} @ {p['Local']}  ({p['Fecha']})")

pd.DataFrame(partidos).to_csv('partidos_hoy.csv', index=False)
print("\nGuardado en partidos_hoy.csv")
