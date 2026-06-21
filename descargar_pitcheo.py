from pybaseball.datasources.bref import BRefSession
import pybaseball as pyb
pyb.cache.disable()
import pandas as pd
from io import StringIO
import time

session = BRefSession()

def descargar_pitcheo_equipo(season, team):
    url = f"https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=p&year={season}"
    content = session.get(url).content.decode('utf-8')
    content_sin_comentarios = content.replace('<!--', '').replace('-->', '')

    tablas = pd.read_html(StringIO(content_sin_comentarios))
    df = tablas[1]

    df.columns = [c[1] for c in df.columns]
    df = df.rename(columns={'Unnamed: 3_level_1': 'Home_Marker'})

    df['Gtm'] = pd.to_numeric(df['Gtm'], errors='coerce')
    df = df.dropna(subset=['Gtm']).copy()

    cols_numericas = ['IP', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'RS', 'RA']
    for c in cols_numericas:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    def convertir_ip(ip):
        entero = int(ip)
        decimal = round(ip - entero, 1)
        extra = {0.0: 0, 0.1: 1, 0.2: 2}.get(decimal, 0)
        return entero + extra / 3

    df['IP_dec'] = df['IP'].apply(convertir_ip)

    df['K9']   = df['SO'] * 9 / df['IP_dec']
    df['BB9']  = df['BB'] * 9 / df['IP_dec']
    df['HR9']  = df['HR'] * 9 / df['IP_dec']
    df['WHIP'] = (df['BB'] + df['H']) / df['IP_dec']

    df['Season'] = season
    df['Team']   = team

    df = df.sort_values('Gtm').reset_index(drop=True)
    df['Game_Seq'] = range(1, len(df) + 1)

    # Columna clave: Pitchers Used (para extraer el abridor)
    col_pitchers = [c for c in df.columns if 'Pitchers' in str(c) or 'pitchers' in str(c).lower()]
    col_pitchers = col_pitchers[0] if col_pitchers else None

    cols_guardar = ['Team','Season','Game_Seq','Date','Opp',
                    'IP_dec','H','ER','BB','SO','HR','K9','BB9','HR9','WHIP']
    if col_pitchers:
        df = df.rename(columns={col_pitchers: 'Pitchers_Used'})
        cols_guardar.append('Pitchers_Used')

    return df[cols_guardar]

equipos = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
]
temporadas = [2022, 2023, 2024, 2025, 2026]

todas_las_filas = []
total    = len(equipos) * len(temporadas)
contador = 0

for temporada in temporadas:
    for equipo in equipos:
        contador += 1
        print(f"[{contador}/{total}] Descargando pitcheo {equipo} - {temporada}...")
        try:
            datos = descargar_pitcheo_equipo(temporada, equipo)
            todas_las_filas.append(datos)
        except Exception as e:
            print(f"  -> Error con {equipo} {temporada}: {e}")
        time.sleep(1.5)

df = pd.concat(todas_las_filas, ignore_index=True)
print(f"\nTotal filas: {len(df)}")
print(f"Columnas: {df.columns.tolist()}")
df.to_csv('datos_pitcheo_crudo.csv', index=False)
print("Guardado en datos_pitcheo_crudo.csv")
