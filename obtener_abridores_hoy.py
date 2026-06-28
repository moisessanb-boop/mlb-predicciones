import requests
import pandas as pd
from datetime import datetime

hoy = datetime.now().strftime('%Y-%m-%d')
temporada = datetime.now().year

print(f"Obteniendo abridores probables para {hoy}...")

# ===== 1. Obtener abridores probables =====
url_schedule = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}&hydrate=probablePitcher,team"
headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}

resp = requests.get(url_schedule, headers=headers, timeout=15)
partidos = resp.json().get('dates', [{}])[0].get('games', [])

abridores = {}  # {team_abbr: {nombre, id}}
ABREV_MLB = {
    'AZ':'ARI','ARI':'ARI','ATL':'ATL','BAL':'BAL','BOS':'BOS','CHC':'CHC',
    'CWS':'CHW','CIN':'CIN','CLE':'CLE','COL':'COL','DET':'DET',
    'HOU':'HOU','KC': 'KCR','LAA':'LAA','LAD':'LAD','MIA':'MIA',
    'MIL':'MIL','MIN':'MIN','NYM':'NYM','NYY':'NYY','ATH':'ATH',
    'PHI':'PHI','PIT':'PIT','SD': 'SDP','SEA':'SEA','SF': 'SFG',
    'STL':'STL','TB': 'TBR','TEX':'TEX','TOR':'TOR','WSH':'WSN',
}

for p in partidos:
    for lado in ['away','home']:
        equipo = p['teams'][lado]
        abbr_mlb = equipo['team'].get('abbreviation','')
        abbr_br  = ABREV_MLB.get(abbr_mlb, abbr_mlb)
        sp       = equipo.get('probablePitcher', {})
        abridores[abbr_br] = {
            'nombre': sp.get('fullName', 'TBD'),
            'id':     sp.get('id', None)
        }

print(f"Equipos con abridor: {len(abridores)}")

# ===== 2. Descargar stats de todos los pitchers =====
print("Descargando stats de pitchers 2026...")
url_stats = "https://statsapi.mlb.com/api/v1/stats"
params = {
    'stats': 'season', 'group': 'pitching',
    'season': temporada, 'playerPool': 'All',
    'limit': 1000, 'sortStat': 'inningsPitched', 'order': 'desc'
}
resp_stats = requests.get(url_stats, params=params, headers=headers, timeout=15)
splits = resp_stats.json().get('stats', [{}])[0].get('splits', [])

# Crear lookup: nombre -> stats
stats_lookup = {}
for s in splits:
    nombre = s['player']['fullName']
    st     = s['stat']
    gs     = int(st.get('gamesStarted', 0))
    if gs >= 1:  # solo abridores
        stats_lookup[nombre] = {
            'ERA':  float(st.get('era',  4.50) or 4.50),
            'WHIP': float(st.get('whip', 1.30) or 1.30),
            'K9':   float(st.get('strikeoutsPer9Inn', 8.0) or 8.0),
            'IP':   float(st.get('inningsPitched', 0) or 0),
            'GS':   gs
        }

print(f"Pitchers con stats: {len(stats_lookup)}")

# ===== 3. Cruzar abridores con stats =====
resultados = []
DEFAULTS = {'ERA': 4.50, 'WHIP': 1.30, 'K9': 8.0, 'IP': 0, 'GS': 0}

for equipo, sp in abridores.items():
    nombre = sp['nombre']
    if nombre == 'TBD' or nombre not in stats_lookup:
        # Buscar por apellido
        apellido = nombre.split()[-1] if nombre != 'TBD' else ''
        match = [k for k in stats_lookup if apellido and apellido in k]
        if match:
            stats = stats_lookup[match[0]]
            nombre_encontrado = match[0]
        else:
            stats = DEFAULTS.copy()
            nombre_encontrado = 'N/A (usando defaults)'
    else:
        stats = stats_lookup[nombre]
        nombre_encontrado = nombre

    resultados.append({
        'Team':   equipo,
        'SP':     nombre,
        'SP_ERA': stats['ERA'],
        'SP_WHIP': stats['WHIP'],
        'SP_K9':  stats['K9'],
        'SP_IP':  stats['IP'],
        'SP_GS':  stats['GS'],
    })

df = pd.DataFrame(resultados)
df.to_csv('abridores_hoy.csv', index=False)

print("\nAbridores de hoy:")
print(df[['Team','SP','SP_ERA','SP_WHIP','SP_K9']].to_string(index=False))
print("\nGuardado en abridores_hoy.csv")
