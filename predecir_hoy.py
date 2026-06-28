import pandas as pd
import joblib
from datetime import datetime

# ============================================================
# CONFIGURACION
# ============================================================
PARK_FACTORS = {
    'ARI': 100, 'ATL': 98,  'BAL': 103, 'BOS': 104, 'CHC': 101,
    'CHW': 99,  'CIN': 107, 'CLE': 100, 'COL': 118, 'DET': 99,
    'HOU': 100, 'KCR': 100, 'LAA': 97,  'LAD': 96,  'MIA': 91,
    'MIL': 99,  'MIN': 102, 'NYM': 100, 'NYY': 106, 'ATH': 97,
    'PHI': 103, 'PIT': 101, 'SDP': 93,  'SEA': 95,  'SFG': 95,
    'STL': 99,  'TBR': 100, 'TEX': 105, 'TOR': 102, 'WSN': 99,
}

base_v2 = [
    'Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10',
    'Pct_Victorias_L10','Diff_Carreras_L10',
    'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp',
    'Pct_Victorias_Temp','Racha'
]
vars_v4 = ['PF_norm','SP_Rest_Local','SP_GS_Local','SP_Rest_Visit','SP_GS_Visit']
vars_v5 = ['SP_ERA_Local','SP_WHIP_Local','SP_K9_Local',
           'SP_ERA_Visit','SP_WHIP_Visit','SP_K9_Visit']
features = (['Es_Local'] + base_v2 + [c+'_Rival' for c in base_v2] + vars_v4 + vars_v5)

# ============================================================
# CARGAR DATOS
# ============================================================
df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(['Team', 'Fecha'])

df_pitch = pd.read_csv('datos_pitcheo_crudo.csv')
df_pitch['Fecha'] = pd.to_datetime(
    df_pitch['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True), errors='coerce')
df_pitch = df_pitch.sort_values(['Team','Fecha'])

partidos  = pd.read_csv('partidos_hoy.csv')
modelo_ml     = joblib.load('modelo_moneyline_v2.pkl')
modelo_tot    = joblib.load('modelo_total_v2.pkl')
modelo_margen = joblib.load('modelo_margen.pkl')

# Cargar abridores de hoy (con stats individuales)
try:
    abridores = pd.read_csv('abridores_hoy.csv')
    sp_hoy = {row['Team']: row for _, row in abridores.iterrows()}
    print(f"Abridores cargados: {len(sp_hoy)}")
except FileNotFoundError:
    sp_hoy = {}
    print("No hay abridores_hoy.csv - usando defaults")

hoy = pd.Timestamp(datetime.now().date())

# ============================================================
# PERFILES
# ============================================================
def calcular_perfil_actual(equipo):
    datos_equipo = df[df['Team'] == equipo]
    if datos_equipo.empty:
        return None
    temporada_actual = datos_equipo['Season'].max()
    datos_temporada  = datos_equipo[datos_equipo['Season'] == temporada_actual]
    datos_antes_hoy  = datos_temporada[datos_temporada['Fecha'] < hoy]
    if datos_antes_hoy.empty:
        return None
    ultimos_10 = datos_antes_hoy.tail(10)
    perfil = {
        'Prom_Carreras_Anotadas_L10':    ultimos_10['R'].mean(),
        'Prom_Carreras_Permitidas_L10':  ultimos_10['RA'].mean(),
        'Pct_Victorias_L10':             ultimos_10['Gano'].mean(),
        'Prom_Carreras_Anotadas_Temp':   datos_antes_hoy['R'].mean(),
        'Prom_Carreras_Permitidas_Temp': datos_antes_hoy['RA'].mean(),
        'Pct_Victorias_Temp':            datos_antes_hoy['Gano'].mean(),
        'Racha':                         datos_antes_hoy['Streak'].iloc[-1],
    }
    perfil['Diff_Carreras_L10'] = perfil['Prom_Carreras_Anotadas_L10'] - perfil['Prom_Carreras_Permitidas_L10']
    return perfil

def stats_abridor(equipo):
    """Stats individuales del abridor de hoy desde abridores_hoy.csv."""
    if equipo in sp_hoy:
        r = sp_hoy[equipo]
        return {'ERA': r['SP_ERA'], 'WHIP': r['SP_WHIP'], 'K9': r['SP_K9']}
    return {'ERA': 4.50, 'WHIP': 1.30, 'K9': 8.0}

def gs_proxy(equipo):
    """GameScore proxy del equipo (para vars_v4)."""
    datos_sp = df_pitch[(df_pitch['Team'] == equipo) & (df_pitch['Fecha'] < hoy)].tail(5)
    if datos_sp.empty or 'K9' not in datos_sp.columns:
        return 50.0
    gs = (datos_sp['K9'] * 3 + (9 - datos_sp['BB9']) * 2).mean()
    return max(20, min(80, gs))

# ============================================================
# PREDECIR
# ============================================================
resultados = []
for _, partido in partidos.iterrows():
    local, visitante = partido['Local'], partido['Visitante']
    perfil_local = calcular_perfil_actual(local)
    perfil_visit = calcular_perfil_actual(visitante)
    if perfil_local is None or perfil_visit is None:
        print(f"Sin datos para {visitante} @ {local}")
        continue

    sp_local = stats_abridor(local)
    sp_visit = stats_abridor(visitante)
    pf = PARK_FACTORS.get(local, 100) / 100.0

    fila = {'Es_Local': 1}
    fila.update(perfil_local)
    for k, v in perfil_visit.items():
        fila[k + '_Rival'] = v
    fila['PF_norm']       = pf
    fila['SP_Rest_Local'] = 5.0
    fila['SP_GS_Local']   = gs_proxy(local)
    fila['SP_Rest_Visit'] = 5.0
    fila['SP_GS_Visit']   = gs_proxy(visitante)
    fila['SP_ERA_Local']  = sp_local['ERA']
    fila['SP_WHIP_Local'] = sp_local['WHIP']
    fila['SP_K9_Local']   = sp_local['K9']
    fila['SP_ERA_Visit']  = sp_visit['ERA']
    fila['SP_WHIP_Visit'] = sp_visit['WHIP']
    fila['SP_K9_Visit']   = sp_visit['K9']

    X_nuevo = pd.DataFrame([fila])[features]

    prob_local      = modelo_ml.predict_proba(X_nuevo)[0][1]
    total_estimado  = modelo_tot.predict(X_nuevo)[0]
    margen_estimado = modelo_margen.predict(X_nuevo)[0]

    resultados.append({
        'Visitante':         visitante,
        'Local':             local,
        f'Prob_{local}':     round(prob_local * 100, 1),
        f'Prob_{visitante}': round((1 - prob_local) * 100, 1),
        'Total_Estimado':    round(total_estimado, 1),
        'Margen_Estimado':   round(margen_estimado, 2),
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
df_resultados.to_csv('predicciones_hoy.csv', index=False)
print("\nGuardado en predicciones_hoy.csv")
