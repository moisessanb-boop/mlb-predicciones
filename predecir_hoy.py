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
vars_nuevas = ['PF_norm','SP_Rest_Local','SP_GS_Local','SP_Rest_Visit','SP_GS_Visit']
features = ['Es_Local'] + base_v2 + [c+'_Rival' for c in base_v2] + vars_nuevas

# ============================================================
# CARGAR DATOS
# ============================================================
df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(['Team', 'Fecha'])

df_pitch = pd.read_csv('datos_pitcheo_crudo.csv')
df_pitch['Fecha'] = pd.to_datetime(
    df_pitch['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True))
df_pitch = df_pitch.sort_values(['Team','Fecha'])

partidos  = pd.read_csv('partidos_hoy.csv')
modelo_ml  = joblib.load('modelo_moneyline_v2.pkl')
modelo_tot = joblib.load('modelo_total_v2.pkl')

hoy = pd.Timestamp(datetime.now().date())

# ============================================================
# FUNCIONES DE PERFIL
# ============================================================
def calcular_perfil_actual(equipo):
    datos_equipo    = df[df['Team'] == equipo]
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

def calcular_sp_perfil(equipo):
    """GameScore promedio de los últimos 5 juegos del equipo (proxy del abridor)."""
    datos_sp = df_pitch[
        (df_pitch['Team'] == equipo) & (df_pitch['Fecha'] < hoy)
    ].tail(5)
    if datos_sp.empty or 'K9' not in datos_sp.columns:
        return {'SP_GS': 50.0, 'SP_Rest': 5.0}
    # Aproximar GameScore desde K9 y ERA del equipo (proxy razonable)
    gs_proxy = (datos_sp['K9'] * 3 + (9 - datos_sp['BB9']) * 2).mean()
    gs_proxy = max(20, min(80, gs_proxy))  # clip en rango razonable
    return {'SP_GS': gs_proxy, 'SP_Rest': 5.0}

# ============================================================
# GENERAR PREDICCIONES
# ============================================================
resultados = []
for _, partido in partidos.iterrows():
    local, visitante = partido['Local'], partido['Visitante']

    perfil_local = calcular_perfil_actual(local)
    perfil_visit = calcular_perfil_actual(visitante)
    if perfil_local is None or perfil_visit is None:
        print(f"Sin datos para {visitante} @ {local}")
        continue

    sp_local = calcular_sp_perfil(local)
    sp_visit = calcular_sp_perfil(visitante)

    # Factor de parque (aplica al estadio del equipo LOCAL)
    pf = PARK_FACTORS.get(local, 100) / 100.0

    fila = {'Es_Local': 1}
    fila.update(perfil_local)
    for k, v in perfil_visit.items():
        fila[k + '_Rival'] = v
    fila['PF_norm']        = pf
    fila['SP_Rest_Local']  = sp_local['SP_Rest']
    fila['SP_GS_Local']    = sp_local['SP_GS']
    fila['SP_Rest_Visit']  = sp_visit['SP_Rest']
    fila['SP_GS_Visit']    = sp_visit['SP_GS']

    X_nuevo = pd.DataFrame([fila])[features]

    prob_local    = modelo_ml.predict_proba(X_nuevo)[0][1]
    total_estimado = modelo_tot.predict(X_nuevo)[0]

    resultados.append({
        'Visitante':             visitante,
        'Local':                 local,
        f'Prob_{local}':         round(prob_local * 100, 1),
        f'Prob_{visitante}':     round((1 - prob_local) * 100, 1),
        'Total_Estimado':        round(total_estimado, 1)
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
df_resultados.to_csv('predicciones_hoy.csv', index=False)
print("\nGuardado en predicciones_hoy.csv")
