import pandas as pd
import joblib

# ====================================================
# CONFIGURA AQUI EL PARTIDO QUE QUIERES PREDECIR
# Usa las abreviaciones de Baseball Reference, ej:
# ARI, ATL, BAL, BOS, CHC, CHW, CIN, CLE, COL, DET, HOU,
# KCR, LAA, LAD, MIA, MIL, MIN, NYM, NYY, OAK, PHI, PIT,
# SDP, SEA, SFG, STL, TBR, TEX, TOR, WSN
# ====================================================
EQUIPO_LOCAL = 'NYY'
EQUIPO_VISITANTE = 'BOS'
# ====================================================

df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(['Team', 'Fecha'])

def calcular_perfil_actual(equipo):
    datos_equipo = df[df['Team'] == equipo]
    if datos_equipo.empty:
        raise ValueError(f"No hay datos para el equipo '{equipo}'")

    temporada_actual = datos_equipo['Season'].max()
    datos_temporada = datos_equipo[datos_equipo['Season'] == temporada_actual]
    ultimos_10 = datos_temporada.tail(10)

    perfil = {
        'Prom_Carreras_Anotadas_L10': ultimos_10['R'].mean(),
        'Prom_Carreras_Permitidas_L10': ultimos_10['RA'].mean(),
        'Pct_Victorias_L10': ultimos_10['Gano'].mean(),
        'Prom_Carreras_Anotadas_Temp': datos_temporada['R'].mean(),
        'Prom_Carreras_Permitidas_Temp': datos_temporada['RA'].mean(),
        'Pct_Victorias_Temp': datos_temporada['Gano'].mean(),
        'Racha': datos_temporada['Streak'].iloc[-1],
    }
    perfil['Diff_Carreras_L10'] = perfil['Prom_Carreras_Anotadas_L10'] - perfil['Prom_Carreras_Permitidas_L10']
    return perfil, temporada_actual, datos_temporada['Fecha'].max()

perfil_local, temp_local, fecha_local = calcular_perfil_actual(EQUIPO_LOCAL)
perfil_visit, temp_visit, fecha_visit = calcular_perfil_actual(EQUIPO_VISITANTE)

print(f"Forma actual de {EQUIPO_LOCAL}: temporada {temp_local}, datos hasta {fecha_local.date()}")
print(f"Forma actual de {EQUIPO_VISITANTE}: temporada {temp_visit}, datos hasta {fecha_visit.date()}")

# Construir la fila de variables (perspectiva del equipo LOCAL)
fila = {'Es_Local': 1}
fila.update(perfil_local)
for k, v in perfil_visit.items():
    fila[k + '_Rival'] = v

features = [
    'Es_Local',
    'Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
    'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha',
    'Prom_Carreras_Anotadas_L10_Rival','Prom_Carreras_Permitidas_L10_Rival','Pct_Victorias_L10_Rival','Diff_Carreras_L10_Rival',
    'Prom_Carreras_Anotadas_Temp_Rival','Prom_Carreras_Permitidas_Temp_Rival','Pct_Victorias_Temp_Rival','Racha_Rival',
]
X_nuevo = pd.DataFrame([fila])[features]

modelo_ml = joblib.load('modelo_moneyline_v2.pkl')
modelo_tot = joblib.load('modelo_total_v2.pkl')

prob_local = modelo_ml.predict_proba(X_nuevo)[0][1]
total_estimado = modelo_tot.predict(X_nuevo)[0]

print(f"\n=== {EQUIPO_LOCAL} (local) vs {EQUIPO_VISITANTE} (visitante) ===")
print(f"Probabilidad de victoria {EQUIPO_LOCAL}: {prob_local:.1%}")
print(f"Probabilidad de victoria {EQUIPO_VISITANTE}: {1 - prob_local:.1%}")
print(f"Total de carreras estimado: {total_estimado:.1f}")
