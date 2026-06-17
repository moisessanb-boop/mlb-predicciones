import pandas as pd
import joblib

df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(['Team', 'Fecha'])

partidos = pd.read_csv('partidos_manana.csv')

modelo_ml = joblib.load('modelo_moneyline_v2.pkl')
modelo_tot = joblib.load('modelo_total_v2.pkl')

features = [
    'Es_Local',
    'Prom_Carreras_Anotadas_L10','Prom_Carreras_Permitidas_L10','Pct_Victorias_L10','Diff_Carreras_L10',
    'Prom_Carreras_Anotadas_Temp','Prom_Carreras_Permitidas_Temp','Pct_Victorias_Temp','Racha',
    'Prom_Carreras_Anotadas_L10_Rival','Prom_Carreras_Permitidas_L10_Rival','Pct_Victorias_L10_Rival','Diff_Carreras_L10_Rival',
    'Prom_Carreras_Anotadas_Temp_Rival','Prom_Carreras_Permitidas_Temp_Rival','Pct_Victorias_Temp_Rival','Racha_Rival',
]

def calcular_perfil_actual(equipo):
    datos_equipo = df[df['Team'] == equipo]
    if datos_equipo.empty:
        return None
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
    return perfil

resultados = []
for _, partido in partidos.iterrows():
    local, visitante = partido['Local'], partido['Visitante']

    perfil_local = calcular_perfil_actual(local)
    perfil_visit = calcular_perfil_actual(visitante)

    if perfil_local is None or perfil_visit is None:
        print(f"Sin datos suficientes para {visitante} @ {local}, se omite.")
        continue

    fila = {'Es_Local': 1}
    fila.update(perfil_local)
    for k, v in perfil_visit.items():
        fila[k + '_Rival'] = v

    X_nuevo = pd.DataFrame([fila])[features]

    prob_local = modelo_ml.predict_proba(X_nuevo)[0][1]
    total_estimado = modelo_tot.predict(X_nuevo)[0]

    resultados.append({
        'Visitante': visitante,
        'Local': local,
        f'Prob_{local}': round(prob_local * 100, 1),
        f'Prob_{visitante}': round((1 - prob_local) * 100, 1),
        'Total_Estimado': round(total_estimado, 1)
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
df_resultados.to_csv('predicciones_manana.csv', index=False)
print("\nGuardado en predicciones_manana.csv")
