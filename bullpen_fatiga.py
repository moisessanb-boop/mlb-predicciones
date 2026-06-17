import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

hoy = pd.Timestamp(datetime.now().date())

equipos_activos = df[df['Season'] == hoy.year]['Team'].unique()

# ---------- Calcular índice de fatiga por equipo ----------
indices = {}
detalle = []

for equipo in equipos_activos:
    grupo = df[(df['Team'] == equipo) & (df['Fecha'] <= hoy)].sort_values('Fecha')
    if grupo.empty:
        continue

    ultimo_partido = grupo['Fecha'].max()
    dias_descanso = (hoy - ultimo_partido).days

    ultimos_2 = grupo.tail(2).copy()
    ultimos_2['Inn'] = pd.to_numeric(ultimos_2['Inn'], errors='coerce').fillna(9)
    entradas_extra = (ultimos_2['Inn'] - 9).clip(lower=0).sum()

    ult_2dias = grupo[grupo['Fecha'] >= hoy - timedelta(days=1)]
    dobles_juegos = int((ult_2dias.groupby('Fecha').size() >= 2).sum())

    indice = entradas_extra + (dobles_juegos * 3) - (dias_descanso * 2)
    indices[equipo] = indice

    detalle.append({
        'Team': equipo,
        'Ultimo_partido': ultimo_partido.date(),
        'Dias_descanso': dias_descanso,
        'Entradas_extra': entradas_extra,
        'Dobles_juegos': dobles_juegos,
        'Indice': indice
    })

df_detalle = pd.DataFrame(detalle).sort_values('Indice', ascending=False).reset_index(drop=True)

# ---------- Top 3 cansado / fresco ----------
print("=== TOP 3 BULLPEN MAS CANSADO ===")
print(df_detalle.head(3).to_string(index=False))
print("\n=== TOP 3 BULLPEN MAS FRESCO ===")
print(df_detalle.tail(3).sort_values('Indice').to_string(index=False))

# ---------- Señal Over/Under para los partidos de HOY ----------
try:
    partidos = pd.read_csv('partidos_hoy.csv')
    pred = pd.read_csv('predicciones_hoy.csv')
except FileNotFoundError:
    print("\nNo hay partidos de hoy todavia. Corre obtener_partidos_hoy.py y predecir_hoy.py primero.")
    exit()

# Extraer total estimado por partido
totales = {}
for _, fila in pred.iterrows():
    llave = f"{fila['Visitante']} @ {fila['Local']}"
    totales[llave] = fila['Total_Estimado']

resultados = []
for _, p in partidos.iterrows():
    local, visitante = p['Local'], p['Visitante']
    fat_l = indices.get(local, 0)
    fat_v = indices.get(visitante, 0)
    fat_comb = fat_l + fat_v
    llave = f"{visitante} @ {local}"
    total_modelo = totales.get(llave, None)

    if fat_comb >= 3:
        senal = 'OVER  ▲'
        nota  = 'Bullpen cansado → más carreras tardías esperadas'
    elif fat_comb <= -6:
        senal = 'UNDER ▼'
        nota  = 'Ambos bullpens frescos → pueden cerrar entradas'
    else:
        senal = 'NEUTRO ─'
        nota  = 'Sin señal clara de bullpen'

    resultados.append({
        'Partido': llave,
        'Bullpen_Local': fat_l,
        'Bullpen_Visit': fat_v,
        'Fat_Comb': fat_comb,
        'Total_Modelo': total_modelo,
        'Señal_Bullpen': senal,
        'Nota': nota
    })

df_r = pd.DataFrame(resultados).sort_values('Fat_Comb', ascending=False).reset_index(drop=True)
df_r.index += 1

print(f"\n=== SEÑAL OVER/UNDER POR BULLPEN — {hoy.date()} ===\n")
print(f"{'#':<3} {'Partido':<18} {'B.Local':>8} {'B.Visit':>8} {'Comb':>6}  {'Total Mod':>10}  {'Señal'}")
print("-" * 72)
for i, row in df_r.iterrows():
    total_str = f"{row['Total_Modelo']:.1f}" if row['Total_Modelo'] is not None else "  N/A"
    print(f"{i:<3} {row['Partido']:<18} {row['Bullpen_Local']:>8.1f} {row['Bullpen_Visit']:>8.1f} {row['Fat_Comb']:>6.1f}  {total_str:>10}  {row['Señal_Bullpen']}")

print("\nLeyenda:")
print("  OVER  ▲ = Bullpen cansado, más carreras tardías esperadas")
print("  UNDER ▼ = Ambos bullpens frescos, pueden cerrar entradas")
print("  NEUTRO ─ = Sin señal clara de bullpen")
print("\nNota: esta señal complementa el Total del modelo, no lo reemplaza.")
print("  Mayor impacto cuando AMBAS señales apuntan en la misma dirección.")

df_r.to_csv('bullpen_fatiga.csv', index=False)
print("\nGuardado en bullpen_fatiga.csv")
