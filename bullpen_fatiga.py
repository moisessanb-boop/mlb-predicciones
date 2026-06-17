import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('datos_mlb_limpio.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Inn'] = pd.to_numeric(df['Inn'], errors='coerce').fillna(9)
df['R']   = pd.to_numeric(df['R'],   errors='coerce').fillna(0)
df['RA']  = pd.to_numeric(df['RA'],  errors='coerce').fillna(0)

hoy = pd.Timestamp(datetime.now().date())
equipos_activos = df[df['Season'] == hoy.year]['Team'].unique()

def calcular_indice(grupo, hoy):
    ultimo = grupo['Fecha'].max()
    dias_descanso = (hoy - ultimo).days
    ultimos_4 = grupo.sort_values('Fecha').tail(4).copy()
    ultimos_4['Margen'] = abs(ultimos_4['R'] - ultimos_4['RA'])
    fatiga = 0
    for _, g in ultimos_4.iterrows():
        extras = max(0, g['Inn'] - 9)
        margen = g['Margen']
        if margen <= 1:   peso = 3
        elif margen <= 2: peso = 2
        elif margen <= 4: peso = 1
        else:             peso = 0
        fatiga += extras * 2 + peso
    fatiga -= dias_descanso * 2
    return fatiga, ultimo.date(), dias_descanso

indices = {}
detalle = []
for equipo in equipos_activos:
    grupo = df[(df['Team'] == equipo) & (df['Fecha'] <= hoy)].sort_values('Fecha')
    if len(grupo) < 2:
        continue
    indice, ultimo, dias = calcular_indice(grupo, hoy)
    indices[equipo] = indice
    detalle.append({'Team': equipo, 'Ultimo_partido': ultimo,
                    'Dias_descanso': dias, 'Indice': indice})

df_detalle = pd.DataFrame(detalle).sort_values('Indice', ascending=False).reset_index(drop=True)

print("=== TOP 3 BULLPEN MAS CANSADO (juegos recientes muy cerrados) ===")
print(df_detalle.head(3).to_string(index=False))
print("\n=== TOP 3 BULLPEN MAS FRESCO (goleadas o descanso reciente) ===")
print(df_detalle.tail(3).sort_values('Indice').to_string(index=False))

try:
    partidos = pd.read_csv('partidos_hoy.csv')
    pred     = pd.read_csv('predicciones_hoy.csv')
except FileNotFoundError:
    print("\nNo hay partidos de hoy. Corre obtener_partidos_hoy.py y predecir_hoy.py primero.")
    exit()

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

    if fat_comb >= 6:
        senal = 'OVER  ▲'
    elif fat_comb <= -2:
        senal = 'UNDER ▼'
    else:
        senal = 'NEUTRO ─'

    resultados.append({
        'Partido':       llave,
        'Bullpen_Local': fat_l,
        'Bullpen_Visit': fat_v,
        'Fat_Comb':      fat_comb,
        'Total_Modelo':  total_modelo,
        'Señal_Bullpen': senal,
    })

df_r = pd.DataFrame(resultados).sort_values('Fat_Comb', ascending=False).reset_index(drop=True)
df_r.index += 1

print(f"\n=== SEÑAL OVER/UNDER POR BULLPEN — {hoy.date()} ===\n")
print(f"{'#':<3} {'Partido':<18} {'B.Local':>8} {'B.Visit':>8} {'Comb':>6}  {'Total Mod':>10}  {'Señal'}")
print("-" * 72)
for i, row in df_r.iterrows():
    total_str = f"{row['Total_Modelo']:.1f}" if row['Total_Modelo'] is not None else "  N/A"
    print(f"{i:<3} {row['Partido']:<18} {row['Bullpen_Local']:>8.1f} "
          f"{row['Bullpen_Visit']:>8.1f} {row['Fat_Comb']:>6.1f}  "
          f"{total_str:>10}  {row['Señal_Bullpen']}")

print("\nLeyenda:")
print("  OVER  ▲ = Ambos bullpens exigidos recientemente (juegos cerrados)")
print("  UNDER ▼ = Bullpens descansados o con victorias/derrotas holgadas")
print("  NEUTRO ─ = Sin señal clara")
print("\nNota: esta señal complementa el Total del modelo, no lo reemplaza.")

df_r.to_csv('bullpen_fatiga.csv', index=False)
print("\nGuardado en bullpen_fatiga.csv")

TOP_N = 2
over   = df_r[df_r['Señal_Bullpen'] == 'OVER  ▲'].head(TOP_N)
under  = df_r[df_r['Señal_Bullpen'] == 'UNDER ▼'].head(TOP_N)
neutro = df_r[df_r['Señal_Bullpen'] == 'NEUTRO ─'].head(TOP_N)

print("\n=== RESUMEN DE SEÑALES BULLPEN ===\n")
for titulo, grupo in [
    ('OVER  ▲  (bullpens exigidos → más carreras esperadas)', over),
    ('UNDER ▼  (bullpens frescos → menos carreras esperadas)', under),
    ('NEUTRO ─  (sin señal clara)', neutro),
]:
    print(f"  {titulo}")
    if grupo.empty:
        print("    Sin partidos con esta señal hoy")
    else:
        for _, row in grupo.iterrows():
            print(f"    {row['Partido']:<18}  Comb: {row['Fat_Comb']:>5.1f}  Total modelo: {row['Total_Modelo']:.1f}")
    print()
