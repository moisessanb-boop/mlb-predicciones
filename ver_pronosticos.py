import pandas as pd
from datetime import datetime

ARCHIVO_PRED = 'predicciones_hoy.csv'

try:
    pred = pd.read_csv(ARCHIVO_PRED)
except FileNotFoundError:
    print("No hay predicciones todavia. Corre predecir_hoy.py primero.")
    exit()

# Cargar bullpen si existe
try:
    bull = pd.read_csv('bullpen_fatiga.csv')
    tiene_bullpen = True
except FileNotFoundError:
    tiene_bullpen = False

hoy = datetime.now().strftime('%Y-%m-%d')

def recomendar_spread(prob_fav, margen):
    if prob_fav >= 62 and margen >= 3.5: return "-3.5"
    elif prob_fav >= 57 and margen >= 2.5: return "-2.5"
    elif prob_fav >= 53 and margen >= 1.5: return "-1.5"
    else: return "ML only"

filas = []
for _, fila in pred.iterrows():
    local, visitante = fila['Local'], fila['Visitante']
    prob_local = fila[f'Prob_{local}']
    prob_visit = fila[f'Prob_{visitante}']
    margen     = fila.get('Margen_Estimado', 2.5)

    if prob_local >= prob_visit:
        favorito, prob_fav = local, prob_local
        underdog, prob_und = visitante, prob_visit
    else:
        favorito, prob_fav = visitante, prob_visit
        underdog, prob_und = local, prob_local

    partido_key = f"{visitante} @ {local}"
    spread      = recomendar_spread(prob_fav, margen)

    # Bullpen
    alineacion = 'NEUTRO     ─'
    if tiene_bullpen:
        bull_row = bull[bull['Partido'] == partido_key]
        if not bull_row.empty:
            b     = bull_row.iloc[0]
            fat_l = b['Bullpen_Local']
            fat_v = b['Bullpen_Visit']
            fat_fav = fat_l if favorito == local else fat_v
            fat_dog = fat_v if favorito == local else fat_l
            if fat_fav < fat_dog:
                alineacion = 'ALINEADO  ✓'
            elif fat_fav > fat_dog:
                alineacion = 'CONTRADICE ✗'

    filas.append({
        'Partido':         partido_key,
        'Seleccion':       favorito,
        'Probabilidad':    prob_fav,
        'Underdog':        underdog,
        'Prob_Dog':        prob_und,
        'Total_Estimado':  fila['Total_Estimado'],
        'Margen_Estimado': round(margen, 2),
        'Spread':          spread,
        'Bullpen_ML':      alineacion,
    })

df_out = pd.DataFrame(filas).sort_values('Probabilidad', ascending=False).reset_index(drop=True)
df_out.index += 1

print(f"\n=== PRONOSTICOS {hoy} ({len(df_out)} partidos) ===\n")
print(f"{'#':<3} {'Partido':<18} {'Fav':<6} {'Prob':>6}  {'Dog':<6} {'Prob':>6}  {'Total':>6}  {'Spread':<10}  {'Bullpen'}")
print("-" * 80)
for i, row in df_out.iterrows():
    marca = '★' if i <= 5 else ' '
    print(f"{marca}{i:<2} {row['Partido']:<18} {row['Seleccion']:<6} {row['Probabilidad']:>5.1f}%  "
          f"{row['Underdog']:<6} {row['Prob_Dog']:>5.1f}%  {row['Total_Estimado']:>5.1f}  "
          f"{row['Spread']:<10}  {row['Bullpen_ML']}")

# Guardar top_selecciones.csv con bullpen
df_out[['Partido','Seleccion','Probabilidad','Total_Estimado',
        'Margen_Estimado','Spread','Bullpen_ML']].to_csv('top_selecciones.csv', index=False)
print("\ntop_selecciones.csv guardado con Bullpen_ML")
