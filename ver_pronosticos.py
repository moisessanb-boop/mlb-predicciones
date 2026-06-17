import pandas as pd
from datetime import datetime

ARCHIVO_PRED = 'predicciones_hoy.csv'

try:
    df = pd.read_csv(ARCHIVO_PRED)
except FileNotFoundError:
    print("No hay predicciones todavia. Corre predecir_hoy.py primero.")
    exit()

hoy = datetime.now().strftime('%Y-%m-%d')

filas = []
for _, fila in df.iterrows():
    local, visitante = fila['Local'], fila['Visitante']
    prob_local = fila[f'Prob_{local}']
    prob_visit = fila[f'Prob_{visitante}']

    if prob_local >= prob_visit:
        favorito, prob_fav = local, prob_local
        underdog, prob_und = visitante, prob_visit
    else:
        favorito, prob_fav = visitante, prob_visit
        underdog, prob_und = local, prob_local

    filas.append({
        'Partido': f"{visitante} @ {local}",
        'Favorito': favorito,
        'Prob_Fav': prob_fav,
        'Underdog': underdog,
        'Prob_Und': prob_und,
        'Total_Est': fila['Total_Estimado']
    })

df_out = pd.DataFrame(filas).sort_values('Prob_Fav', ascending=False).reset_index(drop=True)
df_out.index += 1

print(f"\n=== PRONOSTICOS {hoy} ({len(df_out)} partidos) ===\n")
print(f"{'#':<3} {'Partido':<18} {'Favorito':<10} {'Prob':>6}  {'Underdog':<10} {'Prob':>6}  {'Total':>6}")
print("-" * 65)
for i, row in df_out.iterrows():
    # Marcar los top 5 con una estrella
    marca = '★' if i <= 5 else ' '
    print(f"{marca}{i:<3} {row['Partido']:<18} {row['Favorito']:<10} {row['Prob_Fav']:>5.1f}%  {row['Underdog']:<10} {row['Prob_Und']:>5.1f}%  {row['Total_Est']:>5.1f}")

print("\n★ = Top 5 picks (mayor probabilidad)")

# Guardar top_selecciones.csv para registrar_prediccion.py
df_out['Seleccion'] = df_out['Favorito']
df_out['Probabilidad'] = df_out['Prob_Fav']
df_out[['Partido','Seleccion','Probabilidad','Total_Est']].rename(
    columns={'Total_Est':'Total_Estimado'}).to_csv('top_selecciones.csv', index=False)
