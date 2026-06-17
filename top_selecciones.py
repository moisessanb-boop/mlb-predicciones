import pandas as pd

ARCHIVO = 'predicciones_hoy.csv'  # cambia a 'predicciones_manana.csv' si corres esa version

df = pd.read_csv(ARCHIVO)

selecciones = []
for _, fila in df.iterrows():
    col_local = f"Prob_{fila['Local']}"
    col_visit = f"Prob_{fila['Visitante']}"

    if fila[col_local] >= fila[col_visit]:
        equipo_favorito = fila['Local']
        prob_favorito = fila[col_local]
        rival = fila['Visitante']
    else:
        equipo_favorito = fila['Visitante']
        prob_favorito = fila[col_visit]
        rival = fila['Local']

    selecciones.append({
        'Partido': f"{fila['Visitante']} @ {fila['Local']}",
        'Seleccion': equipo_favorito,
        'Probabilidad': prob_favorito,
        'Total_Estimado': fila['Total_Estimado']
    })

df_selecciones = pd.DataFrame(selecciones).sort_values('Probabilidad', ascending=False)

print("=== TOP 5 SELECCIONES MONEYLINE (mayor probabilidad) ===\n")
print(df_selecciones.head(5).to_string(index=False))

df_selecciones.to_csv('top_selecciones.csv', index=False)
print("\nGuardado en top_selecciones.csv (todas las selecciones, ordenadas)")
