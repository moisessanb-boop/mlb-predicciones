import pandas as pd
from datetime import datetime

try:
    pred = pd.read_csv('predicciones_hoy.csv')
    bull = pd.read_csv('bullpen_fatiga.csv')
except FileNotFoundError as e:
    print(f"Archivo no encontrado: {e}")
    print("Corre primero: predecir_hoy.py y bullpen_fatiga.py")
    exit()

filas = []
for _, p in pred.iterrows():
    local, visit = p['Local'], p['Visitante']
    prob_local = p[f'Prob_{local}']
    prob_visit = p[f'Prob_{visit}']

    if prob_local >= prob_visit:
        favorito, prob_fav = local, prob_local
        underdog, prob_und = visit, prob_visit
    else:
        favorito, prob_fav = visit, prob_visit
        underdog, prob_und = local, prob_local

    partido_key = f"{visit} @ {local}"
    bull_row = bull[bull['Partido'] == partido_key]

    if bull_row.empty:
        senal_total = 'N/A'
        fat_l, fat_v = 0, 0
    else:
        b = bull_row.iloc[0]
        senal_total = b['Señal_Bullpen']
        fat_l = b['Bullpen_Local']
        fat_v = b['Bullpen_Visit']

    fat_fav = fat_l if favorito == local else fat_v
    fat_dog = fat_v if favorito == local else fat_l

    if fat_fav < fat_dog:
        alineacion = 'ALINEADO  ✓'
    elif fat_fav > fat_dog:
        alineacion = 'CONTRADICE ✗'
    else:
        alineacion = 'NEUTRO     ─'

    filas.append({
        'Partido':     partido_key,
        'Favorito':    favorito,
        'Prob':        prob_fav,
        'Underdog':    underdog,
        'Prob_Dog':    prob_und,
        'Total':       p['Total_Estimado'],
        'Señal_Total': senal_total,
        'Bullpen_ML':  alineacion,
    })

df_out = pd.DataFrame(filas).sort_values('Prob', ascending=False).reset_index(drop=True)
df_out.index += 1

hoy = datetime.now().strftime('%Y-%m-%d')

print(f"\n=== PICKS DEL DIA {hoy} ===\n")
print(f"{'#':<3} {'Partido':<18} {'Fav':<6} {'Prob':>6}  {'Dog':<6} {'Prob':>6}  "
      f"{'Total':>6}  {'Señal Total':<12}  {'Bullpen ML'}")
print("-" * 85)
for i, row in df_out.iterrows():
    marca = '★' if i <= 3 else ' '
    print(f"{marca}{i:<2} {row['Partido']:<18} {row['Favorito']:<6} {row['Prob']:>5.1f}%  "
          f"{row['Underdog']:<6} {row['Prob_Dog']:>5.1f}%  "
          f"{row['Total']:>6.1f}  {row['Señal_Total']:<12}  {row['Bullpen_ML']}")

# --- Top 3 recomendados ---
alineados = df_out[df_out['Bullpen_ML'] == 'ALINEADO  ✓'].head(3)
if len(alineados) < 3:
    resto = df_out[df_out['Bullpen_ML'] != 'ALINEADO  ✓'].head(3 - len(alineados))
    top3 = pd.concat([alineados, resto]).reset_index(drop=True)
else:
    top3 = alineados.reset_index(drop=True)

print("\n=== TOP 3 PICKS RECOMENDADOS ===\n")
print("Criterio: mayor probabilidad + bullpen ALINEADO con el favorito\n")
for i, row in top3.iterrows():
    confianza = '🔥 Alta' if row['Bullpen_ML'] == 'ALINEADO  ✓' else '⚠️  Moderada'
    print(f"  {i+1}. {row['Partido']} → {row['Favorito']} ({row['Prob']:.1f}%)")
    print(f"     Total: {row['Total']:.1f}  |  Bullpen: {row['Bullpen_ML']}  |  {confianza}")
    print()

print("Leyenda:")
print("  ALINEADO  ✓ = Bullpen del favorito más fresco → pick más sólido")
print("  CONTRADICE ✗ = Bullpen del underdog más fresco → pick más riesgoso")
print("  NEUTRO     ─ = Sin diferencia clara de bullpen")

# --- Underdogs fuertes ---
df_out['Es_underdog_fuerte'] = (
    (df_out['Prob_Dog'] >= 38) &
    (df_out['Prob_Dog'] <= 48) &
    (df_out['Bullpen_ML'] == 'CONTRADICE ✗')
)
underdogs = df_out[df_out['Es_underdog_fuerte']].sort_values(
    'Prob_Dog', ascending=False).head(2)

print("\n=== TOP 2 UNDERDOGS FUERTES ===\n")
print("Criterio: probabilidad 38-48% + bullpen del underdog más fresco que el favorito\n")
if underdogs.empty:
    print("  Sin underdogs fuertes hoy\n")
else:
    for i, (_, row) in enumerate(underdogs.iterrows(), 1):
        print(f"  {i}. {row['Partido']} → {row['Underdog']} ({row['Prob_Dog']:.1f}%)")
        print(f"     Favorito: {row['Favorito']} ({row['Prob']:.1f}%)  |  Total: {row['Total']:.1f}  |  Bullpen: favorito más cansado")
        print()

# --- Guardar para registro ---
top3_save = top3[['Partido','Favorito','Prob','Total','Bullpen_ML']].rename(
    columns={'Favorito':'Seleccion','Prob':'Probabilidad','Total':'Total_Estimado'})
top3_save.to_csv('top3_recomendados.csv', index=False)

if not underdogs.empty:
    und_save = underdogs[['Partido','Underdog','Prob_Dog','Total','Señal_Total','Bullpen_ML']].rename(
        columns={'Underdog':'Seleccion','Prob_Dog':'Probabilidad','Total':'Total_Estimado'})
    und_save.to_csv('underdogs_fuertes.csv', index=False)
else:
    pd.DataFrame(columns=['Partido','Seleccion','Probabilidad',
                          'Total_Estimado','Señal_Total','Bullpen_ML']).to_csv(
        'underdogs_fuertes.csv', index=False)

print("top3_recomendados.csv y underdogs_fuertes.csv guardados.")
