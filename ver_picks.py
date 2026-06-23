import pandas as pd
from datetime import datetime

hoy = datetime.now().strftime('%Y-%m-%d')

print(f"\n{'='*50}")
print(f"  PICKS DEL DIA {hoy}")
print(f"{'='*50}")

def recomendar_spread(prob_favorito, margen_predicho):
    if prob_favorito >= 62 and margen_predicho >= 3.5:
        return "-3.5"
    elif prob_favorito >= 57 and margen_predicho >= 2.5:
        return "-2.5"
    elif prob_favorito >= 53 and margen_predicho >= 1.5:
        return "-1.5"
    else:
        return "ML only"

try:
    pred = pd.read_csv('predicciones_hoy.csv')
    bull = pd.read_csv('bullpen_fatiga.csv')
except FileNotFoundError as e:
    print(f"\n  Archivo no encontrado: {e}")
    exit()

filas = []
for _, p in pred.iterrows():
    local, visit = p['Local'], p['Visitante']
    prob_local = p[f'Prob_{local}']
    prob_visit = p[f'Prob_{visit}']
    margen     = p.get('Margen_Estimado', 2.5)

    if prob_local >= prob_visit:
        fav, prob_fav, dog, prob_dog = local, prob_local, visit, prob_visit
    else:
        fav, prob_fav, dog, prob_dog = visit, prob_visit, local, prob_local

    partido_key = f"{visit} @ {local}"
    bull_row    = bull[bull['Partido'] == partido_key]

    if bull_row.empty:
        senal, fat_l, fat_v = 'N/A', 0, 0
    else:
        b      = bull_row.iloc[0]
        senal  = b['Señal_Bullpen']
        fat_l  = b['Bullpen_Local']
        fat_v  = b['Bullpen_Visit']

    fat_fav = fat_l if fav == local else fat_v
    fat_dog = fat_v if fav == local else fat_l

    if fat_fav < fat_dog:
        alineacion = 'ALINEADO  ✓'
    elif fat_fav > fat_dog:
        alineacion = 'CONTRADICE ✗'
    else:
        alineacion = 'NEUTRO     ─'

    spread = recomendar_spread(prob_fav, margen)

    filas.append({
        'Partido':     partido_key,
        'Fav':         fav,
        'Prob':        prob_fav,
        'Dog':         dog,
        'Prob_Dog':    prob_dog,
        'Total':       p['Total_Estimado'],
        'Margen':      margen,
        'Spread':      spread,
        'Señal_Total': senal,
        'Bullpen_ML':  alineacion,
    })

df = pd.DataFrame(filas).sort_values('Prob', ascending=False).reset_index(drop=True)
df.index += 1

# --- Pronóstico completo ---
print(f"\n📊 PRONOSTICO COMPLETO ({len(df)} partidos)\n")
print(f"{'#':<3} {'Partido':<18} {'Fav':<6} {'Prob':>6}  {'Dog':<6} {'Prob':>6}  {'Total':>6}  {'Spread':<10}")
print("-" * 68)
for i, row in df.iterrows():
    marca = '★' if i <= 3 else ' '
    print(f"{marca}{i:<2} {row['Partido']:<18} {row['Fav']:<6} {row['Prob']:>5.1f}%  "
          f"{row['Dog']:<6} {row['Prob_Dog']:>5.1f}%  {row['Total']:>5.1f}  {row['Spread']:<10}")

# --- Top 3 recomendados ---
alineados = df[df['Bullpen_ML'] == 'ALINEADO  ✓'].head(3)
if len(alineados) < 3:
    resto = df[df['Bullpen_ML'] != 'ALINEADO  ✓'].head(3 - len(alineados))
    top3  = pd.concat([alineados, resto]).reset_index(drop=True)
else:
    top3 = alineados.reset_index(drop=True)

print("\n🔥 TOP 3 RECOMENDADOS (prob + bullpen alineado)\n")
for i, row in top3.iterrows():
    confianza = '🔥 Alta' if row['Bullpen_ML'] == 'ALINEADO  ✓' else '⚠️  Moderada'
    print(f"  {i+1}. {row['Partido']} → {row['Fav']} ({row['Prob']:.1f}%)")
    print(f"     Total: {row['Total']:.1f}  |  Spread: {row['Spread']}  |  Margen est: {row['Margen']:.1f}c  |  {confianza}")
    print()

# --- Underdogs fuertes ---
df['Es_underdog_fuerte'] = (
    (df['Prob_Dog'] >= 38) &
    (df['Prob_Dog'] <= 48) &
    (df['Bullpen_ML'] == 'CONTRADICE ✗')
)
underdogs = df[df['Es_underdog_fuerte']].sort_values('Prob_Dog', ascending=False).head(2)

print("⚡ TOP 2 UNDERDOGS FUERTES\n")
if underdogs.empty:
    print("  Sin underdogs fuertes hoy\n")
else:
    for i, (_, row) in enumerate(underdogs.iterrows(), 1):
        print(f"  {i}. {row['Partido']} → {row['Dog']} ({row['Prob_Dog']:.1f}%)")
        print(f"     Total: {row['Total']:.1f}  |  Favorito más cansado")
        print()
