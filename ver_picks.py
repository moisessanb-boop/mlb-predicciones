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

# ========== FILTROS DE ENTRADA ==========
UMBRAL_ML     = 55.0
UMBRAL_SPREAD = 57.0

print(f"\n{'='*50}")
print(f"  FILTROS DE ENTRADA")
print(f"{'='*50}")

# Picks ML recomendados (≥55%)
picks_ml = df[df['Prob'] >= UMBRAL_ML]
print(f"\n✅ PICKS ML (prob ≥{UMBRAL_ML}%)\n")
if picks_ml.empty:
    print("  Sin picks que cumplan el umbral hoy — no entrar")
else:
    for i, (_, row) in enumerate(picks_ml.iterrows(), 1):
        print(f"  {i}. {row['Partido']} → {row['Fav']} ({row['Prob']:.1f}%)  |  Bullpen: {row['Bullpen_ML']}")
    print()

# Picks Spread recomendados (≥57% + ALINEADO)
picks_spread = df[
    (df['Prob'] >= UMBRAL_SPREAD) &
    (df['Bullpen_ML'] == 'ALINEADO  ✓') &
    (df['Spread'] != 'ML only')
]
print(f"✅ PICKS SPREAD (prob ≥{UMBRAL_SPREAD}% + bullpen ALINEADO)\n")
if picks_spread.empty:
    print("  Sin picks de spread que cumplan los filtros hoy — no entrar")
else:
    for i, (_, row) in enumerate(picks_spread.iterrows(), 1):
        print(f"  {i}. {row['Partido']} → {row['Fav']} {row['Spread']} ({row['Prob']:.1f}%)  |  Bullpen: {row['Bullpen_ML']}")
    print()

print("Umbrales: ML ≥55% | Spread ≥57% + bullpen ALINEADO")

# --- Filtro Underdogs ---
picks_dog_ml = df[
    (df['Prob_Dog'] >= 42) &
    (df['Prob_Dog'] <= 48) &
    (df['Bullpen_ML'] == 'CONTRADICE ✗') &
    (df['Margen'] < 2.5)
]

picks_dog_spread = df[
    (df['Prob_Dog'] >= 40) &
    (
        (df['Bullpen_ML'] == 'CONTRADICE ✗') |
        (df['Margen'] < 2.0)
    ) &
    (df['Spread'] != 'ML only')
]

print(f"\n✅ UNDERDOGS ML (42-48% + bullpen CONTRADICE + margen <2.5)\n")
if picks_dog_ml.empty:
    print("  Sin underdogs ML que cumplan los filtros hoy")
else:
    for i, (_, row) in enumerate(picks_dog_ml.iterrows(), 1):
        print(f"  {i}. {row['Partido']} → {row['Dog']} ML ({row['Prob_Dog']:.1f}%)  |  Margen est: {row['Margen']:.1f}c  |  Bullpen: {row['Bullpen_ML']}")
print()

print(f"✅ UNDERDOGS +1.5 (≥40% + bullpen CONTRADICE o margen <2.0)\n")
if picks_dog_spread.empty:
    print("  Sin underdogs +1.5 que cumplan los filtros hoy")
else:
    for i, (_, row) in enumerate(picks_dog_spread.iterrows(), 1):
        print(f"  {i}. {row['Partido']} → {row['Dog']} +1.5 ({row['Prob_Dog']:.1f}%)  |  Margen est: {row['Margen']:.1f}c  |  Bullpen: {row['Bullpen_ML']}")
print()
