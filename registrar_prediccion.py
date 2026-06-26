import pandas as pd
from datetime import datetime

ARCHIVO_PICKS       = 'top_selecciones.csv'
HISTORIAL_TOP5      = 'historial_top5.csv'
HISTORIAL_TODOS     = 'historial_todos.csv'
HISTORIAL_TOP3      = 'historial_top3.csv'
HISTORIAL_UNDERDOGS = 'historial_underdogs.csv'
HISTORIAL_SPREAD    = 'historial_spread.csv'
TOP_N = 5

hoy = datetime.now().strftime('%Y-%m-%d')
df_todos = pd.read_csv(ARCHIVO_PICKS)

def preparar_filas(df, fecha):
    filas = df.copy()
    filas.insert(0, 'Fecha', fecha)
    for col in ['Resultado','Total_Real','Resultado_Total','Margen_Real','Spread_Resultado','Bullpen_ML']:
        filas[col] = None
        filas[col] = filas[col].astype('object')
    # Asegurar columnas de spread
    if 'Spread' not in filas.columns:
        filas['Spread'] = 'ML only'
    if 'Margen_Estimado' not in filas.columns:
        filas['Margen_Estimado'] = 0.0
    return filas

def registrar(df_nuevo, archivo, etiqueta):
    try:
        df_hist = pd.read_csv(archivo)
        # Agregar columnas nuevas si no existen
        for col in ['Spread','Margen_Estimado','Margen_Real','Spread_Resultado']:
            if col not in df_hist.columns:
                df_hist[col] = None
        if hoy in df_hist['Fecha'].values:
            print(f"[{etiqueta}] Ya registrado para hoy ({hoy}), se omite.")
            return
        df_hist = pd.concat([df_hist, df_nuevo], ignore_index=True)
    except FileNotFoundError:
        df_hist = df_nuevo
    df_hist.to_csv(archivo, index=False)
    print(f"[{etiqueta}] {len(df_nuevo)} picks registrados para {hoy}:")
    cols_show = ['Partido','Seleccion','Probabilidad','Spread']
    cols_show = [c for c in cols_show if c in df_nuevo.columns]
    print(df_nuevo[cols_show].to_string(index=False))
    print()

registrar(preparar_filas(df_todos.head(TOP_N), hoy), HISTORIAL_TOP5,  'TOP 5')
registrar(preparar_filas(df_todos, hoy),            HISTORIAL_TODOS, 'TODOS')

try:
    df_top3 = pd.read_csv('top3_recomendados.csv')
    registrar(preparar_filas(df_top3, hoy), HISTORIAL_TOP3, 'TOP 3 RECOMENDADOS')
except FileNotFoundError:
    print("[TOP 3] No encontrado — corre picks_del_dia.py primero.")

try:
    df_dogs = pd.read_csv('underdogs_fuertes.csv')
    if len(df_dogs) > 0:
        registrar(preparar_filas(df_dogs, hoy), HISTORIAL_UNDERDOGS, 'TOP 2 UNDERDOGS')
    else:
        print("[TOP 2 UNDERDOGS] Sin underdogs fuertes hoy.")
except FileNotFoundError:
    print("[TOP 2 UNDERDOGS] No encontrado — corre picks_del_dia.py primero.")

# Top 3 Spread
try:
    df_spread = pd.read_csv('top3_spread.csv')
    if len(df_spread) > 0:
        registrar(preparar_filas(df_spread, hoy), HISTORIAL_SPREAD, 'TOP 3 SPREAD')
    else:
        print("[TOP 3 SPREAD] Sin partidos con spread hoy.")
except FileNotFoundError:
    print("[TOP 3 SPREAD] No encontrado — corre picks_del_dia.py primero.")

# Underdogs +1.5
try:
    df_dog_sp = pd.read_csv('underdogs_spread.csv')
    if len(df_dog_sp) > 0:
        registrar(preparar_filas(df_dog_sp, hoy), 'historial_underdogs_spread.csv', 'UNDERDOGS +1.5')
    else:
        print("[UNDERDOGS +1.5] Sin candidatos hoy.")
except FileNotFoundError:
    print("[UNDERDOGS +1.5] No encontrado — corre picks_del_dia.py primero.")
