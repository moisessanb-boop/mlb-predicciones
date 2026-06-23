import pandas as pd
from datetime import datetime

ARCHIVO_PICKS       = 'top_selecciones.csv'
HISTORIAL_TOP5      = 'historial_top5.csv'
HISTORIAL_TODOS     = 'historial_todos.csv'
HISTORIAL_TOP3      = 'historial_top3.csv'
HISTORIAL_UNDERDOGS = 'historial_underdogs.csv'
TOP_N = 5

hoy = datetime.now().strftime('%Y-%m-%d')

df_todos = pd.read_csv(ARCHIVO_PICKS)

def preparar_filas(df, fecha):
    filas = df.copy()
    filas.insert(0, 'Fecha', fecha)
    filas['Resultado']       = None
    filas['Total_Real']      = None
    filas['Resultado_Total'] = None
    for col in ['Resultado','Total_Real','Resultado_Total']:
        filas[col] = filas[col].astype('object')
    return filas

def registrar(df_nuevo, archivo, etiqueta):
    try:
        df_hist = pd.read_csv(archivo)
        for col in ['Total_Real','Resultado_Total']:
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
    print(df_nuevo[['Partido','Seleccion','Probabilidad']].to_string(index=False))
    print()

# Top 5 y todos
registrar(preparar_filas(df_todos.head(TOP_N), hoy), HISTORIAL_TOP5,  'TOP 5')
registrar(preparar_filas(df_todos, hoy),            HISTORIAL_TODOS, 'TODOS')

# Top 3 recomendados
try:
    df_top3 = pd.read_csv('top3_recomendados.csv')
    registrar(preparar_filas(df_top3, hoy), HISTORIAL_TOP3, 'TOP 3 RECOMENDADOS')
except FileNotFoundError:
    print("[TOP 3 RECOMENDADOS] No encontrado — corre picks_del_dia.py primero.")

# Underdogs fuertes
try:
    df_dogs = pd.read_csv('underdogs_fuertes.csv')
    if len(df_dogs) > 0:
        registrar(preparar_filas(df_dogs, hoy), HISTORIAL_UNDERDOGS, 'TOP 2 UNDERDOGS')
    else:
        print("[TOP 2 UNDERDOGS] Sin underdogs fuertes hoy, no se registra.")
except FileNotFoundError:
    print("[TOP 2 UNDERDOGS] No encontrado — corre picks_del_dia.py primero.")
