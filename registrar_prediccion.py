import pandas as pd
from datetime import datetime

ARCHIVO_PICKS   = 'top_selecciones.csv'
HISTORIAL_TOP5  = 'historial_top5.csv'
HISTORIAL_TODOS = 'historial_todos.csv'
TOP_N = 5

hoy = datetime.now().strftime('%Y-%m-%d')

df_todos = pd.read_csv(ARCHIVO_PICKS)

def preparar_filas(df, fecha):
    filas = df.copy()
    filas.insert(0, 'Fecha', fecha)
    filas['Resultado']       = None
    filas['Total_Real']      = None
    filas['Resultado_Total'] = None
    for col in ['Resultado', 'Total_Real', 'Resultado_Total']:
        filas[col] = filas[col].astype('object')
    return filas

def registrar(df_nuevo, archivo, etiqueta):
    try:
        df_hist = pd.read_csv(archivo)
        # Agregar columnas nuevas si no existen (compatibilidad con historial anterior)
        for col in ['Total_Real', 'Resultado_Total']:
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
    print(df_nuevo[['Partido','Seleccion','Probabilidad','Total_Estimado']].to_string(index=False))
    print()

registrar(preparar_filas(df_todos.head(TOP_N), hoy), HISTORIAL_TOP5,  'TOP 5')
registrar(preparar_filas(df_todos, hoy),            HISTORIAL_TODOS, 'TODOS')
