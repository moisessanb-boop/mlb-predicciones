import pandas as pd

HISTORIAL_TOP5  = 'historial_top5.csv'
HISTORIAL_TODOS = 'historial_todos.csv'

df_datos = pd.read_csv('datos_mlb_limpio.csv')
df_datos['Fecha'] = pd.to_datetime(df_datos['Fecha']).dt.strftime('%Y-%m-%d')

def get_total_real(partido, fecha):
    partes = partido.split(' @ ')
    if len(partes) != 2:
        return None
    visitante, local = partes[0].strip(), partes[1].strip()
    for equipo in [local, visitante]:
        match = df_datos[(df_datos['Team'] == equipo) & (df_datos['Fecha'] == fecha)]
        if not match.empty:
            return match.iloc[0]['Total_Carreras']
    return None

def verificar_archivo(archivo, etiqueta):
    try:
        df = pd.read_csv(archivo)
    except FileNotFoundError:
        print(f"\n[{etiqueta}] Archivo no encontrado.")
        return

    df['Resultado'] = df['Resultado'].astype('object')
    for col in ['Total_Real', 'Resultado_Total']:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype('object')

    # --- Verificar Moneyline pendientes ---
    for idx, pick in df[df['Resultado'].isna()].iterrows():
        match = df_datos[
            (df_datos['Team'] == pick['Seleccion']) &
            (df_datos['Fecha'] == str(pick['Fecha']))
        ]
        if match.empty:
            continue
        df.at[idx, 'Resultado'] = 'CORRECTO' if match.iloc[0]['Gano'] == 1 else 'INCORRECTO'

    # --- Verificar Totales pendientes ---
    for idx, pick in df[df['Total_Real'].isna()].iterrows():
        total_real = get_total_real(pick['Partido'], str(pick['Fecha']))
        if total_real is None:
            continue
        df.at[idx, 'Total_Real'] = total_real
        diff = total_real - pick['Total_Estimado']
        if abs(diff) <= 0.5:
            df.at[idx, 'Resultado_Total'] = 'EXACTO  ='
        elif diff > 0:
            df.at[idx, 'Resultado_Total'] = 'OVER    ▲'
        else:
            df.at[idx, 'Resultado_Total'] = 'UNDER   ▼'

    df.to_csv(archivo, index=False)

    print(f"\n{'='*55}")
    print(f"  {etiqueta}")
    print(f"{'='*55}")
    print(df[['Fecha','Partido','Seleccion',
              'Resultado','Total_Estimado',
              'Total_Real','Resultado_Total']].to_string(index=False))

    # --- Stats Moneyline ---
    ml_ver = df[df['Resultado'].notna()]
    if not ml_ver.empty:
        total_ml  = len(ml_ver)
        correctos = (ml_ver['Resultado'] == 'CORRECTO').sum()
        print(f"\n  --- MONEYLINE ---")
        print(f"  Total verificados : {total_ml}")
        print(f"  Correctos         : {correctos} ({correctos/total_ml*100:.1f}%)")
        print(f"  Incorrectos       : {total_ml-correctos} ({(total_ml-correctos)/total_ml*100:.1f}%)")

    # --- Stats Totales ---
    tot_ver = df[df['Total_Real'].notna()].copy()
    if not tot_ver.empty:
        tot_ver['Total_Real']     = pd.to_numeric(tot_ver['Total_Real'])
        tot_ver['Total_Estimado'] = pd.to_numeric(tot_ver['Total_Estimado'])
        tot_ver['Error']          = abs(tot_ver['Total_Real'] - tot_ver['Total_Estimado'])
        mae    = tot_ver['Error'].mean()
        over   = (tot_ver['Resultado_Total'] == 'OVER    ▲').sum()
        under  = (tot_ver['Resultado_Total'] == 'UNDER   ▼').sum()
        exacto = (tot_ver['Resultado_Total'] == 'EXACTO  =').sum()
        print(f"\n  --- TOTAL DE CARRERAS ---")
        print(f"  MAE acumulado     : {mae:.2f} carreras promedio de error")
        print(f"  Over vs modelo    : {over}  (real > estimado)")
        print(f"  Under vs modelo   : {under}  (real < estimado)")
        print(f"  Exacto (+-0.5)    : {exacto}")

    pendientes = df[df['Resultado'].isna() | df['Total_Real'].isna()]
    if not pendientes.empty:
        print(f"\n  Sin resultado aun : {len(pendientes)} partidos")

verificar_archivo(HISTORIAL_TOP5,  'TOP 5 PICKS')
verificar_archivo(HISTORIAL_TODOS, 'TODOS LOS PARTIDOS')
