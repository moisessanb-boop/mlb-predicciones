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

    for idx, pick in df[df['Resultado'].isna()].iterrows():
        match = df_datos[
            (df_datos['Team'] == pick['Seleccion']) &
            (df_datos['Fecha'] == str(pick['Fecha']))
        ]
        if match.empty:
            continue
        df.at[idx, 'Resultado'] = 'CORRECTO' if match.iloc[0]['Gano'] == 1 else 'INCORRECTO'

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
    print("\n--- RESULTADOS POR DIA ---\n")

    for fecha in sorted(df['Fecha'].unique()):
        dia         = df[df['Fecha'] == fecha]
        verificados = dia[dia['Resultado'].notna()]
        pendientes  = dia[dia['Resultado'].isna()]

        print(f"  [{fecha}]")
        print(dia[['Partido','Seleccion','Resultado',
                   'Total_Estimado','Total_Real',
                   'Resultado_Total']].to_string(index=False, col_space=10))

        if not verificados.empty:
            total_ml  = len(verificados)
            correctos = (verificados['Resultado'] == 'CORRECTO').sum()
            resumen   = f"  ML: {correctos}/{total_ml} ({correctos/total_ml*100:.0f}%)"
            con_total = verificados[verificados['Total_Real'].notna()].copy()
            if not con_total.empty:
                con_total['Error'] = abs(
                    pd.to_numeric(con_total['Total_Real']) -
                    pd.to_numeric(con_total['Total_Estimado']))
                resumen += f"  |  MAE Total: {con_total['Error'].mean():.2f} carreras"
            print(resumen)

        if not pendientes.empty:
            print(f"  Pendientes: {len(pendientes)} partidos")
        print()

    verificados_tot = df[df['Resultado'].notna()]
    pendientes_tot  = df[df['Resultado'].isna()]

    if not verificados_tot.empty:
        total_ml  = len(verificados_tot)
        correctos = (verificados_tot['Resultado'] == 'CORRECTO').sum()
        print(f"  --- ACUMULADO TOTAL ---")
        print(f"  ML verificados  : {total_ml}")
        print(f"  Correctos       : {correctos} ({correctos/total_ml*100:.1f}%)")
        print(f"  Incorrectos     : {total_ml-correctos} ({(total_ml-correctos)/total_ml*100:.1f}%)")

        con_total = verificados_tot[verificados_tot['Total_Real'].notna()].copy()
        if not con_total.empty:
            con_total['Error'] = abs(
                pd.to_numeric(con_total['Total_Real']) -
                pd.to_numeric(con_total['Total_Estimado']))
            mae    = con_total['Error'].mean()
            over   = (con_total['Resultado_Total'] == 'OVER    ▲').sum()
            under  = (con_total['Resultado_Total'] == 'UNDER   ▼').sum()
            exacto = (con_total['Resultado_Total'] == 'EXACTO  =').sum()
            print(f"\n  MAE acumulado   : {mae:.2f} carreras promedio de error")
            print(f"  Over vs modelo  : {over}")
            print(f"  Under vs modelo : {under}")
            print(f"  Exacto (+-0.5)  : {exacto}")

    if not pendientes_tot.empty:
        print(f"\n  Sin resultado aun: {len(pendientes_tot)} partidos")

verificar_archivo(HISTORIAL_TOP5,  'TOP 5 PICKS')
verificar_archivo(HISTORIAL_TODOS, 'TODOS LOS PARTIDOS')
