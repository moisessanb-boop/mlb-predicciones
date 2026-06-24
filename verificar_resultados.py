import pandas as pd

HISTORIALES = [
    ('historial_top5.csv',      'TOP 5 PICKS'),
    ('historial_todos.csv',     'TODOS LOS PARTIDOS'),
    ('historial_top3.csv',      'TOP 3 RECOMENDADOS'),
    ('historial_underdogs.csv', 'TOP 2 UNDERDOGS FUERTES'),
]

df_datos = pd.read_csv('datos_mlb_limpio.csv')
df_datos['Fecha'] = pd.to_datetime(df_datos['Fecha']).dt.strftime('%Y-%m-%d')
df_datos['R']  = pd.to_numeric(df_datos['R'],  errors='coerce')
df_datos['RA'] = pd.to_numeric(df_datos['RA'], errors='coerce')

def get_total_real(partido, fecha):
    partes = partido.split(' @ ')
    if len(partes) != 2: return None
    visitante, local = partes[0].strip(), partes[1].strip()
    for equipo in [local, visitante]:
        match = df_datos[(df_datos['Team']==equipo) & (df_datos['Fecha']==fecha)]
        if not match.empty:
            return match.iloc[0]['Total_Carreras']
    return None

def verificar_spread(spread_str, margen_real, gano):
    if spread_str in ['ML only', None, 'N/A'] or pd.isna(str(spread_str)):
        return 'N/A'
    if not gano:
        return 'NO CUBIERTO ✗'
    try:
        spread_val = abs(float(spread_str))
        return 'CUBIERTO  ✓' if margen_real > spread_val else 'NO CUBIERTO ✗'
    except:
        return 'N/A'

def verificar_archivo(archivo, etiqueta):
    try:
        df = pd.read_csv(archivo)
    except FileNotFoundError:
        print(f"\n[{etiqueta}] Archivo no encontrado aun.")
        return

    df['Resultado'] = df['Resultado'].astype('object')
    for col in ['Total_Real','Resultado_Total','Margen_Real','Spread_Resultado']:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype('object')
    if 'Spread' not in df.columns:
        df['Spread'] = 'ML only'

    # Verificar ML y Totales
    for idx, pick in df[df['Resultado'].isna()].iterrows():
        match = df_datos[
            (df_datos['Team']==pick['Seleccion']) &
            (df_datos['Fecha']==str(pick['Fecha']))
        ]
        if match.empty: continue
        row   = match.iloc[0]
        gano  = int(row['Gano'])
        margen = row['R'] - row['RA']

        df.at[idx,'Resultado']   = 'CORRECTO' if gano==1 else 'INCORRECTO'
        df.at[idx,'Margen_Real'] = margen

        # Spread
        df.at[idx,'Spread_Resultado'] = verificar_spread(pick['Spread'], margen, gano==1)

        # Total
        total_real = get_total_real(pick['Partido'], str(pick['Fecha']))
        if total_real is not None:
            df.at[idx,'Total_Real'] = total_real
            diff = total_real - pick['Total_Estimado']
            if abs(diff) <= 0.5:
                df.at[idx,'Resultado_Total'] = 'EXACTO  ='
            elif diff > 0:
                df.at[idx,'Resultado_Total'] = 'OVER    ▲'
            else:
                df.at[idx,'Resultado_Total'] = 'UNDER   ▼'

    df.to_csv(archivo, index=False)

    print(f"\n{'='*55}")
    print(f"  {etiqueta}")
    print(f"{'='*55}")
    print("\n--- RESULTADOS POR DIA ---\n")

    for fecha in sorted(df['Fecha'].unique()):
        dia         = df[df['Fecha']==fecha]
        verificados = dia[dia['Resultado'].notna()]
        pendientes  = dia[dia['Resultado'].isna()]

        print(f"  [{fecha}]")
        print(dia[['Partido','Seleccion','Spread','Resultado',
                   'Margen_Real','Spread_Resultado',
                   'Total_Estimado','Total_Real',
                   'Resultado_Total']].to_string(index=False))

        if not verificados.empty:
            total_ml  = len(verificados)
            correctos = (verificados['Resultado']=='CORRECTO').sum()
            resumen   = f"  ML: {correctos}/{total_ml} ({correctos/total_ml*100:.0f}%)"

            # Spread stats
            con_spread = verificados[verificados['Spread_Resultado'].notna() &
                                     (verificados['Spread_Resultado'] != 'N/A')]
            if not con_spread.empty:
                sp_ok  = (con_spread['Spread_Resultado']=='CUBIERTO  ✓').sum()
                sp_tot = len(con_spread)
                resumen += f"  |  Spread: {sp_ok}/{sp_tot} ({sp_ok/sp_tot*100:.0f}%)"

            con_total = verificados[verificados['Total_Real'].notna()].copy()
            if not con_total.empty:
                con_total['Error'] = abs(pd.to_numeric(con_total['Total_Real']) -
                                         pd.to_numeric(con_total['Total_Estimado']))
                resumen += f"  |  MAE: {con_total['Error'].mean():.2f}c"
            print(resumen)

        if not pendientes.empty:
            print(f"  Pendientes: {len(pendientes)} partidos")
        print()

    # Acumulado
    verificados_tot = df[df['Resultado'].notna()]
    if not verificados_tot.empty:
        total_ml  = len(verificados_tot)
        correctos = (verificados_tot['Resultado']=='CORRECTO').sum()
        print(f"  --- ACUMULADO TOTAL ---")
        print(f"  ML  : {correctos}/{total_ml} ({correctos/total_ml*100:.1f}%)")

        con_spread = verificados_tot[
            verificados_tot['Spread_Resultado'].notna() &
            (verificados_tot['Spread_Resultado'] != 'N/A')
        ]
        if not con_spread.empty:
            sp_ok  = (con_spread['Spread_Resultado']=='CUBIERTO  ✓').sum()
            sp_tot = len(con_spread)
            print(f"  Spread cubierto: {sp_ok}/{sp_tot} ({sp_ok/sp_tot*100:.1f}%)")

        con_total = verificados_tot[verificados_tot['Total_Real'].notna()].copy()
        if not con_total.empty:
            con_total['Error'] = abs(pd.to_numeric(con_total['Total_Real']) -
                                     pd.to_numeric(con_total['Total_Estimado']))
            print(f"  MAE Total: {con_total['Error'].mean():.2f} carreras")

    if df['Resultado'].isna().any():
        print(f"\n  Sin resultado aun: {df['Resultado'].isna().sum()} partidos")

for archivo, etiqueta in HISTORIALES:
    verificar_archivo(archivo, etiqueta)
