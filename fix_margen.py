import pandas as pd

df_datos = pd.read_csv('datos_mlb_limpio.csv')
df_datos['Fecha'] = pd.to_datetime(df_datos['Fecha']).dt.strftime('%Y-%m-%d')
df_datos['R']  = pd.to_numeric(df_datos['R'],  errors='coerce')
df_datos['RA'] = pd.to_numeric(df_datos['RA'], errors='coerce')

for archivo in ['historial_top5.csv','historial_todos.csv',
                'historial_top3.csv','historial_underdogs.csv']:
    try:
        df = pd.read_csv(archivo)
        if 'Margen_Real' not in df.columns:
            df['Margen_Real'] = None
        df['Margen_Real'] = df['Margen_Real'].astype('object')

        # Rellenar Margen_Real donde ya hay Resultado pero falta el margen
        sin_margen = df[df['Resultado'].notna() & df['Margen_Real'].isna()]
        actualizados = 0
        for idx, pick in sin_margen.iterrows():
            match = df_datos[
                (df_datos['Team'] == pick['Seleccion']) &
                (df_datos['Fecha'] == str(pick['Fecha']))
            ]
            if match.empty:
                continue
            row    = match.iloc[0]
            margen = row['R'] - row['RA']
            df.at[idx, 'Margen_Real'] = margen
            actualizados += 1

        df.to_csv(archivo, index=False)
        print(f"{archivo}: {actualizados} margenes actualizados")
    except FileNotFoundError:
        print(f"{archivo}: no existe")
