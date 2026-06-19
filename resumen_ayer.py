import pandas as pd
from datetime import datetime, timedelta

ayer = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

print(f"\n{'='*50}")
print(f"  RESUMEN DEL {ayer}")
print(f"{'='*50}")

HISTORIALES = [
    ('historial_todos.csv',     'TODOS LOS PARTIDOS'),
    ('historial_top3.csv',      'TOP 3 RECOMENDADOS'),
    ('historial_underdogs.csv', 'TOP 2 UNDERDOGS FUERTES'),
]

def resumen_ayer(archivo, etiqueta, ayer):
    try:
        df = pd.read_csv(archivo)
    except FileNotFoundError:
        print(f"\n[{etiqueta}] Archivo no encontrado aun.")
        return

    dia_ayer = df[df['Fecha'] == ayer]
    todos    = df[df['Resultado'].notna()]

    print(f"\n{'─'*50}")
    print(f"  {etiqueta}")
    print(f"{'─'*50}")

    if dia_ayer.empty:
        print(f"  Sin picks registrados para {ayer}")
    else:
        ver_ayer = dia_ayer[dia_ayer['Resultado'].notna()]
        print(f"\n  Picks del {ayer}:")
        print(dia_ayer[['Partido','Seleccion','Resultado',
                        'Total_Estimado','Total_Real',
                        'Resultado_Total']].to_string(index=False))

        if not ver_ayer.empty:
            total     = len(ver_ayer)
            correctos = (ver_ayer['Resultado'] == 'CORRECTO').sum()
            print(f"\n  ML ayer     : {correctos}/{total} ({correctos/total*100:.0f}%)")
            con_total = ver_ayer[ver_ayer['Total_Real'].notna()].copy()
            if not con_total.empty:
                con_total['Error'] = abs(
                    pd.to_numeric(con_total['Total_Real']) -
                    pd.to_numeric(con_total['Total_Estimado']))
                mae   = con_total['Error'].mean()
                over  = (con_total['Resultado_Total'] == 'OVER    ▲').sum()
                under = (con_total['Resultado_Total'] == 'UNDER   ▼').sum()
                print(f"  MAE ayer    : {mae:.2f} carreras")
                print(f"  Over/Under  : {over} Over / {under} Under")
        else:
            print(f"  Pendientes  : {len(dia_ayer)} partidos sin resultado")

    if not todos.empty:
        total_ac   = len(todos)
        correct_ac = (todos['Resultado'] == 'CORRECTO').sum()
        con_total  = todos[todos['Total_Real'].notna()].copy()
        print(f"\n  Acumulado   : {correct_ac}/{total_ac} ({correct_ac/total_ac*100:.1f}%)", end="")
        if not con_total.empty:
            con_total['Error'] = abs(
                pd.to_numeric(con_total['Total_Real']) -
                pd.to_numeric(con_total['Total_Estimado']))
            print(f"  |  MAE acum: {con_total['Error'].mean():.2f} carreras", end="")
        print()

for archivo, etiqueta in HISTORIALES:
    resumen_ayer(archivo, etiqueta, ayer)

print()
