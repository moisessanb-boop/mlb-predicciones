#!/bin/bash
cd ~/mlb_predicciones
source venv/bin/activate

echo "==============================="
echo " RUTINA MLB - $(date '+%Y-%m-%d %H:%M')"
echo "==============================="

python3 actualizar_datos.py
python3 verificar_resultados.py
python3 resumen_ayer.py
python3 estadisticas.py
python3 obtener_partidos_hoy.py
python3 obtener_abridores_hoy.py
python3 predecir_hoy.py
python3 bullpen_fatiga.py
python3 picks_del_dia.py
python3 ver_pronosticos.py
python3 registrar_prediccion.py
python3 ver_picks.py

echo "==============================="
echo " RUTINA COMPLETADA"
echo "==============================="
