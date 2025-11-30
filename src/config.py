"""
config.py

Parámetros globales del proyecto:
- Rutas a carpetas de datos
- Parámetros de tiempo (Δt)
- Patrones de nombres de archivos
"""

from pathlib import Path

# Carpeta raíz del proyecto (algeciras_milp/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]



# Carpetas de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Fichero de red de SUMO (net.xml) del Puerto de Algeciras
NET_FILE = RAW_DIR / "simulation" / "algeciras.net.xml"  
# Horizonte temporal (segundos)
# Debe coincidir con el 'freq' usado en edgeData de SUMO
DELTA_T = 60.0  # por ejemplo, 60 s

# Patrones de archivos procesados
EDGE_DATA_PATTERN = "edgeData_scen*.csv"
TRIPINFO_PATTERN = "tripinfo_scen*.csv"

# Archivo de salida con datos ya listos para el MILP
MILP_INPUTS_FILE = PROCESSED_DIR / "milp_inputs.pkl"
