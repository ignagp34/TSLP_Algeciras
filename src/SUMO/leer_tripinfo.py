import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import sys
from typing import Optional


def leer_tripinfo(path_xml: str, scenario: Optional[str] = None) -> pd.DataFrame:
    """
    Lee un archivo tripinfo.xml de SUMO y lo convierte en un DataFrame.

    Cada <tripinfo .../> del XML se convierte en una fila del DataFrame.

    Añade (si se pasa) una columna 'scenario' para poder distinguir
    entre distintos escenarios (scen1, scen2, ...).
    """
    path = Path(path_xml)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    filas = []
    for trip in root.iter("tripinfo"):
        attrs = trip.attrib.copy()
        if scenario is not None:
            attrs["scenario"] = scenario
        filas.append(attrs)

    if not filas:
        print("Aviso: no se han encontrado etiquetas <tripinfo> en el XML.")
        return pd.DataFrame()

    df = pd.DataFrame(filas)

    # columnas numéricas típicas de tripinfo
    cols_numericas = [
        "depart",
        "arrival",
        "duration",
        "routeLength",
        "waitingTime",
        "departDelay",
        "arrivalDelay",
        "timeLoss",
        "rerouteNo",
    ]

    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    """
    Uso por línea de comandos:

        python leer_tripinfo.py tripinfo_scen1.xml [scenario] [salida_csv]

    Ejemplo:

        python leer_tripinfo.py data/raw/tripinfo_scen1.xml scen1 \
               data/processed/tripinfo_scen1.csv
    """
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python leer_tripinfo.py tripinfo_file.xml [scenario] [salida_csv]")
        sys.exit(1)

    path_xml = sys.argv[1]
    scenario = sys.argv[2] if len(sys.argv) >= 3 else None
    salida_csv = sys.argv[3] if len(sys.argv) >= 4 else "tripinfo.csv"

    print(f"Leyendo tripinfo de: {path_xml}")
    if scenario is not None:
        print(f"Escenario: {scenario}")

    df = leer_tripinfo(path_xml, scenario=scenario)

    if df.empty:
        print("DataFrame vacío. No se guardará CSV.")
        sys.exit(0)

    print("\nPrimeras filas del DataFrame:")
    print(df.head())

    salida_path = Path(salida_csv)
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_path, index=False)
    print(f"\nDatos guardados en: {salida_path}")


if __name__ == "__main__":
    main()
