import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys
import math
from typing import Optional


def leer_edgedata(path_xml: str, scenario: Optional[str] = None) -> pd.DataFrame:
    """
    Lee un archivo de salida <edgeData> de SUMO y lo convierte en
    un DataFrame de pandas.

    Estructura típica del XML:

    <edgeData ...>
        <interval id="0" begin="0.00" end="60.00" ...>
            <edge id="edge_1" nVehContrib="10" speed="13.5" .../>
            <edge id="edge_2" nVehContrib="5"  speed="11.2" .../>
            ...
        </interval>
        <interval id="1" begin="60.00" end="120.00" ...>
            ...
        </interval>
    </edgeData>

    Para cada combinación (intervalo, edge) genera una fila con:
      - scenario (si se pasa como argumento)
      - interval_id, begin, end
      - edge_id
      - atributos numéricos típicos (nVehContrib, speed, etc. si existen)
      - columna 'flow' ≈ nVehContrib / (end - begin) [veh/s] si es posible
    """

    path = Path(path_xml)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    # Parsear el XML completo
    tree = ET.parse(path)
    root = tree.getroot()

    filas = []

    # Recorremos todos los <interval> del edgeData
    for interval in root.iter("interval"):
        # Atributos del intervalo (tiempos, id de intervalo, etc.)
        int_attrib = dict(interval.attrib)

        try:
            begin = float(int_attrib.get("begin", 0.0))
        except ValueError:
            begin = 0.0

        try:
            end = float(int_attrib.get("end", begin))
        except ValueError:
            end = begin

        # Recorremos los hijos; suelen ser <edge .../>
        for edge in interval:
            if edge.tag != "edge":
                # Por si hubiera otras etiquetas dentro de <interval>
                continue

            edge_attrib = dict(edge.attrib)

            # Construimos un diccionario de fila combinando info del intervalo y del edge
            fila = {
                "scenario": scenario,                         # puede ser None
                "interval_id": int_attrib.get("id"),
                "begin": begin,
                "end": end,
                "edge_id": edge_attrib.get("id"),
            }

            # Copiamos atributos numéricos típicos si existen
            # (si no existen en el XML, simplemente no se añaden a la fila)
            for key in [
                "nVehContrib",
                "speed",
                "traveltime",
                "occupancy",
                "entered",
                "left",
                "meanSpeed",
                "meanTimeLoss",
                "meanHaltingTime",
            ]:
                if key in edge_attrib:
                    try:
                        fila[key] = float(edge_attrib[key])
                    except ValueError:
                        # Si no se puede convertir, se deja como texto
                        fila[key] = edge_attrib[key]

            filas.append(fila)

    # Construimos el DataFrame final
    df = pd.DataFrame(filas)

    if df.empty:
        print("Aviso: el DataFrame resultante está vacío. "
              "¿El archivo edgeData tiene intervalos y edges?")
        return df

    # Cálculo de flujo medio: nVehContrib / (end - begin) [veh/s]
    if "nVehContrib" in df.columns:
        duracion = df["end"] - df["begin"]
        # Evitar divisiones por 0
        duracion = duracion.replace(0, math.nan)
        df["flow"] = df["nVehContrib"] / duracion

    return df


def main():
    """
    Uso por línea de comandos:

        python leer_edgedata.py edgeData_port_scen1.xml [scenario] [salida_csv]

    Ejemplo:

        python leer_edgedata.py data/raw/edgeData_port_scen1.xml scen1 \
               data/processed/edgeData_scen1.csv
    """
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python leer_edgedata.py edgeData_file.xml [scenario] [salida_csv]")
        sys.exit(1)

    path_arg = sys.argv[1]
    
    # Resolucion robusta de rutas
    posibles_rutas = [
        Path(path_arg),
        Path(__file__).resolve().parents[2] / path_arg, # Root relative
        Path(__file__).resolve().parents[2] / "data" / "raw" / path_arg,
        Path(__file__).resolve().parents[2] / "data" / "raw" / "simulation" / path_arg,
        Path(__file__).resolve().parents[2] / "data" / "processed" / path_arg,
    ]
    
    path_xml = None
    for p in posibles_rutas:
        if p.exists():
            path_xml = p
            break
            
    if path_xml is None:
        print(f"Error: No se encontró el archivo '{path_arg}' en ninguna de las rutas esperadas:")
        for p in posibles_rutas:
            print(f"  - {p}")
        sys.exit(1)
        
    print(f"Archivo encontrado: {path_xml}")
    scenario = sys.argv[2] if len(sys.argv) >= 3 else None
    salida_csv = sys.argv[3] if len(sys.argv) >= 4 else "edgeData.csv"

    print(f"Leyendo edgeData de: {path_xml}")
    if scenario is not None:
        print(f"Escenario: {scenario}")

    df = leer_edgedata(path_xml, scenario=scenario)

    if df.empty:
        print("No se han generado filas. No se guardará CSV.")
        sys.exit(0)

    # Mostrar las primeras filas por consola para comprobar
    print("\nPrimeras filas del DataFrame:")
    print(df.head())

    # Guardar a CSV
    salida_path = Path(salida_csv)
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_path, index=False)
    print(f"\nDatos guardados en: {salida_path}")


if __name__ == "__main__":
    main()
