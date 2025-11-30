import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def leer_detector(path_xml: str) -> pd.DataFrame:
    """
    Lee un archivo de salida de un inductionLoop de SUMO
    tomando únicamente las líneas <interval .../>.
    Es robusto aunque el archivo tenga cabecera XML, etiquetas
    <detectorStats>, etc.
    """
    path = Path(path_xml)
    text = path.read_text(encoding="utf-8", errors="ignore")

    filas = []

    for line in text.splitlines():
        line = line.strip()
        # Nos quedamos solo con las líneas que tengan <interval .../>
        if not line.startswith("<interval"):
            continue

        # Asegurarnos de que termina en '/>' para que sea bien formado
        if not line.endswith("/>"):
            # Por si hubiera algo tipo <interval ...></interval>
            if line.endswith(">") and "</interval>" not in line:
                # lo tratamos igual, ET se apaña
                pass

        try:
            elem = ET.fromstring(line)
        except ET.ParseError:
            # Si alguna línea viene rara, la ignoramos
            continue

        attrs = elem.attrib.copy()
        # detector_id: si hay atributo id, lo usamos; si no, nombre de fichero
        if "id" in attrs:
            attrs["detector_id"] = attrs["id"]
        else:
            attrs["detector_id"] = path.stem

        filas.append(attrs)

    if not filas:
        return pd.DataFrame()

    df = pd.DataFrame(filas)

    # Convertir columnas numéricas si existen
    for col in ["begin", "end", "nVehContrib", "flow", "occupancy", "speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # Tiempo central del intervalo
    if {"begin", "end"} <= set(df.columns):
        df["t"] = (df["begin"] + df["end"]) / 2.0

    return df


if __name__ == "__main__":
    archivos = ["det_entrada_norte.xml", "det_salida_sur.xml"]

    dfs = []
    for f in archivos:
        print(f"Leyendo {f} ...")
        df_f = leer_detector(f)
        if not df_f.empty:
            dfs.append(df_f)
            print(f"  {len(df_f)} filas leídas.")
        else:
            print(f"  (archivo {f} vacío o sin intervalos)")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        print("\nPrimeras filas:")
        print(df_all.head())
        df_all.to_csv("detectors_data.csv", index=False)
        print("\nDatos guardados en detectors_data.csv")
    else:
        print("No se ha leído ningún dato de detectores.")

