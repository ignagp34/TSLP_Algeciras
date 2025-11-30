"""
make_sensor_pois.py

Lee la lista de arcos con sensor (selected_edges_B*.txt)
y la red SUMO (algeciras.net.xml) y genera un fichero .add.xml
con POIs en el punto medio del primer carril de cada arco.

Uso (desde la raíz del proyecto):

    python -m scripts.make_sensor_pois \
        data/processed/selected_edges_B20.txt \
        data/raw/simulation/algeciras.net.xml \
        data/processed/sensors_B20.add.xml
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def leer_selected_edges(path_txt: str | Path) -> list[str]:
    """
    Lee un archivo .txt con líneas tipo:
        edge:ID
    y devuelve una lista [ID1, ID2, ...].
    """
    path = Path(path_txt)
    edge_ids: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("edge:"):
                eid = line.split("edge:", 1)[1].strip()
                if eid:
                    edge_ids.append(eid)

    return edge_ids


def cargar_shapes_net(path_net: str | Path) -> dict:
    """
    Lee el .net.xml de SUMO y devuelve un diccionario:

        edge_id -> lista de coordenadas (x, y)

    Tomamos el 'shape' del primer carril de cada edge si existe.
    """
    path = Path(path_net)
    tree = ET.parse(path)
    root = tree.getroot()

    # En SUMO, los elementos 'edge' están en el espacio de nombres vacío
    edge_shapes: dict[str, list[tuple[float, float]]] = {}

    for edge in root.findall("edge"):
        eid = edge.get("id")
        if eid is None:
            continue

        # Buscamos el primer 'lane' hijo con atributo 'shape'
        lane = edge.find("lane")
        if lane is None:
            continue

        shape_str = lane.get("shape")
        if not shape_str:
            continue

        coords: list[tuple[float, float]] = []
        for token in shape_str.split():
            # token: "x,y"
            try:
                xs, ys = token.split(",")
                x = float(xs)
                y = float(ys)
                coords.append((x, y))
            except ValueError:
                continue

        if coords:
            edge_shapes[eid] = coords

    return edge_shapes


def punto_medio(coords: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Calcula un punto medio simple entre el primer y último punto de la polyline.
    (Para visualización es suficiente.)
    """
    if not coords:
        return 0.0, 0.0
    if len(coords) == 1:
        return coords[0]

    (x1, y1) = coords[0]
    (x2, y2) = coords[-1]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def escribir_pois(path_out: str | Path, edge_ids: list[str], edge_shapes: dict) -> None:
    """
    Genera un .add.xml con POIs para cada edge_id en edge_ids.

    Cada POI se llama 'sensor_<idx>' y se coloca en el punto medio
    del primer carril del edge correspondiente.
    """
    path = Path(path_out)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("additional")

    for idx, eid in enumerate(edge_ids):
        if eid not in edge_shapes:
            # Si no tenemos shape para este edge, lo ignoramos
            continue

        coords = edge_shapes[eid]
        x, y = punto_medio(coords)

        poi = ET.SubElement(root, "poi")
        poi.set("id", f"sensor_{idx}")
        poi.set("x", f"{x:.2f}")
        poi.set("y", f"{y:.2f}")

        # Color amarillo (R,G,B en [0,1])
        poi.set("color", "1,0,0")

        # Capa alta para que se vean por encima de todo
        poi.set("layer", "10")

        # Tamaño del símbolo (aumentar ancho/alto)
        poi.set("width", "40")   # antes 2 → ahora mucho más grande
        poi.set("height", "40")

        # Opcional: texto emergente con el id del edge
        poi.set("name", eid)

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"POIs de sensores guardados en: {path}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) != 3:
        print("Uso: python -m scripts.make_sensor_pois "
              "<selected_edges_txt> <net_xml> <output_add_xml>")
        sys.exit(1)

    selected_txt = args[0]
    net_xml = args[1]
    output_add = args[2]

    edge_ids = leer_selected_edges(selected_txt)
    print(f"Leídos {len(edge_ids)} edges con sensor de {selected_txt}")

    edge_shapes = cargar_shapes_net(net_xml)
    print(f"Shapes disponibles para {len(edge_shapes)} edges en {net_xml}")

    escribir_pois(output_add, edge_ids, edge_shapes)


if __name__ == "__main__":
    main()
