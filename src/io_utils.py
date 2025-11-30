from pathlib import Path

def leer_critical_edges(path_txt: str | Path) -> list[str]:
    """
    Lee un archivo .txt con líneas tipo:
        edge:22690125#1
        edge:559367542
    y devuelve la lista de IDs de edge (strings SUMO).
    """
    path = Path(path_txt)
    edge_ids = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("edge:"):
                eid = line.split("edge:", 1)[1]
                eid = eid.strip()
                if eid:
                    edge_ids.append(eid)

    return edge_ids
