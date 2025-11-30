"""
preprocessing.py

Funciones para pasar de los CSV procesados de SUMO
(edgeData, tripinfo) a tablas limpias con índices
(a, tau, omega) y flujos listos para el MILP.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from typing import Tuple

from . import config


def cargar_todos_los_edgeData() -> pd.DataFrame:
    """
    Carga todos los CSV de edgeData_scen*.csv en data/processed/
    y los concatena en un único DataFrame.

    Se espera que cada CSV tenga al menos:
      - scenario
      - begin, end
      - edge_id
      - (opcional) nVehContrib, entered, left, speed, occupancy, ...

    Devuelve un DataFrame con todas las filas de todos los escenarios.
    """
    processed_dir = config.PROCESSED_DIR
    pattern = config.EDGE_DATA_PATTERN

    archivos = sorted(processed_dir.glob(pattern))

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos con patrón {pattern} en {processed_dir}"
        )

    dfs = []
    for f in archivos:
        print(f"Cargando edgeData desde: {f}")
        df_f = pd.read_csv(f)
        dfs.append(df_f)

    df = pd.concat(dfs, ignore_index=True)

    return df


def preparar_flows_para_milp(df_edge: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del DataFrame de edgeData concatenado, añade:

      - tau: índice de tiempo discreto (begin / DELTA_T)
      - flow_veh_s: flujo medio [veh/s]
      - flow_veh_h: flujo medio [veh/h]

    Estrategia para el flujo:
      - Si existe 'nVehContrib', usa nVehContrib / (end - begin)
      - Si NO existe 'nVehContrib' pero existe 'entered',
        usa entered / (end - begin)
      - Si no hay ninguna de las dos, el flujo queda como NaN.

    Devuelve un DataFrame reducido con columnas clave:
      - scenario, edge_id, tau, begin, end, flow_veh_s, flow_veh_h
    """
    df = df_edge.copy()

    # Aseguramos tipo numérico en begin/end
    for col in ["begin", "end"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise KeyError(f"Falta la columna {col} en edgeData.")

    # Duración del intervalo
    df["duration"] = df["end"] - df["begin"]

    # Cálculo de flujo: preferimos nVehContrib, si no, entered
    flow_num = None
    if "nVehContrib" in df.columns:
        print("Usando 'nVehContrib' para el cálculo de flujo.")
        df["nVehContrib"] = pd.to_numeric(df["nVehContrib"], errors="coerce")
        flow_num = df["nVehContrib"]
    elif "entered" in df.columns:
        print("No se encontró 'nVehContrib'. Usando 'entered' para el flujo.")
        df["entered"] = pd.to_numeric(df["entered"], errors="coerce")
        flow_num = df["entered"]
    else:
        print(
            "Aviso: no se encontraron columnas 'nVehContrib' ni 'entered'. "
            "El flujo quedará como NaN."
        )

    if flow_num is not None:
        # Evitamos divisiones por 0
        dur = df["duration"].replace(0, np.nan)
        df["flow_veh_s"] = flow_num / dur
    else:
        df["flow_veh_s"] = np.nan

    # Flujo en veh/h por comodidad
    df["flow_veh_h"] = df["flow_veh_s"] * 3600.0

    # Índice de tiempo tau (entero) a partir del inicio del intervalo
    delta_t = config.DELTA_T
    df["tau"] = (df["begin"] / delta_t).astype(int)

    # Seleccionamos columnas clave para el MILP
    columnas_salida = [
        "scenario",
        "edge_id",
        "tau",
        "begin",
        "end",
        "flow_veh_s",
        "flow_veh_h",
    ]

    # Algunas columnas podrían no existir (por ejemplo, scenario si no se pasó),
    # así que filtramos por las que realmente estén presentes.
    columnas_salida = [c for c in columnas_salida if c in df.columns]

    df_out = df[columnas_salida].copy()

    return df_out


def construir_milp_inputs() -> dict:
    """
    Función de alto nivel que:

      1) Carga todos los edgeData_scen*.csv
      2) Calcula tau y flujos medios
      3) Lee la red desde net.xml y construye arcos/nodos
      4) Calcula nodos de entrada/salida y sus O/D
      5) Calcula la cota M

    Devuelve un diccionario con:
      - 'flows': DataFrame (scenario, edge_id, tau, flow_veh_h, ...)
      - 'edges': DataFrame (edge_id, from_node, to_node)
      - 'nodes': DataFrame (node_id)
      - 'entry_nodes': lista de node_id
      - 'exit_nodes': lista de node_id
      - 'O': DataFrame (scenario, node_id, tau, O_veh_h)
      - 'D': DataFrame (scenario, node_id, tau, D_veh_h)
      - 'M': float
    """
    # 1) Flujos
    df_edge_all = cargar_todos_los_edgeData()
    df_flows = preparar_flows_para_milp(df_edge_all)

    # 2) Red (nodos y arcos)
    df_edges = leer_red_desde_netxml()
    red_info = construir_nodos_y_fronteras(df_edges)
    df_nodes = red_info["nodes"]
    entry_nodes = red_info["entry_nodes"]
    exit_nodes = red_info["exit_nodes"]

    # 3) O_i^{tau,omega} y D_i^{tau,omega}
    O_df, D_df = calcular_O_y_D(df_flows, df_edges, entry_nodes, exit_nodes)

    # 4) Big-M
    M = calcular_big_M(df_flows, factor_seguridad=1.5)

    milp_inputs = {
        "flows": df_flows,
        "edges": df_edges,
        "nodes": df_nodes,
        "entry_nodes": entry_nodes,
        "exit_nodes": exit_nodes,
        "O": O_df,
        "D": D_df,
        "M": M,
    }

    return milp_inputs


def guardar_milp_inputs(milp_inputs: dict, path: Path = None) -> None:
    """
    Guarda el diccionario milp_inputs en un archivo pickle
    (por defecto, config.MILP_INPUTS_FILE).
    """
    if path is None:
        path = config.MILP_INPUTS_FILE

    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(path, "wb") as f:
        pickle.dump(milp_inputs, f)

    print(f"milp_inputs guardado en: {path}")

def construir_nodos_y_fronteras(df_edges: pd.DataFrame) -> dict:
    """
    A partir del DataFrame de arcos (edge_id, from_node, to_node),
    construye:

      - conjunto de nodos V
      - nodos que solo tienen arcos salientes (candidatos a nodos de entrada)
      - nodos que solo tienen arcos entrantes (candidatos a nodos de salida)

    Devuelve un diccionario con:
      - 'nodes': DataFrame con una columna 'node_id'
      - 'entry_nodes': lista de node_id
      - 'exit_nodes':  lista de node_id
    """
    if df_edges.empty:
        return {
            "nodes": pd.DataFrame(columns=["node_id"]),
            "entry_nodes": [],
            "exit_nodes": [],
        }

    # Todos los nodos
    from_nodes = set(df_edges["from_node"].unique())
    to_nodes = set(df_edges["to_node"].unique())
    all_nodes = sorted(from_nodes.union(to_nodes))

    df_nodes = pd.DataFrame({"node_id": all_nodes})

    # Nodos que solo tienen salidas (no tienen entradas)
    entry_nodes = sorted(list(from_nodes - to_nodes))

    # Nodos que solo tienen entradas (no tienen salidas)
    exit_nodes = sorted(list(to_nodes - from_nodes))

    print(f"Total de nodos: {len(all_nodes)}")
    print(f"Nodos solo-entrada (candidatos O): {len(entry_nodes)}")
    print(f"Nodos solo-salida (candidatos D): {len(exit_nodes)}")

    return {
        "nodes": df_nodes,
        "entry_nodes": entry_nodes,
        "exit_nodes": exit_nodes,
    }

def calcular_big_M(df_flows: pd.DataFrame, factor_seguridad: float = 1.5) -> float:
    """
    Calcula una cota M para el MILP a partir de los flujos observados.

    Estrategia sencilla:
      - Tomamos el máximo flujo observado (en veh/h) en df_flows
      - Multiplicamos por un factor de seguridad > 1

    M se usará luego en las restricciones tipo:
      f_a^{tau,omega} <= M * x_a
    """
    if "flow_veh_h" not in df_flows.columns:
        raise KeyError("df_flows no tiene columna 'flow_veh_h'.")

    max_flow = df_flows["flow_veh_h"].max(skipna=True)

    if pd.isna(max_flow):
        raise ValueError("No se ha podido calcular M: 'flow_veh_h' está vacío o es NaN.")

    M = factor_seguridad * max_flow

    print(f"Máximo flujo observado (veh/h): {max_flow:.2f}")
    print(f"Big-M calculado con factor {factor_seguridad}: {M:.2f}")

    return float(M)

def calcular_O_y_D(
    df_flows: pd.DataFrame,
    df_edges: pd.DataFrame,
    entry_nodes: list,
    exit_nodes: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula:

      - O_df: (scenario, node_id, tau, O_veh_h)
      - D_df: (scenario, node_id, tau, D_veh_h)
    """
    df = df_flows.copy()

    # Unimos info topológica al DataFrame de flujos
    df = df.merge(df_edges, on="edge_id", how="left")

    if "from_node" not in df.columns or "to_node" not in df.columns:
        raise KeyError("Tras el merge, faltan columnas 'from_node' o 'to_node'.")

    # --- O_i^{tau,omega}: nodos de entrada -----------------------------------
    if entry_nodes:
        df_entry = df[df["from_node"].isin(entry_nodes)].copy()
        df_entry = df_entry.rename(columns={"from_node": "node_id"})
        O_df = (
            df_entry.groupby(["scenario", "node_id", "tau"], as_index=False)["flow_veh_h"]
            .sum()
            .rename(columns={"flow_veh_h": "O_veh_h"})
        )
    else:
        O_df = pd.DataFrame(columns=["scenario", "node_id", "tau", "O_veh_h"])

    # --- D_i^{tau,omega}: nodos de salida ------------------------------------
    if exit_nodes:
        df_exit = df[df["to_node"].isin(exit_nodes)].copy()
        df_exit = df_exit.rename(columns={"to_node": "node_id"})
        D_df = (
            df_exit.groupby(["scenario", "node_id", "tau"], as_index=False)["flow_veh_h"]
            .sum()
            .rename(columns={"flow_veh_h": "D_veh_h"})
        )
    else:
        D_df = pd.DataFrame(columns=["scenario", "node_id", "tau", "D_veh_h"])

    print(f"O calculado para {len(O_df)} combinaciones (scenario, node, tau).")
    print(f"D calculado para {len(D_df)} combinaciones (scenario, node, tau).")

    return O_df, D_df


def leer_red_desde_netxml(net_path: Path = None) -> pd.DataFrame:
    """
    Lee el fichero net.xml de SUMO y construye un DataFrame de arcos (edges):

        edge_id, from_node, to_node

    Ignora edges internos (function="internal") para quedarnos solo
    con los arcos "reales" de la red.
    """
    if net_path is None:
        net_path = config.NET_FILE

    if not net_path.exists():
        raise FileNotFoundError(f"No se encontró el net.xml: {net_path}")

    print(f"Leyendo red desde: {net_path}")

    tree = ET.parse(net_path)
    root = tree.getroot()

    filas = []

    for edge in root.iter("edge"):
        attrib = edge.attrib

        # Ignoramos edges internos de SUMO
        if attrib.get("function") == "internal":
            continue

        edge_id = attrib.get("id")
        from_node = attrib.get("from")
        to_node = attrib.get("to")

        if edge_id is None or from_node is None or to_node is None:
            continue

        filas.append(
            {
                "edge_id": edge_id,
                "from_node": from_node,
                "to_node": to_node,
            }
        )

    df_edges = pd.DataFrame(filas)

    if df_edges.empty:
        print("Aviso: no se han encontrado edges 'reales' en el net.xml.")

    return df_edges


