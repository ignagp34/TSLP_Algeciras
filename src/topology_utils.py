import networkx as nx
import pandas as pd
from typing import Dict, Any

def construir_grafo_desde_milp_inputs(milp_inputs: Dict[str, Any]) -> tuple[nx.DiGraph, dict]:
    """
    Construye un grafo dirigido de networkx a partir de milp_inputs['edges'].

    Devuelve:
      - G: DiGraph con nodos = node_id y aristas con atributo 'edge_id'
      - edge_id_to_uv: dict {edge_id: (u, v)} para acceso rápido
    """
    edges: pd.DataFrame = milp_inputs["edges"]

    G = nx.DiGraph()
    edge_id_to_uv: dict[str, tuple[str, str]] = {}

    for _, row in edges.iterrows():
        u = row["from_node"]
        v = row["to_node"]
        eid = row["edge_id"]

        G.add_edge(u, v, edge_id=eid)
        edge_id_to_uv[eid] = (u, v)

    return G, edge_id_to_uv

def ciclos_en_edges_criticos(
    milp_inputs: Dict[str, Any],
    critical_edges: list[str],
) -> list[list[str]]:
    """
    Devuelve una lista de ciclos, cada uno como lista de edge_id,
    contenidos en el subgrafo formado por los edges críticos.

    Uso típico: para cada ciclo C, añadir restricción sum_{a in C} x_a >= 1.
    """
    import networkx as nx

    G, edge_id_to_uv = construir_grafo_desde_milp_inputs(milp_inputs)

    # Filtramos únicamente arcos críticos que existan en el grafo
    critical_uv = []
    for eid in critical_edges:
        if eid in edge_id_to_uv:
            critical_uv.append(edge_id_to_uv[eid])

    # Subgrafo dirigido con solo esos arcos
    Gc = G.edge_subgraph(critical_uv).copy()

    # Trabajamos sobre no dirigido para sacar un ciclo base
    Gc_und = Gc.to_undirected()

    # Lista de ciclos como lista de nodos
    node_cycles = nx.cycle_basis(Gc_und)

    cycles_edge_ids: list[list[str]] = []

    for cyc_nodes in node_cycles:
        # cerramos ciclo: n0, n1, ..., nk, n0
        cyc_edges: list[str] = []
        for u, v in zip(cyc_nodes, cyc_nodes[1:] + [cyc_nodes[0]]):
            if Gc.has_edge(u, v):
                eid = Gc[u][v]["edge_id"]
            elif Gc.has_edge(v, u):
                eid = Gc[v][u]["edge_id"]
            else:
                continue
            cyc_edges.append(eid)

        # eliminamos duplicados preservando orden
        if cyc_edges:
            seen = set()
            cyc_edges_unique = []
            for eid in cyc_edges:
                if eid not in seen:
                    seen.add(eid)
                    cyc_edges_unique.append(eid)

            if len(cyc_edges_unique) >= 2:
                cycles_edge_ids.append(cyc_edges_unique)

    return cycles_edge_ids

def cortes_minimos_en_zona_critica(
    milp_inputs: Dict[str, Any],
    critical_edges: list[str],
    max_pairs: int = 50,
) -> list[list[str]]:
    """
    Calcula algunos cortes mínimos (edge cuts) en el subgrafo de edges críticos.

    Devuelve lista de cortes, cada uno como lista de edge_id.

    max_pairs limita el nº de pares s-t para no explotar.
    """
    import networkx as nx

    G, edge_id_to_uv = construir_grafo_desde_milp_inputs(milp_inputs)

    # Subgrafo con solo edges críticos
    critical_uv = [
        edge_id_to_uv[eid]
        for eid in critical_edges
        if eid in edge_id_to_uv
    ]
    Gc = G.edge_subgraph(critical_uv).copy()

    entry_nodes = milp_inputs.get("entry_nodes", [])
    exit_nodes = milp_inputs.get("exit_nodes", [])

    cuts: list[list[str]] = []
    pair_count = 0

    for s in entry_nodes:
        for t in exit_nodes:
            if pair_count >= max_pairs:
                return cuts

            # Solo si s y t están conectados en el subgrafo
            if s in Gc and t in Gc and nx.has_path(Gc, s, t):
                cut_uv = nx.minimum_edge_cut(Gc, s, t)
                cut_eids = [Gc[u][v]["edge_id"] for u, v in cut_uv]
                if cut_eids:
                    cuts.append(cut_eids)
                    pair_count += 1

    return cuts
