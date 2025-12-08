# src/ga_algorithm.py

"""
ga_algorithm.py

Definición de la clase TrafficNetwork utilizada por el GA y SA
para evaluar configuraciones de sensores.

Reutiliza preprocessing.py para:
  - Leer la red real de SUMO (net.xml)
  - Construir la topología (grafo dirigido de tramos)
  - Construir una matriz de "conservación" sencilla basada en conexiones
    reales entre edges (to(u) = from(v)).

La función de fitness:
  - Reconstruye un vector de flujos x_est resolviendo un LSQ:
        A x ≈ b
    combinando:
        * restricciones de conservación
        * pseudo-medidas en los arcos con sensor
  - Calcula el RMSE entre x_est y el flujo "verdadero" (flow_truth)
  - Añade penalizaciones por pasarse / quedarse corto respecto al presupuesto B.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import lsq_linear

from .preprocessing import leer_red_desde_netxml


CsvPathLike = Union[str, Path]
NetPathLike = Union[str, Path]


class TrafficNetwork:
    def __init__(
        self,
        csv_path: CsvPathLike,
        net_xml_path: Optional[NetPathLike] = None,
    ):
        """
        Parameters
        ----------
        csv_path : str or Path
            Ruta al CSV de edgeData (p.ej. edgeData_scen1.csv).
        net_xml_path : str or Path, opcional
            Ruta al fichero net.xml de SUMO. Si es None, se usará
            el valor por defecto de leer_red_desde_netxml (config.NET_FILE).
        """
        csv_path = Path(csv_path)
        self.data = pd.read_csv(csv_path)

        # --- Identificar tramos únicos -------------------------------------
        self.edge_ids: List[str] = sorted(self.data["edge_id"].unique())
        self.num_edges: int = len(self.edge_ids)
        self.edge_map: Dict[str, int] = {eid: i for i, eid in enumerate(self.edge_ids)}

        # --- Construir flujo "verdadero" -----------------------------------
        # Preferimos 'flow_veh_h' si ya vienes del preprocessing;
        # si no, usamos 'entered' como proxy de flujo.
        if "flow_veh_h" in self.data.columns:
            grp = self.data.groupby("edge_id")["flow_veh_h"].mean()
        elif "entered" in self.data.columns:
            grp = self.data.groupby("edge_id")["entered"].mean()
        else:
            raise KeyError(
                "El CSV no contiene ni 'flow_veh_h' ni 'entered' "
                "para construir flow_truth."
            )

        self.flow_truth: np.ndarray = (
            grp.reindex(self.edge_ids).fillna(0.0).astype(float).values
        )

        # --- Topología de la red: usar net.xml real ------------------------
        # Leemos la tabla de edges desde el net.xml (función de preprocessing)
        if net_xml_path is not None:
            net_xml_path = Path(net_xml_path)
            df_edges = leer_red_desde_netxml(net_xml_path)
        else:
            # Usa el net.xml por defecto definido en config.NET_FILE
            df_edges = leer_red_desde_netxml()

        # Filtramos a los edges presentes en nuestros datos
        df_edges = df_edges[df_edges["edge_id"].isin(self.edge_ids)].copy()

        # edge_id -> (from_node, to_node)
        self.edge_endpoints: Dict[str, Tuple[str, str]] = {}
        for row in df_edges.itertuples(index=False):
            self.edge_endpoints[row.edge_id] = (row.from_node, row.to_node)

        # Construir grafo de edges (nodos = edge_id)
        self.graph: nx.DiGraph = self._build_edge_graph()

        # Matriz de "conservación" simple a partir de ese grafo
        self.A_conservation: np.ndarray = self._build_conservation_matrix()

        print(
            f"Sistema inicializado: {self.num_edges} tramos, "
            f"{self.A_conservation.shape[0]} ecuaciones de conservación."
        )

    # ------------------------------------------------------------------ #
    # Construcción de topología y matriz de conservación
    # ------------------------------------------------------------------ #
    def _build_edge_graph(self) -> nx.DiGraph:
        """
        Construye un grafo dirigido donde cada nodo es un tramo (edge_id)
        y añadimos arcos u -> v si to(u) == from(v) según la red SUMO.
        """
        G = nx.DiGraph()

        # Crear todos los nodos, incluso si algún edge no está en edge_endpoints
        for eid in self.edge_ids:
            G.add_node(eid)

        # edge_endpoints[eid] = (from_node, to_node)
        edges_from: Dict[str, List[str]] = {}
        for eid, (u_from, u_to) in self.edge_endpoints.items():
            edges_from.setdefault(u_from, []).append(eid)

        # Conexión real: u -> v si to(u) == from(v)
        for eid_u, (u_from, u_to) in self.edge_endpoints.items():
            if u_to not in edges_from:
                continue
            for eid_v in edges_from[u_to]:
                if eid_u == eid_v:
                    continue
                G.add_edge(eid_u, eid_v)

        return G

    def _build_conservation_matrix(self) -> np.ndarray:
        """
        Construimos una matriz A_conservation sencilla donde cada fila
        representa una ecuación del tipo:

            flujo(u) - flujo(v) = 0

        para cada arco u -> v del grafo de edges.

        No es una conservación nodal completa, pero introduce consistencia
        entre flujos de edges consecutivos según la topología de SUMO.
        """
        constraints: List[np.ndarray] = []

        for u, v in self.graph.edges():
            row = np.zeros(self.num_edges, dtype=float)
            row[self.edge_map[u]] = 1.0   # sale de u
            row[self.edge_map[v]] = -1.0  # entra en v
            constraints.append(row)

        if not constraints:
            return np.zeros((0, self.num_edges), dtype=float)
        return np.vstack(constraints)

    # ------------------------------------------------------------------ #
    # Función de evaluación (fitness) para GA/SA
    # ------------------------------------------------------------------ #
    def evaluate_sensor_placement(self, sensor_mask, budget: Optional[int] = None):
        """
        Evalúa una configuración de sensores (máscara binaria 0/1 por tramo).

        Pasos:
          1) Construir ecuaciones de conservación (A_cons x = 0).
          2) Añadir ecuaciones de pseudo-medida en los edges con sensor:
               x_a = flow_truth[a]
          3) Resolver el LSQ:
               min ||A x - b||_2  con x >= 0
          4) Calcular RMSE entre x_est y flow_truth.
          5) Añadir penalizaciones por violar el presupuesto (si budget no es None).

        Devuelve:
          (fitness,)  donde fitness es un escalar a minimizar.
        """
        sensor_mask = np.asarray(sensor_mask, dtype=int)
        num_sensors = int(sensor_mask.sum())

        # 1) Validación básica -> si no hay sensores, fitness enorme
        if num_sensors == 0:
            return (1.0e7,)

        # 2) Construir matrices
        A_cons = self.A_conservation
        b_cons = np.zeros(A_cons.shape[0], dtype=float)

        sensor_indices = np.where(sensor_mask == 1)[0]
        A_meas = np.zeros((len(sensor_indices), self.num_edges), dtype=float)
        b_meas = np.zeros(len(sensor_indices), dtype=float)

        for i, idx in enumerate(sensor_indices):
            A_meas[i, idx] = 1.0
            b_meas[i] = self.flow_truth[idx]

        if A_cons.shape[0] > 0:
            A = np.vstack([A_cons, A_meas])
            b = np.concatenate([b_cons, b_meas])
        else:
            A = A_meas
            b = b_meas

        # 3) Resolver LSQ con x >= 0
        res = lsq_linear(A, b, bounds=(0.0, np.inf), method="trf")
        x_est = res.x

        # 4) RMSE global respecto a flow_truth
        rmse = float(np.sqrt(np.mean((x_est - self.flow_truth) ** 2)))
        fitness = rmse

        # 5) Penalizaciones de presupuesto (si se especifica)
        if budget is not None:
            if num_sensors > budget:
                # Pasarse de B es muy malo
                fitness += (num_sensors - budget) * 10000.0
            elif num_sensors < budget:
                # Quedarse corto se penaliza suavemente
                """ fitness += (budget - num_sensors) * 50.0 """

        return (fitness,)


# ---------------------------------------------------------------------- #
# Funciones auxiliares de visualización / guardado
# ---------------------------------------------------------------------- #
def plot_solution(sensor_mask, domain: TrafficNetwork):
    """
    Visualiza la distribución de sensores (barra 0/1 por tramo)
    e imprime el fitness de la solución.
    """
    import matplotlib.pyplot as plt

    fitness_val = domain.evaluate_sensor_placement(sensor_mask)[0]
    print(
        f"Solución visualizada. Sensores: {int(sum(sensor_mask))}. "
        f"Fitness (RMSE + penalización): {fitness_val:.4f}"
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        range(len(sensor_mask)),
        sensor_mask,
        alpha=0.6,
        label="Sensores (0/1)",
    )
    plt.xlabel("Índice de tramo (edge)")
    plt.ylabel("Presencia de sensor")
    plt.title(f"Distribución de sensores (fitness: {fitness_val:.2f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_ga_solution(sensor_mask, domain: TrafficNetwork, output_txt_path: CsvPathLike):
    """
    Guarda la solución del GA/SA en un .txt con formato:
        edge:<ID_DEL_ARCO>

    una línea por cada arco donde sensor_mask == 1.
    Compatible con critical_edges.txt y selected_edges_BXX.txt.
    """
    path = Path(output_txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for bit, eid in zip(sensor_mask, domain.edge_ids):
            if bit:
                f.write(f"edge:{eid}\n")

    print(f"[GA] Solución guardada en: {path}")

