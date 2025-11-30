import pandas as pd
import numpy as np
import networkx as nx
from scipy.optimize import lsq_linear

from pathlib import Path
import json
from datetime import datetime

class TrafficNetwork:
    def __init__(self, csv_path):
        # Cargar datos
        self.data = pd.read_csv(csv_path)

        # Identificar tramos únicos
        self.edge_ids = sorted(self.data['edge_id'].unique())
        self.num_edges = len(self.edge_ids)
        self.edge_map = {eid: i for i, eid in enumerate(self.edge_ids)}

        # Reconstuimos el Flujo Real promedio para la evaluación
        # Usamos 'entered' para reconstuir el flujo
        self.flow_truth = self.data.groupby('edge_id')['entered'].mean().reindex(self.edge_ids).fillna(0).values

        # Inferencia de topología y matriz de conservación
        # Esta primera línea es topología de la red de tráfico (grafo dirigido)
        self.graph = self._infer_topology()
        # La matriz A_conservation se usará para la fitness
        # y sirve para crear matriz de conservación de flujo
        self.A_conservation = self._build_conservation_matrix()

        print(f"Sistema inicializado: {self.num_edges} tramos, {self.A_conservation.shape[0]} ecuaciones de conservación.")

    def _infer_topology(self):
        """
        Intenta deducir conexiones basándose en nombres secuenciales de tramos.
        """
        G = nx.DiGraph()
        for eid in self.edge_ids:
            G.add_node(eid)

        for u in self.edge_ids:
            for v in self.edge_ids:
                if u == v: continue
                # Parsear nombres: separar base e índice (ej. "-123#0" -> "-123", 0)
                u_base, u_idx = self._parse_id(u)
                v_base, v_idx = self._parse_id(v)

                # Si comparten base y son consecutivos, asumimos conexión
                if u_base == v_base and v_idx == u_idx + 1:
                    G.add_edge(u, v)
        return G

    def _parse_id(self, eid):
        if '#' in eid:
            parts = eid.split('#')
            try:
                return parts[0], int(parts[1])
            except:
                return eid, 0
        return eid, 0

    def _build_conservation_matrix(self):
        """
        Constuimos la matriz A para las restricciones: Flujo_u - Flujo_v = 0 (Conservación simple)
        """
        constraints = []
        for u, v in self.graph.edges():
            row = np.zeros(self.num_edges)
            row[self.edge_map[u]] = 1  # Sale de u
            row[self.edge_map[v]] = -1 # Entra en v
            constraints.append(row)

        if not constraints:
            return np.zeros((0, self.num_edges))
        return np.array(constraints)

    def evaluate_sensor_placement(self, sensor_mask, budget=None):
      """
      Función de Fitness:
      Buscamos minimizar el error Y maximizar el uso del presupuesto disponible.
      Todavía no es definitivo pues decidimos usar el multiobjetivo en el siguiente
      entregable.
      """

      # Contar sensores activos
      num_sensors = np.sum(sensor_mask)

      # 1. Validación básica -> Si no hay sensores pena de muerte
      if num_sensors == 0:
        return 10000000.0,

      # 2. Constuimos matrices (Igual que antes)
      A_cons = self.A_conservation
      b_cons = np.zeros(A_cons.shape[0])

      sensor_indices = np.where(np.array(sensor_mask) == 1)[0]
      A_meas = np.zeros((len(sensor_indices), self.num_edges))
      b_meas = np.zeros(len(sensor_indices))

      for i, idx in enumerate(sensor_indices):
        A_meas[i, idx] = 1
        b_meas[i] = self.flow_truth[idx]

      # 3. Resolvemos el sistema
      if A_cons.shape[0] > 0:
        A = np.vstack([A_cons, A_meas])
        b = np.concatenate([b_cons, b_meas])
      else:
        A = A_meas
        b = b_meas

      res = lsq_linear(A, b, bounds=(0, np.inf), method='trf')
      x_est = res.x

      # 4. Calcular RMSE
      rmse = np.sqrt(np.mean((x_est - self.flow_truth)**2))

      # Aquí se acabaría el problema si no tiviesemos presupuesto de cámaras
      fitness = rmse

      if budget is not None:

        # Caso A: Te has pasado del presupuesto (Muy malo) -> pena de muerte
        if num_sensors > budget:
            fitness += (num_sensors - budget) * 10000.0
        # Caso B: Te has quedado corto (No malo pero no ideal)
        # Si usamos menos de 'budget', añadimos una pequeña penalización
        # para animar al algoritmo a usar más sensores si eso ayuda aunque sea un poco
        # (sin pasar de presupuesto que sino se mete arriba).
        elif num_sensors < budget:
          # Penalización suave: preferimos usar 10 sensores a usar 5 si el error es similar
          fitness += (budget - num_sensors) * 50.0

      return fitness,



# Función auxiliar para visualización
def plot_solution(sensor_mask, domain):
    import matplotlib.pyplot as plt

    x_est_error = domain.evaluate_sensor_placement(sensor_mask)[0]
    print(f"Solución visualizada. Sensores: {sum(sensor_mask)}. RMSE: {x_est_error:.4f}")

    # Aquí podrías añadir un gráfico de barras comparando Real vs Estimado
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sensor_mask)), sensor_mask, color='blue', alpha=0.6, label='Sensores')
    plt.xlabel('ID de Arco')
    plt.ylabel('Presencia de Sensor')
    plt.title(f'Distribución de Sensores (RMSE: {x_est_error:.2f})')
    plt.show()


# Función auxiliar para visualización

def save_ga_solution(sensor_mask, domain, output_txt_path):
    """
    Guarda la solución del GA en un .txt con formato:
        edge:<ID_DEL_ARCO>
    una línea por cada arco donde sensor_mask == 1.

    Esto es compatible con lo que ya usas para:
    - critical_edges.txt
    - selected_edges_BXX.txt
    """
    path = Path(output_txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for bit, eid in zip(sensor_mask, domain.edge_ids):
            if bit:  # sensor activo en ese arco
                f.write(f"edge:{eid}\n")

    print(f"[GA] Solución guardada en: {path}")

