# sa_algorithm.py
"""
Búsqueda por Simulated Annealing para colocación de sensores
usando directamente los flujos reales del preprocesado (sin
resolver sistemas de ecuaciones ni inferir flujos).

Idea del dominio:
- A partir de los CSV de edgeData y el preprocessing,
  construimos un vector de "flujo medio" por tramo (edge).
- Una solución es una máscara binaria de longitud = nº de edges.
- La fitness es el flujo NO cubierto:
      fitness(mask) = total_flow - covered_flow
  donde covered_flow = sum(mask[i] * flow_edge[i]).
  Es un problema de MINIMIZACIÓN:
      cuanto menor fitness, mejor (más flujo cubierto).

Esto evita la parte costosa de least-squares y hace el SA
muy ligero incluso con muchas iteraciones.

Se mantiene el módulo ga_algorithm.py tal cual.
"""

from __future__ import annotations

import numpy as np
import random
import math
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

from . import preprocessing  # usamos el pipeline de flujos ya definido


# ----------------------------------------------------------------------
# Dominio basado en flujos reales del preprocessing
# ----------------------------------------------------------------------
class FlowCoverageDomain:
    """
    Dominio sencillo para el SA:

    - edge_ids: lista de IDs de tramo (edge_id)
    - flow_vector: np.array con flujo medio (veh/h) por edge_id

    La evaluación de una máscara de sensores se basa en:
        flujo_cubierto = sum(mask[i] * flow_vector[i])
        flujo_no_cubierto = total_flow - flujo_cubierto  (fitness)
    """

    def __init__(self, edge_ids: Iterable[str], flow_vector: np.ndarray):
        edge_ids = list(edge_ids)
        flow_vector = np.asarray(flow_vector, dtype=float)

        assert len(edge_ids) == len(flow_vector), (
            "edge_ids y flow_vector deben tener la misma longitud"
        )

        self.edge_ids = edge_ids
        self.num_edges = len(edge_ids)
        self.flow_vector = flow_vector
        self.total_flow = float(flow_vector.sum())

        if self.total_flow <= 0:
            print(
                "[FlowCoverageDomain] Aviso: total_flow <= 0. "
                "Revisa los flujos de entrada."
            )

    def evaluate_mask(self, sensor_mask: np.ndarray) -> float:
        """
        Devuelve la fitness de la máscara:
            fitness = flujo_no_cubierto = total_flow - covered_flow
        Si no hay sensores, devolvemos un valor grande.
        """
        sensor_mask = np.asarray(sensor_mask, dtype=int)
        num_sensors = int(sensor_mask.sum())

        if num_sensors == 0:
            # "pena de muerte" si no se coloca ningún sensor
            return 1e9

        covered_flow = float(sensor_mask @ self.flow_vector)
        uncovered_flow = self.total_flow - covered_flow

        # Nos aseguramos de que sea no negativa numéricamente
        return max(uncovered_flow, 0.0)


def build_domain_from_preprocessing() -> FlowCoverageDomain:
    """
    Construye automáticamente un FlowCoverageDomain usando el
    pipeline de preprocessing:

      1) cargar_todos_los_edgeData()
      2) preparar_flows_para_milp()
      3) agrupar por edge_id el flujo medio en veh/h

    Devuelve un FlowCoverageDomain listo para usar en el SA.
    """
    # 1) Cargar todos los edgeData_scen*.csv
    df_edge_all = preprocessing.cargar_todos_los_edgeData()

    # 2) Calcular tau, flow_veh_s, flow_veh_h...
    df_flows = preprocessing.preparar_flows_para_milp(df_edge_all)

    if df_flows.empty:
        raise ValueError(
            "[build_domain_from_preprocessing] df_flows está vacío. "
            "Revisa los CSV en data/processed/."
        )

    if "edge_id" not in df_flows.columns or "flow_veh_h" not in df_flows.columns:
        raise KeyError(
            "[build_domain_from_preprocessing] df_flows debe tener columnas "
            "'edge_id' y 'flow_veh_h'."
        )

    # 3) Flujo medio por tramo (agregando sobre tau y escenarios)
    agg = (
        df_flows.groupby("edge_id", as_index=True)["flow_veh_h"]
        .mean()
        .sort_index()
    )

    edge_ids = list(agg.index)
    flow_vector = agg.to_numpy()

    print(
        f"[build_domain_from_preprocessing] Dominio construido con "
        f"{len(edge_ids)} edges."
    )

    return FlowCoverageDomain(edge_ids, flow_vector)


# ----------------------------------------------------------------------
# Simulated Annealing
# ----------------------------------------------------------------------
class SimulatedAnnealing:
    def __init__(
        self,
        domain: FlowCoverageDomain,
        budget: int,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.99,
        min_temp: float = 1.0,
        max_iter: int = 1000,
        fitness_func=None,              # <--- NUEVO
    ):
        self.domain = domain
        self.budget = int(budget)

        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.min_temp = float(min_temp)
        self.max_iter = int(max_iter)

        self.num_edges = self.domain.num_edges

        if self.budget <= 0 or self.budget > self.num_edges:
            raise ValueError(
                f"Presupuesto de sensores inválido: {self.budget} "
                f"(num_edges = {self.num_edges})."
            )

        # Si no nos pasan una función de fitness, usamos la del dominio
        if fitness_func is None:
            self.fitness_func = lambda mask: self.domain.evaluate_mask(mask)
        else:
            self.fitness_func = fitness_func

    def _evaluate(self, mask) -> float:
        """Evalúa la máscara con la función de fitness configurada."""
        return float(self.fitness_func(mask))


    # ----------------- generación de soluciones -----------------------
    def _generate_initial_solution(self) -> np.ndarray:
        """
        Genera una máscara inicial aleatoria con exactamente `budget`
        sensores activos (1) y el resto a 0.
        """
        mask = np.zeros(self.num_edges, dtype=int)
        indices = np.random.choice(self.num_edges, self.budget, replace=False)
        mask[indices] = 1
        return mask

    def _get_neighbor(self, current_mask: np.ndarray) -> np.ndarray:
        """
        Genera una solución vecina intercambiando un sensor (1)
        con una posición vacía (0). Mantiene constante el presupuesto.
        """
        neighbor_mask = current_mask.copy()

        # Índices con sensor y sin sensor
        sensor_indices = np.where(neighbor_mask == 1)[0]
        empty_indices = np.where(neighbor_mask == 0)[0]

        if len(sensor_indices) > 0 and len(empty_indices) > 0:
            remove_idx = np.random.choice(sensor_indices)
            add_idx = np.random.choice(empty_indices)

            neighbor_mask[remove_idx] = 0
            neighbor_mask[add_idx] = 1

        return neighbor_mask

    # ---------------------- bucle principal ---------------------------
    def run(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta el recocido simulado.

        Devuelve un diccionario con:
          - 'best_solution': np.ndarray de 0/1
          - 'best_fitness': valor mínimo encontrado
          - 'history': lista con la evolución (iter, temp, fitness, best_fitness)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 1. Inicialización
        current_solution = self._generate_initial_solution()
        current_fitness = self._evaluate(current_solution)

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        temp = self.initial_temp
        history: list[Dict[str, float]] = []

        for i in range(self.max_iter):
            # Condición de parada por temperatura
            if temp < self.min_temp:
                break

            # 2. Generar vecino
            neighbor_solution = self._get_neighbor(current_solution)
            neighbor_fitness = self._evaluate(neighbor_solution)

            # 3. Regla de aceptación (minimización)
            delta = neighbor_fitness - current_fitness

            if delta < 0:
                # Mejora: siempre aceptamos
                accept = True
            else:
                # Empeora: aceptamos con probabilidad e^{-delta/T}
                prob = math.exp(-delta / temp) if temp > 0 else 0.0
                accept = random.random() < prob

            if accept:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness

                # Actualizar mejor global
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()

            # 4. Enfriamiento
            temp *= self.cooling_rate

            history.append(
                {
                    "iter": i,
                    "temp": temp,
                    "fitness": float(current_fitness),
                    "best_fitness": float(best_fitness),
                }
            )

        return {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "history": history,
        }


# ----------------------------------------------------------------------
# Pequeño main opcional (para uso desde terminal)
# ----------------------------------------------------------------------
def main():
    """
    Ejemplo sencillo para lanzar el SA desde línea de comandos:

        python -m src.sa_algorithm

    (Asumiendo que data/processed/edgeData_scen*.csv existen
     y que config.DELTA_T está bien definido).
    """
    print("=== Construyendo dominio desde preprocessing ===")
    domain = build_domain_from_preprocessing()

    B = 50  # por ejemplo, 50 sensores
    sa = SimulatedAnnealing(
        domain,
        budget=B,
        initial_temp=500.0,
        cooling_rate=0.995,
        min_temp=0.1,
        max_iter=1000,
    )

    print(f"=== Lanzando SA con B = {B} ===")
    result = sa.run(seed=42)

    print("\n--- Resultado SA ---")
    print(f"Mejor fitness (flujo NO cubierto): {result['best_fitness']:.2f}")
    print(f"Nº de sensores activos: {int(result['best_solution'].sum())}")
    print(f"Nº total de edges: {domain.num_edges}")


if __name__ == "__main__":
    main()
