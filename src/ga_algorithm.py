# src/ga_algorithm.py

"""
ga_algorithm.py

Definición de la clase TrafficNetwork utilizada por el GA y SA
para evaluar configuraciones de sensores, ADAPTADA a la metodología del paper.

Metodología:
  - Fitness: Weighted L1-norm objective function.
  - Evaluación Híbrida:
      * Exact (LP): Resuelve el problema de optimización completo.
      * Surrogate (Fast): Aproximación heurística rápida usando LSQ para proyección de flujo.
  - Operadores:
      * Repair: Asegura presupuesto y cobertura (k+1 coverage).
"""

from pathlib import Path
from typing import Optional, Union, Dict, Tuple, List, Set, Any
import random

import numpy as np
import pandas as pd
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
from scipy.optimize import lsq_linear

from .preprocessing import leer_red_desde_netxml, construir_milp_inputs
from .topology_utils import ciclos_en_edges_criticos, cortes_minimos_en_zona_critica
from .io_utils import leer_critical_edges
from .config import PROJECT_ROOT
from .milp_model import build_sensor_placement_model


CsvPathLike = Union[str, Path]
NetPathLike = Union[str, Path]


class TrafficNetwork:
    def __init__(
        self,
        csv_path: Optional[CsvPathLike] = None,
        net_xml_path: Optional[NetPathLike] = None,
        critical_edges_path: Optional[Path] = None,
        k_coverage: int = 1,
        milp_inputs: Optional[Dict[str, Any]] = None, # Permitir inyección directa
        epsilon_weight: float = 1.0,
        lambda_flow_balance: float = 0.1,
    ):

        """
        Inicializa el entorno para el GA.
        Carga datos, topología, y pre-calcula ciclos y cortes críticos.
        """
        print("--- [TrafficNetwork] Inicializando ---")
        
        # 1. Cargar inputs (inyectados o por defecto)
        if milp_inputs is not None:
            print("    -> Usando milp_inputs proporcionados externally.")
            self.milp_inputs = milp_inputs
        else:
            print("    -> Cargando milp_inputs por defecto desde disco.")
            self.milp_inputs = construir_milp_inputs()

        self.flows = self.milp_inputs["flows"]
        self.edges = self.milp_inputs["edges"]
        
        # Identificar tramos únicos y mapeo
        # USAR SOLO ARCOS CON FLUJO (flows) PARA EVITAR ARCOS VACÍOS
        self.edge_ids: List[str] = sorted(self.flows["edge_id"].unique())
        self.num_edges: int = len(self.edge_ids)
        self.edge_map: Dict[str, int] = {eid: i for i, eid in enumerate(self.edge_ids)}
        
        # Construir matrices para Surrogate (LSQ)
        # Necesitamos grafo para matriz de conservación
        self._build_matrices_for_surrogate()

        # 2. Configurar "Flow Truth" y Pesos
        self.flow_truth_map = {}
        self.weights_map = {}
        self.epsilon_weight = epsilon_weight
        
        # Pre-calcular pesos custom
        # Estructura: dict[(scen, tau, edge)] -> float
        for _, row in self.flows.iterrows():
            key = (row["scenario"], int(row["tau"]), row["edge_id"])
            val = float(row["flow_veh_h"])
            self.flow_truth_map[key] = val
            self.weights_map[key] = 1.0 / max(abs(val), self.epsilon_weight)
            
        # 2b. Compute aggregated flow_truth (average over scenarios/time) for SA
        self.flow_truth = np.zeros(self.num_edges)
        if not self.flows.empty:
            # Group by edge_id and mean
            avg_flows = self.flows.groupby("edge_id")["flow_veh_h"].mean()
            for i, eid in enumerate(self.edge_ids):
                self.flow_truth[i] = avg_flows.get(eid, 0.0)

        # 3. Cargar robustez (Ciclos y Cortes)
        if critical_edges_path is None:
             critical_edges_path = PROJECT_ROOT / "data" / "processed" / "critical_edges.txt"
        
        print(f"    -> Buscando critical_edges en: {critical_edges_path}")
        self.critical_edges: List[str] = []
        if critical_edges_path.exists():
            self.critical_edges = leer_critical_edges(critical_edges_path)
            self.critical_edges = [e for e in self.critical_edges if e in self.edge_ids]
            print(f"    -> Critical Edges cargados: {len(self.critical_edges)}")
        else:
            print("    -> [WARNING] critical_edges.txt NO ENCONTRADO.")
        
        self.cycles = ciclos_en_edges_criticos(self.milp_inputs, self.critical_edges)
        self.cuts = cortes_minimos_en_zona_critica(self.milp_inputs, self.critical_edges)
        print(f"    -> Ciclos cargados: {len(self.cycles)}, Cortes cargados: {len(self.cuts)}")
        if len(self.cycles) > 0:
            print(f"    -> DEBUG First Cycle: {self.cycles[0]}")
        print(f"    -> DEBUG First 5 Edges: {self.edge_ids[:5]}")
        
        self.k_coverage = k_coverage
        self.lambda_flow_balance = float(lambda_flow_balance)
        self.epsilon_weight = float(epsilon_weight)

        # Pre-calcular frecuencias para repair
        self.edge_freq = {eid: 0 for eid in self.edge_ids}
        for s in self.cycles + self.cuts:
            for eid in s:
                if eid in self.edge_freq:
                    self.edge_freq[eid] += 1
        
        self.costs = {eid: 1.0 for eid in self.edge_ids} 

        # Promedio de pesos
        self.avg_weights = {eid: 0.0 for eid in self.edge_ids}
        counts = {eid: 0 for eid in self.edge_ids}
        for (scen, tau, eid), w in self.weights_map.items():
            if eid in self.avg_weights:
                self.avg_weights[eid] += w
                counts[eid] += 1
        for eid in self.avg_weights:
            if counts[eid] > 0:
                self.avg_weights[eid] /= counts[eid]

            # 4. Pre-calcular vectores para Surrogate (LSQ) para evitar pandas en el loop
        print("    -> Pre-calculando vectores Surrogate...")
        self.surrogate_data = {} # (sc, t) -> (b_bal_vec, f_prior_vec, total_weights_vec)
        
        # Iterar todos los snapshots disponibles
        # Usamos los keys de b_balance_map que ya filtramos/creamos
        all_keys = list(self.b_balance_map.keys())
        
        for (sc, t) in all_keys:
            b_bal = self.b_balance_map[(sc, t)]
            
            # f_prior vec
            f_vec = np.zeros(self.num_edges)
            w_vec = np.zeros(self.num_edges)
            
            # Optimización: acceder al df filtrado es lento.
            # Mejor: iterar sobre 'self.flow_truth_map' que ya es un dict
            # Pero iterar edges es 2500 iteraciones.
            # Vectorizado:
            for idx, eid in enumerate(self.edge_ids):
                # flow_truth_map tiene (sc, t, eid)
                val = self.flow_truth_map.get((sc, t, eid), 0.0)
                f_vec[idx] = val
                w_vec[idx] = self.weights_map.get((sc, t, eid), 1.0)
                
            self.surrogate_data[(sc, t)] = (b_bal, f_vec, w_vec)
            
        self.snapshot_keys = all_keys
        print("--- [TrafficNetwork] Inicialización completa ---")

    def _build_matrices_for_surrogate(self):
        """
        Construye la matriz de conservación A_cons para el LSQ surrogate.
        """
        G = nx.DiGraph()
        for eid in self.edge_ids:
            G.add_node(eid)
            
        self.node_ids = sorted(self.milp_inputs["nodes"])
        self.node_map = {nid: i for i, nid in enumerate(self.node_ids)}
        
        rows = []
        cols = []
        vals = []
        
        for idx_e, row in self.edges.iterrows():
            eid = row["edge_id"]
            if eid not in self.edge_map: continue
            j = self.edge_map[eid]
            u = row["from_node"]
            v = row["to_node"]
            
            if u in self.node_map:
                rows.append(self.node_map[u])
                cols.append(j)
                vals.append(1.0) # Sale
            
            if v in self.node_map:
                rows.append(self.node_map[v])
                cols.append(j)
                vals.append(-1.0) # Entra
                
        from scipy.sparse import coo_matrix
        # Guardar como densa para lsq_linear (versiones antiguas de scipy requieren densa)
        self.A_balance = coo_matrix((vals, (rows, cols)), shape=(len(self.node_ids), self.num_edges)).toarray()
        
        # Pre-calcular b_balance map
        self.b_balance_map = {}
        O_df = self.milp_inputs.get("O", pd.DataFrame())
        D_df = self.milp_inputs.get("D", pd.DataFrame())
        
        # Necesitamos iterar sobre las combinaciones presentes en flows
        # Groupby rápido para obtener keys
        if "tau" in self.flows.columns and "scenario" in self.flows.columns:
            groups = self.flows[["scenario", "tau"]].drop_duplicates()
            keys_st = list(groups.itertuples(index=False, name=None))
        else:
            keys_st = []

        # Convertir O y D a dict para acceso rápido
        # (sc, tau, node) -> val
        O_dict = {}
        if not O_df.empty:
            for r in O_df.itertuples():
                O_dict[(r.scenario, r.tau, r.node_id)] = r.O_veh_h
        
        D_dict = {}
        if not D_df.empty:
            for r in D_df.itertuples():
                D_dict[(r.scenario, r.tau, r.node_id)] = r.D_veh_h

        for (sc, t) in keys_st:
            b = np.zeros(len(self.node_ids))
            
            # Sumar O
            # Iterar nodos es rápido (pocos nodos)
            # O mejor: iterar solo entries/exits conocidos?
            # Iteramos entries
            for nid in self.milp_inputs.get("entry_nodes", []):
                val = O_dict.get((sc, t, nid), 0.0)
                if val > 0 and nid in self.node_map:
                     b[self.node_map[nid]] += val
            
            for nid in self.milp_inputs.get("exit_nodes", []):
                val = D_dict.get((sc, t, nid), 0.0)
                if val > 0 and nid in self.node_map:
                     b[self.node_map[nid]] -= val
                     
            self.b_balance_map[(sc, t)] = b


    # ------------------------------------------------------------------ #
    #  REPAIR OPERATOR
    # ------------------------------------------------------------------ #
    def calculate_benefit_cost_ratio(self, edge_id: str, gamma1: float = 1.0, gamma2: float = 1.0) -> float:
        freq = self.edge_freq.get(edge_id, 0)
        w_bar = self.avg_weights.get(edge_id, 0.0)
        c_a = self.costs.get(edge_id, 1.0)
        if c_a == 0: return 999999.0
        return (gamma1 * freq + gamma2 * w_bar) / c_a

    def repair_individual(self, individual: List[int], budget: int) -> List[int]:
        active_indices = [i for i, x in enumerate(individual) if x == 1]
        
        if len(active_indices) > budget:
            active_indices.sort(key=lambda idx: self.calculate_benefit_cost_ratio(self.edge_ids[idx]))
            while len(active_indices) > budget:
                individual[active_indices.pop(0)] = 0
        
        sets_to_check = self.cycles + self.cuts
        active_set = set(self.edge_ids[i] for i, x in enumerate(individual) if x == 1)
        dirty = True
        iter_count = 0
        
        while dirty and iter_count < 100:
            dirty = False
            iter_count += 1
            
            for component_edges in sets_to_check:
                count = sum(1 for e in component_edges if e in active_set)
                needed = (self.k_coverage + 1) - count
                if needed > 0:
                    candidates = [e for e in component_edges if e not in active_set]
                    candidates.sort(key=lambda e: self.calculate_benefit_cost_ratio(e), reverse=True)
                    for _ in range(needed):
                        if not candidates: break
                        best = candidates.pop(0)
                        individual[self.edge_map[best]] = 1
                        active_set.add(best)
                        dirty = True
        
        if len(active_set) > budget:
            active_indices = [i for i, x in enumerate(individual) if x == 1]
            active_indices.sort(key=lambda idx: self.calculate_benefit_cost_ratio(self.edge_ids[idx]))
            while len(active_indices) > budget:
                individual[active_indices.pop(0)] = 0
                
        return individual

    # ------------------------------------------------------------------ #
    #  EVALUATION: Exact vs Surrogate
    # ------------------------------------------------------------------ #
    def _get_cached_exact_model(self):
        if not hasattr(self, "_cached_model"):
            print("    -> Building EXACT model (cached)...")
            model, x_vars = build_sensor_placement_model(
                self.milp_inputs,
                max_sensors=None,
                weight_scheme="inv_abs", 
                lambda_flow_balance=self.lambda_flow_balance,
                epsilon_weight=self.epsilon_weight,
                forced_cycles=self.cycles,
                forced_cuts=self.cuts
            )
            self._cached_model = model
            self._cached_x_vars = x_vars
        return self._cached_model, self._cached_x_vars

    def evaluate_exact(self, individual: List[int]) -> float:
        if sum(individual) == 0: return 1e9
        
        model, x_vars = self._get_cached_exact_model()
        
        # Update variable bounds to fix them to the individual's values
        for i, eid in enumerate(self.edge_ids):
            if eid in x_vars:
                val = individual[i]
                # Modifying bounds is faster than re-creating variables
                x_vars[eid].lowBound = val
                x_vars[eid].upBound = val
        
        solver = PULP_CBC_CMD(msg=False)
        model.solve(solver)
        
        if model.status != 1: return 1e9
        if model.status != 1: return 1e9
        return value(model.objective)

    def evaluate_surrogate(self, individual: List[int], sample_size: int = 5) -> float:
        """
        Evaluación Surrogate con ITERATIVE NODAL BALANCING (Heurística O(N)).
        Reemplaza a lsq_linear para extrema velocidad y robustez ante "congestión" (infeasibility).
        
        Algoritmo (para un snapshot):
          1. f_tilde = f_prior
          2. Repetir k veces (o hasta convergencia suave):
             Para cada nodo n:
                calc imbalance I = sum(f_in) - sum(f_out) + net_source(n)
                repartir error I entre arcos incidentes para reducirlo.
          3. Calcular residual sobre sensores.
        """
        if sum(individual) == 0: return 1e9
        sensor_idxs = [i for i, x in enumerate(individual) if x == 1]
        
        n_snapshots = len(self.snapshot_keys)
        if n_snapshots == 0: return 1e9
        k = min(sample_size, n_snapshots)
        indices = random.sample(range(n_snapshots), k)

        total_weighted_residual = 0.0
        
        # Pre-computar estructura de adyacencia rápida si no existe
        if not hasattr(self, "_adj_struct"):
            # self.node_map ya existe.
            # Necesitamos: node_idx -> list of (edge_idx, direction +1/-1)
            # direction: +1 sale (out) -> reduce balance (out increases => total balance decreases?)
            # balance = sum(in) - sum(out) + source.
            # if edge j enters node (direction -1 in A_matrix): contributes +f_j
            # if edge j leaves node (direction +1 in A_matrix): contributes -f_j
            # Wait, my A matrix definition was: +1 sale, -1 entra.
            # A_row * f = sum(f_out) - sum(f_in) = b (source - sink)
            # Imbalance = sum(f_out) - sum(f_in) - b.
            # Queremos Imbalance = 0.
            
            self._adj_struct = [[] for _ in range(len(self.node_ids))]
            # Iterar A_balance sparse o edges
            # A_balance es denso ahora. Usar edges es mejor.
            
            for idx_e, row in self.edges.iterrows():
                eid = row["edge_id"]
                if eid not in self.edge_map: continue
                j = self.edge_map[eid]
                
                u = row["from_node"]
                v = row["to_node"]
                
                if u in self.node_map:
                    u_idx = self.node_map[u]
                    self._adj_struct[u_idx].append((j, 1.0)) # Sale
                
                if v in self.node_map:
                    v_idx = self.node_map[v]
                    self._adj_struct[v_idx].append((j, -1.0)) # Entra

        adj = self._adj_struct
        num_nodes = len(adj)
        num_edges = self.num_edges
        
        for idx_snap in indices:
            key = self.snapshot_keys[idx_snap]
            b_bal, f_prior, w_vec = self.surrogate_data[key]
            
            # 1. Init f_tilde con f_prior
            f_tilde = f_prior.copy()
            
            # 2. Iterative Balancing (5 passes)
            # Esto difunde el error de conservación localmente
            # Maneja bien la congestión (donde in != out) suavizándola
            for _ in range(5):
                # Calcular imbalances y corregir nodo a nodo
                # (Gauss-Seidel style update)
                for n_idx in range(num_nodes):
                    # Calc current imbalance: sum(sign * f) - b
                    imbalance = -b_bal[n_idx]
                    incident_edges = adj[n_idx]
                    if not incident_edges: continue
                    
                    for (e_idx, sign) in incident_edges:
                        imbalance += sign * f_tilde[e_idx]
                    
                    # Si imbalance != 0, distribuir corrección
                    # Correction delta para cada edge: -sign * (imbalance / degrees)
                    # Relax factor 0.5 para estabilidad
                    if abs(imbalance) > 1e-4:
                        delta = -(imbalance * 0.5) / len(incident_edges)
                        for (e_idx, sign) in incident_edges:
                            # Apply delta * sign?
                            # Queremos reducir imbalance.
                            # New f = f + correction
                            # New imbalance = Old + sum(sign * correction)
                            # = Old + sum(sign * sign * delta) = Old + sum(delta) = Old + N*delta
                            # Queremos New = 0 => N*delta = -Old => delta = -Old/N. Correcto.
                            
                            f_tilde[e_idx] += sign * delta
                            # Proyectar a >= 0
                            if f_tilde[e_idx] < 0: f_tilde[e_idx] = 0
            
            # 3. Calc fitness
            snapshot_res = 0.0
            for s_idx in sensor_idxs:
                diff = abs(f_tilde[s_idx] - f_prior[s_idx])
                w = w_vec[s_idx]
                snapshot_res += w * diff
                # 4) Slack/balance term 
                imb = self.A_balance @ f_tilde - b_bal
                balance_term = self.lambda_flow_balance * float(np.sum(np.abs(imb)))
                snapshot_res += balance_term
            total_weighted_residual += snapshot_res

        avg_residual = total_weighted_residual / k
        base_fitness = avg_residual * n_snapshots
        
        # --- PENALTY FOR UNSATISFIED CONSTRAINTS ---
        # The surrogate must guide the GA towards feasible regions.
        # Check cycles and cuts.
        
        active_set_check = set(self.edge_ids[i] for i in sensor_idxs)
        violations = 0
        total_missing = 0
        
        # Check both cycles and cuts
        for component_edges in (self.cycles + self.cuts):
            # Count sensors in this component
            count = sum(1 for e in component_edges if e in active_set_check)
            needed = (self.k_coverage + 1)
            if count < needed:
                violations += 1
                total_missing += (needed - count)
        
        # Penalty calculation
        # If violations > 0, we add a huge penalty.
        # We scale it by total_missing to provide gradient.
        penalty = 0.0
        if violations > 0:
            penalty = 1e6 + (total_missing * 1000.0)
            
        return base_fitness + penalty

    def fitness_function(self, individual: List[int], use_surrogate: bool = True) -> Tuple[float]:
        if use_surrogate:
            raw_residual = self.evaluate_surrogate(individual)
        else:
            raw_residual = self.evaluate_exact(individual)
            
        num_sensors = sum(individual)
        max_sensors = max(1, num_sensors)
        # N(x) normaliza por el tamaño del problema (T*Omega) y num_sensores
        num_t_omega = len(self.snapshot_keys)
        
        normalization = 1.0 / (num_t_omega * max_sensors) if num_t_omega > 0 else 1.0
        
        # Factor de seguridad para evitar 0 absoluto si hay redondeo
        if raw_residual < 1e-9: raw_residual = 1e-6 
        
        fitness_val = raw_residual * normalization
        return (fitness_val,)


def plot_solution(sensor_mask, domain: TrafficNetwork):
    print(f"Solución: {sum(sensor_mask)} sensores.")

def save_ga_solution(sensor_mask, domain: TrafficNetwork, output_txt_path: CsvPathLike):
    path = Path(output_txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, bit in enumerate(sensor_mask):
            if bit:
                eid = domain.edge_ids[i]
                f.write(f"edge:{eid}\n")
    print(f"[GA] Solución guardada en: {path}")