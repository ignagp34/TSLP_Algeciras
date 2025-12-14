from typing import Dict, Any, Optional, Tuple, List
from .io_utils import leer_critical_edges
from .config import PROJECT_ROOT
from .topology_utils import (
    ciclos_en_edges_criticos, 
    cortes_minimos_en_zona_critica,
)

import pandas as pd
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    LpBinary,
    lpSum,
)


def build_sensor_placement_model(
    milp_inputs: Dict[str, Any],
    max_sensors: Optional[int] = 20,
    weight_scheme: str = "inv_abs",
    lambda_flow_balance: float = 50,
    epsilon_weight: float = 1.0,
    alpha_sensor_error: float = 1.0,
    forced_cycles: Optional[List[List[str]]] = None,
    forced_cuts: Optional[List[List[str]]] = None,
) -> Tuple[LpProblem, Dict[str, LpVariable]]:

    """
    MILP para colocación de sensores.

    Variables:
    - x_a ∈ {0,1}: sensor en el arco a
    - f_{a}^{τ,ω} ≥ 0: flujo reconstruido
    - r_{a}^{τ,ω} ≥ 0: residuo absoluto en edges con sensor
    - z_{i}^{τ,ω} ≥ 0: violación de balance en nodo

    Función objetivo:
    min  sum_{τ,ω,a} w_{a}^{τ,ω} r_{a}^{τ,ω} + λ * sum_{τ,ω,i} z_{i}^{τ,ω}

    Restricciones:
    - r >= α*f - f_prior - M(1-x)  (constraint 8)
    - r >= f_prior - α*f - M(1-x)  (constraint 9)
    - sum x_a = max_sensors
    - |∑out f - ∑in f - (O-D)| <= z

    Parámetros:
    alpha_sensor_error : float
        Factor de error del sensor (α). Valor 1.0 = sensor perfecto.
    """
    # ------------------------------------------------------------------
    # 1) Extraer datos
    # ------------------------------------------------------------------
    flows: pd.DataFrame = milp_inputs["flows"]
    edges: pd.DataFrame = milp_inputs["edges"]
    O_df: pd.DataFrame = milp_inputs.get("O", pd.DataFrame())
    D_df: pd.DataFrame = milp_inputs.get("D", pd.DataFrame())
    M: float = float(milp_inputs.get("M", 0.0))

    for col in ["scenario", "edge_id", "tau", "flow_veh_s"]:
        if col not in flows.columns:
            raise KeyError("El DataFrame 'flows' debe contener la columna '%s'" % col)
    for col in ["edge_id", "from_node", "to_node"]:
        if col not in edges.columns:
            raise KeyError("El DataFrame 'edges' debe contener la columna '%s'" % col)

    # Copia local de flujos y tau como entero
    flows_local = flows.copy()
    flows_local["tau"] = flows_local["tau"].astype(int)

    # Conjunto de arcos candidatos a sensor:
    # SOLO ARCOS CON FLUJO HISTÓRICO (evitar arcos vacíos)
    candidate_edge_ids = sorted(flows_local["edge_id"].unique().tolist())

    # Mapas from/to
    edge_from = dict(zip(edges["edge_id"], edges["from_node"]))
    edge_to = dict(zip(edges["edge_id"], edges["to_node"]))

    # Pseudomediciones y pesos
    meas_hat: Dict[tuple, float] = {}
    weights: Dict[tuple, float] = {}

    for _, row in flows_local.iterrows():
        key = (row["scenario"], int(row["tau"]), row["edge_id"])
        hat = float(row["flow_veh_s"])
        meas_hat[key] = hat

        if weight_scheme == "uniform":
            w = 1.0
        elif weight_scheme == "inv_abs":
            # w ≈ 1 / max(|hat|, ε)
            denom = max(abs(hat), epsilon_weight)
            w = 1.0 / denom
        else:
            # por defecto, si ponemos algo raro, dejamos 1.0
            w = 1.0

        weights[key] = w


    fr_keys = list(meas_hat.keys())

    # Mapas O y D
    O_map: Dict[tuple, float] = {}
    if not O_df.empty:
        O_df_local = O_df.copy()
        O_df_local["tau"] = O_df_local["tau"].astype(int)
        for _, row in O_df_local.iterrows():
            key = (row["scenario"], int(row["tau"]), row["node_id"])
            O_map[key] = float(row["O_veh_h"]) / 3600.0

    D_map: Dict[tuple, float] = {}
    if not D_df.empty:
        D_df_local = D_df.copy()
        D_df_local["tau"] = D_df_local["tau"].astype(int)
        for _, row in D_df_local.iterrows():
            key = (row["scenario"], int(row["tau"]), row["node_id"])
            D_map[key] = float(row["D_veh_h"]) / 3600.0

    # ------------------------------------------------------------------
    # 2) Modelo y variables
    # ------------------------------------------------------------------
    model = LpProblem("Algeciras_MILP_SensorPlacement", LpMinimize)

    # x_a: sensor en arco a (solo candidatos)
    x_vars: Dict[str, LpVariable] = {
        e: LpVariable("x_%s" % str(e), lowBound=0, upBound=1, cat=LpBinary)
        for e in candidate_edge_ids
    }

    # Identificar snapshots activos (scenario, tau)
    # fr_keys son (scen, tau, edge_id) donde hay medicion
    active_snapshots = set((k[0], k[1]) for k in fr_keys)
    all_edge_ids = sorted(edges["edge_id"].unique().tolist()) # Todos los arcos de la red

    # f_{a}^{τ,ω}: Flujo en CUALQUIER arco (para conservar masa)
    # r_{a}^{τ,ω}: Residuo SOLO en arcos con medición
    f_vars: Dict[tuple, LpVariable] = {}
    r_vars: Dict[tuple, LpVariable] = {}

    # Generar f para todo (snapshot, edge)
    for (scen, tau) in active_snapshots:
        for eid in all_edge_ids:
            f_key = (scen, tau, eid)
            # Bounds: 0 to infinity (or BigM)
            f_vars[f_key] = LpVariable("f_%s_%s_%s" % (scen, tau, eid), lowBound=0)

    # Generar r solo para measured keys
    for key in fr_keys:
        scen, tau, edge_id = key
        r_vars[key] = LpVariable("r_%s_%s_%s" % (scen, tau, edge_id), lowBound=0)

    # ------------------------------------------------------------------
    # 3) Preparar claves para balance y slacks z
    # ------------------------------------------------------------------
    outgoing = {}  # (scen, tau, node) -> lista de keys de f_vars
    incoming = {}

    # Ahora mapeamos TODOS los arcos para el balance
    for (scen, tau) in active_snapshots:
        for eid in all_edge_ids:
            i = edge_from.get(eid)
            j = edge_to.get(eid)
            f_key = (scen, tau, eid) # Key for f_var
            
            if i is not None:
                # Flow leaves i
                k_out = (scen, tau, i)
                outgoing.setdefault(k_out, []).append(f_key) # Store variable KEY
            if j is not None:
                # Flow enters j
                k_in = (scen, tau, j)
                incoming.setdefault(k_in, []).append(f_key)

    cons_keys = set(outgoing.keys()) | set(incoming.keys()) | set(O_map.keys()) | set(D_map.keys())

    # z_{i}^{τ,ω} ≥ 0
    z_vars: Dict[tuple, LpVariable] = {
        (scen, tau, node): LpVariable(
            "z_%s_%s_%s" % (scen, tau, node),
            lowBound=0.0,
        )
        for (scen, tau, node) in cons_keys
    }

    # ------------------------------------------------------------------
    # 4) Función objetivo: sum w*r + λ sum z
    # ------------------------------------------------------------------
    model += (
        lpSum(weights[key] * r_vars[key] for key in fr_keys)
        + lambda_flow_balance * lpSum(z_vars[k] for k in cons_keys),
        "WeightedResidualsPlusSlack",
    )

    # ------------------------------------------------------------------
    # 5) Presupuesto de sensores: sum x_a = max_sensors
    # ------------------------------------------------------------------
    if max_sensors is not None:
        model += (
            lpSum(x_vars[e] for e in candidate_edge_ids) == max_sensors,
            "SensorBudget",
        )

    # ------------------------------------------------------------------
    # 6) Big-M para residuos
    # ------------------------------------------------------------------
    for key in fr_keys:
        scen, tau, edge_id = key
        f_var = f_vars[key]
        r_var = r_vars[key]
        hat = meas_hat[key]
        x_var = x_vars[edge_id] if edge_id in x_vars else None

        if x_var is None:
            # Si por algún motivo el edge_id no es candidato, forzamos r=0 y no lo ligamos a x
            model += (r_var == 0, f"ResFix_{scen}_{tau}_{edge_id}")
            continue

        # r <= M x
        model += (
            r_var <= M * x_var,
            "ResUpper_%s_%s_%s" % (scen, tau, edge_id),
        )
        model += (
            r_var >= alpha_sensor_error * f_var - hat - M * (1 - x_var),
            "ResPos_%s_%s_%s" % (scen, tau, edge_id),
        )
        model += (
            r_var >= hat - alpha_sensor_error * f_var - M * (1 - x_var),
            "ResNeg_%s_%s_%s" % (scen, tau, edge_id),
        )

    # ------------------------------------------------------------------
    # 7) Balance relajado con slack z
    # ------------------------------------------------------------------
    for (scen, tau, node_id) in cons_keys:
        out_list = outgoing.get((scen, tau, node_id), [])
        in_list = incoming.get((scen, tau, node_id), [])

        lhs = lpSum(f_vars[k] for k in out_list) - lpSum(f_vars[k] for k in in_list)

        O_val = O_map.get((scen, tau, node_id), 0.0)
        D_val = D_map.get((scen, tau, node_id), 0.0)
        rhs = float(O_val) - float(D_val)

        z_var = z_vars[(scen, tau, node_id)]

        model += (
            lhs - rhs <= z_var,
            "FlowBalPos_%s_%s_%s" % (scen, tau, node_id),
        )
        model += (
            rhs - lhs <= z_var,
            "FlowBalNeg_%s_%s_%s" % (scen, tau, node_id),
        )
    # ------------------------------------------------------------------
    # 8) Cortes y ciclos críticos
    # ------------------------------------------------------------------    
    
    # 8) Cortes y ciclos críticos
    # ------------------------------------------------------------------    
    
    if forced_cycles is not None:
        ciclos_criticos = forced_cycles
    else:
        critical_path = PROJECT_ROOT / "data" / "processed" / "critical_edges.txt"
        critical_edges = leer_critical_edges(critical_path)
        ciclos_criticos = ciclos_en_edges_criticos(milp_inputs, critical_edges)

    if forced_cuts is not None:
        cortes_criticos = forced_cuts
    else:
        critical_path = PROJECT_ROOT / "data" / "processed" / "critical_edges.txt"
        critical_edges = leer_critical_edges(critical_path)
        cortes_criticos = cortes_minimos_en_zona_critica(milp_inputs, critical_edges)

    for idx, ciclo in enumerate(ciclos_criticos):
        model += (
            lpSum(x_vars[e] for e in ciclo if e in x_vars) >= 1,
            f"CycleCov_{idx}",
        )

    for idx, corte in enumerate(cortes_criticos):
        model += (
            lpSum(x_vars[e] for e in corte if e in x_vars) >= 1,
            f"CutCov_{idx}",
        )

                                               
    return model, x_vars
