"""
milp_solve.py

Carga milp_inputs, construye el modelo MILP de colocación de sensores
y lo resuelve con CBC.
"""

from typing import Dict, Any, Optional
import pickle

from pulp import LpStatus, PULP_CBC_CMD

from . import config
from .milp_model import build_sensor_placement_model


def load_milp_inputs(path=None) -> Dict[str, Any]:
    """
    Carga el diccionario milp_inputs generado por preprocessing.

    Si no se pasa path, usa config.MILP_INPUTS_FILE.
    """
    if path is None:
        path = config.MILP_INPUTS_FILE

    with open(path, "rb") as f:
        milp_inputs = pickle.load(f)

    return milp_inputs


def solve_sensor_placement(
    milp_inputs: Optional[Dict[str, Any]] = None,
    max_sensors: Optional[int] = 20,
    msg: bool = True,
) -> Dict[str, Any]:
    """
    Construye y resuelve el modelo de colocación de sensores
    con la formulación tipo paper (simplificada).

    Parámetros
    ----------
    milp_inputs : dict, opcional
        Si es None, se cargará desde config.MILP_INPUTS_FILE.
    max_sensors : int
        Presupuesto máximo de sensores B (sum x_a <= B).
        Por defecto B = 20 (ajústalo según el caso de estudio).
    msg : bool
        Si True, muestra información del solver CBC.

    Devuelve
    --------
    result : dict
        - 'status': estado del solver (string)
        - 'objective_value': valor óptimo de la función objetivo
        - 'selected_edges': lista de edge_id con x_a = 1
        - 'x_values': diccionario {edge_id: valor de x_a}
    """
    # 1) Cargar datos si no se han pasado
    if milp_inputs is None:
        milp_inputs = load_milp_inputs()

    # 2) Construir modelo (MILP tipo paper simplificado)
    model, x_vars = build_sensor_placement_model(
        milp_inputs,
        max_sensors=max_sensors,
        weight_scheme="inv_abs",   # usamos la fórmula práctica
        lambda_flow_balance=0.1,   # como antes
        epsilon_weight=1.0,        # puedes probar también 10.0 etc.
    )


    # 3) Resolver con CBC
    solver = PULP_CBC_CMD(msg=msg)
    model.solve(solver)

    status = LpStatus[model.status]
    obj_value = model.objective.value()

    # 4) Extraer solución de las variables x
    x_values = {e: var.value() for e, var in x_vars.items()}
    selected_edges = [e for e, val in x_values.items() if val is not None and val > 0.5]

    result = {
        "status": status,
        "objective_value": obj_value,
        "selected_edges": selected_edges,
        "x_values": x_values,
    }

    return result
