"""
evaluation.py

Funciones para evaluar la calidad de una configuración de sensores:

- Cobertura de flujo: qué porcentaje del flujo total pasa por arcos sensorizados.
- Métricas por escenario y globales.

Se apoya en:
  - milp_inputs["flows"]: pseudomediciones ˆf_{a}^{τ,ω} (flow_veh_h)
  - lista de arcos con sensor (selected_edges)
"""

from typing import Dict, Any, List
import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import re


def compute_flow_coverage(
    milp_inputs: Dict[str, Any],
    selected_edges: List[str],
) -> pd.DataFrame:
    """
    Calcula métricas de cobertura de flujo para una configuración de sensores.

    Parámetros
    ----------
    milp_inputs : dict
        Diccionario con al menos la clave 'flows':
          - 'flows': DataFrame con columnas
                scenario, edge_id, tau, flow_veh_h

    selected_edges : lista de str
        Lista de edge_id donde el MILP ha colocado sensores (x_a = 1).

    Devuelve
    --------
    df_metrics : DataFrame
        Cada fila es un escenario (más una fila 'ALL' global) con:
          - scenario
          - total_flow_veh_h      (flujo total en la red)
          - sensed_flow_veh_h     (flujo que pasa por arcos con sensor)
          - coverage_ratio        (sensed_flow / total_flow)
          - num_sensors           (nº de arcos con sensor en ese escenario: es global)
    """
    flows = milp_inputs["flows"].copy()

    # Aseguramos tipos
    if "flow_veh_h" not in flows.columns:
        raise KeyError("El DataFrame 'flows' debe contener 'flow_veh_h'.")

    # Marcamos qué arcos tienen sensor
    flows["has_sensor"] = flows["edge_id"].isin(selected_edges)

    # Métricas por escenario
    grouped = flows.groupby("scenario", as_index=False).agg(
        total_flow_veh_h=("flow_veh_h", "sum"),
        sensed_flow_veh_h=("flow_veh_h", lambda x: x[flows.loc[x.index, "has_sensor"]].sum()),
    )

    grouped["coverage_ratio"] = grouped["sensed_flow_veh_h"] / grouped["total_flow_veh_h"]
    grouped["num_sensors"] = len(selected_edges)

    # Fila global (todos los escenarios)
    total_flow_all = flows["flow_veh_h"].sum()
    sensed_flow_all = flows.loc[flows["has_sensor"], "flow_veh_h"].sum()

    global_row = pd.DataFrame(
        {
            "scenario": ["ALL"],
            "total_flow_veh_h": [total_flow_all],
            "sensed_flow_veh_h": [sensed_flow_all],
            "coverage_ratio": [sensed_flow_all / total_flow_all if total_flow_all > 0 else 0.0],
            "num_sensors": [len(selected_edges)],
        }
    )

    df_metrics = pd.concat([grouped, global_row], ignore_index=True)

    return df_metrics

def _load_selected_edges_from_txt(path_txt: Path) -> List[str]:
    """
    Lee un fichero selected_edges_*.txt con líneas tipo:
        edge:<edge_id>
    y devuelve la lista de IDs de arcos.
    """
    path_txt = Path(path_txt)
    edge_ids: List[str] = []

    with path_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("edge:"):
                eid = line.split("edge:", 1)[1].strip()
                if eid:
                    edge_ids.append(eid)

    return edge_ids


def _extract_B_from_stem(stem: str) -> int | None:
    """
    Extrae el número de sensores B a partir del nombre base del fichero.

    Soporta nombres del tipo:
        selected_edges_B25
        selected_edges_B40_GA

    Devuelve B como entero, o None si no se puede parsear.
    """
    m = re.search(r"_B(\d+)", stem)
    if not m:
        return None
    return int(m.group(1))


def compare_milp_vs_ga(
    milp_inputs: Dict[str, Any],
    selected_dir: str | Path = "data/processed/selected",
    results_dir: str | Path = "data/Results/evaluation",
) -> pd.DataFrame:
    """
    Compara la cobertura de flujo de MILP vs GA para todos los budgets B
    donde existan ambos ficheros:

        - MILP: selected_edges_B{B}.txt
        - GA:   selected_edges_B{B}_GA.txt

    Genera una figura por cada B con barras (MILP vs GA) para la fila 'ALL'
    de cobertura de flujo, y la guarda en:

        data/Results/evaluation/comparison_B{B}.png

    Devuelve un DataFrame resumen a nivel 'ALL' con columnas:
        B, algo (MILP/GA), coverage_ratio, total_flow_veh_h, sensed_flow_veh_h
    """
    selected_dir = Path(selected_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Buscar ficheros MILP y GA ---
    milp_files = {}
    ga_files = {}

    for txt_path in selected_dir.glob("selected_edges_B*.txt"):
        stem = txt_path.stem  # p.ej. selected_edges_B25 o selected_edges_B25_GA
        B = _extract_B_from_stem(stem)
        if B is None:
            continue

        if stem.endswith("_GA"):
            ga_files[B] = txt_path
        else:
            milp_files[B] = txt_path

    common_B = sorted(set(milp_files.keys()) & set(ga_files.keys()))

    if not common_B:
        print("No se encontraron budgets B comunes entre MILP y GA.")
        return pd.DataFrame()

    print("Budgets B comunes encontrados (MILP & GA):", common_B)

    summary_rows = []

    for B in common_B:
        print(f"\n=== Comparando B = {B} ===")

        # --- MILP ---
        milp_txt = milp_files[B]
        milp_edges = _load_selected_edges_from_txt(milp_txt)
        df_milp = compute_flow_coverage(milp_inputs, milp_edges)

        # Fila 'ALL' (global)
        row_milp_all = df_milp[df_milp["scenario"] == "ALL"].iloc[0]

        summary_rows.append(
            {
                "B": B,
                "algo": "MILP",
                "scenario": "ALL",
                "total_flow_veh_h": row_milp_all["total_flow_veh_h"],
                "sensed_flow_veh_h": row_milp_all["sensed_flow_veh_h"],
                "coverage_ratio": row_milp_all["coverage_ratio"],
            }
        )

        # --- GA ---
        ga_txt = ga_files[B]
        ga_edges = _load_selected_edges_from_txt(ga_txt)
        df_ga = compute_flow_coverage(milp_inputs, ga_edges)
        row_ga_all = df_ga[df_ga["scenario"] == "ALL"].iloc[0]

        summary_rows.append(
            {
                "B": B,
                "algo": "GA",
                "scenario": "ALL",
                "total_flow_veh_h": row_ga_all["total_flow_veh_h"],
                "sensed_flow_veh_h": row_ga_all["sensed_flow_veh_h"],
                "coverage_ratio": row_ga_all["coverage_ratio"],
            }
        )

        # --- Figura de comparación para este B (ALL) ---
        labels = ["MILP", "GA"]
        covs = [row_milp_all["coverage_ratio"], row_ga_all["coverage_ratio"]]

        plt.figure(figsize=(5, 4))
        plt.bar(labels, covs)
        plt.ylabel("Coverage Ratio (ALL)")
        plt.ylim(0, max(covs) * 1.1 if max(covs) > 0 else 1.0)
        plt.title(f"MILP vs GA Comparison (B = {B})")
        plt.grid(axis="y", alpha=0.3)

        fig_path = results_dir / f"comparison_MILP_vs_GA_B{B}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"Figura de comparación guardada en: {fig_path}")

    df_summary = pd.DataFrame(summary_rows)
    return df_summary
