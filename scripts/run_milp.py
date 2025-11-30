# scripts/run_milp.py

from pathlib import Path

from src.milp_solve import solve_sensor_placement, load_milp_inputs
from src.evaluation import compute_flow_coverage


def write_selection_txt(edge_ids, path_txt: str | Path) -> None:
    """
    Guarda un fichero .txt de selección de edges para SUMO/NETEDIT
    con el mismo formato que critical_edges.txt:

        edge:<edge_id>
        edge:<edge_id>
        ...

    Luego se puede cargar como selección desde SUMO/NETEDIT.
    """
    path = Path(path_txt)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for eid in edge_ids:
            f.write(f"edge:{eid}\n")

    print(f"Selection file saved to: {path}")


def run_milp_experiment(
    B: int = 20,
    selection_dir: Path | None = None,
) -> dict:
    """
    Ejecuta el MILP para un presupuesto B de sensores, imprime
    el resumen por consola y devuelve:

      - result: salida de solve_sensor_placement
      - df_cov: DataFrame con métricas de cobertura
      - selection_path: ruta al .txt con edges seleccionados

    Además, guarda un fichero selected_edges_B{B}.txt en selection_dir
    (formato edge:<ID>) para poder cargarlo en SUMO/NETEDIT.
    """
    print("=== Ejecutando MILP de colocación de sensores (formulación tipo paper) ===")
    print(f"Presupuesto de sensores B = {B}")

    if selection_dir is None:
        selection_dir = Path("data/processed/selected")

    selection_dir = Path(selection_dir)
    selection_dir.mkdir(parents=True, exist_ok=True)

    # 1) Resolver MILP
    result = solve_sensor_placement(
        max_sensors=B,
        msg=True,
    )

    print(f"\nEstado del solver: {result['status']}")
    print(f"Valor óptimo de la función objetivo (suma de residuos + slacks): {result['objective_value']}")
    print(f"Número de arcos con sensor: {len(result['selected_edges'])} (B = {B})")

    print("\nPrimeros arcos seleccionados:")
    for e in result["selected_edges"][:20]:
        print(f"  - {e}")

    # 2) Cargar milp_inputs y calcular cobertura
    milp_inputs = load_milp_inputs()
    df_cov = compute_flow_coverage(milp_inputs, result["selected_edges"])

    print("\n=== Métricas de cobertura de flujo ===")
    print(df_cov)

    # 3) Guardar selección en formato txt (edge:ID)
    selection_path = selection_dir / f"selected_edges_B{B}.txt"
    write_selection_txt(result["selected_edges"], selection_path)

    return {
        "result": result,
        "df_cov": df_cov,
        "selection_path": selection_path,
    }


def main():
    # Comportamiento por defecto cuando se llama como script
    B_SENSORS = 20
    run_milp_experiment(B=B_SENSORS)


if __name__ == "__main__":
    main()
