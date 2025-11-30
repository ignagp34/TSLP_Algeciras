"""
run_preprocessing.py

Script para lanzar desde línea de comandos el preprocesado
de datos SUMO → inputs para el MILP.
"""

from src import preprocessing


def main():
    print("=== PREPROCESSING: Construyendo milp_inputs ===")
    milp_inputs = preprocessing.construir_milp_inputs()

    # Información básica por pantalla
    flows = milp_inputs.get("flows")
    if flows is not None:
        print("\nResumen de 'flows':")
        print(flows.head())
        print(f"\nTotal de filas en flows: {len(flows)}")
        print(f"Escenarios encontrados: {flows['scenario'].unique() if 'scenario' in flows.columns else 'N/A'}")

    # Guardar en pickle
    preprocessing.guardar_milp_inputs(milp_inputs)


if __name__ == "__main__":
    main()
