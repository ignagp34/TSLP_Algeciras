from deap import base, creator, tools, algorithms

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.ga_algorithm import TrafficNetwork, plot_solution, save_ga_solution
from pathlib import Path


# --- RUTAS BASE ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "Results"
SELECTED_DIR = DATA_PROCESSED / "selected"

SELECTED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ga_experiment(
    csv_path: Path | None = None,
    population_size: int = 50,
    max_generations: int = 50,
    p_crossover: float = 0.8,
    p_mutation: float = 0.2,
    sensor_budget: int = 20,
    tournament_size: int = 3,
    random_seed: int = 42,
):
    """
    Ejecuta el Algoritmo Genético para colocación de sensores.

    - csv_path: ruta al CSV de edgeData (por defecto edgeData_scen1.csv)
    - population_size, max_generations, etc.: parámetros del GA

    Guarda automáticamente:
      - data/processed/selected/selected_edges_B{B}_GA.txt
      - data/Results/GA_convergence_B{B}.png

    Devuelve un dict con info básica de la mejor solución.
    """
    # --- CSV por defecto ---
    if csv_path is None:
        csv_path = DATA_PROCESSED / "edgeData_scen1.csv"

    csv_path = Path(csv_path)
    print("=== Ejecutando Algoritmo Genético para Sensores ===")
    print(f"CSV de entrada: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra el CSV: {csv_path}")

    # --- Semillas ---
    random.seed(random_seed)
    np.random.seed(random_seed)

    # --- Inicializar dominio ---
    domain_problem = TrafficNetwork(str(csv_path))
    NUM_EDGES = domain_problem.num_edges
    print(f"Número total de tramos (edges): {NUM_EDGES}")

    # --- SETUP DEAP ---
    toolbox = base.Toolbox()

    # Evitamos recrear las clases si se reimporta el módulo desde el notebook
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def create_sparse_individual():
        """
        Crea un individuo con un número aleatorio de sensores 
        entre 1 y sensor_budget.
        """
        ind = [0] * NUM_EDGES
        num_active = random.randint(1, sensor_budget)
        idxs = random.sample(range(NUM_EDGES), num_active)
        for i in idxs:
            ind[i] = 1
        return ind

    toolbox.register("individualCreator", tools.initIterate, creator.Individual, create_sparse_individual)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # Función de evaluación (usa el método del dominio)
    def evaluate_wrapper(individual):
        return domain_problem.evaluate_sensor_placement(individual, budget=sensor_budget)

    toolbox.register("evaluate", evaluate_wrapper)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / NUM_EDGES)

    # --- FLUJO PRINCIPAL ---
    print("Iniciando Algoritmo Genético...")
    population = toolbox.populationCreator(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(5)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=p_crossover,
        mutpb=p_mutation,
        ngen=max_generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # --- RESULTADOS ---
    print("\n--- Mejores soluciones (Hall of Fame) ---")
    best = hof[0]
    best_fitness = best.fitness.values[0]
    num_sensors = int(sum(best))

    print(f"Mejor Fitness (RMSE): {best_fitness:.4f}")
    print(f"Número de Sensores: {num_sensors}")
    print(f"Número total de tramos: {len(best)}")

    # --- GUARDAR CONFIGURACIÓN GANADORA ---
    txt_path = SELECTED_DIR / f"selected_edges_B{num_sensors}_GA.txt"
    save_ga_solution(best, domain_problem, txt_path)
    print(f"Fichero de arcos seleccionados guardado en: {txt_path}")

    # --- GRÁFICOS DE CONVERGENCIA ---
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mejor fitness
    sns.lineplot(x=range(len(minFitnessValues)), y=minFitnessValues, ax=axes[0])
    axes[0].set_xlabel('Generación')
    axes[0].set_ylabel('Mejor Fitness (RMSE)')
    axes[0].set_title('Convergencia: Mejor Individuo')
    axes[0].grid(True)

    # Fitness medio
    sns.lineplot(x=range(len(meanFitnessValues)), y=meanFitnessValues, ax=axes[1])
    axes[1].set_xlabel('Generación')
    axes[1].set_ylabel('Fitness Promedio')
    axes[1].set_title('Convergencia: Promedio Población')
    axes[1].grid(True)

    plt.tight_layout()
    png_path = RESULTS_DIR / f"GA_convergence_B{num_sensors}.png"
    fig.savefig(png_path, dpi=200)
    print(f"Figura de convergencia guardada en: {png_path}")
    plt.show()

    # Visualizar la mejor solución
    plot_solution(best, domain_problem)

    # Devolvemos info útil al notebook
    return {
        "best": best,
        "best_fitness": best_fitness,
        "num_sensors": num_sensors,
        "txt_path": txt_path,
        "png_path": png_path,
        "logbook": logbook,
    }


def main():
    # Para poder seguir usando: python -m scripts.run_ga
    run_ga_experiment()


if __name__ == "__main__":
    main()
