import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import random  # para el GA

from src.ga_algorithm import TrafficNetwork
from src.sa_algorithm import SimulatedAnnealing, FlowCoverageDomain
from deap import algorithms, base, creator, tools


def run_multi_seed_experiment(
    network: TrafficNetwork,
    algorithm_name: str,
    budget: int,
    seeds,
    **kwargs,
) -> pd.DataFrame:
    """
    Ejecuta GA o SA varias veces con diferentes seeds.

    Args:
        network: instancia de TrafficNetwork (se usa siempre para calcular RMSE).
        algorithm_name: 'GA' o 'SA'.
        budget: número de sensores (B).
        seeds: iterable de semillas.
        **kwargs:
            - SA:
                initial_temp, cooling_rate, min_temp, max_iter, ...
            - GA:
                population_size, max_generations, ...

    Returns:
        DataFrame con columnas:
            algorithm, budget, seed, fitness, rmse, solution
        donde:
            - Para GA:
                fitness = valor de la función objetivo del GA
                          (evaluate_sensor_placement con budget).
                rmse    = RMSE de network.evaluate_sensor_placement
                          (sin budget).
            - Para SA:
                fitness = RMSE de network.evaluate_sensor_placement
                          (misma métrica que GA, pasada como fitness_func).
                rmse    = igual que fitness.
    """
    results = []

    # ------------------------------------------------------------------
    # Dominio para SA: mismo orden de edge_ids que GA.
    # Lo usamos solo como "portador" de edge_ids y tamaño.
    # ------------------------------------------------------------------
    if algorithm_name == "SA":
        flow_vector = np.asarray(network.flow_truth, dtype=float)
        domain_for_sa = FlowCoverageDomain(network.edge_ids, flow_vector)
    else:
        domain_for_sa = None

    for seed in tqdm(seeds, desc=f"Running {algorithm_name} B={budget}"):

        if algorithm_name == "SA":
            # --------- Simulated Annealing con misma métrica que GA ---------

            def rmse_fitness(mask):
                # Misma función objetivo que el GA (con budget)
                val = network.evaluate_sensor_placement(mask, budget=budget)
                return val[0] if isinstance(val, tuple) else float(val)

            sa = SimulatedAnnealing(
                domain=domain_for_sa,
                budget=budget,
                fitness_func=rmse_fitness,
                **kwargs,
            )
            res = sa.run(seed=seed)

            best_fit = float(res["best_fitness"])   # RMSE
            best_sol = res["best_solution"]

            results.append(
                {
                    "algorithm": "SA",
                    "budget": budget,
                    "seed": seed,
                    "fitness": best_fit,
                    "rmse": best_fit,  # misma métrica
                    "solution": np.array(best_sol, dtype=int),
                }
            )

        elif algorithm_name == "GA":
            # ----------------- Algoritmo Genético (como en run_ga.py) --------------------
            # Evitar recrear creadores DEAP si ya existen
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMin)

            NUM_EDGES = network.num_edges

            toolbox = base.Toolbox()

            # ---- creación de individuos escasos (<= budget sensores) ----
            def create_sparse_individual():
                """
                Crea un individuo con un número aleatorio de sensores
                entre 1 y 'budget' (igual que en scripts/run_ga.py).
                """
                ind = [0] * NUM_EDGES
                num_active = random.randint(1, budget)
                idxs = random.sample(range(NUM_EDGES), num_active)
                for i in idxs:
                    ind[i] = 1
                return ind

            toolbox.register(
                "individualCreator",
                tools.initIterate,
                creator.Individual,
                create_sparse_individual,
            )
            toolbox.register(
                "populationCreator",
                tools.initRepeat,
                list,
                toolbox.individualCreator,
            )

            # Función de evaluación (usa el mismo método que en run_ga.py)
            def eval_fitness(individual):
                return network.evaluate_sensor_placement(individual, budget=budget)

            toolbox.register("evaluate", eval_fitness)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / NUM_EDGES)

            pop_size = kwargs.get("population_size", 50)
            n_gen = kwargs.get("max_generations", 50)

            # Seed reproducible
            np.random.seed(seed)
            random.seed(seed)

            pop = toolbox.populationCreator(n=pop_size)
            hoff = tools.HallOfFame(1)

            algorithms.eaSimple(
                pop,
                toolbox,
                cxpb=0.8,   # como en run_ga.py por defecto
                mutpb=0.2,
                ngen=n_gen,
                stats=None,
                halloffame=hoff,
                verbose=False,
            )

            best_ind = hoff[0]
            best_fit = float(best_ind.fitness.values[0])

            # RMSE "limpio" SIN pasar budget (para no volver a penalizar)
            rmse_val = network.evaluate_sensor_placement(best_ind)
            rmse_val = rmse_val[0] if isinstance(rmse_val, tuple) else float(rmse_val)

            results.append(
                {
                    "algorithm": "GA",
                    "budget": budget,
                    "seed": seed,
                    "fitness": best_fit,
                    "rmse": rmse_val,
                    "solution": np.array(best_ind, dtype=int),
                }
            )

        else:
            raise ValueError(f"algorithm_name desconocido: {algorithm_name}")

    return pd.DataFrame(results)


def plot_variance_analysis(
    df_results: pd.DataFrame, output_path: Path | None = None
):
    """
    Boxplot de comparación de varianza de la métrica (rmse) entre algoritmos/budgets.

    Aquí 'rmse' es SIEMPRE network.evaluate_sensor_placement(...) para
    ambas familias de algoritmos, por lo que las comparaciones son justas.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="algorithm", y="rmse", hue="budget")
    plt.title("Variance Analysis: RMSE distribution across seeds")
    plt.grid(True, alpha=0.3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    plt.show()

