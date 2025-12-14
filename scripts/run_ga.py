# scripts/run_ga.py
from __future__ import annotations

from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional

from src.ga_algorithm import TrafficNetwork, save_ga_solution


# --- RUTAS BASE ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "Results"
SELECTED_DIR = DATA_PROCESSED / "selected"

SELECTED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ga_experiment(
    csv_path: Path | None = None,
    net_xml_path: Path | None = None,
    population_size: int = 50,
    max_generations: int = 50,
    p_crossover: float = 0.8,
    p_mutation: float | None = None,  
    sensor_budget: int = 20,
    tournament_size: int = 2,
    random_seed: int = 42,
    k_coverage: int = 1,
    validation_freq: int = 0,  # 0 = desactivado
    milp_inputs: Optional[Dict[str, Any]] = None,
    lambda_flow_balance: float = 100.0,
    alpha_sensor_error: float = 1.0,
):
    """
    Ejecuta un GA para colocación de sensores con presupuesto fijo B.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    # --- Dominio / red ---
    domain = TrafficNetwork(
        csv_path=csv_path,
        net_xml_path=net_xml_path,
        k_coverage=k_coverage,
        milp_inputs=milp_inputs,
        lambda_flow_balance=lambda_flow_balance,
        alpha_sensor_error=alpha_sensor_error,
    )
    NUM_EDGES = domain.num_edges


    if p_mutation is None:
        p_mutation = 0.3

    print("=== GA Setup ===")
    print(f"Edges: {NUM_EDGES}, Budget: {sensor_budget}, k: {k_coverage}")
    print(
        f"Pop: {population_size}, Gens: {max_generations}, "
        f"Cross: {p_crossover:.2f}, Mut(ind): {p_mutation:.2f}, "
        f"Tourn: {tournament_size}"
    )

    # --- DEAP SETUP ---

    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()


    def repair_budget(individual: creator.Individual) -> creator.Individual:
        """Fuerza que el individuo tenga exactamente sensor_budget unos."""
        ones = [i for i, b in enumerate(individual) if b == 1]
        zeros = [i for i, b in enumerate(individual) if b == 0]

        
        if len(ones) > sensor_budget:
            to_off = random.sample(ones, len(ones) - sensor_budget)
            for i in to_off:
                individual[i] = 0

        
        elif len(ones) < sensor_budget:
            need = sensor_budget - len(ones)
            if need > len(zeros):
                
                need = len(zeros)
            to_on = random.sample(zeros, need)
            for i in to_on:
                individual[i] = 1

        return individual


    def make_budget_individual() -> creator.Individual:
        ind = [0] * NUM_EDGES
        for idx in random.sample(range(NUM_EDGES), sensor_budget):
            ind[idx] = 1
        return creator.Individual(ind)

    def create_population() -> list[creator.Individual]:
        pop = [make_budget_individual() for _ in range(population_size)]
        # Seguridad extra
        for ind in pop:
            repair_budget(ind)
        return pop


    def mutate_swap(individual: creator.Individual, n_swaps: int = 2):
        """
        Realiza n_swaps intercambios: apaga un 1 y enciende un 0.
        Mantiene el presupuesto exactamente.
        """
        ones = [i for i, b in enumerate(individual) if b == 1]
        zeros = [i for i, b in enumerate(individual) if b == 0]
        if not ones or not zeros:
            return (individual,)

        for _ in range(n_swaps):
            i_off = random.choice(ones)
            i_on = random.choice(zeros)

            individual[i_off] = 0
            individual[i_on] = 1

            # actualiza listas
            ones.remove(i_off)
            zeros.remove(i_on)
            ones.append(i_on)
            zeros.append(i_off)

        return (individual,)


    def evaluate_surrogate(ind):
        return domain.fitness_function(ind, use_surrogate=True)

    def evaluate_exact(ind):
        return domain.fitness_function(ind, use_surrogate=False)

    # --- Operadores DEAP ---
    toolbox.register("mate", tools.cxTwoPoint)  
    toolbox.register("mutate", mutate_swap, n_swaps=2)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("evaluate", evaluate_surrogate)
    toolbox.register("evaluate_exact", evaluate_exact)

    # --- Inicialización ---
    population = create_population()

    # Debug: diversidad real al inicio
    uniq = len({tuple(ind) for ind in population})
    print(f"[DEBUG] Individuos únicos al inicio: {uniq}/{len(population)}")

    print("Evaluando población inicial (Surrogate)...")
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # --- Estadísticas ---
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    best_ind = deepcopy(tools.selBest(population, 1)[0])

    # --- Bucle evolutivo ---
    elite_size = 2

    for gen in range(1, max_generations + 1):

        # 1) Elitismo
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
        elites = [toolbox.clone(ind) for ind in sorted_pop[:elite_size]]

        # 2) Selección
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # 3) Cruce
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_crossover:
                toolbox.mate(c1, c2)
                
                repair_budget(c1)
                repair_budget(c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

        # 4) Mutación 
        for mutant in offspring:
            if random.random() < p_mutation:
                toolbox.mutate(mutant)
                repair_budget(mutant)
                if hasattr(mutant.fitness, "values"):
                    del mutant.fitness.values

        # 5) Evaluación 
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # 6) Revalidación exacta opcional 
        if validation_freq and (gen % validation_freq == 0):

            current_best = tools.selBest(offspring + elites, 1)[0]
            _ = toolbox.evaluate_exact(current_best)

        # 7) Reemplazo
        population[:] = elites + offspring

        # Log
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Actualiza best global (surrogate)
        current_best = tools.selBest(population, 1)[0]
        if current_best.fitness.values[0] < best_ind.fitness.values[0]:
            best_ind = deepcopy(current_best)

    # --- Finalización ---
    print("\n--- GA Terminada ---")
    print(f"Mejor Fitness (Surrogate): {best_ind.fitness.values[0]:.4f}")
    print(f"Sensores activos: {sum(best_ind)} (debería ser {sensor_budget})")

    # Guardado
    txt_path = SELECTED_DIR / f"selected_edges_B{sensor_budget}_GA.txt"
    save_ga_solution(best_ind, domain, txt_path)

    # Plot convergencia
    min_vals = logbook.select("min")
    avg_vals = logbook.select("avg")
    plt.figure()
    plt.plot(min_vals, label="Min Fitness")
    plt.plot(avg_vals, label="Avg Fitness")
    plt.legend()
    plt.title(f"GA Convergence (B={sensor_budget})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Surrogate)")
    png_path = RESULTS_DIR / f"GA_convergence_B{sensor_budget}.png"
    plt.savefig(png_path)
    print(f"Convergence plot saved: {png_path}")

    return {
        "best": best_ind,
        "logbook": logbook,
        "txt_path": txt_path,
        "png_path": png_path,
    }


if __name__ == "__main__":
    run_ga_experiment()
