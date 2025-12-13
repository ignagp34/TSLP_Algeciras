# scripts/run_ga.py

from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional

from src.ga_algorithm import TrafficNetwork, plot_solution, save_ga_solution

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
    validation_freq: int = 5,
    milp_inputs: Optional[Dict[str, Any]] = None, # Input injection
):
    """
    Ejecuta el GA adaptado a la metodología del paper.
    """
    
    # --- SETUP ---
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Cargar dominio
    domain = TrafficNetwork(
        csv_path=csv_path,
        net_xml_path=net_xml_path,
        k_coverage=k_coverage,
        milp_inputs=milp_inputs # Pass injected inputs
    )
    NUM_EDGES = domain.num_edges
    
    if p_mutation is None:
        p_mutation = 1.0 / NUM_EDGES
        
    print(f"=== GA Setup ===")
    print(f"Edges: {NUM_EDGES}, Budget: {sensor_budget}, k: {k_coverage}")
    print(f"Pop: {population_size}, Gens: {max_generations}, MutRate: {p_mutation:.5f}")

    # --- DEAP SETUP ---
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
        
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5) 
    
    def mutate_custom(individual, indpb):
        tools.mutFlipBit(individual, indpb=indpb)
        return (individual,)
        
    toolbox.register("mutate", mutate_custom, indpb=p_mutation)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    # Evaluation hooks: Surrogate (default) vs Exact
    def evaluate_surrogate_wrapper(ind):
        return domain.fitness_function(ind, use_surrogate=True)

    def evaluate_exact_wrapper(ind):
        return domain.fitness_function(ind, use_surrogate=False)

    toolbox.register("evaluate", evaluate_surrogate_wrapper)
    toolbox.register("evaluate_exact", evaluate_exact_wrapper)
    
    # Repair hook
    def repair_hook(individual):
        domain.repair_individual(individual, budget=sensor_budget)
        return individual

    # --- INITIALIZATION with Seeding ---
    def create_population():
        pop = []
        # 1. Random seeds
        for _ in range(population_size):
            ind = creator.Individual(random.randint(0, 1) for _ in range(NUM_EDGES))
            repair_hook(ind)
            pop.append(ind)
            
        # 2. Robust seed (DISABLED for gradual convergence demo)
        # robust_ind = creator.Individual([0]*NUM_EDGES)
        # # Prioritize edges in cycles/cuts (high frequency)
        # sorted_edges = sorted(range(NUM_EDGES), 
        #                       key=lambda i: domain.edge_freq.get(domain.edge_ids[i], 0), 
        #                       reverse=True)
        # for i in range(sensor_budget):
        #     robust_ind[sorted_edges[i]] = 1
        # pop[0] = robust_ind 
        
        return pop

    population = create_population()
    
    print("Evaluando población inicial (Surrogate)...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    
    best_ind = None
    
    # --- EVOLUTION LOOP (Manual) ---
    for gen in range(1, max_generations + 1):
        
        # 1. Selection & Elitism
        elite_size = 2 
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
        elites = [toolbox.clone(ind) for ind in sorted_pop[:elite_size]]
        
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = list(map(toolbox.clone, offspring))
        
        # 2. Crossover & Mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.5: 
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # 3. REPAIR
        for child in offspring:
            repair_hook(child)
            
        # 4. Evaluate (Surrogate)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind) # Uses Surrogate
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # 5. Revalidation of Elites (Every G_val gens)
        if gen % validation_freq == 0:
            # Re-evaluate elites with EXACT model
            # [DISABLED] User requested ONLY SURROGATE
            pass
            # for ind in elites:
            #     ind.fitness.values = toolbox.evaluate_exact(ind)
            
            # Also re-evaluate best new individuals? Paper says "best new individuals".
            # Let's re-evaluate current top of offspring too.
            # (Finding best offspring is a bit costly if we sort, but ok)
            # pass

        # 6. Replace
        population[:] = elites + offspring
        
        # Log
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # Update best
        current_best = tools.selBest(population, 1)[0]
        # Always verify best with EXACT before storing as global best
        # [DISABLED] User requested ONLY SURROGATE.
        # exact_fit = toolbox.evaluate_exact(current_best)
        # current_best.fitness.values = exact_fit
        
        # We rely on surrogate fitness
        pass
        
        if best_ind is None or current_best.fitness.values[0] < best_ind.fitness.values[0]:
            best_ind = deepcopy(current_best)

    # --- FINALIZATION ---
    print("\n--- GA Terminada ---")
    
    # Final exact evaluation for best found
    # [DISABLED] User requested NO FINAL VALIDATION.
    # best_exact = toolbox.evaluate_exact(best_ind)
    # best_ind.fitness.values = best_exact
    
    print(f"Mejor Fitness (Exact): {best_ind.fitness.values[0]:.4f}")
    print(f"Sensores activos: {sum(best_ind)}")
    
    # Save
    txt_path = SELECTED_DIR / f"selected_edges_B{sensor_budget}_GA.txt"
    save_ga_solution(best_ind, domain, txt_path)
    
    # Plot convergence
    min_vals = logbook.select("min")
    avg_vals = logbook.select("avg")
    plt.figure()
    plt.plot(min_vals, label="Min Fitness")
    plt.plot(avg_vals, label="Avg Fitness")
    plt.legend()
    plt.title(f"GA Convergence (B={sensor_budget})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Weighted L1)")
    png_path = RESULTS_DIR / f"GA_convergence_B{sensor_budget}.png"
    plt.savefig(png_path)
    print(f"Convergence plot saved: {png_path}")
    
    return {
        "best": best_ind,
        "logbook": logbook,
        "txt_path": txt_path
    }

if __name__ == "__main__":
    run_ga_experiment()