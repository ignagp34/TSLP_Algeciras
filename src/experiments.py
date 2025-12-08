
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from src.ga_algorithm import TrafficNetwork
from src.sa_algorithm import SimulatedAnnealing
from deap import algorithms, base, creator, tools

def run_multi_seed_experiment(
    network, 
    algorithm_name, 
    budget, 
    seeds, 
    **kwargs
):
    """
    Runs an algorithm (GA or SA) multiple times with different seeds.
    
    Args:
        network: TrafficNetwork instance
        algorithm_name: 'GA' or 'SA'
        budget: Number of sensors
        seeds: List of seeds to run
        **kwargs: Algorithm specific parameters
        
    Returns:
        DataFrame including seed, best_fitness, best_solution
    """
    results = []
    
    for seed in tqdm(seeds, desc=f"Running {algorithm_name} B={budget}"):
        
        if algorithm_name == 'SA':
            sa = SimulatedAnnealing(network, budget, **kwargs)
            res = sa.run(seed=seed)
            best_fit = res['best_fitness']
            best_sol = res['best_solution']
            # Compute RMSE specifically if fitness includes penalties (although for SA/GA usually fitness=RMSE if feasible)
            rmse = network.evaluate_sensor_placement(best_sol)  # budget is None to just get RMSE
            
            results.append({
                'algorithm': 'SA',
                'budget': budget,
                'seed': seed,
                'fitness': best_fit,
                'rmse': rmse[0] if isinstance(rmse, tuple) else rmse,
                'solution': best_sol
            })
            
        elif algorithm_name == 'GA':
            # We need to replicate the GA config from scripts/run_ga.py or creating a standard one here
            # For simplicity, we define a quick setup here or we import if possible.
            # Assuming we need to define DEAP stuff locally to avoid global conflicts if re-running
            
            # Note: DEAP creators might scream if created multiple times. 
            # Check if exists
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMin)
                
            IND_SIZE = network.num_edges
            
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", np.random.randint, 0, 2)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evalOneMax(individual):
                return network.evaluate_sensor_placement(individual, budget=budget)
            
            toolbox.register("evaluate", evalOneMax)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            pop_size = kwargs.get('population_size', 50)
            n_gen = kwargs.get('max_generations', 50)
            
            # Set seed
            np.random.seed(seed)
            random.seed(seed)
            
            pop = toolbox.population(n=pop_size)
            hoff = tools.HallOfFame(1)
            
            algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, 
                                stats=None, halloffame=hoff, verbose=False)
            
            best_ind = hoff[0]
            best_fit = best_ind.fitness.values[0]
            # Clean RMSE calculation
            rmse = network.evaluate_sensor_placement(best_ind)
            
            results.append({
                'algorithm': 'GA',
                'budget': budget,
                'seed': seed,
                'fitness': best_fit,
                'rmse': rmse[0] if isinstance(rmse, tuple) else rmse,
                'solution': np.array(best_ind)
            })

    return pd.DataFrame(results)

def plot_variance_analysis(df_results, output_path=None):
    """
    Plots a boxplot comparison of RMSE variance between algorithms/budgets.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x='algorithm', y='rmse', hue='budget')
    plt.title('Variance Analysis: RMSE Distribution across Seeds')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

import random # Ensure random is imported for GA
