# scripts/ga_sensitivity_analysis.py
"""
GA Parameter Sensitivity Analysis

Evaluates the impact of different GA parameters on performance:
- Initialization method (random vs hybrid)
- Population size
- Crossover probability

Addresses reviewer feedback about lack of experimental evaluation.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ga import run_ga_experiment
from src.preprocessing import construir_milp_inputs

# Output directory
RESULTS_DIR = PROJECT_ROOT / "data" / "Results" / "sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_sensitivity_analysis(
    sensor_budget: int = 20,
    max_generations: int = 30,
    n_trials: int = 3,
    milp_inputs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run GA sensitivity analysis across different parameter configurations.
    
    Parameters
    ----------
    sensor_budget : int
        Number of sensors (B)
    max_generations : int
        Maximum generations per run
    n_trials : int
        Number of random seeds per configuration
    milp_inputs : dict, optional
        Pre-loaded MILP inputs (if None, will load from disk)
    
    Returns
    -------
    pd.DataFrame
        Results with columns: init_method, pop_size, p_crossover, trial, 
        best_fitness, avg_fitness, convergence_gen, runtime
    """
    
    if milp_inputs is None:
        print("Loading MILP inputs...")
        milp_inputs = construir_milp_inputs()
    
    # Parameter configurations to test
    population_sizes = [30, 50, 100]
    crossover_probs = [0.6, 0.8, 0.9]
    init_methods = ["random"]  # Note: hybrid initialization would require MILP solution
    
    results = []
    total_runs = len(population_sizes) * len(crossover_probs) * len(init_methods) * n_trials
    run_count = 0
    
    print(f"\n{'='*60}")
    print(f"GA PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total configurations: {total_runs}")
    print(f"Budget B: {sensor_budget}")
    print(f"Max generations: {max_generations}")
    print(f"Trials per config: {n_trials}")
    print(f"{'='*60}\n")
    
    for pop_size in population_sizes:
        for p_cross in crossover_probs:
            for init_method in init_methods:
                for trial in range(n_trials):
                    run_count += 1
                    seed = 42 + trial  # Different seed per trial
                    
                    config_name = f"Pop{pop_size}_Cross{p_cross}_Init{init_method}_Trial{trial}"
                    print(f"\n[{run_count}/{total_runs}] Running: {config_name}")
                    
                    start_time = time.time()
                    
                    try:
                        # Run GA with current configuration
                        ga_result = run_ga_experiment(
                            population_size=pop_size,
                            max_generations=max_generations,
                            p_crossover=p_cross,
                            p_mutation=0.3,  # Keep constant
                            sensor_budget=sensor_budget,
                            tournament_size=2,  # Keep constant
                            random_seed=seed,
                            k_coverage=1,  # Keep constant
                            validation_freq=0,
                            milp_inputs=milp_inputs,
                            lambda_flow_balance=50.0,  # Current value
                        )
                        
                        runtime = time.time() - start_time
                        
                        # Extract metrics from logbook
                        logbook = ga_result["logbook"]
                        min_fitness_history = logbook.select("min")
                        avg_fitness_history = logbook.select("avg")
                        
                        best_fitness = min_fitness_history[-1]
                        final_avg_fitness = avg_fitness_history[-1]
                        
                        # Find convergence generation (when improvement < 1%)
                        convergence_gen = max_generations
                        for gen in range(1, len(min_fitness_history)):
                            if gen > 0:
                                improvement = abs(min_fitness_history[gen] - min_fitness_history[gen-1])
                                relative_improvement = improvement / (min_fitness_history[gen-1] + 1e-9)
                                if relative_improvement < 0.01:
                                    convergence_gen = gen
                                    break
                        
                        results.append({
                            "init_method": init_method,
                            "pop_size": pop_size,
                            "p_crossover": p_cross,
                            "trial": trial,
                            "seed": seed,
                            "best_fitness": best_fitness,
                            "avg_fitness": final_avg_fitness,
                            "convergence_gen": convergence_gen,
                            "runtime": runtime,
                            "min_history": min_fitness_history,
                            "avg_history": avg_fitness_history,
                        })
                        
                        print(f"  ✓ Best fitness: {best_fitness:.4f}, Runtime: {runtime:.1f}s")
                        
                    except Exception as e:
                        print(f"  ✗ ERROR: {e}")
                        continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    csv_path = RESULTS_DIR / "ga_sensitivity_results.csv"
    df_results_save = df_results.drop(columns=["min_history", "avg_history"])
    df_results_save.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    _plot_convergence_comparison(df_results)
    _plot_fitness_boxplots(df_results)
    _plot_parameter_heatmaps(df_results)
    
    return df_results


def _plot_convergence_comparison(df: pd.DataFrame):
    """Plot convergence curves grouped by parameter."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: By population size
    ax = axes[0]
    for pop_size in sorted(df["pop_size"].unique()):
        subset = df[df["pop_size"] == pop_size]
        # Average across trials and crossover probs
        histories = subset["min_history"].values
        max_len = max(len(h) for h in histories)
        
        # Pad histories to same length
        padded = np.array([
            list(h) + [h[-1]] * (max_len - len(h)) 
            for h in histories
        ])
        
        mean_history = padded.mean(axis=0)
        std_history = padded.std(axis=0)
        
        generations = range(len(mean_history))
        ax.plot(generations, mean_history, label=f"Pop={pop_size}", linewidth=2)
        ax.fill_between(generations, 
                        mean_history - std_history, 
                        mean_history + std_history, 
                        alpha=0.2)
    
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Convergence by Population Size", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: By crossover probability
    ax = axes[1]
    for p_cross in sorted(df["p_crossover"].unique()):
        subset = df[df["p_crossover"] == p_cross]
        histories = subset["min_history"].values
        max_len = max(len(h) for h in histories)
        
        padded = np.array([
            list(h) + [h[-1]] * (max_len - len(h)) 
            for h in histories
        ])
        
        mean_history = padded.mean(axis=0)
        std_history = padded.std(axis=0)
        
        generations = range(len(mean_history))
        ax.plot(generations, mean_history, label=f"p_cross={p_cross}", linewidth=2)
        ax.fill_between(generations, 
                        mean_history - std_history, 
                        mean_history + std_history, 
                        alpha=0.2)
    
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Convergence by Crossover Probability", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "convergence_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def _plot_fitness_boxplots(df: pd.DataFrame):
    """Plot box plots of final fitness by parameter."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: By population size
    ax = axes[0]
    df_plot = df.copy()
    df_plot["pop_size"] = df_plot["pop_size"].astype(str)
    sns.boxplot(data=df_plot, x="pop_size", y="best_fitness", ax=ax, palette="Set2")
    ax.set_xlabel("Population Size", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Final Fitness Distribution by Population Size", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Plot 2: By crossover probability
    ax = axes[1]
    df_plot["p_crossover"] = df_plot["p_crossover"].astype(str)
    sns.boxplot(data=df_plot, x="p_crossover", y="best_fitness", ax=ax, palette="Set2")
    ax.set_xlabel("Crossover Probability", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Final Fitness Distribution by Crossover Probability", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "fitness_boxplots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def _plot_parameter_heatmaps(df: pd.DataFrame):
    """Plot heatmaps showing parameter interactions."""
    
    # Aggregate by parameter combinations
    agg_df = df.groupby(["pop_size", "p_crossover"]).agg({
        "best_fitness": "mean",
        "convergence_gen": "mean",
        "runtime": "mean"
    }).reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ("best_fitness", "Mean Best Fitness"),
        ("convergence_gen", "Mean Convergence Generation"),
        ("runtime", "Mean Runtime (s)")
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        pivot = agg_df.pivot(index="pop_size", columns="p_crossover", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Crossover Probability", fontsize=11)
        ax.set_ylabel("Population Size", fontsize=11)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "parameter_heatmaps.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


if __name__ == "__main__":
    # Run sensitivity analysis
    results_df = run_sensitivity_analysis(
        sensor_budget=20,
        max_generations=30,
        n_trials=3
    )
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(results_df.groupby(["pop_size", "p_crossover"])["best_fitness"].describe())
    print("\n✓ Sensitivity analysis complete!")
