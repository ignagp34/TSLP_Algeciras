
import json
from pathlib import Path

# Paths
notebook_path = Path("notebooks/01_experimentos.ipynb")

# New cells to append
new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simulated Annealing (SA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.sa_algorithm import SimulatedAnnealing\n",
    "from src.ga_algorithm import TrafficNetwork, save_ga_solution\n",
    "\n",
    "# Load network\n",
    "csv_path = DATA_PROCESSED / \"edgeData_scen2.csv\"\n",
    "network = TrafficNetwork(csv_path)\n",
    "\n",
    "budget = 20\n",
    "\n",
    "print(f\"Running SA with B={budget}...\")\n",
    "sa = SimulatedAnnealing(network, budget=budget, max_iter=3000, initial_temp=10000)\n",
    "sa_result = sa.run(seed=42)\n",
    "\n",
    "print(\"Best SA Fitness (RMSE):\", sa_result['best_fitness'])\n",
    "\n",
    "# Save solution\n",
    "output_sa_path = DATA_PROCESSED / f\"selected/selected_edges_B{budget}_SA.txt\"\n",
    "save_ga_solution(sa_result['best_solution'], network, output_sa_path)\n",
    "print(f\"SA Solution saved to {output_sa_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multi-Seed Experiments & Variance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.experiments import run_multi_seed_experiment, plot_variance_analysis\n",
    "import pandas as pd\n",
    "\n",
    "# Define seeds\n",
    "seeds = [42, 10, 2023, 777, 99]\n",
    "budget = 20\n",
    "\n",
    "# Run GA\n",
    "print(\"Running Multi-seed GA...\")\n",
    "df_ga = run_multi_seed_experiment(network, 'GA', budget, seeds, population_size=50, max_generations=50)\n",
    "\n",
    "# Run SA\n",
    "print(\"Running Multi-seed SA...\")\n",
    "df_sa = run_multi_seed_experiment(network, 'SA', budget, seeds, max_iter=3000)\n",
    "\n",
    "# Combine results\n",
    "df_all = pd.concat([df_ga, df_sa], ignore_index=True)\n",
    "\n",
    "# Display results\n",
    "display(df_all.groupby('algorithm')['rmse'].describe())\n",
    "\n",
    "# Plot Variance\n",
    "output_plot = RESULTS_DIR / \"variance_analysis.png\"\n",
    "plot_variance_analysis(df_all, output_path=output_plot)"
   ]
  }
]

def update_notebook():
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Append cells
    nb['cells'].extend(new_cells)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully appended {len(new_cells)} cells to {notebook_path}")

if __name__ == "__main__":
    update_notebook()
