
from scripts.run_ga import run_ga_experiment
from src.config import PROJECT_ROOT

print("Running short verification GA...")
res = run_ga_experiment(
    max_generations=2,
    population_size=10,
    sensor_budget=20,
    k_coverage=1
)
print("Verification complete.")
