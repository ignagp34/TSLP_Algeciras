# Test GA Components

import sys
from pathlib import Path
sys.path.append(str(Path(".").resolve()))

from src.ga_algorithm import TrafficNetwork

def test_ga_components():
    print("--- Testing TrafficNetwork Initialization ---")
    try:
        domain = TrafficNetwork(k_coverage=1)
        print("Initialization successful.")
        print(f"Num edges: {domain.num_edges}")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("\n--- Testing Repair Operator ---")
    budget = 10
    # Create individual with all 1s (huge budget violation)
    ind = [1] * domain.num_edges
    print(f"Initial sensors: {sum(ind)}")
    
    repaired_ind = domain.repair_individual(ind, budget=budget)
    curr_sensors = sum(repaired_ind)
    print(f"Repaired sensors: {curr_sensors}")
    
    if curr_sensors <= budget:
        print("Budget constraint: PASSED")
    else:
        print(f"Budget constraint: FAILED (Expected <= {budget}, got {curr_sensors})")

    # Test coverage (harder to assert without known topology violations, but we check it runs)
    # Force 0 sensors and see if it adds any for coverage
    ind_empty = [0] * domain.num_edges
    # This might take a while if many cycles
    # repaired_empty = domain.repair_individual(ind_empty, budget=budget)
    # print(f"Repaired empty (added for coverage?): {sum(repaired_empty)}")

    print("\n--- Testing Fitness Function (Exact) ---")
    # Take a small sample valid individual
    # repair_individual returns a list, we need it as list for fitness
    try:
        fit = domain.fitness_function(repaired_ind)
        print(f"Fitness value: {fit}")
        if isinstance(fit, tuple) and len(fit) == 1 and isinstance(fit[0], float):
             print("Fitness format: PASSED")
        else:
             print(f"Fitness format: FAILED {fit}")
    except Exception as e:
        print(f"Fitness evaluation failed: {e}")

if __name__ == "__main__":
    test_ga_components()
