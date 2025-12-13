
import sys
from pathlib import Path
from pulp import value, PULP_CBC_CMD

sys.path.append(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras")

from src.ga_algorithm import TrafficNetwork
from src.milp_model import build_sensor_placement_model
from src.preprocessing import construir_milp_inputs

def solve_verbose():
    print("Loading Inputs...")
    milp_inputs = construir_milp_inputs()
    # Force larger M just in case
    # milp_inputs["M"] = 100000.0 
    
    selected_path = r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras\data\processed\selected\selected_edges_B20_GA.txt"
    selected = set()
    with open(selected_path, 'r') as f:
        for line in f:
            if line.startswith("edge:"):
                selected.add(line.strip().split(":")[1])
    
    print(f"Loaded {len(selected)} sensors.")
    
    # We must replicate the cycle loading of the GA exactly if we want to test that hypothesis
    # But for now, let's just use the DEFAULT cycles loaded by milp loop (which I updated to use file)
    # The GA uses the same file logic now.
    
    # But wait, GA calculates cycles from `critical_edges.txt`.
    # Exact model also calculates from `critical_edges.txt` (unless forced).
    # Since I didn't verify if `forced_cycles` works in the *standalone* check, 
    # let's assume `build_sensor_placement_model` loads from txt if not forced.
    # Determinism issue: `ciclos_en_edges_criticos` MIGHT be non-deterministic if `nx.cycle_basis` is unstable.
    # If the GA found a solution for Basis A, and we load Basis B here, we might fail.
    # However, running this script will tell us if it IS infeasible for the *current* basis logic.
    
    print("Building Model...")
    model, x_vars = build_sensor_placement_model(
        milp_inputs,
        max_sensors=None,
        weight_scheme="inv_abs", 
        lambda_flow_balance=100.0,
        epsilon_weight=1e-3,
        # forced_cycles=None # Let it load from file
    )
    
    # Fix variables
    for eid, var in x_vars.items():
        val = 1 if eid in selected else 0
        var.setInitialValue(val)
        var.fixValue()
        
    print("Solving with Verbose Output...")
    solver = PULP_CBC_CMD(msg=True)
    model.solve(solver)
    
    print(f"Status: {model.status}")
    print(f"Objective: {value(model.objective)}")

if __name__ == "__main__":
    solve_verbose()
