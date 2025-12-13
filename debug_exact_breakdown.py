
import sys
from pathlib import Path
from pulp import value

# Add project root to path
sys.path.append(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras")

from src.ga_algorithm import TrafficNetwork
from src.milp_model import build_sensor_placement_model
from src.io_utils import leer_critical_edges
from src.preprocessing import construir_milp_inputs

def debug_exact_breakdown(selected_edges_path):
    print(f"Analyzing solution from: {selected_edges_path}")
    
    # 1. Load Solution
    selected = set()
    with open(selected_edges_path, 'r') as f:
        for line in f:
            if line.startswith("edge:"):
                selected.add(line.strip().split(":")[1])
    
    print(f"Loaded {len(selected)} sensors.")
    
    # 2. Setup Env
    print("Loading Traffic Network...")
    milp_inputs = construir_milp_inputs()
    milp_inputs["M"] = 100000.0 # Override Big-M
    domain = TrafficNetwork(milp_inputs=milp_inputs)
    
    # 3. Build Exact Model
    print("Building Exact Model...")
    model, x_vars = build_sensor_placement_model(
        milp_inputs,
        max_sensors=None, # Fix vars manually
        weight_scheme="inv_abs", 
        lambda_flow_balance=100.0,
        epsilon_weight=1e-3
    )
    
    # 4. Fix variables
    for eid, var in x_vars.items():
        val = 1 if eid in selected else 0
        var.setInitialValue(val)
        var.fixValue()
        
    # 5. Solve
    print("Solving Exact Model...")
    from pulp import PULP_CBC_CMD
    solver = PULP_CBC_CMD(msg=True)
    model.solve(solver)
    
    print(f"Model Status: {model.status}")
    print(f"Total Objective: {value(model.objective)}")
    
    # 6. Breakdown
    # Inspect variables from model
    # Only iterate vars if model solved
    
    residual_sum = 0.0
    slack_sum = 0.0
    
    # We need to access variables by name or reconstruct access
    # Easier to iterate model.variables() and check names
    for v in model.variables():
        if v.name.startswith("r_"):
            # This is weighted residual? No, v is just the variable value.
            # The objective coeff is w.
            # Hard to reconstruct w from here without map.
            pass
        elif v.name.startswith("z_"):
            slack_sum += v.varValue
            
    # To get proper weighted sums, likely need to re-calc using maps
    # Re-using logic from build_model is hard without returning the terms.
    
    # Alternative: use the internal maps if I can access them?
    # No, local to function.
    
    # Let's iterate the 'weights' map from domain (which matches model construction logic hopefully)
    # The domain has weights_map.
    
    total_w_r = 0.0
    total_lambda_z = 0.0
    lambda_val = 100.0
    
    # Iterate known keys
    # r vars
    for key, w in domain.weights_map.items():
        scen, tau, eid = key
        # var name: r_scen_tau_eid
        var_name = f"r_{scen}_{tau}_{eid}"
        v = model.variablesDict().get(var_name)
        if v:
            r_val = v.varValue
            total_w_r += w * r_val
            
    # z vars
    # keys from domain.b_balance_map keys (sc, t)?
    # Need CONS keys.
    # We can iterate model variables looking for z_
    for v in model.variables():
        if v.name.startswith("z_"):
             total_lambda_z += lambda_val * v.varValue
             
    print(f"--- Breakdown ---")
    print(f"Weighted Residuals Term: {total_w_r:.4f}")
    print(f"Flow Balance Slacks Term (lambda*z): {total_lambda_z:.4f} (lambda={lambda_val})")
    
    # Calc normalization used in GA
    # num_t_omega * max_sensors
    num_t_omega = len(domain.snapshot_keys)
    max_sensors = max(1, len(selected))
    normalization = 1.0 / (num_t_omega * max_sensors) if num_t_omega > 0 else 1.0
    
    print(f"Normalization Constant: {normalization}")
    print(f"Normalized Objective: {value(model.objective) * normalization}")

    return

if __name__ == "__main__":
    txt_path = r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras\data\processed\selected\selected_edges_B20_GA.txt"
    debug_exact_breakdown(txt_path)
