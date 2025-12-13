
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras")

from src.ga_algorithm import TrafficNetwork
from src.preprocessing import construir_milp_inputs
from src.topology_utils import ciclos_en_edges_criticos, cortes_minimos_en_zona_critica
from src.io_utils import leer_critical_edges

def verify_solution_constraints(selected_edges_path):
    print(f"--- Verifying Solution: {selected_edges_path} ---")
    
    # 1. Load Solution
    selected = set()
    with open(selected_edges_path, 'r') as f:
        for line in f:
            if line.startswith("edge:"):
                selected.add(line.strip().split(":")[1])
    print(f"Selected {len(selected)} sensors.")
    
    # 2. Load Inputs
    milp_inputs = construir_milp_inputs()
    flows = milp_inputs["flows"]
    # Check Big-M
    M = float(milp_inputs.get("M", 2520.0))
    print(f"Using Big-M: {M}")
    
    critical_path = Path(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras\data\processed\critical_edges.txt")
    crit_edges = leer_critical_edges(critical_path)
    
    # 3. Check Cycles
    cycles = ciclos_en_edges_criticos(milp_inputs, crit_edges)
    print(f"Checking {len(cycles)} cycles...")
    if len(cycles) > 0:
        print(f"  -> DEBUG First Cycle: {cycles[0]}")
    cycles_violated = 0
    for idx, cycle in enumerate(cycles):
        # exact model logic: sum(x) >= 1
        # count valid sensors
        count = sum(1 for e in cycle if e in selected)
        # However, MILP only counts if e in x_vars (candidates)
        # We know all 'selected' are candidates.
        # But are there edges in cycle that are candidates but not selected? checking coverage.
        # If count >= 1, satisfied.
        if count < 1:
            print(f"  [VIOLATION] Cycle {idx} uncovered! Edges: {cycle}")
            cycles_violated += 1
            
    # 4. Check Cuts
    cuts = cortes_minimos_en_zona_critica(milp_inputs, crit_edges)
    print(f"Checking {len(cuts)} cuts...")
    cuts_violated = 0
    for idx, cut in enumerate(cuts):
        count = sum(1 for e in cut if e in selected)
        if count < 1:
            print(f"  [VIOLATION] Cut {idx} uncovered! Edges: {cut}")
            cuts_violated += 1
            
    print(f"Summary: {cycles_violated} cycles violated, {cuts_violated} cuts violated.")
    
    # 5. Check Big-M Feasibility (Heuristic)
    # If x=0, f must be in [hat - M, hat + M].
    # This is only a problem if valid f MUST be outside this range to satisfy Balance.
    # We can't easily check this without solving the LP. 
    # But checking coverage is the first step.
    
    return

if __name__ == "__main__":
    txt_path = r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras\data\processed\selected\selected_edges_B20_GA.txt"
    verify_solution_constraints(txt_path)
