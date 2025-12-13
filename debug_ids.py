
import sys
from pathlib import Path
sys.path.append(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras")
from src.preprocessing import construir_milp_inputs

def check_ids():
    print("Checking IDs...")
    inputs = construir_milp_inputs()
    valid_edges = set(inputs["edges"]["edge_id"].unique())
    print(f"Total valid edges in system: {len(valid_edges)}")
    
    path = r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras\data\processed\selected\selected_edges_B20_GA.txt"
    selected = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("edge:"):
                # Careful strip
                raw = line.strip()
                val = raw.split(":", 1)[1]
                val_clean = val.strip()
                selected.add(val_clean)
                
    print(f"Read {len(selected)} edges from file.")
    
    # Check intersection
    intersect = valid_edges.intersection(selected)
    print(f"Valid Edges in File: {len(intersect)}")
    
    invalid = selected - valid_edges
    if invalid:
        print(f"INVALID IDs in file: {invalid}")
    else:
        print("All file IDs are valid known edges.")
        
    # Check if a known cycle is covered
    # We need to replicate the cycle load logic exactly
    from src.io_utils import leer_critical_edges
    from src.topology_utils import ciclos_en_edges_criticos
    
    crit_path = Path(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras\data\processed\critical_edges.txt")
    ce = leer_critical_edges(crit_path)
    cycles = ciclos_en_edges_criticos(inputs, ce)
    
    print(f"Loaded {len(cycles)} cycles.")
    violated = 0
    for i, c in enumerate(cycles):
        # c is list of strings
        # count how many in selected
        count = sum(1 for e in c if e in selected)
        if count == 0:
            violated += 1
            if violated == 1:
                print(f"First Violated Cycle: {c}")
                # Check individual edges
                for e in c:
                    print(f"  Edge {e} in selected? {e in selected}")
                    
    print(f"Total Violated Cycles (Verification): {violated}")

if __name__ == "__main__":
    check_ids()
