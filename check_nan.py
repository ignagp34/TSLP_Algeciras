
import sys
import numpy as np
import pandas as pd
sys.path.append(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras")

from src.preprocessing import construir_milp_inputs

def check_inputs():
    print("Checking milp_inputs for NaNs...")
    inputs = construir_milp_inputs()
    
    # Check Flows
    flows = inputs["flows"]
    if flows["flow_veh_h"].isna().any():
        print("FAIL: flows['flow_veh_h'] contains NaNs!")
        print(flows[flows["flow_veh_h"].isna()])
    else:
        print("flows['flow_veh_h'] OK.")
        
    # Check O/D
    O = inputs["O"]
    if O["O_veh_h"].isna().any():
         print("FAIL: O['O_veh_h'] contains NaNs!")
    else:
         print("O OK.")
         
    D = inputs["D"]
    if D["D_veh_h"].isna().any():
         print("FAIL: D['D_veh_h'] contains NaNs!")
    else:
         print("D OK.")
         
    # Check Edges/Nodes
    print(f"Nodes: {len(inputs['nodes'])}, Edges: {len(inputs['edges'])}")

if __name__ == "__main__":
    check_inputs()
