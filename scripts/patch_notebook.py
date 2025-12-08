
import json
from pathlib import Path

notebook_path = Path("notebooks/01_experimentos.ipynb")

def patch_notebook():
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Modify the cell with SimulatedAnnealing instantiation
    patched = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "SimulatedAnnealing(network, budget=budget, max_iter=3000, initial_temp=10000)" in source:
                print("Found SA cell. Patching parameters...")
                new_source = source.replace(
                    "SimulatedAnnealing(network, budget=budget, max_iter=3000, initial_temp=10000)",
                    "SimulatedAnnealing(network, budget=budget, max_iter=500, initial_temp=1000)"
                )
                cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in new_source.splitlines()]
                # fix splitlines removing \n
                cell['source'] = []
                # Reconstruct source list correctly
                lines = new_source.split('\n')
                for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        cell['source'].append(line + '\n')
                    else:
                        if line: # only add last line if not empty, or check how split worked
                             cell['source'].append(line)
                
                patched = True
            
            # Also patch the multi-seed experiment cell
            if "max_iter=3000" in source and "run_multi_seed_experiment" in source:
                 print("Found Multi-seed cell. Patching parameters...")
                 new_source = source.replace("max_iter=3000", "max_iter=500")
                 
                 cell['source'] = []
                 lines = new_source.split('\n')
                 for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        cell['source'].append(line + '\n')
                    else:
                        if line:
                             cell['source'].append(line)
                 patched = True

    if patched:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook patched successfully.")
    else:
        print("Target code patterns not found. Notebook might already be patched or different.")

if __name__ == "__main__":
    patch_notebook()
