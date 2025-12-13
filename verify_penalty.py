
import sys
from pathlib import Path

# Add project root to path
sys.path.append(r"c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras")

from src.ga_algorithm import TrafficNetwork

# Mock TrafficNetwork to isolate surrogate logic
class MockTrafficNetwork(TrafficNetwork):
    def __init__(self):
        self.edge_ids = [f"e{i}" for i in range(10)]
        self.num_edges = 10
        self.edge_map = {eid: i for i, eid in enumerate(self.edge_ids)}
        
        # Define a cycle that requires coverage
        self.cycles = [["e0", "e1", "e2", "e3", "e4"]] 
        self.cuts = []
        self.k_coverage = 4 # Need 5 sensors for this cycle (k+1 coverage? default k=1, logic says (k+1) - count needed)
        # Wait, previous logic was: count < (k+1). if k=1, needed 2 sensors.
        # My reproduction used k=4 -> needed 5.
        
        self.snapshot_keys = []
        self.surrogate_data = {}

    def fitness_function(self, individual, use_surrogate=True):
        # We manually call evaluate_surrogate in tests but let's override parent just in case
        return super().fitness_function(individual, use_surrogate)

    def evaluate_surrogate(self, individual, sample_size=5):
        # Mocking parent implementation just to focus on the penalty part which IS inside parent method
        # But wait, I modified the PARENT method. I need to call the ACTUAL parent method.
        # So I should instantiate the REAL class but mock its attributes?
        pass

# Better: Instantiate REAL TrafficNetwork, but mock its init heavy lifting
# But TrafficNetwork.__init__ does a lot.
# Let's instantiate normally but with dummy paths or bypass init.
# Or just copy the method logic here for verification? No, I want to verify the modified file.

# Let's monkeypatch TrafficNetwork.__init__ to do nothing
old_init = TrafficNetwork.__init__
def new_init(self, k_coverage=1, **kwargs):
    self.edge_ids = [f"e{i}" for i in range(10)]
    self.num_edges = 10
    self.edge_map = {eid: i for i, eid in enumerate(self.edge_ids)}
    self.cycles = [["e0", "e1", "e2"]] # k=1 -> need 2 sensors
    self.cuts = []
    self.k_coverage = k_coverage
    self.snapshot_keys = ["s1"]
    
    # Mock surrogate data for base fitness
    # (b_bal, f_vec, w_vec)
    import numpy as np
    b = np.zeros(10)
    f = np.zeros(10)
    w = np.ones(10)
    self.surrogate_data = {"s1": (b, f, w)}
    self.node_ids = ["n1", "n2"]
    self.node_map = {"n1": 0, "n2": 1}
    self._adj_struct = [[] for _ in range(2)]

TrafficNetwork.__init__ = new_init

try:
    tn = TrafficNetwork(k_coverage=1)
    
    # Case 1: Invalid individual (0 sensors in cycle)
    # Cycle e0,e1,e2. Need 2. Have 0.
    ind_invalid = [0] * 10
    fit_invalid = tn.evaluate_surrogate(ind_invalid)
    print(f"Invalid Fitness (0 sensors): {fit_invalid}")
    
    # Case 2: Valid individual (2 sensors in cycle)
    ind_valid = [0] * 10
    ind_valid[0] = 1 # e0
    ind_valid[1] = 1 # e1
    fit_valid = tn.evaluate_surrogate(ind_valid)
    print(f"Valid Fitness (2 sensors): {fit_valid}")
    
    if fit_invalid > 1e5 and fit_valid < 100:
        print("SUCCESS: Penalty logic works as expected.")
    else:
        print("FAILURE: Penalty logic verification failed.")
        
finally:
    TrafficNetwork.__init__ = old_init
