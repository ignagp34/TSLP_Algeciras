
import numpy as np
import random
import math
import copy

class SimulatedAnnealing:
    def __init__(self, traffic_network, budget, initial_temp=1000, cooling_rate=0.99, min_temp=1, max_iter=1000):
        self.network = traffic_network
        self.budget = budget
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter = max_iter
        self.num_edges = self.network.num_edges

    def _generate_initial_solution(self):
        """Generates a random initial mask respecting the budget."""
        mask = np.zeros(self.num_edges, dtype=int)
        indices = np.random.choice(self.num_edges, self.budget, replace=False)
        mask[indices] = 1
        return mask

    def _get_neighbor(self, current_mask):
        """
        Generates a neighbor solution by moving a sensor to an empty spot.
        Ensures the budget is maintained (swap 1 for 0).
        """
        neighbor_mask = current_mask.copy()
        
        # Find indices where sensors are present (1s) and absent (0s)
        sensor_indices = np.where(neighbor_mask == 1)[0]
        empty_indices = np.where(neighbor_mask == 0)[0]
        
        if len(sensor_indices) > 0 and len(empty_indices) > 0:
            # Pick one sensor to remove
            remove_idx = np.random.choice(sensor_indices)
            # Pick one empty spot to add
            add_idx = np.random.choice(empty_indices)
            
            neighbor_mask[remove_idx] = 0
            neighbor_mask[add_idx] = 1
            
        return neighbor_mask

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 1. Initialization
        current_solution = self._generate_initial_solution()
        current_fitness = self.network.evaluate_sensor_placement(current_solution, self.budget)[0]
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temp = self.initial_temp
        
        history = []
        
        for i in range(self.max_iter):
            # Check stopping condition
            if temp < self.min_temp:
                break
                
            # 2. Generate neighbor
            neighbor_solution = self._get_neighbor(current_solution)
            neighbor_fitness = self.network.evaluate_sensor_placement(neighbor_solution, self.budget)[0]
            
            # 3. Acceptance probability
            # We want to minimize fitness (RMSE)
            delta = neighbor_fitness - current_fitness
            
            if delta < 0:
                # Improvement: accept always
                accept = True
            else:
                # Worsening: accept with probability
                prob = math.exp(-delta / temp)
                accept = random.random() < prob
            
            if accept:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                # Update global best
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
            
            # 4. Cooling
            temp *= self.cooling_rate
            
            history.append({
                'iter': i,
                'temp': temp,
                'fitness': current_fitness,
                'best_fitness': best_fitness
            })
            
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': history
        }
