import numpy as np
from itertools import permutations

class TSP_brute_force:
    def __init__(self, s):
        self.s = s
        self.N = s.shape[0] - 1  # Exclude depot
        self.iterations = 0

    def run(self):
        best_path, best_cost = self.brute_force_tsp(self.s)
        best_path = np.array([node + 1 for node in best_path] ) # Convert to 1-based indexing
        self.iterations = len(list(permutations(range(self.N))))
        return best_path, best_cost
    
    def brute_force_tsp(self, s):
        cities = list(range(self.N))
        depot = self.N
        best_cost = float('inf')
        best_path = None

        for perm in permutations(cities):
            path = [depot] + list(perm) + [depot]
            cost = sum(s[path[i], path[i+1]] for i in range(len(path)-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path
                
        return best_path, best_cost