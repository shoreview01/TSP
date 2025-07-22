import numpy as np
import time
from methods.original import TSPMaxSum
from methods.brute_force import TSP_brute_force

# Example usage
s = np.array([
    [0.8,   10.1,  12.5,  0.1,  0.6,   8.4,  9.1],
    [0.9,   0.2,   0.9,   0.4,  0.1,   7.8,  6.5],
    [0.1,   0.5,   0.9,   0.9,  0.8,   5.7,  4.9],
    [0.9,   0.9,   0.5,   0.8,  0.9,   6.2,  7.4],
    [0.6,   0.009, 1.8,   0.9,  0.6,   8.3,  6.8],
    [7.7,   6.9,   5.5,   7.2,  8.1,   0.3,  1.2],
    [9.0,   7.8,   5.1,   6.6,  8.9,   1.1,  0.4]
])
np.random.seed(2)  # 결과 재현 가능하게
s = np.random.uniform(0, 20, size=(10, 10))


# TSP solver using the original method
solver = TSPMaxSum(s, verbose=True)
start_time = time.time()
path, history1 = solver.run()
end_time = time.time()
print(f"Time taken: {end_time - start_time:.4f} seconds")
print("Optimal path:", path)
print(f"Optimal cost: {solver.get_cost():.1f}")
print("Number of iterations:", solver.iterations)
print("Convergence history:", history1)

print("\n" + "="*40 + "\n")

# Brute force solver for comparison
solver_brute_force = TSP_brute_force(s)
start_time = time.time()
best_path, best_cost = solver_brute_force.run()
end_time = time.time()
print(f"Brute force time taken: {end_time - start_time:.4f} seconds")
print("Brute force optimal path:", best_path)
print(f"Brute force optimal cost: {best_cost:.1f}")
