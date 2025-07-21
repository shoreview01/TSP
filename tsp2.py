import numpy as np
import time
from methods.original import TSPMaxSum
from methods.hypercube import TSPHC1
from methods.hypercube2 import TSPHC2
from methods.brute_force import TSP_brute_force

# Example usage
s1 = np.array([
    [0.8,   10.1,  12.5,  0.1,  0.6,   8.4,  9.1],
    [0.9,   0.2,   0.9,   0.4,  0.1,   7.8,  6.5],
    [0.1,   0.5,   0.9,   0.9,  0.8,   5.7,  4.9],
    [0.9,   0.9,   0.5,   0.8,  0.9,   6.2,  7.4],
    [0.6,   0.009, 1.8,   0.9,  0.6,   8.3,  6.8],
    [7.7,   6.9,   5.5,   7.2,  8.1,   0.3,  1.2],
    [9.0,   7.8,   5.1,   6.6,  8.9,   1.1,  0.4]
])
s2 = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]
])
s_size = 11
s3 = np.random.uniform(0, 10, size=(s_size, s_size))

verbose = False
win = [0, 0, 0]
banbok = 1 # Number of iterations for testing

for i in range(banbok):
    s = s2
    s_size = s.shape[0]
    print("s matrix : \n", s)

    print("\n" + "="*40 + "\n")

    # TSP solver using the original method
    solver = TSPMaxSum(s, verbose=verbose)
    start_time = time.time()
    path = solver.run()
    end_time = time.time()
    print(f"Time taken (original): {end_time - start_time:.4f} seconds")
    print("Optimal path (original):", path)
    print(f"Optimal cost (original): {solver.get_cost():.1f}")
    print("Number of iterations (original):", solver.iterations)
    sum = np.sum(path)
    if sum != (s_size * (s_size + 1)/2 + s_size):
        print("Warning: The path does not make sense.")
        
    print("\n" + "="*40 + "\n")

    # Using the hypercube method

    # Using the original hypercube method
    start_time = time.time()
    solver1 = TSPHC1(s, verbose=verbose)
    path1 = solver1.run()
    end_time = time.time()
    print(f"Time taken (hypercube1): {end_time - start_time:.4f} seconds")
    print("Optimal path (hypercube1):", path1)
    print(f"Optimal cost (hypercube1): {solver1.get_cost():.1f}")
    print("Number of iterations (hypercube1):", solver1.iterations)
    sum = np.sum(path1)
    if sum != (s_size * (s_size + 1)/2 + s_size):
        print("Warning: The path does not make sense.")

    print("\n" + "="*40 + "\n")

    # Using the new hypercube method
    start_time = time.time()
    solver2 = TSPHC2(s, verbose=verbose)
    path2 = solver2.run()
    end_time = time.time()
    print(f"Time taken (hypercube2): {end_time - start_time:.4f} seconds")
    print("Optimal path (hypercube2):", path2)
    print(f"Optimal cost (hypercube2): {solver2.get_cost():.1f}")
    print("Number of iterations (hypercube2):", solver2.iterations)
    sum = np.sum(path2)
    if sum != (s_size * (s_size + 1)/2 + s_size):
        print("Warning: The path does not make sense.")

    if solver.get_cost() < solver1.get_cost() and solver.get_cost() < solver2.get_cost():
        win[0] += 1
    elif solver1.get_cost() < solver.get_cost() and solver1.get_cost() < solver2.get_cost():
        win[1] += 1
    else:
        win[2] += 1

    print("\n" + "="*40 + "\n")

# Brute force solver for comparison
start_time = time.time()
solver_brute_force = TSP_brute_force(s)
best_path, best_cost = solver_brute_force.run()
end_time = time.time()
print(f"Brute force time taken (brute_force): {end_time - start_time:.4f} seconds")
print("Brute force optimal path (brute_force):", best_path)
print(f"Brute force optimal cost (brute_force): {best_cost:.1f}")

if (banbok > 1):
    print("\n" + "="*40 + "\n")
    print("Comparison of methods:")
    print(f"Original method wins: {win[0]} times")
    print(f"Hypercube method 1 wins: {win[1]} times")
    print(f"Hypercube method 2 wins: {win[2]} times\n")
