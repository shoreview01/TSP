import numpy as np
import time
from methods.hypercube import TSPHC1
from methods.hypercube2 import TSPHC2
from methods.brute_force import TSP_brute_force

def tsp_bitmask_dp(dist):
    n = dist.shape[0]
    cities = list(range(n - 1))  # exclude depot (last city)

    dp = {}  # key: (visited_bitmask, current_city), value: (cost, prev_city)

    # Initialize: start from depot to city i
    for i in cities:
        dp[(1 << i, i)] = (dist[n - 1][i], n - 1)

    # DP: only for non-depot cities
    for visited in range(1 << (n - 1)):
        for u in cities:
            if not (visited & (1 << u)):
                continue
            for v in cities:
                if visited & (1 << v):
                    continue
                new_visited = visited | (1 << v)
                prev_cost = dp.get((visited, u), (np.inf, -1))[0]
                new_cost = prev_cost + dist[u][v]
                if (new_visited, v) not in dp or new_cost < dp[(new_visited, v)][0]:
                    dp[(new_visited, v)] = (new_cost, u)

    # Find best path back to depot
    end_mask = (1 << (n - 1)) - 1
    min_cost = np.inf
    last_city = -1
    for u in cities:
        cost_to_depot = dp.get((end_mask, u), (np.inf, -1))[0] + dist[u][n - 1]
        if cost_to_depot < min_cost:
            min_cost = cost_to_depot
            last_city = u

    # Reconstruct path
    tour = [n - 1]  # start from depot
    mask = end_mask
    curr = last_city
    for _ in range(n - 1):
        tour.append(curr)
        mask, curr = mask ^ (1 << curr), dp[(mask, curr)][1]
    tour.append(n - 1)  # end at depot
    tour.reverse()

    return tour, min_cost


# Sample input
dist = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]  # depot = node 4
])
s_size = 14
np.random.seed(1000)
dist = np.random.uniform(0, 10, size=(s_size, s_size))
N = dist[0].shape
start = time.time()
tour, cost = tsp_bitmask_dp(dist)
end = time.time()
tour_mid = tour[1:-1]
print("Tour:", N, [x+1 for x in tour_mid], N, f"Total Cost: {cost:.4f}")
print(f"Elasped time: {(end - start)*1000:.4f} ms")
solver = TSPHC2(dist, c_old=tour_mid, verbose=True)
path, history = solver.run()
solver2 = TSP_brute_force(dist)
path1, cost1 = solver2.run()
print(f"cost: {cost1:.4f}")