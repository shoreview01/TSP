import numpy as np

class TSPBitmask:
    def __init__(self, dist, max=50, verbose=False):
        self.dist = dist
        self.verbose = verbose
        self.n = dist.shape[0]
        self.dp = {}
        self.tour = []
        self.min_cost = np.inf
        self.max = max

    def run(self):
        n = self.n
        cities = list(range(n - 1))  # exclude depot

        dp = {}  # (visited_mask, current_city) → (cost, prev_city)

        # Initialize: depot → i
        for i in cities:
            dp[(1 << i, i)] = (self.dist[n - 1][i], n - 1)

        for visited in range(1 << (n - 1)):
            for u in cities:
                if not (visited & (1 << u)):
                    continue
                for v in cities:
                    if visited & (1 << v):
                        continue
                    if self.dist[u][v] >= self.max:
                        continue
                    new_visited = visited | (1 << v)
                    prev_cost = dp.get((visited, u), (np.inf, -1))[0]
                    new_cost = prev_cost + self.dist[u][v]
                    if (new_visited, v) not in dp or new_cost < dp[(new_visited, v)][0]:
                        dp[(new_visited, v)] = (new_cost, u)

        end_mask = (1 << (n - 1)) - 1
        min_cost = np.inf
        last_city = -1
        for u in cities:
            cost_to_depot = dp.get((end_mask, u), (np.inf, -1))[0] + self.dist[u][n - 1]
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
        tour.append(n - 1)
        tour.reverse()

        self.dp = dp
        self.tour = tour
        self.min_cost = min_cost

        if self.verbose:
            print(f"Tour (1-based): {[x+1 for x in tour]}")
            print(f"Total Cost: {min_cost:.4f}")

        return tour, min_cost

    def get_path(self):
        return self.tour

    def get_cost(self):
        return self.min_cost
