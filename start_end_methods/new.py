import numpy as np
import pandas as pd

# Reload the distance matrix file
file_path = "map/Capital_Cities_DistanceMatrix_Penalty50.csv"
dist_df = pd.read_csv(file_path, index_col=0)
dist_df = dist_df.iloc[:15,:15]
start_city="서울"
end_city="용인"

class TSPStartEndNew:
    def __init__(self, dist_df, start_city, end_city, max_threshold=50, verbose=False):
        self.city_names = dist_df.index.tolist()
        self.dist_matrix = dist_df.values.astype(float)
        self.max_threshold = max_threshold
        self.verbose = verbose
        
        self.parity_df = (dist_df < max_threshold).astype(int)

        self.start_idx = self.city_names.index(start_city)
        self.end_idx = self.city_names.index(end_city)
        
        # 도시 재정렬: start, end를 맨 뒤로
        remaining = [i for i in range(len(self.city_names)) if i not in [self.start_idx, self.end_idx]]
        self.new_order = remaining + [self.end_idx, self.start_idx]
        self.reordered_matrix = self.dist_matrix[np.ix_(self.new_order, self.new_order)]
        self.reordered_cities = [self.city_names[i] for i in self.new_order]
        self.reordered_parity_matrix = (self.reordered_matrix < max_threshold).astype(int)

        self.start = len(self.reordered_cities) - 1
        self.end = len(self.reordered_cities) - 2
        self.n = len(self.reordered_cities)
        
    def run(self):
        final_mask = None
        last_city = None
        dp = {}
        self.min_cost = np.inf
        n = self.n-1
        cities = self.reordered_cities
        city_range = range(n)
        index_map = {i: city for i, city in enumerate(range(n))}
        reverse_map = {city: i for i, city in index_map.items()}
        
        for v in city_range:
            if self.reordered_parity_matrix[n][v] == 0:
                continue
            if (0,n) not in dp:
                dp[(0, n)] = self.reordered_matrix[n][v]
            else:
                prev_cost = dp[(0,n)]
                if self.reordered_matrix[n][v] < prev_cost:
                    dp[(0, n)] = self.reordered_matrix[n][v]
        
        for visited in range(1,1 << n):
            if visited & self.end:
                continue
            for u in city_range:
                if (visited,u) not in dp:
                    continue
                if u == self.end:
                        total_cost = dp[(visited, u)][0]
                        if total_cost < min_cost:
                            min_cost = total_cost
                            final_mask = visited
                            last_city = u
                        continue
                for v in city_range:
                    if (u == v):
                        continue
                    v_bit = 1 << reverse_map[v]
                    if visited & v_bit:
                        continue
                    if self.reordered_parity_matrix[u][v] == 0:
                        continue
                    new_visited = visited | v_bit
                    prev_cost = dp[(visited, u)][0]
                    new_cost = prev_cost + self.reordered_matrix[u][v]
                    if (new_visited, v) not in dp or new_cost < dp[(new_visited, v)][0]:
                        dp[(new_visited, v)] = (new_cost, u)
        if last_city is None:
            raise ValueError("No valid path from start to end under threshold")

        # 경로 복원
        path = [last_city]
        mask = final_mask
        curr = last_city
        while mask:
            prev = dp[(mask, curr)][1]
            path.append(prev)
            if prev == self.start:
                break
            mask ^= 1 << reverse_map[curr]
            curr = prev

        path.reverse()
        self.dp = dp
        self.tour = path
        self.min_cost = min_cost
        
        if self.verbose:
            print(" → ".join(self.reordered_cities[i] for i in path))
            print(f"Total Cost: {min_cost:.2f}")

        return [self.reordered_cities[i] for i in path], min_cost


solver = TSPStartEndNew(dist_df, start_city=start_city, end_city=end_city, verbose=True)
path, cost = solver.run()