import numpy as np

class TSPBitmaskStartEnd:
    def __init__(self, dist_df, start_city, end_city, max_threshold=50, verbose=False):
        self.city_names = dist_df.index.tolist()
        self.dist_matrix = dist_df.values.astype(float)
        self.max_threshold = max_threshold
        self.verbose = verbose

        self.start_idx = self.city_names.index(start_city)
        self.end_idx = self.city_names.index(end_city)

        # 도시 재정렬: start, end를 맨 뒤로
        remaining = [i for i in range(len(self.city_names)) if i not in [self.start_idx, self.end_idx]]
        self.new_order = remaining + [self.start_idx, self.end_idx]
        self.reordered_matrix = self.dist_matrix[np.ix_(self.new_order, self.new_order)]
        self.reordered_cities = [self.city_names[i] for i in self.new_order]

        self.start = len(self.reordered_cities) - 2
        self.end = len(self.reordered_cities) - 1
        self.n = len(self.reordered_cities)

        self.middle_indices = [i for i in range(self.n) if i != self.start]
        self.k = len(self.middle_indices)
        self.dp = {}
        self.tour = []
        self.min_cost = np.inf

    def run(self):
        k = self.k
        dp = {}

        index_map = {i: city for i, city in enumerate(self.middle_indices)}
        reverse_map = {city: i for i, city in index_map.items()}

        # 초기화
        for i, city in index_map.items():
            if self.reordered_matrix[self.start][city] < self.max_threshold:
                dp[(1 << i, city)] = (self.reordered_matrix[self.start][city], self.start)

        min_cost = np.inf
        final_mask = None
        last_city = None

        for visited in range(1 << k):
            for u in self.middle_indices:
                if (visited, u) not in dp:
                    continue
                if u == self.end:
                    total_cost = dp[(visited, u)][0]
                    if total_cost < min_cost:
                        min_cost = total_cost
                        final_mask = visited
                        last_city = u
                    continue
                for v in self.middle_indices:
                    if v == u:
                        continue
                    v_bit = 1 << reverse_map[v]
                    if visited & v_bit:
                        continue
                    if self.reordered_matrix[u][v] >= self.max_threshold:
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