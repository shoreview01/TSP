import numpy as np
from hypercube3 import TSPHC3

class APViterbiTSP:
    def __init__(self, s, ap_model: TSPHC3):
        self.s = s
        self.N = s.shape[0]
        self.ap_model = ap_model  # 이미 학습된 메시지 상태 보유
        self.dp = {}
        self.reconstruct = {}

    def run(self):
        N = self.N
        cities = list(range(N - 1))  # exclude depot
        dp = {}  # (mask, current) -> cost
        recon = {}

        # 초기 상태 (depot -> i) using AP zeta / lambda values
        for i in cities:
            cost = self.ap_model.lambda_[0, i] + self.ap_model.s_alt(N - 1, i, 0, 0)
            dp[(1 << i, i)] = cost
            recon[(1 << i, i)] = N - 1

        for mask in range(1 << (N - 1)):
            for u in cities:
                if not (mask & (1 << u)):
                    continue
                for v in cities:
                    if (mask & (1 << v)) or (u == v):
                        continue
                    new_mask = mask | (1 << v)
                    cost = dp[(mask, u)] + self.ap_model.s_alt(u, v, mask, bin(mask).count("1"))
                    if (new_mask, v) not in dp or cost < dp[(new_mask, v)]:
                        dp[(new_mask, v)] = cost
                        recon[(new_mask, v)] = u

        # 종료 (i → depot)
        end_mask = (1 << (N - 1)) - 1
        best_cost = np.inf
        last = None
        for u in cities:
            total_cost = dp[(end_mask, u)] + self.ap_model.s_alt(u, N - 1, end_mask, N - 1)
            if total_cost < best_cost:
                best_cost = total_cost
                last = u

        # 경로 복원
        path = [N - 1]
        mask = end_mask
        curr = last
        for _ in range(N - 1):
            path.append(curr)
            curr_prev = recon[(mask, curr)]
            mask ^= (1 << curr)
            curr = curr_prev
        path.append(N - 1)
        path.reverse()

        self.dp = dp
        self.tour = path
        self.min_cost = best_cost
        return path, best_cost

dist = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]
])
model = TSPHC3(dist)
solver = APViterbiTSP(dist, ap_model=model)
path, cost = solver.run()
print(path)