"""
Reweighted-Message-Passing(TSP) — starts at city N+1 (index N),
visits every other city exactly once, and returns to the depot.
Requires:  numpy, scipy (≥ 1.10 for linear_sum_assignment).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from methods.hypercube3 import TSPHC3

np.set_printoptions(precision=3, suppress=True)

class TSPReMP:
    # ------- public API ----------------------------------------------------
    # run()  ->  (tour, history)  where
    #            tour    : [depot, … , depot]  (0‑based indices of s)
    #            history : subtotal cost after each outer iteration
    # ----------------------------------------------------------------------
    def __init__(self, s, rho=0.6, big_penalty=None,
                 remp_iter=100, outer_iter=100, tol=1e-1, verbose=False):
        self.D = s.astype(float)
        self.N_tot = self.D.shape[0]        # includes depot
        self.depot = self.N_tot - 1         # last row/col is depot
        self.rho = rho
        self.remp_iter = remp_iter
        self.outer_iter = outer_iter
        self.tol = tol
        self.verbose = verbose

        if big_penalty is None:
            big_penalty = self.D.max() * 1e4 + 1
        self.big_penalty = big_penalty

        self.cost_hist = []

        # cost matrix used inside message passing
        self.P = self.D.copy()
        np.fill_diagonal(self.P, self.big_penalty)

    # ============== core ===================================================

    def run(self):
        P = self.P.copy()
        for out in range(self.outer_iter):
            X = self._remp_assignment(P)        # 1‑to‑1 assignment
            cycles = self._find_cycles(X)

            if self.verbose:
                print(f"[Iter {out+1:02}]  cycles = {len(cycles)}")

            # Hamiltonian cycle found
            if len(cycles) == 1 and len(cycles[0]) == self.N_tot:
                tour = self._cycle_to_tour(cycles[0])
                cost = self._tour_cost(tour)
                self.cost_hist.append(cost)
                    
                return tour, self.cost_hist

            # break each subtour with a penalty on one of its edges
            for cyc in cycles:
                if len(cyc) == self.N_tot:
                    continue
                i = cyc[0]
                j = np.where(X[i])[0][0]
                P[i, j] += self.big_penalty

            self.cost_hist.append(None)

        raise RuntimeError("Max outer iterations reached without full tour")

    # ============== helpers ===============================================

    def _remp_assignment(self, P):
        """ReMP inner loop + Hungarian to enforce 1-to-1, with min-sum style message passing."""
        N = P.shape[0]
        mu = np.zeros((N, N))
        mu_t = np.zeros((N, N))

        for _ in range(self.remp_iter):
            # row update: i → j
            R = P + mu_t.T
            row_min = np.min(R + np.eye(N)*self.big_penalty, axis=1, keepdims=True)
            row_diff = R - row_min  # how worse each option is compared to best
            mu_new = self.rho * row_diff

            # column update: j → i
            C = P + mu_new
            col_min = np.min(C + np.eye(N)*self.big_penalty, axis=0, keepdims=True)
            col_diff = C - col_min
            mu_t_new = self.rho * col_diff

            # damping
            mu = self.rho * mu + (1 - self.rho) * mu_new
            mu_t = self.rho * mu_t + (1 - self.rho) * mu_t_new

        # score: negative because we want to minimize final cost
        tau = mu + mu_t.T + P
        r, c = linear_sum_assignment(tau)  # maximize tau = minimize cost
        X = np.zeros_like(tau, dtype=int)
        X[r, c] = 1
        return X


    @staticmethod
    def _find_cycles(X):
        N = X.shape[0]
        succ = {i: np.where(X[i])[0][0] for i in range(N)}
        visited, cycles = set(), []
        for start in range(N):
            if start in visited:
                continue
            cyc = [start]
            nxt = succ[start]
            while nxt not in cyc:
                cyc.append(nxt)
                nxt = succ[nxt]
            cyc = cyc[cyc.index(nxt):]
            cycles.append(cyc)
            visited.update(cyc)
        return cycles

    def _cycle_to_tour(self, cyc):
        """Rotate cycle so it starts/ends at the depot."""
        k = cyc.index(self.depot)
        tour = cyc[k:] + cyc[:k] + [self.depot]
        return tour

    def _tour_cost(self, tour):
        D = self.D
        return sum(D[tour[i], tour[i + 1]] for i in range(len(tour) - 1))


# ---------------- minimal demo ----------------
if __name__ == "__main__":

    dist = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]
    ])
    dist = np.random.rand(15,15)
    
    
    solver = TSPReMP(dist, verbose=True)
    best_tour, history = solver.run()
    print("Best tour (1-based):", [x+1 for x in best_tour])
    print("Total distance:", f"{history[-1]:.3f}")
    
    solver2 = TSPHC3(dist, verbose=True)
    best_tour2, history2 = solver2.run()
    print("Best tour (1-based):", [x for x in best_tour2])
    print("Total distance:", f"{history2[-1]:.3f}")
    
    solver3 = TSPHC3(dist, c_old=1, verbose=True)
    best_tour3, history3 = solver3.run()
    print("Best tour (1-based):", [x for x in best_tour3])
    print("Total distance:", f"{history3[-1]:.3f}")
    
