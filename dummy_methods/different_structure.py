import numpy as np
import heapq
import time
from viterbi import TSPBitmask

class TSP_MP_HardBP:
    def __init__(self,
                 distance_matrix,
                 mp_max_iters=200,
                 mp_tconv=5,
                 damp=0.5,
                 gate_with_beam=True,
                 gate_penalty=None,
                 safe_topM=0,
                 use_2_opt=False,
                 rng=None):
        self.s_original = np.asarray(distance_matrix, dtype=float)
        assert self.s_original.ndim == 2 and self.s_original.shape[0] == self.s_original.shape[1]
        self.N = self.s_original.shape[0] - 1
        self.depot = self.N

        D = self.s_original
        s_max = np.max(D)
        self.s = s_max - D  # similarity
        self.K = self.N  # unused in this version

        mu_s, sigma_s = np.mean(D), np.std(D) + 1e-8
        self.s_norm = (D - mu_s) / sigma_s

        self.mp_max_iters = int(mp_max_iters)
        self.mp_tconv = int(mp_tconv)
        self.theta = float(damp)

        self.gate_with_beam = bool(gate_with_beam)
        s_span = float(np.max(self.s) - np.min(self.s) + 1e-8)
        self.gate_penalty = float(gate_penalty if gate_penalty is not None else 10.0 * max(1, self.N) * s_span)
        self.safe_topM = int(safe_topM)

        self.allowed = None
        self.use_2_opt = use_2_opt
        self.rng = np.random.default_rng(rng)

        N = self.N
        self.phi_tilde  = np.zeros((N, N))
        self.gamma_tilde = np.zeros((N, N))
        self.lambda_tilde = np.zeros((N, N))
        self.zeta_tilde = np.zeros((N, N))
        self.delta_tilde = np.zeros((max(0, N-1), N))
        self.beta_tilde  = np.zeros((max(0, N-1), N))

    def _apply_2_opt(self, tour):
        best = tour[:]
        if len(best) <= 4:
            return best
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    a, b = best[i-1], best[i]
                    c, d = best[j], best[j+1]
                    cur = self.s_original[a, b] + self.s_original[c, d]
                    new = self.s_original[a, c] + self.s_original[b, d]
                    if new < cur:
                        best[i:j+1] = reversed(best[i:j+1])
                        improved = True
        return best

    def _cost(self, path):
        return sum(self.s_original[path[i], path[i+1]] for i in range(len(path)-1))

    def _build_beam_gate(self):
        N = self.N
        allowed = [np.zeros((N + 1, N), dtype=bool) for _ in range(N)]
        beam = [(0.0, (self.depot,))]

        for t in range(N):
            cand = []
            for cost, path in beam:
                u = path[-1]
                used = set(path)
                for v in range(N):
                    if v in used:
                        continue
                    allowed[t][u, v] = True
                    new_cost = cost + self.s_norm[u, v]
                    new_path = path + (v,)
                    heapq.heappush(cand, (new_cost, new_path))
            cand.sort()
            beam = cand[:self.K]

        if self.safe_topM > 0:
            for t in range(N):
                for u in range(N + 1):
                    order = np.argsort(self.s_norm[u, :N])[:self.safe_topM]
                    allowed[t][u, order] = True

        return allowed

    def _s_t(self, u, v, t):
        base = self.s[u, v]
        if not self.gate_with_beam:
            return base
        if u == self.depot or t < 0 or t >= self.N:
            return base
        if not self.allowed[t][u, v]:
            return base - self.gate_penalty
        return base

    def _run_message_passing(self):
        N = self.N
        if N <= 0:
            return [self.depot, self.depot], 0.0

        if self.gate_with_beam and self.allowed is None:
            self.allowed = self._build_beam_gate()

        phi = self.phi_tilde
        gamma = self.gamma_tilde
        lam = self.lambda_tilde
        zeta = self.zeta_tilde
        delta = self.delta_tilde
        beta = self.beta_tilde

        theta = self.theta
        tconv_left = self.mp_tconv
        c_prev = None

        for _ in range(self.mp_max_iters):
            # zeta update
            zeta[0, :] = theta * zeta[0, :] + (1 - theta) * self.s[self.depot, :N]
            if N >= 3:
                zeta[1:N-1, :] = theta * zeta[1:N-1, :] + (1 - theta) * delta[0:N-2, :]
            if N >= 2:
                zeta[N-1, :] = theta * zeta[N-1, :] + (1 - theta) * (delta[N-2, :] + self.s[:N, self.depot])

            # phi update
            for t in range(N):
                row = gamma[:, t] + zeta[t, :]
                idx_max = int(np.argmax(row))
                candidates = row[row != row[idx_max]]
                second = np.max(candidates) if candidates.size > 0 else row[idx_max]
                for i in range(N):
                    best_except_i = second if i == idx_max else np.max(row)
                    phi[i, t] = theta * phi[i, t] + (1 - theta) * (-best_except_i + zeta[t, i])

            # gamma update
            for i in range(N):
                row = phi[i, :]
                idx_max = int(np.argmax(row))
                candidates = row[row != row[idx_max]]
                second = np.max(candidates) if candidates.size > 0 else row[idx_max]
                for t in range(N):
                    best_except_t = second if t == idx_max else np.max(row)
                    gamma[i, t] = theta * gamma[i, t] + (1 - theta) * (-best_except_t)

            # lambda update
            lam[:, :] = theta * lam[:, :] + (1 - theta) * gamma.T

            # beta, delta update
            if N >= 2:
                beta[0, :] = theta * beta[0, :] + (1 - theta) * (lam[0, :] + self.s[self.depot, :N])
                for t in range(1, N-1):
                    beta[t, :] = theta * beta[t, :] + (1 - theta) * (lam[t, :] + delta[t-1, :])

                for t in range(N-1):
                    Sm = np.full((N, N), -np.inf)
                    for n in range(N):
                        for m in range(N):
                            if n != m:
                                Sm[n, m] = self._s_t(n, m, t)
                    delta[t, :] = theta * delta[t, :] + (1 - theta) * np.max(Sm + beta[t, :].reshape(-1, 1), axis=0)

            # estimate c_hat from current messages
            c_hat = []
            for t in range(N):
                if t == 0:
                    scores = lam[t, :] + self.s[self.depot, :N]
                elif t == N - 1:
                    scores = delta[N-2, :] + lam[t, :] + self.s[:N, self.depot]
                else:
                    scores = lam[t, :] + delta[t-1, :]
                c_hat.append(int(np.argmax(scores)))

            if c_prev is not None and c_hat == c_prev:
                tconv_left -= 1
            else:
                tconv_left = self.mp_tconv
            c_prev = c_hat

            if tconv_left <= 0:
                break

        path = [self.depot] + c_hat + [self.depot]
        cost = self._cost(path)
        return path, cost

    def run(self):
        tour, cost = self._run_message_passing()
        if self.use_2_opt and len(tour) >= 4:
            tour = self._apply_2_opt(tour)
            cost = self._cost(tour)
        return tour, cost

if __name__ == "__main__":
    import time

    N = 20
    np.random.seed(4402)
    coords = np.random.rand(N + 1, 2)
    s_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords, axis=2)

    solver = TSP_MP_HardBP(
        s_matrix,
        mp_max_iters=200,
        mp_tconv=5,
        damp=0.5,
        gate_with_beam=True,
        safe_topM=1,
        use_2_opt=False,
        rng=42
    )

    start = time.time()
    path, cost = solver.run()
    elapsed = time.time() - start
    
    print("\n--- Results ---")
    print("Tour:", path)
    print("Cost:", cost)
    print("Time: {:.3f}s".format(elapsed))
    
    
    if N <= 16:
        start2 = time.time()
        solver2 = TSPBitmask(s_matrix)
        path2, cost2 = solver2.run()
        elapsed2 = time.time() - start2

        print("Tour:", path2)
        print("Cost:", cost2)
        print("Time: {:.3f}s".format(elapsed2))
