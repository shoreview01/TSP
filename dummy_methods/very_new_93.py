import numpy as np
from dummy_methods.original import TSPMaxSum as OriginalTSPMaxSum

class TSPMaxSum:
    def __init__(self, s, damp=0.5, t_max=1000, t_conv=5, verbose=False):
        self.s_original = s
        self.damp = damp
        self.t_max = t_max
        self.t_conv = t_conv
        self.verbose = verbose
        self.s = np.max(s) - s  # Convert to similarity
        #self.s = np.exp(-s)/np.exp(-np.min(s))  # Convert to similarity
        self.N = self.s.shape[0] - 1  # Exclude depot
        
        # Initialize messages
        N = self.N
        self.eta = np.zeros((N, N))
        self.phi = np.zeros((N, N))
        self.gamma = np.zeros((N, N))
        self.zeta = np.zeros((N, N))
        self.delta = np.zeros((N + 1, N))
        self.rho = np.zeros((N + 1, N))
        self.lambda_ = np.zeros((N, N))
        self.c = np.zeros(N, dtype=int)

    def run(self):
        # --- 최소 수정: eta 버퍼가 없다면 __init__에 추가하세요 ---
        # self.eta = np.zeros((self.N, self.N))

        history = []
        iter_conv_check = 0
        N = self.N
        s = self.s

        iter = 1
        while iter <= self.t_max:
            c_old = self.c.copy()

            # =========================
            # 1) omega, eta, gamma, phi, zeta
            # =========================
            # omega(i,t) = phi(i,t) + lambda(t,i)  (축 정합 주의)
            # eta(i,t)   = - max_{t' != t} omega(i,t')
            for i in range(N):
                for t in range(N):
                    omega_itprime = [self.phi[i, tp] + self.lambda_[tp, i] for tp in range(N) if tp != t]
                    self.eta[i, t] = -max(omega_itprime) if omega_itprime else 0.0

            # gamma = eta + lambda  (동일 좌표)
            for i in range(N):
                for t in range(N):
                    self.gamma[i, t] = self.eta[i, t] + self.lambda_[t, i]

            # phi(i,t) = - max_{i' != i} gamma(i',t)
            for t in range(N):
                for i in range(N):
                    self.phi[i, t] = -max(self.gamma[ip, t] for ip in range(N) if ip != i) if N > 1 else 0.0

            # zeta = eta + phi
            self.zeta = self.eta + self.phi

            # =========================
            # 2) delta (Forward)  + F 캐시
            # =========================
            F = np.zeros((N, N))  # F[t, m]
            # t = 0: delta[0,m] = s(N,m)   (너 논리 유지: ζ는 t>=1에서 t-1로 들어감)
            for m in range(N):
                self.delta[0, m] = self.delta[0, m] * self.damp + s[N, m] * (1 - self.damp)

            # t >= 1: F_t(m) = max_{n!=m} [ s(n,m) + delta[t-1,n] ]
            #          delta[t,m] = F_t(m) + zeta[m, t-1]   (너의 t-오프셋 해석 유지)
            for t in range(1, N + 1):
                if t == N:  # delta[N,*]는 계산하지 않아도 되지만 형상 유지 차원에서 그대로 둔다
                    break
                for m in range(N):
                    max_val = max(s[n, m] + self.delta[t-1, n] for n in range(N) if n != m) if N > 1 else -np.inf
                    F[t-1, m] = max_val
                    self.delta[t, m] = self.delta[t, m] * self.damp + (max_val + self.zeta[m, t-1]) * (1 - self.damp)

            # 정규화(선택)
            self.delta = self.delta - np.max(self.delta, axis=1, keepdims=True)

            # =========================
            # 3) rho (Backward)  — 반드시 rho[t+1] 참조
            # =========================
            # 경계: rho[N, m] = s(m, N)
            for m in range(N):
                self.rho[N, m] = self.rho[N, m] * self.damp + s[m, N] * (1 - self.damp)

            # t = N-1 .. 0:
            # rho[t, m] = max_{n!=m} [ s(m,n) + rho[t+1, n] + zeta[n, t] ]
            for t in range(N - 1, -1, -1):
                for m in range(N):
                    max_val = max(s[m, n] + self.rho[t+1, n] + self.zeta[n, t] for n in range(N) if n != m) if N > 1 else -np.inf
                    self.rho[t, m] = self.rho[t, m] * self.damp + max_val * (1 - self.damp)

            # 정규화(선택)
            self.rho = self.rho - np.max(self.rho, axis=1, keepdims=True)

            # =========================
            # 4) lambda 업데이트
            # =========================
            # A_t(i) = F_t(i) + rho_t(i)   (시점 맞춤!)
            # B_t(i) = max_{i' != i} [ A_t(i') + zeta[i', t] ]
            for t in range(N):
                A_t = np.array([F[t, i] + self.rho[t, i] for i in range(N)])
                for i in range(N):
                    B_t_i = max(A_t[ip] + self.zeta[ip, t] for ip in range(N) if ip != i) if N > 1 else 0.0
                    new_val = A_t[i] - B_t_i
                    self.lambda_[t, i] = self.lambda_[t, i] * self.damp + new_val * (1 - self.damp)

            # =========================
            # 5) c 추정 (네 방식 유지)
            # =========================
            for t in range(N):
                self.c[t] = np.argmax([self.rho[t, m] + self.delta[t, m] for m in range(N)])

            # =========================
            # 6) 로그/수렴
            # =========================
            hamiltonized_c = self.hamiltonianize(self.c)
            self.c = hamiltonized_c
            cost = self.get_cost(end=False)
            history.append(cost)
            if self.verbose and (iter <= 5 or iter % 10 == 0):
                path_str = ' → '.join(str(x) for x in self.get_path())
                print(f"Iter {iter}: path = {path_str}, cost = {cost:.6f}")

            if np.array_equal(self.c, c_old):
                iter_conv_check += 1
                if iter_conv_check >= self.t_conv:
                    if self.verbose:
                        print("Convergence achieved.")
                    break
            else:
                iter_conv_check = 0

            iter += 1

        self.iterations = iter
        return self.get_path(), history


    def get_path(self):
        # Convert to 1-based indexing including depot
        N = self.N
        path = np.zeros(N + 2, dtype=int)
        path[0] = N + 1
        path[1:N + 1] = self.c + 1
        path[N + 1] = N + 1
        return path

    def get_cost(self, end):
        path = self.get_path()
        if np.sum(path) != ((self.N+1) * (self.N + 2) / 2 + self.N+1) and end==True:
            return np.inf  # Return infinity if path does not match expected value
        else:
            return np.sum(self.s_original[path[:-1] - 1, path[1:] - 1])

    def hamiltonianize(self, c):
        visited = set()
        c_new = []
        for t in range(self.N):
            current = c[t]
            if current not in visited:
                c_new.append(current)
                visited.add(current)
            else:
                # 중복일 경우 새 노드 찾기
                candidates = [m for m in range(self.N) if m not in visited]
                if not candidates:
                    break
                prev = c_new[-1]
                best = max(candidates, key=lambda m: self.s[prev, m])  # 유사도 최대인 노드 선택
                c_new.append(best)
                visited.add(best)
        # 경로 길이 부족 시 나머지 node 채움
        if len(c_new) < self.N:
            rest = [m for m in range(self.N) if m not in visited]
            c_new.extend(rest)
        return np.array(c_new)


if __name__ == "__main__":
    import time

    N = 5
    np.random.seed(3061)
    coords = np.random.rand(N, 2)
    s_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords, axis=2)
    d = np.array([
        [0.8, 10.1, 12.5, 0.1, 0.6],
        [0.9, 0.2, 0.9, 0.4, 0.1],
        [0.1, 0.5, 0.9, 0.9, 0.8],
        [0.9, 0.9, 0.5, 0.8, 0.9],
        [0.6, 0.009, 1.8, 0.9, 0.6]
    ])
    solver = TSPMaxSum(s_matrix, verbose=True)
    start = time.time()
    path, cost = solver.run()
    elapsed = time.time() - start
    
    solver_original = OriginalTSPMaxSum(s_matrix, verbose=True)
    start = time.time()
    path_original, cost_original = solver_original.run()
    elapsed_original = time.time() - start
    
    print("\n--- Results ---")
    print("Tour:", path)
    print("Cost:", cost[-1])
    print("Time: {:.3f}s".format(elapsed))
    
