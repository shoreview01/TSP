import numpy as np

class TSPMaxSum:
    def __init__(self, s, damp=0.5, t_max=1000, t_conv=5, verbose=False):
        self.s_original = s
        self.damp = damp
        self.t_max = t_max
        self.t_conv = t_conv
        self.verbose = verbose
        #self.s = np.max(s) - s  # Convert to similarity
        self.s = np.exp(-s)/np.exp(-np.min(s))  # Convert to similarity
        self.N = self.s.shape[0] - 1  # Exclude depot
        
        # Initialize messages
        N = self.N
        self.phi = np.zeros((N, N))
        self.gamma = np.zeros((N, N))
        self.zeta = np.zeros((N, N))
        self.beta = np.zeros((N - 1, N))
        self.delta = np.zeros((N - 1, N))
        self.lambda_ = np.zeros((N, N))
        self.c = np.zeros(N, dtype=int)

    def run(self):
        history = []
        iter_conv_check = 0
        iter = 1
        N = self.N
        s = self.s

        while iter <= self.t_max:
            c_old = self.c.copy()

            # phi update
            for t in range(N):
                for i in range(N):
                    max_val = max([self.gamma[i_prime, t] + self.zeta[t, i_prime]
                                   for i_prime in range(N) if i_prime != i])
                    self.phi[i, t] = self.phi[i, t] * self.damp + (-max_val + self.zeta[t, i]) * (1 - self.damp)

            # gamma update
            for i in range(N):
                for t in range(N):
                    self.gamma[i, t] = self.gamma[i, t] * self.damp + (-max([self.phi[i, t_prime]
                                                                             for t_prime in range(N) if t_prime != t])) * (1 - self.damp)

            # beta update
            for t in range(N - 1):
                for m in range(N):
                    if t == 0:
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + s[N, m]) * (1 - self.damp)
                    else:
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + self.delta[t - 1, m]) * (1 - self.damp)

            # delta update
            for t in range(N - 1):
                for m in range(N):
                    self.delta[t, m] = self.delta[t, m] * self.damp + max(
                        [self.beta[t, m_prime] + s[m_prime, m] for m_prime in range(N) if m_prime != m]) * (1 - self.damp)

            # lambda update
            for t in range(N):
                for m in range(N):
                    self.lambda_[t, m] = self.lambda_[t, m] * self.damp + self.gamma[m, t] * (1 - self.damp)

            # zeta update
            for t in range(N):
                for m in range(N):
                    if t == 0:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + s[N, m] * (1 - self.damp)
                    elif t == N - 1:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + (self.delta[t - 1, m] + s[m, N]) * (1 - self.damp)
                    else:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + self.delta[t - 1, m] * (1 - self.damp)

            # c estimate
            self.c[0] = np.argmax([self.lambda_[0, m] + s[N, m] for m in range(N)])
            self.c[N - 1] = np.argmax([self.lambda_[N - 1, m] + self.delta[N - 2, m] + s[m, N] for m in range(N)])
            for t in range(1, N - 1):
                self.c[t] = np.argmax([self.lambda_[t, m] + self.delta[t - 1, m] for m in range(N)])
            
            self.c = self.hamiltonianize(self.c)
            
            cost = self.get_cost(end=False)
            
            if self.verbose:
                path_str = ' → '.join(str(x) for x in self.get_path())
                print(f"Iter {iter}: path = {path_str}, cost = {cost:.4f}")
                
            history.append(cost)
            
            # convergence check
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
