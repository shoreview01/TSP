import numpy as np

class TSPHC3:
    def __init__(self, s, damp=0.5, t_max=1000, t_conv=5, c_old=False, verbose=False):
        self.s_original = s
        self.damp = damp
        self.t_max = t_max
        self.t_conv = t_conv
        self.verbose = verbose
        similarity = np.max(s) - s  # Convert to similarity
        similarity = np.exp(similarity) / np.exp(np.max(similarity))
        self.N = self.s_original.shape[0] - 1  # Exclude depot
        self.s = similarity.copy()  # shape (N, N)
        self.penalty = 0.5

        # 노드를 Hamming weight별로 그룹화
        self.hypercube = [[] for _ in range(self.N+1)]
        for i in range(2**self.N):
            w = bin(i).count('1')
            self.hypercube[w].append(i)
        self.hypercube = [np.array(group) for group in self.hypercube]

        # Initialize messages
        N = self.N
        self.phi = np.zeros((N, N))
        self.gamma = np.zeros((N, N))
        self.zeta = np.zeros((N, N))
        self.beta = np.zeros((N - 1, N))
        self.delta = np.zeros((N - 1, N))
        self.lambda_ = np.zeros((N, N))
        self.rho = np.zeros((N, N))  # backward message

        if c_old is False:
            self.c = self.init_c_trellis_feasible()
        else:
            self.c = np.array(c_old).copy()

    def run(self):
        history = []
        iter_conv_check = 0
        iter = 1
        N = self.N
        cost = np.sum(self.s_original)

        while iter <= self.t_max:
            c_old = self.c.copy()
            c_to_binary = self.c_to_binary(c_old)

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
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + self.s_alt(self.N, m, c_to_binary[t], t)) * (1 - self.damp)
                    else:
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + self.delta[t - 1, m]) * (1 - self.damp)

            # delta update
            for t in range(N - 1):
                for m in range(N):
                    self.delta[t, m] = self.delta[t, m] * self.damp + max(
                        [self.beta[t, m_prime] + self.s_alt(m_prime, m, c_to_binary[t], t) for m_prime in range(N) if m_prime != m]) * (1 - self.damp)

            # lambda update
            for t in range(N):
                for m in range(N):
                    self.lambda_[t, m] = self.lambda_[t, m] * self.damp + (self.gamma[m, t] + self.rho[t, m]) * (1 - self.damp)

            # backward rho update
            for t in reversed(range(N - 1)):
                for m in range(N):
                    self.rho[t, m] = max(
                        self.rho[t + 1, m_next] + self.s_alt(m, m_next, c_to_binary[t], t)
                        for m_next in range(N) if m_next != m
                    )

            # zeta update
            for t in range(N):
                for m in range(N):
                    if t == 0:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + self.s_alt(self.N, m, 0, 0) * (1 - self.damp)
                    elif t == N - 1:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + (self.delta[t - 1, m] + self.s_alt(m, self.N, c_to_binary[t], t)) * (1 - self.damp)
                    else:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + self.delta[t - 1, m] * (1 - self.damp)

            # c estimate
            self.c[0] = np.argmax([self.lambda_[0, m] + self.s_alt(self.N, m, 0, 0) for m in range(N)])
            self.c[N - 1] = np.argmax([self.lambda_[N - 1, m] + self.delta[N - 2, m] + self.s_alt(m, self.N, c_to_binary[N-1], N-1) for m in range(N)])
            for t in range(1, N - 1):
                self.c[t] = np.argmax([self.lambda_[t, m] + self.delta[t - 1, m] for m in range(N)])

            self.c = self.hamiltonianize(self.c)

            # cost update
            new_cost = self.get_cost(end=False)
            if new_cost <= cost:
                cost = new_cost
            else:
                self.c = c_old

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
        N = self.N
        path = np.zeros(N + 2, dtype=int)
        path[0] = N + 1
        path[1:N + 1] = self.c + 1
        path[N + 1] = N + 1
        return path

    def get_cost(self, end=False):
        path = self.get_path()
        if np.sum(path) != ((self.N+1) * (self.N + 2) / 2 + self.N+1) and end:
            return np.inf
        else:
            return np.sum(self.s_original[path[:-1] - 1, path[1:] - 1])

    def init_c_trellis_feasible(self):
        visited = set()
        current = self.N
        c = []
        for _ in range(self.N):
            next_city = min(
                (j for j in range(self.N) if j not in visited),
                key=lambda j: self.s_original[current, j]
            )
            c.append(next_city)
            visited.add(next_city)
            current = next_city
        return np.array(c)

    def c_to_binary(self, c):
        bin_path = np.zeros(self.N, dtype=int)
        visited = 0
        for i in range(self.N):
            visited |= (1 << c[i])
            bin_path[i] = visited
        return bin_path

    def s_alt(self, m_prime, m, c_to_binary_t, t):
        if m_prime == self.N:
            return self.s[self.N, m]
        elif t == self.N - 1:
            return self.s[m_prime, m]
        else:
            if c_to_binary_t not in self.hypercube[t + 1]:
                return self.s[m_prime, m] * self.penalty_factor(t)
            c_to_binary_next = c_to_binary_t | (1 << m)
            if c_to_binary_next not in self.hypercube[t + 2]:
                return self.s[m_prime, m] * self.penalty_factor(t)
            else:
                return self.s[m_prime, m]

    def hamiltonianize(self, c):
        visited = set()
        c_new = []
        for t in range(self.N):
            current = c[t]
            if current not in visited:
                c_new.append(current)
                visited.add(current)
            else:
                candidates = [m for m in range(self.N) if m not in visited]
                if not candidates:
                    break
                prev = c_new[-1]
                best = max(candidates, key=lambda m: self.s[prev, m])
                c_new.append(best)
                visited.add(best)
        if len(c_new) < self.N:
            rest = [m for m in range(self.N) if m not in visited]
            c_new.extend(rest)
        return np.array(c_new)

    def penalty_factor(self, t):
        return self.penalty * np.exp(-t / 10)
