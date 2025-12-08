import numpy as np
from scipy.optimize import linear_sum_assignment

class TSPHC2:
    def __init__(self, s, damp=0.5, t_max=1000, t_conv=5, c_old=False, verbose=False):
        self.s_original = s
        self.damp = damp
        self.t_max = t_max
        self.t_conv = t_conv
        self.verbose = verbose
        similarity = np.max(s) - s  # Convert to similarity
        self.N = self.s_original.shape[0] - 1  # Exclude depot
        self.s = similarity.copy()  # shape (N, N)
        self.penalty = 0#np.min(s) 

                
        # 노드를 Hamming weight별로 그룹화
        self.hypercube = [[] for _ in range(self.N+1)]
        for i in range(2**self.N):
            w = bin(i).count('1')
            self.hypercube[w].append(i)
        self.hypercube = [np.array(group) for group in self.hypercube]
        
        self.similarity_by_trellis = np.zeros((self.N+1, self.N+1, 2**self.N, 1))  # Trellis info for each node
        for m_prime in range(1, self.N+2):
            for m in range(1, self.N+2):
                for h in range(2**self.N):
                    t = bin(h).count('1')
                    current_binary = h
                    next_binary = current_binary + 2**(m-1) if m != m_prime else current_binary
                    if t == self.N:
                        if m_prime != self.N+1:
                            self.similarity_by_trellis[m_prime-1, m-1, h, 0] = similarity[m_prime-1, m-1]
                        continue
                    elif current_binary in self.hypercube[t] and next_binary in self.hypercube[t+1]:
                        self.similarity_by_trellis[m_prime-1, m-1, h, 0] = similarity[m_prime-1, m-1]
                    else:
                        self.similarity_by_trellis[m_prime-1, m-1, h, 0] = self.penalty

        # Initialize messages
        N = self.N
        self.phi = np.zeros((N, N))
        self.gamma = np.zeros((N, N))
        self.zeta = np.zeros((N, N))
        self.beta = np.zeros((N - 1, N))
        self.delta = np.zeros((N - 1, N))
        self.lambda_ = np.zeros((N, N))
        if c_old==False:
            self.c = self.init_c_trellis_feasible()
        else:
            self.c = np.array(c_old).copy()


    def run(self):
        history = []
        iter_conv_check = 0
        iter = 1
        N = self.N
        s = self.similarity_by_trellis

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
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + s[N, m, 0, 0]) * (1 - self.damp)
                    else:
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + self.delta[t - 1, m]) * (1 - self.damp)

            # delta update
            for t in range(N - 1):
                for m in range(N):
                    self.delta[t, m] = self.delta[t, m] * self.damp + max(
                        [self.beta[t, m_prime] + s[m_prime, m, c_to_binary[t], 0] for m_prime in range(N) if m_prime != m]) * (1 - self.damp)

            # lambda update
            for t in range(N):
                for m in range(N):
                    self.lambda_[t, m] = self.lambda_[t, m] * self.damp + self.gamma[m, t] * (1 - self.damp)

            # zeta update
            for t in range(N):
                for m in range(N):
                    if t == 0:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + s[N, m, 0, 0] * (1 - self.damp)
                    elif t == N - 1:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + (self.delta[t - 1, m] + s[m, N, c_to_binary[t], 0]) * (1 - self.damp)
                    else:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + self.delta[t - 1, m] * (1 - self.damp)

            # c estimate
            self.c[0] = np.argmax([self.lambda_[0, m] + s[N, m, 0, 0] for m in range(N)])
            self.c[N - 1] = np.argmax([self.lambda_[N - 1, m] + self.delta[N - 2, m] + s[m, N, c_to_binary[N-1], 0] for m in range(N)])
            for t in range(1, N - 1):
                self.c[t] = np.argmax([self.lambda_[t, m] + self.delta[t - 1, m] for m in range(N)])

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

    def get_cost(self, end=False):
        path = self.get_path()
        return np.sum(self.s_original[path[:-1] - 1, path[1:] - 1])
    
    def init_c_trellis_feasible(self):
        visited = set()
        current = self.N  # depot
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
