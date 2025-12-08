import numpy as np

class TSPHC1:
    def __init__(self, s, damp=0.5, t_max=1000, t_conv=5, c_old=False, verbose=False):
        self.s_original = s
        self.damp = damp
        self.t_max = t_max
        self.t_conv = t_conv
        self.verbose = verbose
        similarity = np.max(s) - s  # Convert to similarity
        self.N = self.s_original.shape[0] - 1  # Exclude depot
        self.s = np.stack([similarity] * self.N, axis=0)  # shape (t, N, N)
        self.penalty = -1 

        # Initialize messages
        N = self.N
        self.phi = np.zeros((N, N))
        self.gamma = np.zeros((N, N))
        self.zeta = np.zeros((N, N))
        self.beta = np.zeros((N - 1, N))
        self.delta = np.zeros((N - 1, N))
        self.lambda_ = np.zeros((N, N))
        if c_old==False:
            self.c = np.zeros(N, dtype=int)
        else:
            self.c = np.array(c_old).copy()
        
        # 노드를 Hamming weight별로 그룹화
        self.nodes_by_weight = [[] for _ in range(self.N+1)]
        for i in range(2**self.N):
            w = bin(i).count('1')
            self.nodes_by_weight[w].append(i)
        self.nodes_by_weight = [np.array(group) for group in self.nodes_by_weight]


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
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + s[0, N, m]) * (1 - self.damp)
                    else:
                        self.beta[t, m] = self.beta[t, m] * self.damp + (self.lambda_[t, m] + self.delta[t - 1, m]) * (1 - self.damp)

            # delta update
            for t in range(N - 1):
                for m in range(N):
                    self.delta[t, m] = self.delta[t, m] * self.damp + max(
                        [self.beta[t, m_prime] + s[t, m_prime, m] for m_prime in range(N) if m_prime != m]) * (1 - self.damp)

            # lambda update
            for t in range(N):
                for m in range(N):
                    self.lambda_[t, m] = self.lambda_[t, m] * self.damp + self.gamma[m, t] * (1 - self.damp)

            # zeta update
            for t in range(N):
                for m in range(N):
                    if t == 0:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + s[t, N, m] * (1 - self.damp)
                    elif t == N - 1:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + (self.delta[t - 1, m] + s[t, m, N]) * (1 - self.damp)
                    else:
                        self.zeta[t, m] = self.zeta[t, m] * self.damp + self.delta[t - 1, m] * (1 - self.damp)

            # c estimate
            self.c[0] = np.argmax([self.lambda_[0, m] + s[0, N, m] for m in range(N)])
            self.c[N - 1] = np.argmax([self.lambda_[N - 1, m] + self.delta[N - 2, m] + s[N-1, m, N] for m in range(N)])
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
            
            bin_path = np.zeros(N, dtype=int)
            temp = 0
            for i in range(N):
                if i > 0 and self.c[i] == self.c[i - 1]:
                    temp = temp
                else:
                    temp += 1 << self.c[i]
                bin_path[i] = temp
            
            # Update similarity matrix based on current path
            s = self.s_update_by_trellis(s, self.c, self.N, self.nodes_by_weight)
            
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
        if np.sum(path) != ((self.N+1) * (self.N + 2) / 2 + self.N+1) and end==True:
            return np.inf  # Return infinity if path does not match expected value
        else:
            return np.sum(self.s_original[path[:-1] - 1, path[1:] - 1])
    
    def s_update_by_trellis(self, s, c, N, nodes_by_weight):

        # Update the similarity matrix based on the current path c
        availablity_front = np.ones(N)
        availablity_back = np.ones(N)
        bin_path = np.zeros(N, dtype=int)
        temp = 0
        for i in range(N):
            if i > 0 and c[i] == c[i - 1]:
                temp = temp
            else:
                temp += 1 << c[i]
            bin_path[i] = temp
            
        for t in range(N-1):
            state_from = bin_path[t]
            state_to = bin_path[t + 1]
            
            level_nodes = nodes_by_weight[t]
            next_level_nodes = nodes_by_weight[t + 1]
            
            if state_from not in level_nodes or state_to not in next_level_nodes:
                availablity_front[t+1] = self.penalty
                break
        
        for t in range(N-1):
            state_from = bin_path[N-2 - t]
            state_to = bin_path[N-1 - t]
            
            level_nodes = nodes_by_weight[N-2 - t]
            next_level_nodes = nodes_by_weight[N-1 - t]
            
            if state_from not in level_nodes or state_to not in next_level_nodes:
                availablity_back[N-2 - t] = self.penalty
                break
            
        # Update the similarity matrix based on the availability
        for t in range(N-2):
            if availablity_front[t + 1] == self.penalty:
                s[t + 1, c[t + 1], c[t + 2]] = availablity_front[t + 1]
            '''if availablity_back[N - 2 - t] == self.penalty:
                s[N - 2 - t, c[N - 2 - t], c[N - 1 - t]] = availablity_back[N - 2 - t]'''
            
        return s