import numpy as np
from collections import defaultdict

class TSPSolverSOVA:
    def __init__(self, dist_matrix, bp_iterations=20, damping=0.7, verbose=True):
        self.dist_matrix = np.array(dist_matrix)
        self.num_nodes = self.dist_matrix.shape[0]
        self.N = self.num_nodes - 1
        self.depot = self.N
        self.bp_iterations = bp_iterations
        self.damping = damping
        self.verbose = verbose
        self.INF = 1e12
        self.FULL_MASK = (1 << self.N) - 1
        
        max_dist = np.max(self.dist_matrix)
        self.S = max_dist - self.dist_matrix
        np.fill_diagonal(self.S, -self.INF) 
        
        # Messages initialization
        self.tilde_rho = np.zeros((self.N, self.N))
        self.tilde_eta = np.zeros((self.N, self.N))
        self.tilde_phi = np.zeros((self.N, self.N))

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _calc_lambda_sum_bias(self):
        """PDF Eq (1.2) Sum(lambda) 계산 (Beta 역할)"""
        N = self.N
        sum_rho_t = np.sum(self.tilde_rho, axis=1, keepdims=True)
        lambda_sum_bias = -self.tilde_rho + ((N - 1) / N) * sum_rho_t
        return lambda_sum_bias

    def _run_trellis(self):
        N, S, depot = self.N, self.S, self.depot
        bias = self._calc_lambda_sum_bias()
        
        # --- [1] Forward (Alpha) ---
        alpha = [defaultdict(lambda: -self.INF) for _ in range(N + 1)]
        alpha[0][(0, depot)] = 0.0
        
        # 최적화: 루프 내 변수 접근 최소화
        for t in range(N):
            current_bias = bias[t]
            next_alpha = alpha[t+1]
            # items() 복사 오버헤드 방지
            for state, prev_score in alpha[t].items():
                if prev_score < -self.INF / 2: continue
                    
                mask, prev_node = state
                
                # 방문 가능한 노드만 순회 (비트 연산 최적화 가능하지만 가독성 유지)
                # N이 작으므로 range(N) 루프는 빠름
                for i in range(N):
                    if not (mask & (1 << i)):
                        new_mask = mask | (1 << i)
                        val = prev_score + S[prev_node, i] + current_bias[i]
                        
                        if val > next_alpha[(new_mask, i)]:
                            next_alpha[(new_mask, i)] = val

        # --- [2] Backward (Beta) ---
        # Beta는 로직 동일, 코드 생략 없이 최적화만 적용
        beta = [defaultdict(lambda: -self.INF) for _ in range(N + 2)]
        final_mask = self.FULL_MASK
        
        for i in range(N):
            beta[N][(final_mask, i)] = S[i, depot]
            
        for t in range(N - 1, -1, -1):
            curr_bias = bias[t]
            curr_beta = beta[t]
            
            for next_state, next_beta_val in beta[t+1].items():
                if next_beta_val < -self.INF / 2: continue
                next_mask, next_node = next_state
                
                # Xi 값 미리 계산
                xi_val = next_beta_val + curr_bias[next_node]
                
                prev_mask = next_mask & ~(1 << next_node)
                
                # 후보군 추출
                if prev_mask == 0:
                    cands = [depot]
                else:
                    cands = [j for j in range(N) if prev_mask & (1 << j)]
                    
                for prev in cands:
                    val = S[prev, next_node] + xi_val
                    state = (prev_mask, prev)
                    if val > curr_beta[state]:
                        curr_beta[state] = val

        # --- [3] Soft Output (Delta) 최적화 (핵심!) ---
        # 기존: O(N^2 * States) -> 최적화: O(States + N^2)
        tilde_delta = np.zeros((N, N))
        
        for t in range(N):
            # 1. 해당 시간 t의 도시별 최대 점수(City Max Scores)를 한 번의 루프로 수집
            city_max_scores = np.full(N, -self.INF)
            
            # alpha와 beta가 존재하는 상태만 빠르게 스캔
            for state, f_score in alpha[t+1].items():
                if f_score < -self.INF / 2: continue
                if state in beta[t+1]:
                    b_score = beta[t+1][state]
                    if b_score > -self.INF / 2:
                        total = f_score + b_score
                        city = state[1]
                        if total > city_max_scores[city]:
                            city_max_scores[city] = total
            
            # 2. '자기 자신 제외 최대값'을 구하기 위한 전처리
            # 전체 최대값과 두 번째 최대값을 찾음
            sorted_indices = np.argsort(city_max_scores)[::-1] # 내림차순 정렬 인덱스
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            
            global_max = city_max_scores[best_idx]
            global_second = city_max_scores[second_best_idx]

            # 3. Delta 계산 (벡터화)
            # lam_i_for_i = -1/N * rho
            # lam_i_for_j = (N-1)/N * rho
            rho_t = self.tilde_rho[t]
            lam_i_for_i = -(1.0/N) * rho_t
            lam_i_for_j = ((N-1.0)/N) * rho_t
            
            # max_in: 내가 선택된 경우의 최대 점수
            max_in = city_max_scores - lam_i_for_i
            
            # max_out: 내가 선택되지 않았을 때(다른 도시 중) 최대 점수
            # 내가 1등이면 -> 2등 점수 사용, 내가 1등 아니면 -> 1등 점수 사용
            max_out_raw = np.full(N, global_max)
            max_out_raw[best_idx] = global_second # 1등 자리는 2등 점수로 교체
            
            max_out = max_out_raw - lam_i_for_j
            
            # 최종 차이 계산 (Inf 처리 포함)
            diff = np.where(max_in < -self.INF/2, -self.INF,
                            np.where(max_out < -self.INF/2, self.INF, max_in - max_out))
            
            tilde_delta[t] = diff
                
        return alpha, tilde_delta

    def _run_bp(self, tilde_delta):
        """
        Numpy Broadcasting을 이용한 BP 완전 최적화
        O(N^3) -> O(N^2)
        """
        N = self.N
        
        # 1. Omega 계산
        t_omega = self.tilde_phi + tilde_delta
        
        # 2. Eta 계산 (Row 방향 Max Excluding Self)
        # 각 열(Column)에서 특정 행(t)을 제외한 최대값 구하기
        # - 전체 최대값(max1)과 두번째 최대값(max2)을 구해서 처리
        col_max_idx = np.argmax(t_omega, axis=0) # 각 열의 최대값 위치(행 인덱스)
        col_max_val = np.max(t_omega, axis=0)    # 각 열의 최대값
        
        # 두 번째 최대값을 구하기 위해 최대값 위치를 -INF로 잠시 변경
        temp_omega = t_omega.copy()
        for c in range(N):
            temp_omega[col_max_idx[c], c] = -self.INF
        col_second_max = np.max(temp_omega, axis=0)
        
        # new_eta 구성
        # t가 최대값 위치가 아니면 -> 최대값 사용
        # t가 최대값 위치면 -> 두 번째 최대값 사용
        new_eta = np.zeros((N, N))
        for t in range(N):
            # t행이 해당 열(c)의 최대값 위치인지 체크
            is_max_pos = (col_max_idx == t)
            # True면 second_max, False면 max_val
            new_eta[t] = np.where(is_max_pos, col_second_max, col_max_val)
        new_eta = -new_eta # 부호 반전

        # 3. Gamma 계산
        t_gamma = new_eta + tilde_delta
        
        # 4. Phi 계산 (Col 방향 Max Excluding Self)
        # 이번엔 각 행(Row)에서 특정 열(i)을 제외한 최대값
        row_max_idx = np.argmax(t_gamma, axis=1)
        row_max_val = np.max(t_gamma, axis=1)
        
        temp_gamma = t_gamma.copy()
        for r in range(N):
            temp_gamma[r, row_max_idx[r]] = -self.INF
        row_second_max = np.max(temp_gamma, axis=1)
        
        new_phi = np.zeros((N, N))
        for i in range(N):
            is_max_pos = (row_max_idx == i)
            # 전치(Transpose) 주의: new_phi[t, i] 이므로 t 루프 대신 열벡터 연산
            # 여기서는 이중 루프 없이 브로드캐스팅을 위해 전치 사용이 헷갈릴 수 있으므로 
            # 단순하게 열 단위 할당
            new_phi[:, i] = np.where(row_max_idx == i, row_second_max, row_max_val)
            
        new_phi = -new_phi # 부호 반전
        
        # Update with Damping
        self.tilde_eta = self.damping * self.tilde_eta + (1 - self.damping) * new_eta
        self.tilde_phi = self.damping * self.tilde_phi + (1 - self.damping) * new_phi
        self.tilde_rho = self.tilde_eta + self.tilde_phi

    def _extract_path(self, alpha):
        path = []
        curr_mask = self.FULL_MASK
        best_score = -self.INF
        best_last = -1

        # 1) 마지막 도시 선택
        for i in range(self.N):
            state = (curr_mask, i)
            if state in alpha[self.N]:
                score = alpha[self.N][state] + self.S[i, self.depot]
                if score > best_score:
                    best_score = score
                    best_last = i

        if best_last == -1:
            return [], self.INF

        # 초기 상태
        path = [self.depot, best_last]
        curr_node = best_last
        curr_mask = self.FULL_MASK ^ (1 << best_last)

        # 2) 역추적
        for t in range(self.N - 1, 0, -1):

            cands = [j for j in range(self.N) if curr_mask & (1 << j)]  # depot 제거

            best_prev = -1
            best_val = -self.INF

            for prev in cands:
                state = (curr_mask, prev)
                if state in alpha[t]:
                    val = alpha[t][state] + self.S[prev, curr_node]   # bias 제거 (이미 α에 포함됨)

                    if val > best_val:
                        best_val = val
                        best_prev = prev

            if best_prev == -1:
                print(f"Traceback failed at t={t}")
                return [], self.INF

            path.append(best_prev)
            curr_node = best_prev
            curr_mask ^= (1 << best_prev)

        path.append(self.depot)
        path.reverse()

        # cost 계산
        cost = sum(self.dist_matrix[path[k], path[k+1]] for k in range(len(path)-1))
        return path, cost

    def solve(self):
        best_global_path = []
        best_global_cost = self.INF
        
        print(f"Solving TSP (N={self.N}) with Alpha Visualization")
        
        for it in range(self.bp_iterations):
            alpha, tilde_delta = self._run_trellis()
            path, cost = self._extract_path(alpha)
            
            if cost < best_global_cost:
                best_global_cost = cost
                best_global_path = path
                print(f"[Iter {it}] Cost: {cost:.2f} (New Best!) | Path: {path}")
            else:
                if self.verbose:
                     print(f"[Iter {it}] Cost: {cost:.2f} | Path: {path}")

            self._run_bp(tilde_delta)
            
        return best_global_path, best_global_cost