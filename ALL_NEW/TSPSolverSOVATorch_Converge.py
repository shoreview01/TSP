import torch
import numpy as np

class TSPSolverSOVATorch_Converge:
    def __init__(self, dist_matrix, bp_iterations=20, damping=0.7, verbose=True, device='cuda'):
        # Device 설정
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 데이터 초기화 및 Tensor 변환
        self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=self.device)
        self.num_nodes = self.dist_matrix.shape[0]
        self.N = self.num_nodes - 1
        self.depot = self.N
        self.bp_iterations = bp_iterations
        self.damping = damping  # BP 메시지 업데이트 시의 관성(Momentum)
        self.verbose = verbose
        self.INF = 1e8
        
        # [변경] accumulated_bias 제거 (Gradient 방식 폐기)
        # 본래 BP 정의대로 매 iter마다 메시지를 새로 계산하여 전달합니다.

        # 비트마스크 전체 크기 (2^N)
        self.num_states = 1 << self.N
        self.FULL_MASK = self.num_states - 1
        
        # S Matrix 계산 (Max - Dist)
        max_dist = torch.max(self.dist_matrix)
        self.S = max_dist - self.dist_matrix
        self.S.fill_diagonal_(-self.INF)
        
        # Messages initialization (N x N)
        self.tilde_rho = torch.zeros((self.N, self.N), device=self.device)
        self.tilde_eta = torch.zeros((self.N, self.N), device=self.device)
        self.tilde_phi = torch.zeros((self.N, self.N), device=self.device)
        
        # Precompute Masks by Population Count
        self.masks_by_popcount = [[] for _ in range(self.N + 1)]
        for mask in range(self.num_states):
            cnt = bin(mask).count('1')
            if cnt <= self.N:
                self.masks_by_popcount[cnt].append(mask)
        
        # GPU Tensor로 변환
        self.masks_by_popcount_t = [
            torch.tensor(m, dtype=torch.long, device=self.device) 
            for m in self.masks_by_popcount
        ]

    def _calc_lambda_original_def(self):
        """
        [Gauge Fixing 제거됨]
        PDF 식 (2.2)의 원래 정의: lambda_{it} = rho_{it}(b_{it})
        
        - 경로가 i를 방문한다면(b=1): rho(1) 메시지를 받음.
        - 따라서 Trellis Score에 rho(1)에 해당하는 값을 '더해줌(+)' (Positive Feedback).
        - 복잡한 Gauge 계수(-(N-1)/N 등)를 제거하고 Belief(tilde_rho)를 그대로 사용.
        """
        # 1. Bias는 Rho (Belief) 그 자체
        # Positive Feedback: 확률이 높은 경로에 가산점을 부여하여 강화
        lambda_val = self.tilde_rho.clone()
        
        # 2. 수치 안정성(Numerical Stability)을 위한 Normalization
        # Gauge Fixing이 아니라, 값이 무한정 커지는 것을 막기 위해 평균만 0으로 맞춤 (BP 표준 기법)
        lambda_val -= torch.mean(lambda_val)
        
        return lambda_val

    def _run_trellis_gpu(self):
        """GPU Optimized Trellis with defined Lambda"""
        N, S, depot = self.N, self.S, self.depot
        
        # [변경] 원래 정의에 따른 Lambda 계산 호출
        bias = self._calc_lambda_original_def() # (N, N)
        
        # --- [1] Forward (Alpha) ---
        alpha = torch.full((self.num_states, N), -self.INF, device=self.device)
        
        # t=0 초기화
        start_bias = bias[0] 
        start_scores = S[depot, :N] + start_bias 
        
        initial_masks = (1 << torch.arange(N, device=self.device))
        alpha[initial_masks, torch.arange(N, device=self.device)] = start_scores

        # DP Loop
        for t in range(1, N):
            current_bias = bias[t] 
            prev_masks = self.masks_by_popcount_t[t]
            
            # (최적화된 Vectorized DP 로직 유지)
            curr_scores = alpha[prev_masks, :] 
            transition_cost = S[:N, :N] + current_bias.view(1, N)
            
            candidates = curr_scores.unsqueeze(2) + transition_cost.unsqueeze(0)
            
            mask_col = prev_masks.view(-1, 1)
            shifts = torch.arange(N, device=self.device).view(1, -1)
            next_bit_check = (mask_col & (1 << shifts)) == 0
            
            valid_mask = next_bit_check.unsqueeze(1).expand(-1, N, -1)
            candidates = torch.where(valid_mask, candidates, torch.tensor(-self.INF, device=self.device))
            
            new_masks_calc = mask_col | (1 << shifts)
            src_vals = candidates.reshape(-1)
            
            m_idx = new_masks_calc.unsqueeze(1).expand(-1, N, -1).reshape(-1)
            n_idx = torch.arange(N, device=self.device).view(1, 1, N).expand(len(prev_masks), N, -1).reshape(-1)
            flat_indices = m_idx * N + n_idx
            
            alpha.view(-1).scatter_reduce_(0, flat_indices, src_vals, reduce='amax', include_self=True)

        # --- [2] Backward (Beta) ---
        beta = torch.full((self.num_states, N), -self.INF, device=self.device)
        beta[self.FULL_MASK, :] = S[:N, depot]
        
        for t in range(N - 1, -1, -1):
            curr_bias = bias[t]
            next_masks = self.masks_by_popcount_t[t+1]
            if len(next_masks) == 0: continue
            
            next_beta_vals = beta[next_masks, :]
            xi_val = next_beta_vals + curr_bias.view(1, N)
            
            candidates = xi_val.unsqueeze(1) + S[:N, :N].unsqueeze(0)
            
            mask_col = next_masks.view(-1, 1)
            shifts = torch.arange(N, device=self.device).view(1, -1)
            has_next_node = (mask_col & (1 << shifts)) != 0
            prev_masks_calc = mask_col ^ (1 << shifts)
            
            if t > 0:
                 prev_has_j = (prev_masks_calc.unsqueeze(2) & (1 << shifts).unsqueeze(0).unsqueeze(0)) != 0
            else:
                 pass
            
            valid_mask = has_next_node.unsqueeze(1).expand(-1, N, -1)
            if t > 0:
                valid_mask = valid_mask & prev_has_j
            
            vals = torch.where(valid_mask, candidates, torch.tensor(-self.INF, device=self.device))
            
            p_mask_idx = prev_masks_calc.unsqueeze(1).expand(-1, N, -1).reshape(-1)
            p_node_idx = torch.arange(N, device=self.device).view(1, N, 1).expand(len(next_masks), -1, N).reshape(-1)
            flat_indices = p_mask_idx * N + p_node_idx
            src_vals = vals.reshape(-1)
            
            beta.view(-1).scatter_reduce_(0, flat_indices, src_vals, reduce='amax', include_self=True)

        # --- [3] Soft Output (Delta) ---
        tilde_delta = torch.zeros((N, N), device=self.device)
        
        for t in range(N):
            curr_masks = self.masks_by_popcount_t[t+1]
            if len(curr_masks) == 0: continue
            
            a = alpha[curr_masks]
            b = beta[curr_masks]
            valid = (a > -self.INF/2) & (b > -self.INF/2)
            scores = torch.where(valid, a + b, torch.tensor(-self.INF, device=self.device))
            
            city_max_scores = torch.max(scores, dim=0)[0]
            
            top2_vals, top2_idxs = torch.topk(city_max_scores, 2)
            global_max = top2_vals[0]
            global_second = top2_vals[1]
            best_idx = top2_idxs[0]
            
            # [변경] Max-In/Max-Out 계산 시에도 Gauge Fix 계수 제거
            # 순수하게 점수 차이(Diff)만 계산
            # 기존 식: lam_i_for_i = -(1.0/N) * rho_t 등 -> 모두 제거
            # Lambda(Bias)는 이미 Trellis 계산에 녹아있으므로, 
            # 여기서는 순수 Trellis Score(city_max_scores)만으로 Delta를 구함
            
            max_in = city_max_scores # Bias 항 제거됨
            
            max_out_raw = torch.full((N,), global_max, device=self.device)
            max_out_raw[best_idx] = global_second
            max_out = max_out_raw # Bias 항 제거됨
            
            diff = max_in - max_out
            
            diff = torch.where(max_in < -self.INF/2, torch.tensor(-self.INF, device=self.device), diff)
            diff = torch.where(max_out < -self.INF/2, torch.tensor(self.INF, device=self.device), diff)
            
            tilde_delta[t] = diff
            
        return alpha, tilde_delta

    def _run_bp_gpu(self, tilde_delta):
        """
        Matrix Operations Fully on GPU
        Constraint Satisfaction을 위한 BP 업데이트 (Doubly Stochastic)
        """
        N = self.N
        
        # 1. Omega
        t_omega = self.tilde_phi + tilde_delta
        
        # 2. Eta (Column Max)
        vals, idxs = torch.topk(t_omega, 2, dim=0)
        max1 = vals[0]
        max2 = vals[1]
        argmax = idxs[0]
        rows = torch.arange(N, device=self.device).view(N, 1).expand(N, N)
        is_max_pos = (rows == argmax)
        new_eta = -torch.where(is_max_pos, max2, max1)
        
        # 3. Gamma
        t_gamma = new_eta + tilde_delta
        
        # 4. Phi (Row Max)
        vals_r, idxs_r = torch.topk(t_gamma, 2, dim=1)
        max1_r = vals_r[:, 0].unsqueeze(1)
        max2_r = vals_r[:, 1].unsqueeze(1)
        argmax_r = idxs_r[:, 0].unsqueeze(1)
        cols = torch.arange(N, device=self.device).view(1, N).expand(N, N)
        is_max_pos_r = (cols == argmax_r)
        new_phi = -torch.where(is_max_pos_r, max2_r, max1_r)
        
        # Update with Damping (Gradient Descent 대신 관성 사용)
        self.tilde_eta = self.damping * self.tilde_eta + (1 - self.damping) * new_eta
        self.tilde_phi = self.damping * self.tilde_phi + (1 - self.damping) * new_phi
        
        # --- Rho Update ---
        raw_rho = self.tilde_eta + self.tilde_phi
        
        # [중요] Gauge Fixing 제거 후 Normalization
        # PDF 식 (4.2)의 Double Centering은 Gauge Fix에서 나온 것이지만,
        # Max-Sum BP에서 값이 무한정 커지는 것을 막고 행/열 제약을 맞추기 위해 
        # 'Sinkhorn' 스타일의 Normalization은 유지하는 것이 수렴에 유리합니다.
        # 하지만 "Gauge Fixing을 걷어내라"는 요청에 따라, 최소한의 수치 안정성(Mean Subtraction)만 남깁니다.
        
        # 1. Row Centering
        raw_rho -= torch.mean(raw_rho, dim=1, keepdim=True)
        # 2. Col Centering
        raw_rho -= torch.mean(raw_rho, dim=0, keepdim=True)
        
        self.tilde_rho = raw_rho

    def _extract_path_gpu(self, alpha):
            path = []
            curr_mask = self.FULL_MASK
            
            # S 행렬 슬라이싱 주의 (N x N)
            final_scores = alpha[self.FULL_MASK, :] + self.S[:self.N, self.depot]
            best_last = torch.argmax(final_scores).item()
            best_score = final_scores[best_last].item()
            
            if best_score < -self.INF/2:
                return [], self.INF
                
            path = [self.depot, best_last]
            curr_node = best_last
            curr_mask = self.FULL_MASK ^ (1 << best_last)
            
            for t in range(self.N - 1, 0, -1):
                scores = alpha[curr_mask, :] + self.S[:self.N, curr_node]
                best_prev = torch.argmax(scores).item()
                val = scores[best_prev].item()
                
                if val < -self.INF/2:
                    return [], self.INF
                    
                path.append(best_prev)
                curr_node = best_prev
                curr_mask ^= (1 << best_prev)
                
            path.append(self.depot)
            path.reverse()
            
            dist_cpu = self.dist_matrix.cpu().numpy()
            cost = sum(dist_cpu[path[k], path[k+1]] for k in range(len(path)-1))
            
            return path, cost

    def solve(self):
        best_global_path = []
        best_global_cost = self.INF
        
        print(f"Solving TSP (N={self.N}) - Original Definition (Lambda=Rho)")
        
        for it in range(self.bp_iterations):
            alpha, tilde_delta = self._run_trellis_gpu()
            path, cost = self._extract_path_gpu(alpha)
            
            if cost < best_global_cost:
                best_global_cost = cost
                best_global_path = path
                print(f"[Iter {it}] Cost: {cost:.2f} (New Best!)")
            elif self.verbose:
                print(f"[Iter {it}] Cost: {cost:.2f}")

            self._run_bp_gpu(tilde_delta)
            
        return best_global_path, best_global_cost

if __name__ == "__main__":
    # N=14 예제 테스트
    N_CITIES = 20
    #np.random.seed(42)
    data = np.random.rand(N_CITIES, N_CITIES) * 100
    np.fill_diagonal(data, 0)
    
    solver = TSPSolverSOVATorch_Converge(data, bp_iterations=20, damping=0.7, verbose=True)
    path, cost = solver.solve()
    print("Final Path:", path)
    print("Final Cost:", cost)