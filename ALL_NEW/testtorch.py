import torch
import numpy as np

class TSPSolverSOVATorch_Gauge:
    def __init__(self, dist_matrix, bp_iterations=20, damping=0.7, delta_std_target=20, verbose=True, device='cuda'):
        # Device 설정
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 데이터 초기화 및 Tensor 변환
        self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=self.device)
        self.num_nodes = self.dist_matrix.shape[0]
        self.N = self.num_nodes - 1
        self.depot = self.N
        self.bp_iterations = bp_iterations
        self.damping = damping  # BP 메시지 업데이트 시의 관성(Momentum)
        self.delta_std_target = delta_std_target
        self.verbose = verbose
        self.INF = 1e8
        
        # 비트마스크 전체 크기 (2^N)
        self.num_states = 1 << self.N
        self.FULL_MASK = self.num_states - 1
        
        # S Matrix 계산 (Max - Dist)
        max_dist = torch.max(self.dist_matrix)
        self.S = max_dist - self.dist_matrix
        self.S.fill_diagonal_(-self.INF)
        
        # Messages initialization (N x N)
        # 여기서 (row = t, col = city j) 로 사용
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

    # ----------------- 새 λ 정의: per-time gauge fix -----------------
    def _calc_lambda_gauge(self):
        """
        λ_t(j) = ρ̃_t(j) - mean_j ρ̃_t(j)
        -> 각 시간 t마다 sum_j λ_t(j) = 0 이 되게 하는 gauge-fix
        """
        lam = self.tilde_rho.clone()           # (N_t, N_city)
        lam = lam - lam.mean(dim=1, keepdim=True)  # time(t)별 row 평균 제거
        return lam

    # ----------------- Trellis + Soft Output -----------------
    def _run_trellis_gpu(self):
        """GPU Optimized Trellis with gauge-fixed Lambda"""
        N, S, depot = self.N, self.S, self.depot
        
        # Bias 계산 (gauge-fixed λ)
        bias = self._calc_lambda_gauge()  # (N, N)  [row=t, col=city]
        
        # --- [1] Forward (Alpha) ---
        alpha = torch.full((self.num_states, N), -self.INF, device=self.device)
        
        # t=0 초기화
        start_bias = bias[0]                        # (N,)
        start_scores = S[depot, :N] + start_bias    # depot->j 전이 + λ_0(j)
        
        initial_masks = (1 << torch.arange(N, device=self.device))
        alpha[initial_masks, torch.arange(N, device=self.device)] = start_scores

        # DP Loop
        for t in range(1, N):
            current_bias = bias[t]                 # (N,)
            prev_masks = self.masks_by_popcount_t[t]
            if len(prev_masks) == 0:
                continue
            
            curr_scores = alpha[prev_masks, :]     # (num_prev_masks, N)
            transition_cost = S[:N, :N] + current_bias.view(1, N)  # (1,N) broadcast
            
            # i->j 전이 후보
            candidates = curr_scores.unsqueeze(2) + transition_cost.unsqueeze(0)
            # candidates: (num_prev_masks, N(prev), N(next))
            
            mask_col = prev_masks.view(-1, 1)      # (num_prev_masks,1)
            shifts = torch.arange(N, device=self.device).view(1, -1)  # (1,N)
            next_bit_check = (mask_col & (1 << shifts)) == 0          # 방문 안 한 도시만
            
            valid_mask = next_bit_check.unsqueeze(1).expand(-1, N, -1)
            candidates = torch.where(
                valid_mask,
                candidates,
                torch.tensor(-self.INF, device=self.device)
            )
            
            new_masks_calc = mask_col | (1 << shifts)   # (num_prev_masks, N)
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
            if len(next_masks) == 0:
                continue
            
            next_beta_vals = beta[next_masks, :]      # (num_next_masks, N)
            xi_val = next_beta_vals + curr_bias.view(1, N)
            
            candidates = xi_val.unsqueeze(1) + S[:N, :N].unsqueeze(0)
            
            mask_col = next_masks.view(-1, 1)
            shifts = torch.arange(N, device=self.device).view(1, -1)
            has_next_node = (mask_col & (1 << shifts)) != 0
            prev_masks_calc = mask_col ^ (1 << shifts)
            
            if t > 0:
                prev_has_j = (
                    prev_masks_calc.unsqueeze(2) &
                    (1 << shifts).unsqueeze(0).unsqueeze(0)
                ) != 0
            else:
                prev_has_j = None  # t=0은 시작 단계이므로 별 의미 X
            
            valid_mask = has_next_node.unsqueeze(1).expand(-1, N, -1)
            if t > 0:
                valid_mask = valid_mask & prev_has_j
            
            vals = torch.where(
                valid_mask,
                candidates,
                torch.tensor(-self.INF, device=self.device)
            )
            
            p_mask_idx = prev_masks_calc.unsqueeze(1).expand(-1, N, -1).reshape(-1)
            p_node_idx = torch.arange(N, device=self.device).view(1, N, 1).expand(len(next_masks), -1, N).reshape(-1)
            flat_indices = p_mask_idx * N + p_node_idx
            src_vals = vals.reshape(-1)
            
            beta.view(-1).scatter_reduce_(0, flat_indices, src_vals, reduce='amax', include_self=True)

        # --- [3] Soft Output (Delta) : Max-In minus Max-Out (Vectorized) ---
        tilde_delta = torch.zeros((N, N), device=self.device)
        
        for t in range(N):
            curr_masks = self.masks_by_popcount_t[t+1]  # popcount = t+1 상태들
            if len(curr_masks) == 0:
                continue
            
            a = alpha[curr_masks]   # (M,N)
            b = beta[curr_masks]    # (M,N)
            valid = (a > -self.INF/2) & (b > -self.INF/2)
            total_scores = torch.where(
                valid,
                a + b,
                torch.tensor(-self.INF, device=self.device)
            )
            
            # 각 mask별 최댓값
            mask_scores = total_scores.max(dim=1)[0]  # (M,)
            if mask_scores.max() < -self.INF/2:
                continue

            M = len(curr_masks)
            masks_col = curr_masks.view(M, 1)       # (M,1)
            shifts = torch.arange(N, device=self.device).view(1, N)  # (1,N)
            
            has_j_mat = (masks_col & (1 << shifts)) != 0              # (M,N)
            scores_expanded = mask_scores.view(M, 1).expand(M, N)     # (M,N)
            
            in_vals = torch.where(
                has_j_mat,
                scores_expanded,
                torch.tensor(-self.INF, device=self.device)
            )
            max_in = in_vals.max(dim=0)[0]          # (N,)
            
            out_vals = torch.where(
                ~has_j_mat,
                scores_expanded,
                torch.tensor(-self.INF, device=self.device)
            )
            max_out = out_vals.max(dim=0)[0]        # (N,)
            
            diff = max_in - max_out

            # 예외 처리 그대로
            diff = torch.where(
                max_out < -self.INF/2,
                torch.tensor(self.INF/10, device=self.device),
                diff
            )
            diff = torch.where(
                max_in < -self.INF/2,
                torch.tensor(-self.INF/10, device=self.device),
                diff
            )

            # ---- 여기서 tilde_delta[t] 정규화 ----
            # 1) 평균 0
            diff = diff - diff.mean()

            # 2) 표준편차 고정 (soft clipping)
            std = diff.std()
            if std > 1e-8:
                scale = self.delta_std_target / std
                diff = diff * scale

            tilde_delta[t] = diff
            
        return alpha, tilde_delta

    # ----------------- BP Update (tilde_eta, tilde_phi, tilde_rho) -----------------
    def _run_bp_gpu(self, tilde_delta):
        """
        Matrix Operations Fully on GPU
        Constraint Satisfaction을 위한 BP 업데이트
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
        
        # Damping
        self.tilde_eta = self.damping * self.tilde_eta + (1 - self.damping) * new_eta
        self.tilde_phi = self.damping * self.tilde_phi + (1 - self.damping) * new_phi
        
        # --- Rho Update ---
        raw_rho = self.tilde_eta + self.tilde_phi
        
        # row/col gauge-fix (이전 코드 유지)
        raw_rho -= torch.mean(raw_rho, dim=1, keepdim=True)
        raw_rho -= torch.mean(raw_rho, dim=0, keepdim=True)
        
        self.tilde_rho = raw_rho

    # ----------------- Path Extraction -----------------
    def _extract_path_gpu(self, alpha):
        path = []
        curr_mask = self.FULL_MASK
        
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

    # ----------------- Main Solve -----------------
    def solve(self):
        best_global_path = []
        best_global_cost = self.INF
        
        print(f"Solving TSP (N={self.N}) - Gauge-fixed λ (per-time)")
        
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
    # 간단 테스트
    N_CITIES = 20
    #np.random.seed(42)
    data = np.random.rand(N_CITIES, N_CITIES) * 100
    np.fill_diagonal(data, 0)
    
    solver = TSPSolverSOVATorch_Gauge(data, bp_iterations=20, damping=0.8, delta_std_target=np.sqrt(data.mean()),verbose=True)
    path, cost = solver.solve()
    print("Final Path:", path)
    print("Final Cost:", cost)
