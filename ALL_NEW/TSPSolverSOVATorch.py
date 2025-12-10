import torch
import numpy as np
import itertools
from collections import defaultdict

class TSPSolverSOVATorch:
    def __init__(self, dist_matrix, bp_iterations=20, damping=0.7, verbose=True, device='cuda'):
        # Device 설정
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 데이터 초기화 및 Tensor 변환
        self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=self.device)
        self.num_nodes = self.dist_matrix.shape[0]
        self.N = self.num_nodes - 1
        self.depot = self.N
        self.bp_iterations = bp_iterations
        self.damping = damping
        self.verbose = verbose
        self.INF = 1e12
        
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
        
        # Precompute Masks by Population Count (CPU에서 계산 후 GPU 인덱스로 변환)
        # DP 수행 시 같은 '방문 도시 수(popcount)'를 가진 상태끼리 묶어서 병렬 처리
        self.masks_by_popcount = [[] for _ in range(self.N + 1)]
        for mask in range(self.num_states):
            cnt = bin(mask).count('1')
            if cnt <= self.N:
                self.masks_by_popcount[cnt].append(mask)
        
        # GPU Tensor로 변환 (인덱싱용)
        self.masks_by_popcount_t = [
            torch.tensor(m, dtype=torch.long, device=self.device) 
            for m in self.masks_by_popcount
        ]

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _calc_lambda_sum_bias(self):
        """Eq (1.2): Lambda Sum Bias 계산 (Vectorized)"""
        N = self.N
        # dim=1 (Row) 합계
        sum_rho_t = torch.sum(self.tilde_rho, dim=1, keepdim=True)
        lambda_sum_bias = -self.tilde_rho + ((N - 1) / N) * sum_rho_t
        return lambda_sum_bias

    def _run_trellis_gpu(self):
        """
        GPU Optimized Trellis (Forward/Backward)
        Dictionary 대신 Dense Tensor [2^N, N] 사용
        """
        N, S, depot = self.N, self.S, self.depot
        bias = self._calc_lambda_sum_bias() # (N, N)
        
        # --- [1] Forward (Alpha) ---
        # alpha[mask, last_node]
        alpha = torch.full((self.num_states, N), -self.INF, device=self.device)
        
        # 초기 상태: 마스크 0은 없으므로, 첫 방문 노드들에 대해 설정
        # 0번 step(depot에서 출발)은 미리 처리
        # depot -> node i
        # mask: 1 << i, score: S[depot, i] + bias[0, i] (bias는 시간 t=0)
        
        # t=0 초기화
        start_bias = bias[0] # (N,)
        start_scores = S[depot, :N] + start_bias # (N,)
        
        # 각 i에 대해 mask = 1<<i 위치에 값 할당
        initial_masks = (1 << torch.arange(N, device=self.device))
        alpha[initial_masks, torch.arange(N, device=self.device)] = start_scores # deprecated indexing fix applied conceptually

        # DP Loop (t = 1 to N-1)
        for t in range(1, N):
            current_bias = bias[t] # (N,)
            prev_masks = self.masks_by_popcount_t[t]     # 현재 방문 수 t인 마스크들
            next_masks_indices = self.masks_by_popcount_t[t+1] # 방문 수 t+1인 마스크들 (검증용 혹은 인덱싱용)
            
            # 현재 유효한 점수 가져오기: (Num_Masks, N)
            curr_scores = alpha[prev_masks, :] 
            
            # Transition: (Num_Masks, N_prev) -> (Num_Masks, N_prev, N_next)
            # prev에서 next로 갈 때의 비용 추가
            # S: (N, N), current_bias: (N,)
            # cost[prev, next] = S[prev, next] + current_bias[next]
            transition_cost = S[:N, :N] + current_bias.view(1, N) # (N, N)
            
            # BroadCasting:
            # curr_scores: (M, N, 1)
            # transition:  (1, N, N)
            # candidates:  (M, N, N) -> (M, N_prev, N_next)
            candidates = curr_scores.unsqueeze(2) + transition_cost.unsqueeze(0)
            
            # 유효하지 않은 경로(이미 방문한 노드) 필터링은 bit logic으로 해야 함
            # M개의 마스크 각각에 대해 next_node 비트가 0이어야 함
            # Tensor 연산으로 마스크 계산: (M, 1) | (1 << (N)) 
            # -> (M, N_next)
            
            mask_col = prev_masks.view(-1, 1) # (M, 1)
            shifts = torch.arange(N, device=self.device).view(1, -1) # (1, N)
            next_bit_check = (mask_col & (1 << shifts)) == 0 # (M, N) : 방문 안했으면 True
            
            # 방문한 곳은 -INF 처리
            # (M, N, N) 에서 next node(3번째 차원) 기준으로 마스킹
            valid_mask = next_bit_check.unsqueeze(1).expand(-1, N, -1) # (M, N_prev, N_next)
            candidates = torch.where(valid_mask, candidates, torch.tensor(-self.INF, device=self.device))
            
            # Flatten candidates to scatter
            # 목적지 Mask 계산: old_mask | (1 << next_node)
            # (M, N_next) 행렬 생성
            new_masks_calc = mask_col | (1 << shifts) # (M, N_next)
            
            # Scatter를 위해 데이터 평탄화
            # 우리는 각 (new_mask, next_node)에 대해 Max 값을 구해야 함
            
            # Source values: candidates (M, N, N)
            src_vals = candidates.reshape(-1)
            
            # Destination indices construction
            # alpha는 (2^N, N) 형태. flat index = mask * N + node
            
            # 각 후보의 new_mask: (M, 1, N) -> expand -> (M, N, N)
            new_masks_expanded = new_masks_calc.unsqueeze(1).expand(-1, N, -1).reshape(-1)
            next_nodes_expanded = shifts.expand(len(prev_masks), N, N).reshape(-1) # 틀림, logic 수정 필요
            
            # 정확한 인덱스 생성:
            # Outer loop: Masks(M), Middle: Prev(N), Inner: Next(N)
            # new_mask는 Prev와 무관하게 (Mask | Next)임.
            m_idx = new_masks_calc.unsqueeze(1).expand(-1, N, -1).reshape(-1) # (M*N*N)
            n_idx = torch.arange(N, device=self.device).view(1, 1, N).expand(len(prev_masks), N, -1).reshape(-1)
            
            flat_indices = m_idx * N + n_idx
            
            # scatter_reduce_ (PyTorch 1.12+) 사용
            # alpha.view(-1) 에 src_vals를 max로 업데이트
            alpha.view(-1).scatter_reduce_(0, flat_indices, src_vals, reduce='amax', include_self=True)

        # --- [2] Backward (Beta) ---
        # Beta 역시 Dense Tensor [2^N, N]
        beta = torch.full((self.num_states, N), -self.INF, device=self.device)
        
        # t=N (마지막) 초기화: Depot으로 돌아가는 비용
        # beta[FULL_MASK, i] = S[i, depot]
        beta[self.FULL_MASK, :] = S[:N, depot]
        
        # Backward Loop (t = N-1 down to 0)
        for t in range(N - 1, -1, -1):
            curr_bias = bias[t] # (N,)
            
            # t+1 시점의 마스크들
            next_masks = self.masks_by_popcount_t[t+1]
            if len(next_masks) == 0: continue
            
            # (M_next, N)
            next_beta_vals = beta[next_masks, :] 
            
            # Xi 계산: beta + bias[node]
            # (M_next, N)
            xi_val = next_beta_vals + curr_bias.view(1, N)
            
            # 이번에는 '이전 노드(prev)'를 찾아야 함
            # next_mask에서 next_node 비트를 끈 것이 prev_mask
            # prev_node -> next_node
            
            # (M_next, N_next) -> next_node가 i일 때의 xi값
            # 우리는 모든 가능한 prev_node j에 대해:
            # val = S[j, i] + xi_val[mask, i]
            # update beta[mask ^ (1<<i), j] with max(val)
            
            # S: (N, N) (row=prev, col=next)
            # xi_val: (M, N) (row=mask, col=next)
            
            # Broadcasting 합: (N, N) + (M, 1, N) -> (M, prev, next) ?
            # (M, 1, N_next) + (1, N_prev, N_next) -> (M, N_prev, N_next)
            
            candidates = xi_val.unsqueeze(1) + S[:N, :N].unsqueeze(0)
            
            # Valid Check: prev_node가 next_mask에 포함되어 있어야 함
            # (역추적이니까, next_mask에 있는 비트들 중 하나가 prev_node가 됨... 아님)
            # 정확히는: next_mask에서 i를 뺀 것이 t시점의 mask.
            # 그리고 그 t시점 mask에 j가 포함되어 있어야 j -> i가 가능?
            # 아니, Backward는 "미래에 i를 방문했고 상태가 next_mask였다면, 현재 j에서의 가치"
            # 즉, next_mask 상태는 j를 이미 방문한 상태여야 함.
            
            mask_col = next_masks.view(-1, 1) # (M, 1)
            shifts = torch.arange(N, device=self.device).view(1, -1) # (1, N)
            
            # next_node(i)가 mask에 포함되어 있어야 유효한 출발점
            has_next_node = (mask_col & (1 << shifts)) != 0 # (M, N_next)
            
            # prev_mask 계산: mask ^ (1<<i)
            prev_masks_calc = mask_col ^ (1 << shifts)
            
            # j (prev_node)가 prev_mask에 포함되어 있어야 함 (단, t=0일 땐 prev_mask가 0이므로 예외)
            if t > 0:
                 # (M, N_next, 1) & (1, 1, N_prev)
                 prev_has_j = (prev_masks_calc.unsqueeze(2) & (1 << shifts).unsqueeze(0).unsqueeze(0)) != 0
                 # (M, N_next, N_prev)
            else:
                 # t=0이면 prev_mask는 0이어야 하고, prev_node는 없음(Depot).
                 # 하지만 코드 구조상 t=0까지 루프를 돌며 beta[0, :] 등을 채움?
                 # 원본 코드는 t=0까지 돌고, cands=[depot] 처리함.
                 # 여기선 NxN 구조만 다루므로 t=0 단계는 사실상 beta update 필요 없음 (depot 연결은 별도)
                 pass
            
            # --- Backward Scatter Logic ---
            # candidates: (M, N_prev, N_next) = Cost(j->i) + Beta(next_mask, i)
            # Target: beta[prev_mask, j]
            
            # 유효성 마스킹
            valid_mask = has_next_node.unsqueeze(1).expand(-1, N, -1) # i가 mask에 있어야함
            if t > 0:
                valid_mask = valid_mask & prev_has_j
            
            # t=0일때는 prev_mask가 0. (0, j)에 업데이트? 
            # 사실 t=0에서 beta값은 필요 없거나 depot 연결용. 여기선 생략 가능하지만 구조 유지.
            
            vals = torch.where(valid_mask, candidates, torch.tensor(-self.INF, device=self.device))
            
            # Scatter Indices
            # Target Mask: prev_masks_calc (M, N_next) -> (M, N_prev, N_next) 확장
            p_mask_idx = prev_masks_calc.unsqueeze(1).expand(-1, N, -1).reshape(-1)
            p_node_idx = torch.arange(N, device=self.device).view(1, N, 1).expand(len(next_masks), -1, N).reshape(-1)
            
            flat_indices = p_mask_idx * N + p_node_idx
            src_vals = vals.reshape(-1)
            
            beta.view(-1).scatter_reduce_(0, flat_indices, src_vals, reduce='amax', include_self=True)

        # --- [3] Soft Output (Delta) Calculation ---
        # Vectorized Max-In / Max-Out
        tilde_delta = torch.zeros((N, N), device=self.device)
        
        # 모든 시간 t에 대해 병렬 처리는 메모리 부담, t별 Loop
        for t in range(N):
            # Alpha(t) + Beta(t) = Total Score Map
            # shape: (Masks_t, N)
            curr_masks = self.masks_by_popcount_t[t+1] # t+1 시점의 alpha, beta 사용 (원본 코드 참조)
            if len(curr_masks) == 0: continue
            
            # alpha: (M, N), beta: (M, N)
            a = alpha[curr_masks]
            b = beta[curr_masks]
            
            # 유효한 값들만 더함 (-INF 방지)
            valid = (a > -self.INF/2) & (b > -self.INF/2)
            scores = torch.where(valid, a + b, torch.tensor(-self.INF, device=self.device))
            
            # City Max Scores: 해당 시간 t에 각 도시 i를 방문했을 때의 최대 점수
            # Reduce over Masks dimension -> (N,)
            city_max_scores = torch.max(scores, dim=0)[0] # (N,)
            
            # Find Best and Second Best for Max-Out
            # topk (2 values)
            top2_vals, top2_idxs = torch.topk(city_max_scores, 2)
            global_max = top2_vals[0]
            global_second = top2_vals[1]
            best_idx = top2_idxs[0]
            
            # Rho bias
            rho_t = self.tilde_rho[t]
            lam_i_for_i = -(1.0/N) * rho_t
            lam_i_for_j = ((N-1.0)/N) * rho_t
            
            max_in = city_max_scores - lam_i_for_i
            
            # Max Out Vectorized
            # 기본적으로 global_max 사용, best_idx 위치만 global_second 사용
            max_out_raw = torch.full((N,), global_max, device=self.device)
            max_out_raw[best_idx] = global_second
            
            max_out = max_out_raw - lam_i_for_j
            
            # Diff calculation
            diff = max_in - max_out
            
            # INF handling
            diff = torch.where(max_in < -self.INF/2, torch.tensor(-self.INF, device=self.device), diff)
            diff = torch.where(max_out < -self.INF/2, torch.tensor(self.INF, device=self.device), diff)
            
            tilde_delta[t] = diff
            
        return alpha, tilde_delta

    def _run_bp_gpu(self, tilde_delta):
        """Matrix Operations Fully on GPU"""
        N = self.N
        INF = self.INF
        
        # 1. Omega
        t_omega = self.tilde_phi + tilde_delta
        
        # 2. Eta (Column-wise Max excluding self)
        # dim=0 (col) max
        # topk(2)로 1등, 2등 구해서 broadcasting
        vals, idxs = torch.topk(t_omega, 2, dim=0) # (2, N)
        
        max1 = vals[0] # (N,)
        max2 = vals[1] # (N,)
        argmax = idxs[0] # (N,)
        
        # t행이 max위치면 max2, 아니면 max1
        # Grid 생성
        rows = torch.arange(N, device=self.device).view(N, 1).expand(N, N)
        is_max_pos = (rows == argmax)
        
        new_eta = torch.where(is_max_pos, max2, max1)
        new_eta = -new_eta
        
        # 3. Gamma
        t_gamma = new_eta + tilde_delta
        
        # 4. Phi (Row-wise Max excluding self)
        vals_r, idxs_r = torch.topk(t_gamma, 2, dim=1)
        max1_r = vals_r[:, 0].unsqueeze(1) # (N, 1)
        max2_r = vals_r[:, 1].unsqueeze(1)
        argmax_r = idxs_r[:, 0].unsqueeze(1)
        
        cols = torch.arange(N, device=self.device).view(1, N).expand(N, N)
        is_max_pos_r = (cols == argmax_r)
        
        new_phi = torch.where(is_max_pos_r, max2_r, max1_r)
        new_phi = -new_phi
        
        # Update
        self.tilde_eta = self.damping * self.tilde_eta + (1 - self.damping) * new_eta
        self.tilde_phi = self.damping * self.tilde_phi + (1 - self.damping) * new_phi
        self.tilde_rho = self.tilde_eta + self.tilde_phi

    def _extract_path_gpu(self, alpha):
            # CPU로 가져와서 순차적으로 역추적 (이 부분은 계산량이 적으므로 CPU가 편함)
            # alpha_cpu = alpha.cpu().numpy() # (주석 처리됨)
            
            path = []
            curr_mask = self.FULL_MASK
            
            # 1) 마지막 도시 선택
            # alpha[FULL_MASK, i] + S[i, depot]
            # S에서도 N개의 도시에 해당하는 부분만 가져오도록 슬라이싱 [:self.N] 필수
            final_scores = alpha[self.FULL_MASK, :] + self.S[:self.N, self.depot]
            best_last = torch.argmax(final_scores).item()
            best_score = final_scores[best_last].item()
            
            if best_score < -self.INF/2:
                return [], self.INF
                
            path = [self.depot, best_last]
            curr_node = best_last
            curr_mask = self.FULL_MASK ^ (1 << best_last)
            
            # 2) 역추적
            for t in range(self.N - 1, 0, -1):
                # alpha[t] 개념이 아니라, alpha[curr_mask, prev] + S[prev, curr_node]
                
                # [수정된 부분] self.S[:, curr_node] -> self.S[:self.N, curr_node]
                # S 행렬은 (N+1 x N+1) 크기이므로, alpha(N)와 크기를 맞추기 위해 
                # 앞쪽 N개의 행(도시들)만 슬라이싱해야 합니다.
                scores = alpha[curr_mask, :] + self.S[:self.N, curr_node]
                
                best_prev = torch.argmax(scores).item()
                val = scores[best_prev].item()
                
                if val < -self.INF/2:
                    print(f"Traceback failed at t={t}")
                    return [], self.INF
                    
                path.append(best_prev)
                curr_node = best_prev
                curr_mask ^= (1 << best_prev)
                
            path.append(self.depot)
            path.reverse()
            
            # Cost Recalculation (CPU)
            dist_cpu = self.dist_matrix.cpu().numpy()
            cost = sum(dist_cpu[path[k], path[k+1]] for k in range(len(path)-1))
            
            return path, cost

    def solve(self):
        best_global_path = []
        best_global_cost = self.INF
        
        print(f"Solving TSP (N={self.N}) with PyTorch/CUDA")
        
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