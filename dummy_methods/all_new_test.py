import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

class TSP_Hypercube_VBS:
    """
    메시지 패싱(λ=gamma+rho) → Belief 정규화(B=zscore(λ)) → Viterbi 빔 탐색(거리 - w·B)
    → (옵션) 2-opt 후처리까지 수행하는 통합 TSP 솔버.
    """

    def __init__(self, distance_matrix, beam_width=100, mp_iters=5, belief_weight=0.5,
                 damp=0.5, use_2_opt=True, use_return_cost_in_selection=True,
                 schedule_w=False):
        # 기본 데이터
        self.s_original = np.asarray(distance_matrix)
        mu_s, sigma_s = np.mean(self.s_original), np.std(self.s_original)
        self.s_norm = (self.s_original - mu_s) / (sigma_s + 1e-8)
        #print(self.s_original)
        #print(self.s_norm)
        self.N = self.s_original.shape[0] - 1
        self.depot = self.N

        # 하이퍼파라미터
        self.K = int(beam_width)
        self.mp_iters = int(mp_iters)
        self.w = float(belief_weight)
        self.damp = float(damp)
        self.use_2_opt = bool(use_2_opt)
        self.use_return_cost_in_selection = bool(use_return_cost_in_selection)
        self.schedule_w = bool(schedule_w)  # True면 t와 함께 w를 선형 스케줄링

        # 유사도 (max - D)
        self.s = np.max(self.s_original) - self.s_original

        # 메시지 초기화
        self.phi   = np.zeros((self.N, self.N))
        self.gamma = np.zeros((self.N, self.N))
        self.zeta  = np.zeros((self.N, self.N))
        self.beta  = np.zeros((self.N - 1, self.N)) if self.N >= 2 else np.zeros((0, self.N))
        self.delta = np.zeros((self.N - 1, self.N)) if self.N >= 2 else np.zeros((0, self.N))
        self.lambda_ = np.zeros((self.N, self.N))
        self.rho     = np.zeros((self.N, self.N))
        self.beliefs = np.zeros((self.N, self.N))

        # 초기 경로(탐욕)
        self.c = self._init_greedy_path()

    # ---------- 유틸 ----------

    def _init_greedy_path(self):
        path = []
        visited = set()
        current = self.depot
        for _ in range(self.N):
            next_city = min((j for j in range(self.N) if j not in visited),
                            key=lambda j: self.s_original[current, j])
            path.append(next_city)
            visited.add(next_city)
            current = next_city
        return np.array(path, dtype=int)

    def _project_beliefs_to_binary(self, mode='hungarian', temperature=0.5):
        """
        λ(or beliefs)를 T×N 퍼뮤테이션(0/1) 행렬로 프로젝션.
        - mode='hungarian' : 가장 깔끔한 정수해(권장, SciPy 있으면).
        - mode='sinkhorn'  : SciPy 없을 때 무의미한 smear를 줄이는 연화→이진화.
        """
        lam = self.lambda_.copy()  # shape: (N, N) = (step, city)

        if mode == 'hungarian':
            try:
                from scipy.optimize import linear_sum_assignment
                # 최대화이므로 비용을 -lam으로
                row_ind, col_ind = linear_sum_assignment(-lam)
                B = np.zeros_like(lam)
                B[row_ind, col_ind] = 1.0
            except Exception:
                # SciPy가 없으면 그리디 대체(근사). 가장 큰 λ부터 행/열을 배제.
                mask = lam.copy()
                B = np.zeros_like(lam)
                for _ in range(self.N):
                    idx = np.unravel_index(np.argmax(mask), mask.shape)
                    t, m = int(idx[0]), int(idx[1])
                    B[t, m] = 1.0
                    mask[t, :] = -np.inf
                    mask[:, m] = -np.inf
        else:  # 'sinkhorn'
            # 1) softmax로 양수화 & 온도 조절
            A = np.exp((lam - lam.max()) / max(temperature, 1e-6))
            # 2) Sinkhorn 정규화(행/열을 1로)
            for _ in range(60):
                A /= (A.sum(axis=1, keepdims=True) + 1e-12)
                A /= (A.sum(axis=0, keepdims=True) + 1e-12)
            # 3) 행별 argmax로 원-핫 프로젝션 (열 유일성도 거의 만족)
            B = np.zeros_like(A)
            B[np.arange(self.N), np.argmax(A, axis=1)] = 1.0
            # 열 중복이 생기면 간단한 보정(필요시): 큰 값 순서로 하나만 남기기
            used_cols = set()
            for t in np.argsort(-lam.max(axis=1)):  # 강한 행부터
                m = np.argmax(B[t])
                if m in used_cols:
                    # 해당 행에서 두번째로 큰 곳으로 옮기기(간단 보정)
                    order = np.argsort(-lam[t])
                    for m2 in order:
                        if m2 not in used_cols:
                            B[t] *= 0
                            B[t, m2] = 1.0
                            used_cols.add(m2)
                            break
                else:
                    used_cols.add(m)

        self.beliefs = B  # 이제 0/1
    def _path_with_depot(self, path_body):
        return [self.depot] + list(path_body) + [self.depot]

    def _calculate_cost(self, path_with_depot):
        if path_with_depot is None or len(path_with_depot) < 2:
            return np.inf
        return sum(self.s_original[path_with_depot[i], path_with_depot[i+1]]
                   for i in range(len(path_with_depot) - 1))

    def _apply_2_opt(self, path_with_depot):
        best = path_with_depot[:]
        if len(best) <= 4:  # depot, a, depot
            return best
        improved = True
        # depot 고정: i=1..len-3, j=i+1..len-2
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    cur = (self.s_original[best[i-1], best[i]] +
                           self.s_original[best[j],   best[j+1]])
                    new = (self.s_original[best[i-1], best[j]] +
                           self.s_original[best[i],   best[j+1]])
                    if new < cur:
                        best[i:j+1] = reversed(best[i:j+1])
                        improved = True
        return best

    # ---------- 메시지 패싱 (λ=γ+ρ, B=zscore(λ)) ----------

    def _run_full_mp_phase(self):
        N = self.N
        if N == 0:
            self.beliefs = np.zeros((0, 0))
            return

        def s_alt(a, b):
            return self.s[a, b]

        # --- 순방향/역방향 경로 메시지 계산 ---
        for _ in range(self.mp_iters):
            # 순방향(Forward) delta 계산: depot에서 시작하여 t번째에 m에 도착하는 최적 경로 값
            if N >= 2:
                # 임시 beta 행렬을 사용하여 t-1 시점의 delta 값을 저장
                beta_prev_delta = np.zeros(N)
                for t in range(N - 1):
                    for m in range(N):
                        if t == 0:
                             # 첫 스텝은 depot에서 직접 오는 비용
                            beta_prev_delta[m] = s_alt(self.depot, m)
                        else:
                            beta_prev_delta[m] = self.delta[t - 1, m]

                    # t 시점의 delta 값 계산
                    for m in range(N):
                        max_val = np.max([beta_prev_delta[mp] + s_alt(mp, m) for mp in range(N) if mp != m])
                        # 이전 delta 값과 새 값을 damp 비율로 섞어줌
                        self.delta[t, m] = self.damp * self.delta[t, m] + (1 - self.damp) * max_val

            # 역방향(Backward) rho 계산: t번째에 m에서 시작하여 depot으로 돌아가는 최적 경로 값
            self.rho.fill(0.0) # rho는 매번 처음부터 다시 계산
            if N >= 2:
                for t in range(N - 2, -1, -1):
                    for m in range(N):
                        # t+1 시점의 rho 값과 도시 간 유사도를 이용
                        max_val = np.max([self.rho[t + 1, m2] + s_alt(m, m2) for m2 in range(N) if m2 != m])
                        self.rho[t, m] = max_val

        # ------------------------------------------------------------------
        # [최종 수정된 Beliefs 계산]
        # 순방향(delta)과 역방향(rho) 경로 비용을 직접 합산하여 편향 없는 Beliefs 생성
        # ------------------------------------------------------------------
        for t in range(N):
            for m in range(N):
                # forward_cost: (depot -> ... -> t번째 m 도시)
                if t == 0:
                    forward_cost = s_alt(self.depot, m)
                else:
                    # t-1번째까지의 최적 경로에 m으로 오는 비용을 더함
                    forward_cost = np.max([self.delta[t - 1, mp] + s_alt(mp, m) for mp in range(N) if mp != m])

                # backward_cost: (t번째 m 도시 -> ... -> depot)
                backward_cost = self.rho[t, m]

                # 마지막 도시에서 depot으로 돌아가는 비용은 rho에 이미 포함되어 있으나, 명시적으로 더해줄 수도 있음
                # if t == N - 1: backward_cost += s_alt(m, self.depot)

                self.lambda_[t, m] = forward_cost + backward_cost

        # Belief = z-score(λ)
        self.beliefs = self.lambda_.copy()
        if np.std(self.beliefs) > 1e-8:
            self._project_beliefs_to_binary(mode='hungarian')  # 또는 'sinkhorn'
            #mu, sigma = np.mean(self.beliefs), np.std(self.beliefs)
            #self.beliefs = (self.beliefs - mu) / (sigma + 1e-8)
        else:
            self.beliefs.fill(0.0)

        # <<<--- 디버깅 코드 추가 시작 --->>>
        print("Beliefs matrix visualization:")
        if self.N > 0:
            plt.imshow(self.beliefs, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Belief (z-score)')
            plt.title('Beliefs Matrix (City vs. Step)')
            plt.xlabel('Step (t)')
            plt.ylabel('City (i)')
            plt.show()
        # <<<--- 디버깅 코드 추가 끝 --->>>

    # ---------- Viterbi 빔 탐색 (거리 - w·B) ----------

    def _run_viterbi_beam_search_phase(self):
        N = self.N
        if N == 0:
            return [self.depot, self.depot], 0.0
        if N == 1:
            tour = [self.depot, 0, self.depot]
            return tour, (self.s_original[self.depot, 0] + self.s_original[0, self.depot])

        # beam: (eval_cost, true_cost, path)
        beam = [(0.0, 0.0, (self.depot,))]

        for t in range(N):
            cand = []
            w_t = self.w * ((t + 1) / N) if self.schedule_w else self.w

            for eval_cost, true_cost, path in beam:
                last = path[-1]
                used = set(path)  # depot 포함
                for m in range(N):
                    if m in used:
                        continue
                    # 선택 기준(정규화 거리 - w*B)
                    distance_cost_norm = self.s_norm[last, m]        # 정규화된 거리 (선택용)
                    belief_bonus       = self.beliefs[t, m]          # 신뢰도
                    new_eval_cost = eval_cost + distance_cost_norm - w_t * belief_bonus

                    # 진짜 비용(보고용): 원본 거리 누적
                    distance_cost_true = self.s_original[last, m]
                    new_true_cost = true_cost + distance_cost_true

                    new_path = path + (m,)

                    # K-크기 맥스힙 유지
                    if len(cand) < self.K:
                        heapq.heappush(cand, (-new_eval_cost, new_eval_cost, new_true_cost, new_path))
                    else:
                        if new_eval_cost < cand[0][1]:
                            heapq.heapreplace(cand, (-new_eval_cost, new_eval_cost, new_true_cost, new_path))

            if not cand:
                return None

            # eval_cost 기준 오름차순으로 정리해 빔 갱신
            beam = [(ec, tc, p) for _, ec, tc, p in sorted(cand, key=lambda x: x[1])]

        # 최종 선택: (선택은 eval_cost 기준으로 하되) 리턴 간선은
        #   - 선택용: 원하면 self.s_norm[last, depot]를 더해도 되고(선택 일관성↑)
        #   - 보고용: 반드시 self.s_original[last, depot]를 true_cost에 더함
        final_candidates = []
        for ec, tc, p in beam:
            last = p[-1]
            ec_final = ec + self.s_norm[last, self.depot] if self.use_return_cost_in_selection else ec
            tc_final = tc + self.s_original[last, self.depot]  # ← 진짜 투어 비용
            final_candidates.append((ec_final, tc_final, p))

        # eval_cost로 베스트 경로를 고르고, 보고는 true_cost로
        best_ec, best_true_cost, best_body = min(final_candidates, key=lambda x: x[0])
        best_tour = list(best_body + (self.depot,))
        return best_tour, best_true_cost


    # ---------- 전체 실행 ----------

    def run(self):
        # Phase 1: 메시지 패싱 → B 생성
        self._run_full_mp_phase()

        # Phase 2: Viterbi 빔 탐색
        result = self._run_viterbi_beam_search_phase()
        if result is None:
            return None, np.inf
        path_with_depot, cost = result

        # Phase 3: (옵션) 2-opt 후처리
        if self.use_2_opt and len(path_with_depot) >= 4:
            path_with_depot = self._apply_2_opt(path_with_depot)
            cost = self._calculate_cost(path_with_depot)

        return path_with_depot, cost

N = 20
np.random.seed(42)
coords = np.random.rand(N + 1, 2)
s_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords, axis=2)
start = time.time()
solver1 = TSP_Hypercube_VBS(s_matrix, beam_width=100, mp_iters=100, belief_weight=0.5, use_2_opt=False)
path1, cost1 = solver1.run()
elasped1 = time.time() - start
print("\n--- Results ---")
print("Order:", path1)
print(f"TSPLIB-accurate Cost: {cost1}")
print(f"Total Time: {elasped1:.3f}s")