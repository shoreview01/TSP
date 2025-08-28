import numpy as np
import heapq
from viterbi import TSPBitmask
import time
import tsplib95

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
        print(self.s_original)
        print(self.s_norm)
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

        for _ in range(self.mp_iters):
            # phi: 도시→시간 경쟁
            for t in range(N):
                for i in range(N):
                    if N > 1:
                        max_val = np.max([self.gamma[ip, t] + self.zeta[t, ip]
                                          for ip in range(N) if ip != i])
                    else:
                        max_val = -np.inf
                    self.phi[i, t] = self.damp * self.phi[i, t] + (1 - self.damp) * (-max_val + self.zeta[t, i])

            # gamma: 시간→도시 경쟁
            for i in range(N):
                for t in range(N):
                    if N > 1:
                        max_val = np.max([self.phi[i, tp] for tp in range(N) if tp != t])
                    else:
                        max_val = -np.inf
                    self.gamma[i, t] = self.damp * self.gamma[i, t] + (1 - self.damp) * (-max_val)

            # forward β / δ
            if N >= 2:
                for t in range(N - 1):
                    for m in range(N):
                        base = s_alt(self.depot, m) if t == 0 else self.delta[t - 1, m]
                        self.beta[t, m] = self.lambda_[t, m] + base
                    for m in range(N):
                        if N > 1:
                            max_val = np.max([self.beta[t, mp] + s_alt(mp, m)
                                              for mp in range(N) if mp != m])
                        else:
                            max_val = -np.inf
                        self.delta[t, m] = self.damp * self.delta[t, m] + (1 - self.damp) * max_val

            # backward ρ
            self.rho.fill(0.0)
            if N >= 2:
                for t in range(N - 2, -1, -1):
                    for m in range(N):
                        if N > 1:
                            self.rho[t, m] = np.max([self.rho[t + 1, m2] + s_alt(m, m2)
                                                     for m2 in range(N) if m2 != m])
                        else:
                            self.rho[t, m] = 0.0

            # λ = γ + ρ, ζ 경계
            for t in range(N):
                for m in range(N):
                    self.lambda_[t, m] = self.gamma[m, t] + self.rho[t, m]
                    if t == 0:
                        self.zeta[t, m] = s_alt(self.depot, m)
                    elif t == N - 1:
                        self.zeta[t, m] = (self.delta[t - 1, m] if N >= 2 else 0.0) + s_alt(m, self.depot)
                    else:
                        self.zeta[t, m] = self.delta[t - 1, m]

        # Belief = z-score(λ)
        self.beliefs = self.lambda_.copy()
        mu, sigma = np.mean(self.beliefs), np.std(self.beliefs)
        print(self.beliefs)
        self.beliefs = (self.beliefs - mu) / (sigma + 1e-8)
        print(self.beliefs)

    # ---------- Viterbi 빔 탐색 (거리 - w·B) ----------

    def _run_viterbi_beam_search_phase(self):
        N = self.N
        if N == 0:
            return [self.depot, self.depot]
        if N == 1:
            return [self.depot, 0, self.depot]

        beam = [(0.0, (self.depot,))]

        for t in range(N):
            cand = []
            # w 스케줄링 (선택)
            w_t = self.w * ((t + 1) / N) if self.schedule_w else self.w

            for cost, path in beam:
                last = path[-1]
                used = set(path)  # path에 depot 포함됨
                for m in range(N):
                    if m in used:
                        continue
                    distance_cost = self.s_norm[last, m]
                    belief_bonus  = self.beliefs[t, m]
                    new_cost = cost + distance_cost - w_t * belief_bonus
                    new_path = path + (m,)

                    # K-크기 맥스힙 유지 (효율)
                    if len(cand) < self.K:
                        heapq.heappush(cand, (-new_cost, new_cost, new_path))
                    else:
                        if new_cost < cand[0][1]:
                            heapq.heapreplace(cand, (-new_cost, new_cost, new_path))

            if not cand:
                return None

            # cand를 비용 기준 오름차순으로 정렬된 리스트로 변환
            beam = [(c, p) for _, c, p in sorted(cand, key=lambda x: x[1])]

        # 최종 선택 시 '리턴 비용' 반영
        if self.use_return_cost_in_selection:
            final = [(c + self.s_original[p[-1], self.depot], p) for (c, p) in beam]
            best_cost, best_body = min(final, key=lambda x: x[0])
        else:
            best_cost, best_body = min(beam, key=lambda x: x[0])
            best_cost += self.s_original[best_body[-1], self.depot]

        return list(best_body + (self.depot,)), best_cost

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

# --- Example Usage ---
if __name__ == '__main__':
    
    # 1) 문제 로드
    problem = tsplib95.load('tsp_example_file/ALL_tsp/berlin52.tsp')
    nodes = list(problem.get_nodes())             # TSPLIB 라벨 (보통 1..N)
    N = len(nodes)

    # 라벨↔인덱스 매핑
    lab2idx = {lab: i for i, lab in enumerate(nodes)}
    idx2lab = {i: lab for i, lab in enumerate(nodes)}

    # 2) TSPLIB 규칙으로 거리행렬 만들기 (정수 가중치)
    D = np.zeros((N, N), dtype=np.int32)
    for a in nodes:
        ia = lab2idx[a]
        for b in nodes:
            ib = lab2idx[b]
            if ia != ib:
                D[ia, ib] = problem.get_weight(a, b)

    # 3) 시작도시를 depot으로 “복제”해 (N+1)×(N+1) 만들기 (여기선 start=0)
    start = 0
    DD = np.zeros((N+1, N+1), dtype=np.int32)
    DD[:N, :N] = D
    DD[N, :N] = D[start, :]   # depot→도시
    DD[:N, N] = D[:, start]   # 도시→depot
    DD[N, N] = 0

    # 4) 솔버 실행
    print(f"--- Solving TSP for {N} cities (TSPLIB-aligned) ---")
    t0 = time.time()
    solver = TSP_Hypercube_VBS(DD, beam_width=60, mp_iters=10, belief_weight=0.5, use_2_opt=True)
    path_with_depot, _cost_dummy = solver.run()
    dt = time.time() - t0

    # 5) depot 제거 + TSPLIB 방식으로 정확 채점
    #    path_with_depot는 [depot=N, ..., depot=N] 형태라고 가정
    path_idx = [p for p in path_with_depot if p != N]  # depot 제거 (방문 순서만 남김)
    # 시작도시를 맨 앞으로 회전 (선택 사항: 시작 동일성 유지)
    if path_idx[0] != start:
        s = path_idx.index(start)
        path_idx = path_idx[s:] + path_idx[:s]

    def tsplib_cycle_cost(problem, order_idx):
        # 마지막→처음 간선까지 포함 (순환)
        tot = 0
        for i in range(len(order_idx)):
            a = idx2lab[ order_idx[i] ]
            b = idx2lab[ order_idx[(i+1) % len(order_idx)] ]
            tot += problem.get_weight(a, b)
        return tot

    final_cost = tsplib_cycle_cost(problem, path_idx)

    # 6) %gap 계산 (eil51 최적해 = 426) berlin52 = 7542
    OPT = 7542
    gap = (final_cost - OPT) / OPT * 100.0

    print("\n--- Results ---")
    print("Order (0-based idx):", path_idx)
    print(f"TSPLIB-accurate Cost: {final_cost}")
    print(f"%gap: {gap:.2f}%")
    print(f"Total Time: {dt:.3f}s")
