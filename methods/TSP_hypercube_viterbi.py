import numpy as np
import heapq
from viterbi import TSPBitmask
import time
import tsplib95

class TSP_Hypercube_VBS:
    """
    TSPHC3의 메시지 패싱, Viterbi 빔 탐색, 그리고 2-opt 후처리가
    모두 내장된 통합 알고리즘 클래스.
    """
    def __init__(self, distance_matrix, beam_width=100, mp_iters=5, belief_weight=0.5, damp=0.5, use_2_opt=True):
        self.s_original = distance_matrix
        self.N = self.s_original.shape[0] - 1
        self.depot = self.N
        
        # Hyperparameters
        self.K = beam_width
        self.mp_iters = mp_iters
        self.w = belief_weight
        self.damp = damp
        self.use_2_opt = use_2_opt

        # --- TSPHC3의 모든 메시지 초기화 ---
        self.s = np.max(distance_matrix) - distance_matrix
        self.phi = np.zeros((self.N, self.N))
        self.gamma = np.zeros((self.N, self.N))
        self.zeta = np.zeros((self.N, self.N))
        self.beta = np.zeros((self.N - 1, self.N))
        self.delta = np.zeros((self.N - 1, self.N))
        self.lambda_ = np.zeros((self.N, self.N))
        self.rho = np.zeros((self.N, self.N))
        self.beliefs = np.zeros((self.N, self.N))
        self.c = self._init_greedy_path()

    def _init_greedy_path(self):
        path = []
        visited = set()
        current_city = self.depot
        for _ in range(self.N):
            next_city = min(
                (j for j in range(self.N) if j not in visited),
                key=lambda j: self.s_original[current_city, j]
            )
            path.append(next_city)
            visited.add(next_city)
            current_city = next_city
        return np.array(path)

    def _calculate_cost(self, path):
        if path is None: return np.inf
        return sum(self.s_original[path[i], path[i+1]] for i in range(len(path) - 1))

    def _apply_2_opt(self, path):
        """클래스에 내장된 2-opt 후처리 메서드."""
        best_path = path[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_path) - 2):
                for j in range(i + 1, len(best_path) - 1):
                    current_dist = self.s_original[best_path[i-1], best_path[i]] + self.s_original[best_path[j], best_path[j+1]]
                    new_dist = self.s_original[best_path[i-1], best_path[j]] + self.s_original[best_path[i], best_path[j+1]]
                    if new_dist < current_dist:
                        new_segment = best_path[i:j+1]
                        best_path[i:j+1] = new_segment[::-1]
                        improved = True
        return best_path

    def _run_full_mp_phase(self):
        # (이전과 동일한 TSPHC3의 메시지 패싱 로직)
        for _ in range(self.mp_iters):
            def s_alt(m1, m2): return self.s[m1, m2]
            for t in range(self.N):
                for i in range(self.N):
                    max_val = np.max([self.gamma[i_prime, t] + self.zeta[t, i_prime] for i_prime in range(self.N) if i_prime != i]) if self.N > 1 else -np.inf
                    self.phi[i, t] = self.phi[i, t] * self.damp + (-max_val + self.zeta[t, i]) * (1 - self.damp)
            for i in range(self.N):
                for t in range(self.N):
                    max_val = np.max([self.phi[i, t_prime] for t_prime in range(self.N) if t_prime != t]) if self.N > 1 else -np.inf
                    self.gamma[i, t] = self.gamma[i, t] * self.damp + (-max_val) * (1 - self.damp)
            for t in range(self.N - 1):
                for m in range(self.N):
                    self.beta[t, m] = self.lambda_[t, m] + (s_alt(self.depot, m) if t == 0 else self.delta[t-1, m])
                for m in range(self.N):
                    max_val = np.max([self.beta[t, m_prime] + s_alt(m_prime, m) for m_prime in range(self.N) if m_prime != m]) if self.N > 1 else -np.inf
                    self.delta[t, m] = self.delta[t, m] * self.damp + max_val * (1 - self.damp)
            self.rho.fill(0)
            for t in reversed(range(self.N - 1)):
                for m in range(self.N):
                    self.rho[t, m] = np.max([self.rho[t + 1, m_next] + s_alt(m, m_next) for m_next in range(self.N) if m_next != m]) if self.N > 1 else -np.inf
            for t in range(self.N):
                for m in range(self.N):
                    self.lambda_[t, m] = self.gamma[m, t] + self.rho[t, m]
                    if t == 0: self.zeta[t,m] = s_alt(self.depot, m)
                    elif t == self.N -1: self.zeta[t,m] = self.delta[t-1, m] + s_alt(m, self.depot)
                    else: self.zeta[t,m] = self.delta[t-1, m]
        self.beliefs = self.lambda_.copy()
        self.beliefs = (self.beliefs - np.mean(self.beliefs)) / (np.std(self.beliefs) + 1e-8)

    def _run_viterbi_beam_search_phase(self):
        # (이전과 동일한 Viterbi 빔 탐색 로직)
        initial_path = (self.depot,)
        beam = [(0, initial_path)]
        for t in range(self.N):
            candidates = []
            for cost, path in beam:
                last_city = path[-1]
                for next_city in range(self.N):
                    if next_city not in path:
                        distance_cost = self.s_original[last_city, next_city]
                        belief_bonus = self.beliefs[t, next_city]
                        new_cost = cost + distance_cost - self.w * belief_bonus
                        new_path = path + (next_city,)
                        heapq.heappush(candidates, (new_cost, new_path))
            beam = heapq.nsmallest(self.K, candidates)
            if not beam: return None
        _, best_path_body = min(beam, key=lambda x: x[0])
        return list(best_path_body + (self.depot,))

    def run(self):
        """전체 알고리즘을 실행하고, 필요시 2-opt 후처리를 적용합니다."""
        # Phase 1: Belief 생성
        self._run_full_mp_phase()
        
        # Phase 2: Viterbi 빔 탐색으로 경로 초안 생성
        path = self._run_viterbi_beam_search_phase()

        # Phase 3: (선택적) 클래스 내부의 2-opt 메서드 호출
        if path and self.use_2_opt:
            print("--- Applying 2-opt post-processing ---")
            path = self._apply_2_opt(path)
            
        cost = self._calculate_cost(path)
        return path, cost

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
    solver = TSP_Hypercube_VBS(DD, beam_width=50, mp_iters=10, belief_weight=0.1, use_2_opt=True)
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

    # 6) %gap 계산 (eil51 최적해 = 426)
    OPT = 7542
    gap = (final_cost - OPT) / OPT * 100.0

    print("\n--- Results ---")
    print("Order (0-based idx):", path_idx)
    print(f"TSPLIB-accurate Cost: {final_cost}")
    print(f"%gap: {gap:.2f}%")
    print(f"Total Time: {dt:.3f}s")
