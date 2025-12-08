import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import heapq

# viterbi.py에 TSPBitmask가 있다고 가정합니다.
from viterbi import TSPBitmask

# --- 2-opt 후처리 함수 ---
def apply_2_opt(path, dist_matrix):
    """
    Applies the 2-opt heuristic to improve a given TSP path.
    Args:
        path (list): The initial path, e.g., [depot, city1, city2, ..., depot].
        dist_matrix (np.array): The distance matrix.
    Returns:
        list: The improved path.
    """
    best_path = path[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                # 현재 경로: ...A->B...C->D...
                # 바꿀 경로: ...A->C...B->D...
                # A=best_path[i-1], B=best_path[i], C=best_path[j], D=best_path[j+1]
                
                current_dist = dist_matrix[best_path[i-1], best_path[i]] + dist_matrix[best_path[j], best_path[j+1]]
                new_dist = dist_matrix[best_path[i-1], best_path[j]] + dist_matrix[best_path[i], best_path[j+1]]

                if new_dist < current_dist:
                    # 경로를 뒤집어 교차를 해결합니다.
                    new_segment = best_path[i:j+1]
                    best_path[i:j+1] = new_segment[::-1]
                    improved = True
        path = best_path
    return best_path

# --- 기존 클래스 수정 ---
# (각 클래스의 run 메서드 마지막에 2-opt 호출 부분을 추가합니다)

class TSP_MP_VBS:
    """Message-Passing Viterbi Beam Search (MP-VBS) for TSP."""
    def __init__(self, distance_matrix, beam_width=100, mp_iters=5, belief_weight=0.5, damp=0.5, use_2_opt=False):
        self.s_original = distance_matrix
        self.N = self.s_original.shape[0] - 1
        self.depot = self.N
        self.K = beam_width
        self.mp_iters = mp_iters
        self.w = belief_weight
        self.damp = damp
        self.use_2_opt = use_2_opt
        self.delta = np.zeros((self.N - 1, self.N))
        self.beliefs = np.zeros((self.N, self.N))

    def _calculate_cost(self, path):
        if path is None: return np.inf
        return sum(self.s_original[path[i], path[i+1]] for i in range(len(path) - 1))

    def _run_mp_phase(self):
        # ... (이전과 동일) ...
        for iter_num in range(self.mp_iters):
            delta_old = self.delta.copy()
            for t in range(self.N - 1):
                for m in range(self.N):
                    if t == 0: max_val = self.s_original[self.depot, m]
                    else: max_val = np.max([-self.s_original[m_prime, m] + delta_old[t - 1, m_prime] for m_prime in range(self.N) if m_prime != m])
                    self.delta[t, m] = self.delta[t, m] * self.damp + (-max_val) * (1 - self.damp)
        for t in range(self.N):
            for m in range(self.N):
                self.beliefs[t, m] = self.delta[t-1, m] if t > 0 else -self.s_original[self.depot, m]
        self.beliefs = (self.beliefs - np.mean(self.beliefs)) / (np.std(self.beliefs) + 1e-8)


    def _run_viterbi_beam_search_phase(self):
        # ... (이전과 동일, 비용 계산 부분만 분리) ...
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
        self._run_mp_phase()
        path = self._run_viterbi_beam_search_phase()

        if path and self.use_2_opt:
            path = apply_2_opt(path, self.s_original)
            
        cost = self._calculate_cost(path)
        return path, cost

class TSP_Interactive_MPV:
    """An interactive 'Ping-Pong' algorithm."""
    def __init__(self, distance_matrix, max_iters=50, mp_steps=3, damp=0.7, belief_weight=0.5, use_2_opt=False):
        self.s_original = distance_matrix
        self.N = self.s_original.shape[0] - 1
        self.depot = self.N
        self.max_iters = max_iters
        self.mp_steps = mp_steps
        self.damp = damp
        self.w = belief_weight
        self.use_2_opt = use_2_opt
        self.delta = np.zeros((self.N - 1, self.N))
        self.beliefs = np.zeros((self.N, self.N))
        self.best_path = None
        self.best_cost = np.inf
        
    def _calculate_cost(self, path):
        if path is None: return np.inf
        return sum(self.s_original[path[i], path[i+1]] for i in range(len(path) - 1))

    def _mp_step(self, path_evidence):
        # ... (이전과 동일) ...
        for _ in range(self.mp_steps):
            delta_old = self.delta.copy()
            for t in range(self.N - 1):
                for m in range(self.N):
                    if t == 0: max_val = -self.s_original[self.depot, m]
                    else: max_val = np.max([-self.s_original[m_prime, m] + delta_old[t - 1, m_prime] for m_prime in range(self.N) if m_prime != m])
                    evidence_bonus = self.w * self.N if path_evidence and path_evidence[t+1] == m else 0
                    self.delta[t, m] = self.delta[t, m] * self.damp + (-max_val + evidence_bonus) * (1 - self.damp)
        for t in range(self.N):
            for m in range(self.N):
                self.beliefs[t, m] = self.delta[t-1, m] if t > 0 else -self.s_original[self.depot, m]

    def _loose_viterbi_step(self):
        # ... (이전과 동일) ...
        path = [self.depot]
        visited = {self.depot}
        for t in range(self.N):
            last_city = path[-1]
            candidates = [(self.s_original[last_city, nc] - self.beliefs[t, nc], nc) for nc in range(self.N) if nc not in visited]
            if not candidates: return None
            _, best_next_city = min(candidates, key=lambda x: x[0])
            path.append(best_next_city)
            visited.add(best_next_city)
        path.append(self.depot)
        return path

    def run(self):
        current_path_evidence = None
        for i in range(self.max_iters):
            self._mp_step(current_path_evidence)
            new_path = self._loose_viterbi_step()
            new_cost = self._calculate_cost(new_path)
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_path = new_path
            current_path_evidence = new_path
            
        final_path = self.best_path
        if final_path and self.use_2_opt:
            final_path = apply_2_opt(final_path, self.s_original)

        final_cost = self._calculate_cost(final_path)
        return final_path, final_cost


# --- 실험 및 시각화 코드 수정 ---
def run_experiment_with_2_opt():
    city_counts = range(3, 21) # 도시 수를 조절하여 테스트 시간 관리
    num_trials = 20 # 통계적 신뢰도를 위해 시도 횟수 조절
    results = []
    
    mp_vbs_params = {'beam_width': 30, 'mp_iters': 5, 'belief_weight': 0.1, 'damp': 0.5}
    interactive_params = {'max_iters': 20, 'mp_steps': 2, 'belief_weight': 0.2, 'damp': 0.7}
    
    for n_cities in city_counts:
        print(f"\n--- Testing for {n_cities} cities ---")
        for trial in range(num_trials):
            coords = np.random.rand(n_cities + 1, 2)
            s_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords, axis=2)
            
            # 알고리즘 실행 리스트
            solvers_to_run = {
                'MP-VBS': TSP_MP_VBS(s_matrix, **mp_vbs_params, use_2_opt=False),
                'MP-VBS + 2-opt': TSP_MP_VBS(s_matrix, **mp_vbs_params, use_2_opt=True),
                'Interactive MPV': TSP_Interactive_MPV(s_matrix, **interactive_params, use_2_opt=False),
                'Interactive MPV + 2-opt': TSP_Interactive_MPV(s_matrix, **interactive_params, use_2_opt=True),
            }
            if n_cities <= 12:
                 solvers_to_run['Bitmask (Exact)'] = TSPBitmask(s_matrix)

            for name, solver in solvers_to_run.items():
                start_time = time.time()
                _, cost = solver.run()
                duration = time.time() - start_time
                results.append({'Algorithm': name, 'Cities': n_cities, 'Trial': trial, 'Cost': cost, 'Time': duration})

    return pd.DataFrame(results)

def plot_results(df):
    # (이전과 동일한 시각화 코드)
    avg_results = df.groupby(['Algorithm', 'Cities']).mean().reset_index()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    
    # 색상과 마커 스타일을 지정하여 가독성 향상
    styles = {'MP-VBS': ('-','o'), 'MP-VBS + 2-opt': ('--','^'),
              'Interactive MPV': ('-','s'), 'Interactive MPV + 2-opt': ('--','x'),
              'Bitmask (Exact)': ('-','d')}
    colors = plt.cm.viridis(np.linspace(0, 1, len(styles)))
    color_map = {name: colors[i] for i, name in enumerate(styles)}

    for name, group in avg_results.groupby('Algorithm'):
        style = styles.get(name, ('-','.'))
        color = color_map.get(name)
        ax1.plot(group['Cities'], group['Cost'], linestyle=style[0], marker=style[1], label=name, color=color)
        ax2.plot(group['Cities'], group['Time'], linestyle=style[0], marker=style[1], label=name, color=color)
    
    ax1.set_title('Average TSP Cost vs. Number of Cities', fontsize=16)
    ax1.set_ylabel('Average Cost (Path Length)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--')

    ax2.set_title('Average Execution Time vs. Number of Cities (Log Scale)', fontsize=16)
    ax2.set_xlabel('Number of Cities')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--')
    
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    results_df = run_experiment_with_2_opt()
    print("\n--- Experiment Finished. Plotting results... ---")
    plot_results(results_df)