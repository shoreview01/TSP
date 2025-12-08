import numpy as np
import heapq
from viterbi import TSPBitmask

class TSP_MP_VBS:
    """
    Message-Passing Viterbi Beam Search (MP-VBS) for TSP.

    Phase 1: Runs a few iterations of Message-Passing to generate beliefs.
    Phase 2: Uses these beliefs to guide a Viterbi Beam Search to find the optimal path,
             avoiding the O(2^N) complexity of building a full trellis.
    """
    def __init__(self, distance_matrix, beam_width=100, mp_iters=5, belief_weight=0.5, damp=0.5):
        """
        Args:
            distance_matrix (np.array): The distance matrix including the depot (s_original).
            beam_width (int): The number of promising paths to keep at each step (K).
            mp_iters (int): The number of iterations to run the MP phase.
            belief_weight (float): The weight (w) to apply to the MP beliefs in the cost function.
            damp (float): Damping factor for message passing.
        """
        self.s_original = distance_matrix
        self.N = self.s_original.shape[0] - 1  # Number of cities excluding depot
        self.depot = self.N # Depot is indexed as N
        
        # Hyperparameters
        self.K = beam_width
        self.mp_iters = mp_iters
        self.w = belief_weight
        self.damp = damp

        # Initialize messages for MP Phase
        self.lambda_ = np.zeros((self.N, self.N)) # Message from variable c_t to factor f_t
        self.delta = np.zeros((self.N - 1, self.N)) # Forward message in the chain
        self.beliefs = np.zeros((self.N, self.N))

    def _run_mp_phase(self):
        """Phase 1: Generate beliefs using a simplified message-passing scheme."""
        print(f"--- Running Message-Passing Phase for {self.mp_iters} iterations ---")
        
        # Using a simplified version of TSPHC3's message updates for belief generation
        for iter_num in range(self.mp_iters):
            delta_old = self.delta.copy()
            
            # Forward pass (delta update)
            for t in range(self.N - 1):
                for m in range(self.N):
                    if t == 0:
                        # Cost from depot to the first city 'm'
                        cost_from_prev = self.s_original[self.depot, m]
                        max_val = cost_from_prev
                    else:
                        # Max over all previous cities m_prime
                        max_val = np.max([
                            -self.s_original[m_prime, m] + delta_old[t - 1, m_prime]
                            for m_prime in range(self.N) if m_prime != m
                        ])
                    
                    self.delta[t, m] = self.delta[t, m] * self.damp + (-max_val) * (1 - self.damp)

        # Belief is the sum of messages flowing into a variable node c_t
        for t in range(self.N):
            for m in range(self.N):
                if t == 0:
                    self.beliefs[t, m] = -self.s_original[self.depot, m]
                else:
                    self.beliefs[t, m] = self.delta[t-1, m]
        
        # Normalize beliefs to be used as a bonus
        self.beliefs = (self.beliefs - np.mean(self.beliefs)) / (np.std(self.beliefs) + 1e-8)
        print("--- Beliefs generated ---")


    def _run_viterbi_beam_search_phase(self):
        """Phase 2: Construct the path using Viterbi Beam Search guided by MP beliefs."""
        print(f"--- Running Viterbi Beam Search Phase (Beam Width: {self.K}) ---")

        # State: (cost, path_tuple). A min-heap will keep the best states.
        # Start from the depot. Path is represented as a tuple of visited cities.
        initial_path = (self.depot,)
        beam = [(0, initial_path)]

        for t in range(self.N): # Iterate N times to visit all N cities
            
            # A min-heap to store all potential next states for this step
            candidates = []
            
            # Expand each path in the current beam
            for cost, path in beam:
                last_city = path[-1]
                
                # Find all possible next cities
                for next_city in range(self.N):
                    if next_city not in path:
                        # Calculate the cost to move to the next city
                        distance_cost = self.s_original[last_city, next_city]
                        
                        # Apply the belief from MP as a bonus (or penalty)
                        belief_bonus = self.beliefs[t, next_city]
                        
                        # Hybrid cost function
                        new_cost = cost + distance_cost - self.w * belief_bonus
                        new_path = path + (next_city,)
                        
                        heapq.heappush(candidates, (new_cost, new_path))
            
            # Pruning: Select the top K candidates for the next beam
            beam = heapq.nsmallest(self.K, candidates)

            if not beam: # If beam is empty, no valid path was found
                print("Warning: Beam became empty. No solution found.")
                return None, np.inf

        # After visiting all cities, find the best path and add the return cost to depot
        best_cost, best_path_body = min(beam, key=lambda x: x[0])
        
        final_path = best_path_body + (self.depot,)
        
        # Recalculate the final cost using only the original distances
        final_true_cost = 0
        for i in range(len(final_path) - 1):
            final_true_cost += self.s_original[final_path[i], final_path[i+1]]

        return list(final_path), final_true_cost

    def run(self):
        """Executes the full MP-VBS algorithm."""
        # Phase 1
        self._run_mp_phase()
        
        # Phase 2
        path, cost = self._run_viterbi_beam_search_phase()

        # (Optional but recommended) Add a 2-opt post-processing step here
        # path, cost = self.post_process_2opt(path, cost)

        return path, cost

class TSP_Interactive_MPV:
    """
    An interactive "Ping-Pong" algorithm combining Message-Passing and a loose Viterbi.

    - MP runs to create beliefs (Ping).
    - A fast, belief-guided greedy search constructs a path (Pong).
    - The path provides strong evidence for the next MP iteration (Feedback).
    This cycle repeats to progressively refine the solution.
    """
    def __init__(self, distance_matrix, max_iters=50, mp_steps=3, damp=0.7, belief_weight=0.5):
        self.s_original = distance_matrix
        self.N = self.s_original.shape[0] - 1
        self.depot = self.N
        
        # Hyperparameters
        self.max_iters = max_iters
        self.mp_steps = mp_steps # Inner MP iterations per main loop
        self.damp = damp
        self.w = belief_weight # Weight for path feedback

        # Initialize messages and beliefs
        self.delta = np.zeros((self.N - 1, self.N))
        self.beliefs = np.zeros((self.N, self.N))
        
        # Initialize the best path found so far
        self.best_path = None
        self.best_cost = np.inf

    def _mp_step(self, path_evidence):
        """A few steps of MP, guided by evidence from the previous path."""
        for _ in range(self.mp_steps):
            delta_old = self.delta.copy()
            
            # Forward pass (delta update)
            for t in range(self.N - 1):
                for m in range(self.N):
                    if t == 0:
                        max_val = -self.s_original[self.depot, m]
                    else:
                        max_val = np.max([
                            -self.s_original[m_prime, m] + delta_old[t - 1, m_prime]
                            for m_prime in range(self.N) if m_prime != m
                        ])
                    
                    # Apply feedback from the path found in the previous step
                    evidence_bonus = 0
                    if path_evidence is not None and path_evidence[t+1] == m:
                        evidence_bonus = self.w * self.N # Give a strong bonus
                        
                    self.delta[t, m] = self.delta[t, m] * self.damp + (-max_val + evidence_bonus) * (1 - self.damp)
        
        # Update beliefs based on the latest messages
        for t in range(self.N):
            for m in range(self.N):
                self.beliefs[t, m] = self.delta[t-1, m] if t > 0 else -self.s_original[self.depot, m]

    def _loose_viterbi_step(self):
        """A fast, greedy path construction guided by the latest beliefs."""
        path = [self.depot]
        visited = {self.depot}
        
        for t in range(self.N):
            last_city = path[-1]
            
            # Find the best next city based on a mix of distance and belief
            candidates = []
            for next_city in range(self.N):
                if next_city not in visited:
                    cost = self.s_original[last_city, next_city] - self.beliefs[t, next_city]
                    candidates.append((cost, next_city))
            
            if not candidates:
                # Should not happen in a valid TSP
                return None

            # Greedily choose the best next city
            _, best_next_city = min(candidates, key=lambda x: x[0])
            path.append(best_next_city)
            visited.add(best_next_city)
            
        path.append(self.depot)
        return path

    def _calculate_cost(self, path):
        """Calculates the true cost of a path using only original distances."""
        if path is None:
            return np.inf
        cost = 0
        for i in range(len(path) - 1):
            cost += self.s_original[path[i], path[i+1]]
        return cost

    def run(self):
        """Executes the main Ping-Pong loop."""
        print("--- Starting Interactive MP-Viterbi (Ping-Pong) ---")
        
        current_path_evidence = None

        for i in range(self.max_iters):
            # --- PING: MP updates beliefs using evidence from the last path ---
            self._mp_step(current_path_evidence)
            
            # --- PONG: Viterbi constructs a new path based on new beliefs ---
            new_path = self._loose_viterbi_step()
            new_cost = self._calculate_cost(new_path)
            
            # --- Update the best-known solution ---
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_path = new_path
                print(f"Iter {i+1}: New best path found! Cost: {self.best_cost:.4f}")

            # --- FEEDBACK: The new path becomes evidence for the next MP step ---
            current_path_evidence = new_path
            
        print("\n--- Algorithm Finished ---")
        return self.best_path, self.best_cost



# --- Example Usage ---
if __name__ == '__main__':
    # Create a sample distance matrix (N cities + 1 depot)
    N_cities = 100
    #np.random.seed(42)
    coords = np.random.rand(N_cities, 2)
    s_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords, axis=2)
    
    # The depot is the last city (index N_cities)
    print(f"Solving TSP for {N_cities} cities (plus depot).")
    
    # Initialize and run the solver
    solver = TSP_MP_VBS(s_matrix, beam_width=50, mp_iters=5, belief_weight=0.1)
    final_path, final_cost = solver.run()
    
    if final_path:
        path_str = ' -> '.join(str(x+1) for x in final_path)
        print(f"Optimal Path Found: {path_str}")
        print(f"Total Cost: {final_cost:.4f}")
    
    solver3 = TSP_Interactive_MPV(s_matrix, max_iters=30, mp_steps=3, belief_weight=0.2)
    int_path, int_cost = solver3.run()
    
    if int_path:
        int_path_str = ' -> '.join(str(x+1) for x in int_path)
        print(f"\nInteractive MP-Viterbi Path: {int_path_str}")
        print(f"Interactive MP-Viterbi Cost: {int_cost:.4f}")
    
    solver2 = TSPBitmask(s_matrix)
    bf_path, bf_cost = solver2.run()
    
    if bf_path is not None:
        bf_path_str = ' -> '.join(str(x+1) for x in bf_path)
        print(f"\nViterbi Bitmask Path: {bf_path_str}")
        print(f"Viterbi Bitmask Cost: {bf_cost:.4f}")
    
    print("Accuray:", 100 - ((final_cost - bf_cost) / bf_cost * 100) if bf_cost != 0 else 0, "%")