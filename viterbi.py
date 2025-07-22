import itertools
import numpy as np

def hamming_weight(s):
    return bin(s).count('1')

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def generate_states(n):
    """Generate list of states per layer by Hamming weight"""
    layers = [[] for _ in range(n + 1)]
    for s in range(2 ** n):
        wt = hamming_weight(s)
        layers[wt].append(s)
    return layers

def viterbi_hypercube(n, transition_cost):
    layers = generate_states(n)
    dp = [{} for _ in range(n + 1)]  # dp[t][state] = (cost, prev_state)

    # 초기화
    dp[0][0] = (0, None)

    for t in range(1, n + 1):
        for curr in layers[t]:
            min_cost = float('inf')
            best_prev = None
            for prev in layers[t - 1]:
                if hamming_distance(prev, curr) != 1:
                    continue
                trans_c = transition_cost[prev, curr]
                total = dp[t - 1][prev][0] + trans_c
                if total < min_cost:
                    min_cost = total
                    best_prev = prev
            dp[t][curr] = (min_cost, best_prev)

    # Backtrack
    final_state = layers[n][0]  # 111...1
    path = []
    s = final_state
    for t in reversed(range(n + 1)):
        path.append(s)
        s = dp[t][s][1]
    path.reverse()
    total_cost = dp[n][final_state][0]
    return path, total_cost


# Example usage
n = 20
def make_transition_cost(n):
    cost = {}
    for i in range(2**n):
        for j in range(2**n):
            if hamming_distance(i, j) == 1:
                cost[(i, j)] = np.random.rand()  # or custom logic
    return cost

node_cost = make_transition_cost(n)  # random cost per node
path, total_cost = viterbi_hypercube(n, node_cost)

print("Best path (bitstring):", [format(s, f'0{n}b') for s in path])
print("Total cost:", total_cost)
