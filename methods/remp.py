import numpy as np

def remp_assignment(P, rho=0.5, max_iter=200, tol=1e-8):
    """
    Reweighted Message Passing for a square cost matrix P (shape N×N)
    returns binary assignment matrix X (N×N, one‑hot rows & cols)
    """
    N = P.shape[0]
    mu = np.zeros_like(P)        # μ_{n,f}
    mu_tilde = np.zeros_like(P)  # \tilde{μ}_{f,n}

    for _ in range(max_iter):
        # forward messages μ_{n,f}
        min_over_f = np.min(P[:, None, :] + mu_tilde[:, :, None], axis=2)  # shape N×N (n,f)
        mu_new = P - rho * min_over_f + (rho - 1) * (P + mu_tilde.T)

        # backward messages \tilde{μ}_{f,n}
        min_over_n = np.min(mu_new[None, :, :], axis=1)    # shape N×N (f,n)
        mu_tilde_new = -rho * min_over_n + (rho - 1) * mu_new.T

        # convergence check
        if np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(mu_tilde_new - mu_tilde)) < tol:
            mu, mu_tilde = mu_new, mu_tilde_new
            break
        mu, mu_tilde = mu_new, mu_tilde_new

    tau = mu + mu_tilde.T                           # (10)
    X = (tau < 0).astype(int)                       # (11)
    return X

def find_cycles(X):
    """X: binary assignment (N×N), returns list of cycles (each is list of node indices)."""
    N = X.shape[0]
    succ = {i: np.where(X[i])[0][0] for i in range(N)}
    cycles, visited = [], set()
    for start in range(N):
        if start in visited: continue
        cycle = [start]
        nxt = succ[start]
        while nxt not in cycle:
            cycle.append(nxt)
            nxt = succ[nxt]
        cycles.append(cycle[cycle.index(nxt):])  # close the loop
        visited.update(cycle)
    return cycles

def tsp_remp(D, rho=0.5, big_penalty=1e6):
    """
    Solve symmetric TSP distance matrix D (N×N) with ReMP+subtour‑penalty heuristic.
    Returns (tour list, total length).
    """
    N = D.shape[0]
    P = D.copy()
    np.fill_diagonal(P, big_penalty)  # forbid self loops

    while True:
        X = remp_assignment(P, rho=rho)
        cycles = find_cycles(X)
        if len(cycles) == 1 and len(cycles[0]) == N:
            # decode tour order
            tour = [0]
            while len(tour) < N:
                tour.append(np.where(X[tour[-1]])[0][0])
            tour.append(0)  # return to start
            length = sum(D[tour[i], tour[i+1]] for i in range(N))
            return tour, length

        # add penalty to one edge in each subtour to break it
        for cyc in cycles:
            if len(cyc) == N:  # already full tour (shouldn't happen here)
                continue
            i = cyc[0]
            j = np.where(X[i])[0][0]
            P[i, j] += big_penalty  # discourage this edge and rerun

# ------------------ demo ------------------
if __name__ == "__main__":
    np.random.seed(0)
    N = 5
    coords = np.random.rand(N, 2)
    coords = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]
    ])
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    D = coords
    tour, length = tsp_remp(D)
    print("Tour:", tour)
    print("Length:", length)
