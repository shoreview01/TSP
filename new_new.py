import numpy as np

dist = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]
])
N = dist.shape[0]-1
cities = list(range(N))
similarity = np.max(dist) - dist
trellis = {}
for i in cities:
    trellis[(1 << i, i)] = (similarity[N][i], N) 

for visited in range(1 << N):
    for u in cities:
        if not (visited & (1 << u)): # visited에 u가 포함돼있지 않으면 skip
            continue
        for v in cities:
            if visited & (1 << v):
                continue
            new_visited = visited | (1 << v)
            prev_similarity = trellis.get((visited, u), (-np.inf, -1))[0]
            new_similarity = prev_similarity + similarity[u][v]
            if (new_visited, v) not in trellis or new_similarity > trellis[(new_visited, v)][0]:
                trellis[(new_visited, v)] = (new_similarity, u)

end_mask = (1 << N) - 1
min_similarity = -np.inf
last_city = -1
for u in cities:
    sim_to_depot = trellis.get((end_mask, u), (-np.inf, -1))[0] + similarity[u][N]
    if sim_to_depot > min_similarity:
        min_similarity = sim_to_depot
        last_city = u
print(last_city)
print(trellis)

# Reconstruct path
tour = [N]  # start from depot
mask = end_mask
curr = last_city
for _ in range(N):
    tour.append(curr)
    mask, curr = mask ^ (1 << curr), trellis[(mask, curr)][1]
tour.append(N)  # end at depot
tour.reverse()
print([x+1 for x in tour])
print(min_similarity)