import numpy as np
import time
import matplotlib.pyplot as plt
from dummy_methods.hypercube3 import TSPHC3
from dummy_methods.hypercube2 import TSPHC2
from dummy_methods.viterbi import TSPBitmask

s = np.array([
    [0.8, 10.1, 12.5, 0.1, 0.6],
    [0.9, 0.2, 0.9, 0.4, 0.1],
    [0.1, 0.5, 0.9, 0.9, 0.8],
    [0.9, 0.9, 0.5, 0.8, 0.9],
    [0.6, 0.009, 1.8, 0.9, 0.6]
])

hypercube2 = 0
hyp2_time = 0.0
hypercube3 = 0
hyp3_time = 0.0
bitmask = 0
bitmask_time = 0.0

size = 12
N = 10

for i in range(N):
    s = np.random.uniform(0, 10, size=(size,size))
    
    start = time.time()
    solver1 = TSPHC2(s, damp=0.5, c_old=False, verbose=False)
    path1, history1 = solver1.run()
    hyp2_time += time.time() - start
    
    start = time.time()
    solver2 = TSPHC3(s, damp=0.5, c_old=False, verbose=False)
    path2, history2 = solver2.run()
    hyp3_time += time.time() - start
    
    start = time.time()
    solver3 = TSPBitmask(s, verbose=False)
    path3, history3 = solver3.run()
    bitmask_time += time.time() - start
    
    hypercube2 += history1[-1]
    hypercube3 += history2[-1]
    bitmask += history3
    
    print(f"Iter {i+1} finished")

print(f"Hypercube2 average cost: {hypercube2/N:.3f}")
print(f"Average time consumed: {hyp2_time*1000/N:.4f} ms\n")

print(f"Hypercube3 average cost: {hypercube3/N:.3f}")
print(f"Average time consumed: {hyp3_time*1000/N:.4f} ms\n")

print(f"Bitmask average cost: {bitmask/N:.3f}")
print(f"Average time consumed: {bitmask_time*1000/N:.4f} ms")
