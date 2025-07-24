import numpy as np
import time
import matplotlib.pyplot as plt
from methods.hypercube3 import TSPHC3
from methods.hypercube2 import TSPHC2
from methods.viterbi import TSPBitmask

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

size = 17
N = 1

for i in range(N):
    #s = np.random.uniform(0, 10, size=(size,size))
    
    '''start = time.time()
    solver1 = TSPHC2(s, damp=0.5, c_old=False, verbose=False)
    path1, history1 = solver1.run()
    print(history1)
    hyp2_time += time.time() - start'''
    
    start = time.time()
    solver2 = TSPHC3(s, damp=0.5, c_old=False, verbose=True)
    path2, history2 = solver2.run()
    print(history2)
    hyp3_time += time.time() - start
    
    start = time.time()
    solver3 = TSPBitmask(s, verbose=False)
    path3, history3 = solver3.run()
    print(history3)
    bitmask_time += time.time() - start
    
    '''if history1[-1] < history2[-1] and history1[-1] < history3:
        hypercube2 += 1'''
    if history2[-1] <= history3:
        hypercube3 += 1
    else:
        bitmask += 1
    
    print(f"Iter {i+1} finished")

'''print(f"Hypercube2 won {hypercube2:d} times")
print(f"Average time consumed: {hyp2_time*1000/N:.4f} ms")'''
print(f"Hypercube3 won {hypercube3:d} times")
print(f"Average time consumed: {hyp3_time*1000/N:.4f} ms")
print(f"Bitmask won {bitmask:d} times")
print(f"Average time consumed: {bitmask_time*1000/N:.4f} ms")
