import pandas as pd
import numpy as np
import time
from start_end_methods.start_end_viterbi import TSPBitmaskStartEnd
from start_end_methods.dijkstra import DijkstraShortestPath

np.set_printoptions(precision=3, suppress=True)

# Reload the distance matrix file
file_path = "map/Capital_Cities_DistanceMatrix_Penalty50.csv"
dist_df = pd.read_csv(file_path, index_col=0)
dist_df = dist_df.iloc[:15,:15]
start_city="서울"
end_city="연천"

start = time.time()
solver = TSPBitmaskStartEnd(dist_df, start_city=start_city,end_city=end_city,verbose=True)
path, cost = solver.run()
end = time.time()
print(f"Time elasped: {(end-start)*1000:.3f} ms")

start = time.time()
solver2 = DijkstraShortestPath(dist_df, start_city=start_city,end_city=end_city,verbose=True)
path2, cost2 = solver2.run()
end = time.time()
print(f"Time elasped: {(end-start)*1000:.3f} ms")