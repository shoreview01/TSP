import numpy as np
from dummy_methods.original import TSPMaxSum
from dummy_methods.hypercube import TSPHC1
from dummy_methods.hypercube2 import TSPHC2
#from methods.hypercube3 import TSPHC3
from dummy_methods.viterbi import TSPBitmask
import pandas as pd

np.set_printoptions(precision=3, suppress=True)


# CSV 파일 불러오기
file_path = "map/Korean_Cities_DistanceMatrix_Penalty50.csv"
df = pd.read_csv(file_path, index_col=0)
#df = df.iloc[:16,:16]

# 서울이 현재 어디에 있는지 찾기
seoul_index = df.index.get_loc("서울")

# 서울을 마지막으로 옮기기
def move_last(df, label):
    labels = df.index.tolist()
    labels.append(labels.pop(labels.index(label)))
    return df.loc[labels, labels]

df_reordered = move_last(df, "서울")

dist = df_reordered.to_numpy()
N = dist.shape[0]-1
dist += np.eye(N+1)*50
print(dist)

'''solver = TSPHC2(dist, verbose=True)
path, cost = solver.run()
print(path)
print(cost)'''

'''solver = TSPMaxSum(dist, verbose=True)
path, cost = solver.run()'''

solver = TSPBitmask(dist, max=50, verbose=True)
path, cost = solver.run()