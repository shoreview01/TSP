import numpy as np
from methods.original import TSPMaxSum
from methods.hypercube import TSPHC1
from methods.hypercube2 import TSPHC2
#from methods.hypercube3 import TSPHC3
from methods.viterbi import TSPBitmask
import pandas as pd

np.set_printoptions(precision=3, suppress=True)


'''# CSV 파일 불러오기
file_path = "map/Korean_Cities_DistanceMatrix_Penalty50.csv"
df = pd.read_csv(file_path, index_col=0)

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

solver = TSPHC2(dist, verbose=True)
path, cost = solver.run()
print(path)
print(cost)

solver = TSPMaxSum(dist, verbose=True)
path, cost = solver.run()'''

import folium
import pandas as pd

# 도시 좌표 불러오기 (예시 CSV 또는 DataFrame)
df = pd.read_csv("map/Korean_Cities_100_with_Coordinates.csv")  # 'City', 'Latitude', 'Longitude' 컬럼 포함

# 경로 인덱스
path = [1, 33, 16, 37, 52, 41, 50, 30, 17, 38, 3, 19, 36, 10, 51, 69, 84, 83, 85,
        28, 86, 43, 82, 31, 75, 76, 74, 60, 56, 40, 13, 7, 62, 5, 67, 27, 48, 14, 55,
        77, 58, 4, 99, 98, 1, 32, 22, 12, 90, 24, 44, 81, 80, 57, 2, 29, 92, 89, 91,
        88, 87, 6, 47, 21, 65, 93, 59, 94, 66, 95, 8, 11, 45, 23, 15, 26, 73, 72, 49,
        61, 9, 39, 53, 18, 71, 70, 25, 20, 34, 54, 97, 64, 63, 35, 68, 46, 79, 42, 78,
        96, 1]

# 지도 생성
m = folium.Map(location=[36.5, 127.5], zoom_start=7)

# 도시 좌표 리스트 만들기
locations = []
for i in path:
    city = df.iloc[i-1]  # 인덱스 1부터 시작했으므로 -1
    locations.append((city['Latitude'], city['Longitude']))
    folium.CircleMarker(location=(city['Latitude'], city['Longitude']),
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.7).add_to(m)

# 경로 선 추가
folium.PolyLine(locations, color='red', weight=3).add_to(m)

# 저장
m.save("Korean_City_Connections_with_Path.html")
