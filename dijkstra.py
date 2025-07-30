import pandas as pd
import numpy as np
import networkx as nx

# 새 파일 불러오기
file_path = "Korean_Cities_DistanceMatrix_Penalty50.csv"
df = pd.read_csv(file_path, index_col=0)

# 서울이 마지막에 오도록 재정렬
def move_last(df, label):
    labels = df.index.tolist()
    labels.append(labels.pop(labels.index(label)))
    return df.loc[labels, labels]

df = move_last(df, "서울")

# 거리 행렬에서 50인 경우는 연결 안 된 것으로 처리
distance_matrix = df.replace(50, np.inf)

# 그래프 생성
G = nx.from_pandas_adjacency(distance_matrix, create_using=nx.Graph())

'''# Dijkstra 실행: 서울에서 모든 도시까지 최단 거리
lengths, paths = nx.single_source_dijkstra(G, source="서울")

# 결과 정리
distance_from_seoul = pd.DataFrame.from_dict(lengths, orient="index", columns=["Distance_from_Seoul"])
distance_from_seoul = distance_from_seoul.sort_values("Distance_from_Seoul")

print("Dijkstra Distance from Seoul (Thresholded)", distance_from_seoul)'''

length, path = nx.single_source_dijkstra(G, source="광주(경기)", target="인천")
print("거리:", length)
print("경로:", path)