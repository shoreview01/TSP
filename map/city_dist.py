from geopy.distance import geodesic
import numpy as np
import pandas as pd

# 기존 CSV 파일 불러오기
df_csv = pd.read_csv("map/seoul_gyeonggi_incheon_cities.csv")

# 거리 행렬 생성 (50km 이상이면 1000 부여)
N = len(df_csv)
distance_matrix = np.zeros((N, N))


for i in range(N):
    for j in range(N):
        if i == j:
            distance_matrix[i, j] = 50
        else:
            coord_i = (df_csv.loc[i, 'Latitude'], df_csv.loc[i, 'Longitude'])
            coord_j = (df_csv.loc[j, 'Latitude'], df_csv.loc[j, 'Longitude'])
            dist_km = geodesic(coord_i, coord_j).km
            distance_matrix[i, j] = dist_km if dist_km < 50 else 50
similarity_matrix = np.maximum(0, 50 - distance_matrix)

# DataFrame으로 변환
distance_df_csv = pd.DataFrame(distance_matrix, index=df_csv["City"], columns=df_csv["City"])

# 저장
output_path = "Capital_Cities_DistanceMatrix_Penalty50.csv"
distance_df_csv.to_csv(output_path)

# DataFrame으로 변환
similarity_df_csv = pd.DataFrame(similarity_matrix, index=df_csv["City"], columns=df_csv["City"])
np.fill_diagonal(similarity_df_csv.values, 0)

# 저장
output_path = "Capital_Cities_SimilarityMatrix_Penalty0.csv"
similarity_df_csv.to_csv(output_path)