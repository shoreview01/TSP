import pandas as pd
import numpy as np
import folium

# --- 1. 데이터 로딩 ---
# 도시 좌표
df = pd.read_csv("Korean_Cities_100_with_Coordinates.csv")

# 유사도 행렬
similarity_df = pd.read_csv("Korean_Cities_SimilarityMatrix_Penalty0.csv", index_col=0)

# --- 2. 중복 도시 이름 처리 ---
counts = {}
def make_unique(city):
    if city in counts:
        counts[city] += 1
        return f"{city}({counts[city]})"
    else:
        counts[city] = 1
        return city

df["City"] = df["City"].apply(make_unique)
city_coords = df.set_index("City")[["Latitude", "Longitude"]]

# similarity_df의 인덱스와 컬럼 이름도 고유 도시 이름으로 수정
similarity_df.columns = similarity_df.index = df["City"].values

# --- 3. 자기 자신 similarity = 0 ---
np.fill_diagonal(similarity_df.values, 0)

# --- 4. 지도 시각화 ---
m = folium.Map(location=[37.5665, 126.9780], zoom_start=7)

# 마커 추가
for city in similarity_df.index:
    coord = city_coords.loc[city]
    folium.Marker(
        location=[coord["Latitude"], coord["Longitude"]],
        tooltip=city
    ).add_to(m)

# 선 연결 (similarity > 0)
for i, city_i in enumerate(similarity_df.index):
    for j, city_j in enumerate(similarity_df.columns):
        if i < j and similarity_df.iloc[i, j] > 0:
            coord_i = city_coords.loc[city_i]
            coord_j = city_coords.loc[city_j]
            folium.PolyLine(
                locations=[
                    [coord_i["Latitude"], coord_i["Longitude"]],
                    [coord_j["Latitude"], coord_j["Longitude"]]
                ],
                color="red",
                weight=2,
                opacity=0.7
            ).add_to(m)

# --- 5. 저장 ---
m.save("Korean_City_Connections_Map.html")
