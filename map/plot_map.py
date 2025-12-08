import pandas as pd
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

m = 29
# CSV 불러오기: similarity 행렬 (index=도시 이름, column=도시 이름)
similarity_df = pd.read_csv("map/Capital_Cities_SimilarityMatrix_Penalty0.csv", index_col=0).iloc[:m,:m]

# 도시 위도/경도 좌표 불러오기 (index: 도시 이름)
city_coords_df = pd.read_csv("map/seoul_gyeonggi_incheon_cities.csv").iloc[:m,:m]  # columns: City, Latitude, Longitude
coord_map = dict(zip(city_coords_df["City"], zip(city_coords_df["Latitude"], city_coords_df["Longitude"])))

# 연결 정보 추출 (similarity > 0)
edges = []
for i in range(similarity_df.shape[0]):
    for j in range(i+1, similarity_df.shape[1]):
        if similarity_df.iloc[i, j] > 0:
            city1 = similarity_df.index[i]
            city2 = similarity_df.columns[j]
            if city1 in coord_map and city2 in coord_map:
                edges.append((city1, city2))
                
# 위도/경도 범위 자동 계산
lats = [lat for lat, lon in coord_map.values()]
lons = [lon for lat, lon in coord_map.values()]
lat_min, lat_max = round(min(lats))-1, round(max(lats))+1
lon_min, lon_max = round(min(lons))-1, round(max(lons))+1

# 지도 그리기
fig = plt.figure(figsize=(6, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 지도 배경 추가
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAKES, facecolor='lightblue')
ax.add_feature(cfeature.RIVERS)

# 모든 도시 점 찍기
for city, (lat, lon) in coord_map.items():
    ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())

# 연결선 그리기
for city1, city2 in edges:
    lat1, lon1 = coord_map[city1]
    lat2, lon2 = coord_map[city2]
    ax.plot([lon1, lon2], [lat1, lat2], color='gray', linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree())

# 강조 도시 텍스트 표시
for city in ["서울", "부산", "대전"]:
    if city in coord_map:
        lat, lon = coord_map[city]
        ax.plot(lon, lat, marker='o', color='red', markersize=3, transform=ccrs.PlateCarree())
        if city=="서울":
            city = "Seoul"
        elif city=="부산":
            city = "Busan"
        else:
            city = "Daejeon"
        ax.text(
            lon + 0.05, lat + 0.05,  # → 위치 조정: 오른쪽 위로 충분히 떨어뜨림
            city,
            fontsize=12,          # → 크기 키움
            fontweight='bold',    # → 진하게
            color='darkred',      # → 배경과 더 대비되는 색
            transform=ccrs.PlateCarree()
        )

plt.title("Similarity Graph of Korean Cities")
plt.show()
