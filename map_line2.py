import pandas as pd
import folium
from geopy.distance import geodesic

# --- 1. 도시 좌표 로딩 및 중복 처리 ---
df = pd.read_csv("Korean_Cities_100_with_Coordinates.csv")

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

# --- 2. 지도 초기화 ---
m = folium.Map(location=[37.5665, 126.9780], zoom_start=7)

# --- 3. 마커 추가 ---
for city, row in city_coords.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        tooltip=city
    ).add_to(m)

# --- 4. 거리 기반 연결 (100km 이하인 경우에만 선 긋기) ---
cities = city_coords.index.tolist()
for i in range(len(cities)):
    for j in range(i + 1, len(cities)):
        city_i, city_j = cities[i], cities[j]
        coord_i = city_coords.loc[city_i]
        coord_j = city_coords.loc[city_j]
        distance_km = geodesic(
            (coord_i["Latitude"], coord_i["Longitude"]),
            (coord_j["Latitude"], coord_j["Longitude"])
        ).km
        if distance_km <= 100:
            folium.PolyLine(
                locations=[
                    [coord_i["Latitude"], coord_i["Longitude"]],
                    [coord_j["Latitude"], coord_j["Longitude"]]
                ],
                color="blue",
                weight=1,
                opacity=0.5
            ).add_to(m)

# --- 5. 저장 ---
m.save("Korean_City_Distance_Links_100km.html")
