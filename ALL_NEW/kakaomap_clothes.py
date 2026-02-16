import pandas as pd
import numpy as np
import requests
import folium
import json
import time
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import haversine_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 진행상황 표시용 (없으면 pip install tqdm)
from TSPSolverSOVATorch_Converge import TSPSolverSOVATorch_Converge
from TSP_NN import TSP_NearestNeighbor

# [주의] 사용자분의 기존 TSP 클래스(TSPHypercubeBCJR_SOVA, TSP_NearestNeighbor 등)가 
# 반드시 이 코드 위쪽에 선언되어 있거나 import 되어 있어야 합니다.

# =========================================================
# [설정] 카카오 API 키 입력
# =========================================================
KAKAO_API_KEY = "29cf96c3bebe9f8caec569384f45f2b4"  # 실제 키로 변경하세요

# =========================================================
# 유틸리티 함수들 (기존과 동일)
# =========================================================
def get_road_distance(start_x, start_y, end_x, end_y):
    # ... (기존 코드 유지) ...
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    params = {
        "origin": f"{start_x},{start_y}", "destination": f"{end_x},{end_y}",
        "priority": "RECOMMEND", "summary": True
    }
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        resp = requests.get(url, params=params, headers=headers)
        if resp.status_code == 200:
            routes = resp.json().get('routes')
            if routes: return routes[0]['summary']['distance']
    except Exception:
        pass
    return haversine_distance(start_y, start_x, end_y, end_x)

def get_kakao_route_path(start_x, start_y, end_x, end_y):
    # ... (기존 코드 유지) ...
    # (내용 생략 - 위와 동일)
    pass 

def haversine_distance(lat1, lon1, lat2, lon2):
    # ... (기존 코드 유지) ...
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def build_road_dist_matrix(coords_df):
    # ... (기존 코드 유지) ...
    n = len(coords_df)
    matrix = np.zeros((n, n))
    indices = coords_df.index.tolist()
    # 병렬 처리 중 print가 섞일 수 있으므로 제거하거나 최소화
    for i in range(n):
        for j in range(n):
            if i != j:
                src, dst = coords_df.iloc[i], coords_df.iloc[j]
                matrix[i][j] = get_road_distance(src['경도'], src['위도'], dst['경도'], dst['위도'])
    return pd.DataFrame(matrix, index=indices, columns=indices)

# =========================================================
# [핵심] 병렬 처리를 위한 작업 함수 정의
# =========================================================
def process_cluster(cluster_id, labels, centers, df):
    """
    하나의 클러스터에 대해 거리 행렬을 만들고 TSP를 수행하는 함수
    """
    try:
        # 해당 클러스터에 속한 인덱스 추출
        m_idxs = np.where(labels == cluster_id)[0].tolist()
        c_idx = centers[cluster_id]
        
        # 중심점을 리스트의 맨 앞으로 이동 (Head)
        if c_idx in m_idxs: 
            m_idxs.remove(c_idx)
        m_idxs.insert(0, c_idx)
        
        # 서브 데이터프레임 생성
        sub_df = df.iloc[m_idxs]
        
        # [API 호출 구간] 거리 행렬 생성
        road_mat = build_road_dist_matrix(sub_df)
        
        # [TSP 알고리즘 수행]
        # 주의: 사용자 정의 클래스(TSPHypercubeBCJR_SOVA, TSP_NearestNeighbor)가 정의되어 있어야 함
        
        # [A] SOVA
        # (TSP 클래스 정의가 코드에 포함되어 있다고 가정)
        mat_np = road_mat.to_numpy()

        p_opt, c_opt = TSPSolverSOVATorch_Converge(mat_np, verbose=True).solve()
        gp_opt = [m_idxs[x] for x in p_opt] # 전체 인덱스로 변환
        
        # [B] NN
        p_nn, c_nn = TSP_NearestNeighbor(mat_np).run()
        gp_nn = [m_idxs[x] for x in p_nn]   # 전체 인덱스로 변환
        
        return {
            'cluster_id': cluster_id,
            'head': c_idx,
            'opt': (gp_opt, c_opt),
            'nn': (gp_nn, c_nn),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error in Cluster {cluster_id}: {e}")
        return {'cluster_id': cluster_id, 'status': 'fail'}

# =========================================================
# 3. 메인 로직 실행
# =========================================================
if __name__ == "__main__":
    # [1] 데이터 로드
    print("[1] 데이터 로드...")
    try: df = pd.read_csv("서울특별시 성북구_의류수거함 현황_20240307.csv", encoding='cp949')
    except: df = pd.read_csv("서울특별시 성북구_의류수거함 현황_20240307.csv", encoding='utf-8')

    # [2] 클러스터링
    print("[2] Affinity Propagation 클러스터링...")
    coords_rad = np.radians(df[['위도', '경도']].values)
    sim = -(haversine_distances(coords_rad) * 6371000)
    pref = np.percentile(sim, 50) 
    
    af = AffinityPropagation(affinity='precomputed', preference=pref, damping=0.9, random_state=42).fit(sim)
    labels, centers = af.labels_, af.cluster_centers_indices_
    n_clusters = len(centers)
    print(f"  -> {n_clusters}개 클러스터 생성")

    # [3] 경로 계산 (병렬 처리 적용)
    print(f"[3] 경로 계산 시작 (병렬 처리: {n_clusters}개 클러스터)...")
    
    cluster_results = {}
    total_dist_sova = 0
    total_dist_nn = 0

    # max_workers: 동시에 실행할 스레드 수 (API 제한 고려하여 4~8 정도 추천)
    # 카카오 API 무료 사용량 제한이 있으니 너무 높게 설정하면 에러 날 수 있음
    MAX_WORKERS = 8 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 각 클러스터 작업을 스레드풀에 등록
        futures = [
            executor.submit(process_cluster, i, labels, centers, df) 
            for i in range(n_clusters)
        ]
        
        # 작업 완료되는 대로 결과 수집 (tqdm으로 진행바 표시)
        for future in tqdm(as_completed(futures), total=n_clusters, desc="Calculating Routes"):
            res = future.result()
            if res['status'] == 'success':
                cid = res['cluster_id']
                cluster_results[cid] = {
                    'head': res['head'], 
                    'opt': res['opt'], 
                    'nn': res['nn']
                }
                # 총 거리 합산
                total_dist_sova += res['opt'][1]
                total_dist_nn += res['nn'][1]

    # [4] Head TSP (이 부분은 데이터가 작으므로 그냥 단일 실행)
    print("[4] Head 경로 계산...")
    head_mat = build_road_dist_matrix(df.iloc[centers])
    mat_np = head_mat.to_numpy()

    # (주의: Head TSP용 클래스 이름이 TSPSolverSOVA 인지 TSPHypercubeBCJR_SOVA 인지 확인 필요)
    hp_opt, hc_opt = TSPSolverSOVATorch_Converge(mat_np).solve() # 혹은 .solve()
    hgp_opt = [centers[x] for x in hp_opt]
    total_dist_sova += hc_opt 
    
    hp_nn, hc_nn = TSP_NearestNeighbor(mat_np).run()
    hgp_nn = [centers[x] for x in hp_nn]
    total_dist_nn += hc_nn 

    # ★ 거리 출력
    print("\n" + "="*40)
    print(f" [최종 결과 비교]")
    print(f" 1. SOVA (점선) 총 이동 거리: {total_dist_sova:,.0f} m")
    print(f" 2. NN   (실선) 총 이동 거리: {total_dist_nn:,.0f} m")
    print("="*40 + "\n")
    # [5] 시각화 (도로 형상 적용)
    print("[5] 지도 생성 (도로 형상 적용)...")
    folium_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue', 'black', 'pink']
    folium_hex = {'red':'#d63e2a', 'blue':'#38aadd', 'green':'#72b026', 'purple':'#d252b9', 'orange':'#f69730', 
                  'darkred':'#a23336', 'darkblue':'#0067a3', 'cadetblue':'#436978', 'black':'#303030', 'pink':'#ff91ea'}
    
    m = folium.Map(location=[df['위도'].mean(), df['경도'].mean()], zoom_start=14)
    
    # (1) 클러스터 내부 (PolyLine 유지 - API 호출 절약)
    for i, res in cluster_results.items():
        c_name = folium_colors[i % len(folium_colors)]
        c_hex = folium_hex.get(c_name, '#333333')
        
        # NN (실선)
        nn_coords = [[df.iloc[x]['위도'], df.iloc[x]['경도']] for x in res['nn'][0]]
        folium.PolyLine(nn_coords, color=c_hex, weight=8, opacity=0.4, tooltip=f"C{i} NN").add_to(m)
        
        # SOVA (점선)
        opt_coords = [[df.iloc[x]['위도'], df.iloc[x]['경도']] for x in res['opt'][0]]
        folium.PolyLine(opt_coords, color=c_hex, weight=3, opacity=1.0, dash_array='5,5', tooltip=f"C{i} SOVA").add_to(m)

        # 마커
        for idx in res['opt'][0]:
            lat, lon = df.iloc[idx]['위도'], df.iloc[idx]['경도']
            if idx == res['head']:
                folium.Marker([lat,lon], popup=f"Head C{i}", icon=folium.Icon(color=c_name, icon='trash', prefix='fa')).add_to(m)
            else:
                folium.CircleMarker([lat,lon], radius=5, color=c_hex, fill=True, fill_color='white').add_to(m)

    # (2) Head 경로 그리기 (★도로 형상 적용★)
    print("   -> Head 경로 도로 형상 다운로드 중...")
    
    # NN Head -> 실선 (빨강/검정 등 구분을 위해 'Gray' 사용하거나 Black 유지)
    full_path_nn = []
    for k in range(len(hgp_nn)-1):
        s_idx, e_idx = hgp_nn[k], hgp_nn[k+1]
        src, dst = df.iloc[s_idx], df.iloc[e_idx]
        seg_path = get_kakao_route_path(src['경도'], src['위도'], dst['경도'], dst['위도'])
        if not seg_path: seg_path = [[src['위도'], src['경도']], [dst['위도'], dst['경도']]]
        full_path_nn.extend(seg_path)
        
    folium.PolyLine(full_path_nn, color='black', weight=8, opacity=0.4, tooltip="Head NN (Road)").add_to(m)
    
    # SOVA Head -> 점선
    full_path_opt = []
    for k in range(len(hgp_opt)-1):
        s_idx, e_idx = hgp_opt[k], hgp_opt[k+1]
        src, dst = df.iloc[s_idx], df.iloc[e_idx]
        seg_path = get_kakao_route_path(src['경도'], src['위도'], dst['경도'], dst['위도'])
        if not seg_path: seg_path = [[src['위도'], src['경도']], [dst['위도'], dst['경도']]]
        full_path_opt.extend(seg_path)

    folium.PolyLine(full_path_opt, color='black', weight=3, opacity=1.0, dash_array='5,5', tooltip="Head SOVA (Road)").add_to(m)

    m.save("성북구_의류수거함_최종_도로형상적용2.html")
    print("지도 생성 완료!")