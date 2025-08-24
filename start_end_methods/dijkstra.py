import numpy as np
import networkx as nx

class DijkstraShortestPath:
    def __init__(self, dist_df, start_city, end_city, max_threshold=50, verbose=False):
        self.dist_df = dist_df.copy()
        self.start_city = start_city
        self.end_city = end_city
        self.max_threshold = max_threshold
        self.verbose = verbose

        # 거리 50인 경우 연결 끊김 (무한대로 처리)
        self.distance_matrix = self.dist_df.replace(self.max_threshold, np.inf)

        # NetworkX 그래프 생성
        self.G = nx.from_pandas_adjacency(self.distance_matrix, create_using=nx.Graph())

    def run(self):
        if not nx.has_path(self.G, self.start_city, self.end_city):
            raise ValueError("No path exists under given threshold")

        length, path = nx.single_source_dijkstra(self.G, source=self.start_city, target=self.end_city)

        if self.verbose:
            print(" → ".join(path))
            print(f"Total Distance: {length:.2f}")

        return path, length
