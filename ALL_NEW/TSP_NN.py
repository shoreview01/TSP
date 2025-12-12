import numpy as np
import pandas as pd

class TSP_NearestNeighbor:
    """
    단순하고 빠르지만 최적해는 보장하지 않는 탐욕(Greedy) 알고리즘
    비교군(Baseline)으로 사용합니다.
    """
    def __init__(self, dist_matrix):
        self.dist = dist_matrix
        if isinstance(dist_matrix, pd.DataFrame):
            self.n = len(dist_matrix)
        else:
            self.n = dist_matrix.shape[0]

    def run(self):
        if self.n <= 1: return [0, 0], 0
            
        unvisited = set(range(1, self.n))
        current = 0
        path = [0]
        total_cost = 0
        
        get_dist = lambda i, j: self.dist.iloc[i, j] if isinstance(self.dist, pd.DataFrame) else self.dist[i][j]

        while unvisited:
            next_node = min(unvisited, key=lambda x: get_dist(current, x))
            total_cost += get_dist(current, next_node)
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        total_cost += get_dist(current, 0)
        path.append(0)
        return path, total_cost