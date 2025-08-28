import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

N, T = 4, 4
xgap, ygap = 2.2, 1.6
y_top = N + 1.6
y_left = N + 0.5
y_c, y_s = -0.2, -1.0

# 핵심 파라미터: 원본처럼 비스듬하게 만드는 값들
row_tilt  = 0.20   # 행별 가로선 기울기(열이 증가할수록 y가 내려가게)
col_slant = 0.35   # 세로선 x 쏠림(아래로 갈수록 x가 오른쪽/왼쪽으로 이동)
col_jit   = [0.10, 0.00, -0.08, 0.06]  # 열별 미세 흔들림(자연스러움)

G = nx.DiGraph()
for t in range(1, T+1): G.add_node(f"E{t}", kind="E")
for i in range(1, N+1): G.add_node(f"I{i}", kind="I")
for i,t in product(range(1,N+1), range(1,T+1)): G.add_node(f"b{i}{t}", kind="b")
for t in range(1, T+1):
    G.add_node(f"c{t}", kind="c"); G.add_node(f"S{t}", kind="S")

# 에지
for t in range(1, T+1):
    for i in range(1, N+1):
        G.add_edge(f"E{t}", f"b{i}{t}", et="vert")
for i in range(1, N+1):
    for t in range(1, T+1):
        G.add_edge(f"I{i}", f"b{i}{t}", et="horiz")
for t in range(1, T+1):
    G.add_edge(f"b1{t}", f"c{t}", et="down")
    G.add_edge(f"c{t}", f"S{t}", et="beta")
for t in range(1, T):
    G.add_edge(f"c{t}", f"c{t+1}", et="delta")

# ===== 좌표: '기울어진 격자' 만들기 =====
pos = {}
mid_col = (T+1)/2
mid_row = (N+1)/2
xs = {t: t * xgap for t in range(1, T+1)}

# 위 E_t (열마다 약간 y 흔들림)
for t in range(1, T+1):
    pos[f"E{t}"] = (xs[t], y_top + col_jit[(t-1) % len(col_jit)])

# 왼 I_i
for i in range(1, N+1):
    pos[f"I{i}"] = (0.3, y_left - i)

# b_{it}: 열 진행에 따라 행마다 y를 기울이고( row_tilt ),
#         행이 아래로 갈수록 세로선 x를 약간 쏠리게( col_slant )
for i,t in product(range(1,N+1), range(1,T+1)):
    base_y = (N + 0.5 - i)
    y = base_y - row_tilt * (t - mid_col) + col_jit[(t-1) % len(col_jit)]
    x = xs[t] + col_slant * (i - mid_row) * 0.25  # 행별 x 쏠림
    pos[f"b{i}{t}"] = (x, y)

# c_t, S_t도 같은 기울기/쏠림을 따라가게
for t in range(1, T+1):
    yoff = - row_tilt * (t - mid_col) + col_jit[(t-1) % len(col_jit)]
    xoff = col_slant * (0 - mid_row) * 0.25
    pos[f"c{t}"] = (xs[t] + xoff, y_c + yoff)
    pos[f"S{t}"] = (xs[t] + xoff, y_s + yoff)

# ===== 그리기 =====
fig, ax = plt.subplots(figsize=(8, 6)); ax.set_axis_off()

node_groups = {
    "E": dict(node_color="#4CA3A3", node_shape="s", node_size=600),
    "I": dict(node_color="#F29E4C", node_shape="s", node_size=600),
    "b": dict(node_color="white", edgecolors="black", linewidths=1.2, node_size=700),
    "c": dict(node_color="white", edgecolors="black", linewidths=1.2, node_size=700),
    "S": dict(node_color="#BB5A7D", node_shape="s", node_size=600),
}

# 에지 먼저
def draw_edges(et, color, width=1.8, arrows=False, rad=0.0, alpha=1.0):
    es = [(u,v) for u,v,d in G.edges(data=True) if d["et"] == et]
    if not es: return
    nx.draw_networkx_edges(G, pos, edgelist=es, edge_color=color, width=width,
                           arrows=arrows, arrowsize=12,
                           connectionstyle=f"arc3,rad={rad}", alpha=alpha, ax=ax)

draw_edges("horiz", "#F29E4C", width=2.2, rad=0.04, alpha=0.95)  # 살짝 기울어진 주황 띠
draw_edges("vert",  "#3AA6A6", width=2.0, rad=0.0,  alpha=0.9)   # 모아지는 청록
draw_edges("down",  "black",   width=1.2, arrows=True)
draw_edges("beta",  "#BB5A7D", width=2.0, arrows=True)
draw_edges("delta", "#BB5A7D", width=2.0, arrows=True, rad=0.22)  # c_t 곡선

# 노드
for kind, style in node_groups.items():
    nlist = [n for n,d in G.nodes(data=True) if d["kind"]==kind]
    nx.draw_networkx_nodes(G, pos, nodelist=nlist, **style, ax=ax)

# 라벨
labels = {}
labels.update({f"E{t}": rf"$E_{t}$" for t in range(1,T+1)})
labels.update({f"I{i}": rf"$I_{i}$" for i in range(1,N+1)})
labels.update({f"c{t}": rf"$c_{t}$" for t in range(1,T+1)})
labels.update({f"S{t}": rf"$S_{t}$" for t in range(1,T+1)})
labels.update({f"b{i}{t}": rf"$b_{{{i}{t}}}$" for i,t in product(range(1,N+1),range(1,T+1))})
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)

plt.tight_layout(); plt.show()
