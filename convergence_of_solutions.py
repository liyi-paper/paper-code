import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.optimize import root
import matplotlib.pyplot as plt

# 创建 5x5 的格点图
n = 5
grid_graph = nx.grid_2d_graph(n, n)

# 映射二维坐标到节点索引
node_to_index = {node: idx for idx, node in enumerate(grid_graph.nodes)}
b_x = np.zeros(len(grid_graph.nodes))  # 初始化所有节点 b(x) = 0

# 最外圈节点索引
outer_nodes = set()
for i in range(n):
    outer_nodes.add((0, i))  # 第一行
    outer_nodes.add((n - 1, i))  # 最后一行
    outer_nodes.add((i, 0))  # 第一列
    outer_nodes.add((i, n - 1))  # 最后一列

# 为最外圈节点赋值 b(x) = 1
for node in outer_nodes:
    b_x[node_to_index[node]] = 1

# 生成邻接矩阵和度矩阵
adj_matrix = nx.adjacency_matrix(grid_graph).toarray()  # 使用 .toarray() 以避免警告
degree_matrix = np.diag(np.sum(adj_matrix, axis=1))  # 度矩阵
laplacian_matrix = degree_matrix - adj_matrix  # 拉普拉斯矩阵
L = csr_matrix(laplacian_matrix)  # 转换为稀疏矩阵
L2 = L @ L  # 双拉普拉斯算子

# 定义非线性方程
def equation(u, lambda_val, b_x, L, L2, p):
    u_log_term = np.maximum(u, 1e-10)  # 防止除零错误
    u_log_term = np.abs(u_log_term) ** (p - 2) * u_log_term * np.log(u_log_term ** 2)  # 计算对数项
    return L2 @ u - L @ u + (lambda_val * b_x + 1) * u - u_log_term

# 初始化解
u_init = np.ones(len(b_x))

# 设置参数
p = 4  # 非线性项的幂次
lambda_values = np.logspace(0, 4, 8)  # λ 的值从 10^0 到 10^4，取6个点
tol = 1e-6  # 收敛容差
max_iter = 1000  # 最大迭代次数

# 存储每个 λ 对应的解
solutions = []

# 对每个 λ 值求解
for lambda_val in lambda_values:
    result = root(
        equation,
        u_init,
        args=(lambda_val, b_x, L, L2, p),
        tol=tol,
        method='hybr',  # 使用混合方法
    )
    if result.success:
        solutions.append(result.x)
        u_init = result.x  # 将当前解作为下一个解的初值
    else:
        print(f"Solver failed for lambda = {lambda_val}")
        solutions.append(np.nan * np.ones(len(b_x)))  # 如果求解失败，记录为 NaN

# 打印每个 λ 对应的 5x5 结果
for i, lambda_val in enumerate(lambda_values):
    print(f"\nλ = {lambda_val:.1e}")
    print(solutions[i].reshape(n, n))  # 以 5x5 形式打印

# 绘制 u(x) 随 λ 变化的曲线
fig, ax = plt.subplots(figsize=(10, 6))  # 仅创建一个图表

# 线性趋势图：折线图显示解的变化
for i, lambda_val in enumerate(lambda_values):
    ax.plot(range(1, len(grid_graph.nodes) + 1), solutions[i], marker='o', label=f'λ = {lambda_val:.1e}')

# 设置x轴标签为1到25
plt.xticks(ticks=range(1, 26), labels=range(1, 26))

# 设置每5个节点显示一个竖线
ax.xaxis.set_major_locator(plt.MultipleLocator(5))

# 保持横向网格线
ax.grid(True, which='both')  # 显示所有网格线（横竖线）

# 设置标题和标签
ax.set_title('Solution for λ over Node Index')
ax.set_xlabel('Node index')
ax.set_ylabel('u(x)')
ax.legend()

# 显示网格，竖线间隔每5个节点
ax.grid(True, which='both', axis='x')

# 显示图表
plt.tight_layout()
plt.show()
