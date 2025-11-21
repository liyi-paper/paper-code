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
lambda_values = np.logspace(0, 4, 15)[1:]  # 增加λ的个数，从 10^0 到 10^4，取 15 个点，去掉 λ=0 的情况
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


# 计算残差（Residual）
def compute_residual(u_approx, lambda_val, b_x, L, L2, p):
    # 计算右边的非线性项
    u_log_term = np.maximum(u_approx, 1e-10)
    u_log_term = np.abs(u_log_term) ** (p - 2) * u_log_term * np.log(u_log_term ** 2)

    # 计算左边的项
    left_side = L2 @ u_approx - L @ u_approx + (lambda_val * b_x + 1) * u_approx

    # 计算残差
    residual = np.linalg.norm(left_side - u_log_term)  # L2 范数残差
    return residual


# 计算每个 λ 对应的残差（误差）
residuals = []
for i, lambda_val in enumerate(lambda_values):
    residual = compute_residual(solutions[i], lambda_val, b_x, L, L2, p)
    residuals.append(residual)

# 绘制残差（误差）图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambda_values, residuals, marker='o', label="Residual", linestyle='-', color='b')
ax.set_title('Residual (Error) of the solution for λ')
ax.set_xlabel('λ')
ax.set_ylabel('Residual (L2 norm)')
ax.set_xscale('log')  # 使用对数刻度来显示 λ
ax.legend()
ax.grid(True)

# 显示图表
plt.tight_layout()
plt.show()



