import numpy as np
import matplotlib.pyplot as plt
x_nodes_newton = np.array([0.4, 0.55, 0.65, 0.80, 0.90])
y_nodes_newton = np.array([0.41075, 0.57815, 0.69675, 0.88811, 1.02652])
n=len(x_nodes_newton)
diff_table=np.zeros((n,n))
diff_table[:, 0] = y_nodes_newton
for k in range(1, n):  # k是差商的阶数
    for i in range(k, n):  # 第k阶差商从第k行开始计算
        # 核心差商递推公式
        diff_table[i, k] = (diff_table[i, k-1] - diff_table[i-1, k-1]) / (x_nodes_newton[i] - x_nodes_newton[i - k])
# 提取首差商（差商表对角线元素）
coeffs_newton = np.diag(diff_table)
def newton_interp(x, x_nodes, coeffs):
    """
    牛顿差商插值计算函数
    x: 待求的自变量值（可以是单个值/数组）
    x_nodes: 已知节点的x数组
    coeffs: 差商表提取的首差商系数
    return: 插值结果
    """
    n = len(coeffs)
    result = coeffs[0]  # 初始值为0阶项f[x0]
    poly_term = 1.0  # 存储乘积项 (x-x0)(x-x1)...

    for i in range(1, n):
        poly_term *= (x - x_nodes[i - 1])  # 递推乘积项
        result += coeffs[i] * poly_term  # 累加当前阶的项
    return result
x_target_newton = 0.596
y_approx_newton = newton_interp(x_target_newton, x_nodes_newton, coeffs_newton)
print(f"例2：f({x_target_newton})的牛顿插值近似值为：{y_approx_newton:.6f}")