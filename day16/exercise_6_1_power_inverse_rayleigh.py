r"""
练习 6.1：幂法、带位移反幂法与瑞利商迭代（实践 A）
====================================================
测试对象：4x4 实对称矩阵 A = H D H^T，其中 H 是向量 u=(1,2,3,4) 的
Householder 反射矩阵（正交且对称，故 H^T = H），D = diag(10, 4, 2, 1)。
因此 A 的特征值精确已知为 {10, 4, 2, 1}，特征向量为 H 的列向量。

全部运算经 fealpy.backend.backend_manager（bm）完成，同一份代码在
numpy / pytorch 两个后端各跑一次并对比验证误差。

实现内容：
  1. 幂法（返回特征值近似、归一化特征向量、迭代次数、残差历史）
  2. 带位移反幂法（每步解线性方程组 (A-mu I) y = x，不显式求逆；
     固定位移且 (A-mu I) 对称正定时可用 Cholesky 分解复用）
  3. 瑞利商迭代（记录每次迭代的瑞利商与残差）
  4. 与 bm.linalg.eigh 参考结果比较
  5. 改变初始向量 / 位移 / 谱分布，观察并解释行为变化

运行方式：
    python exercise_6_1_power_inverse_rayleigh.py
"""

import numpy as np

from fealpy.backend import backend_manager as bm

from common import (T, fnum, rayleigh, spectral_norm, eig_residual,
                    print_history, print_table, status_of)


# ============================================================
# 测试矩阵：A = H D H^T（H 为 Householder 反射矩阵）
# ============================================================
def householder_matrix(u):
    """向量 u 的 Householder 反射矩阵 H = I - 2 u u^T / (u^T u)（正交且对称）。"""
    u = T(u)
    H = bm.eye(u.shape[0], dtype=bm.float64) - 2.0 * u[:, None] * u[None, :] / bm.sum(u * u)
    return H


def build_matrix(D_diag, u=(1.0, 2.0, 3.0, 4.0)):
    """构造 A = H D H^T，特征值 = D_diag，特征向量 = H 的列。"""
    H = householder_matrix(u)
    D = bm.zeros((len(D_diag), len(D_diag)), dtype=bm.float64)
    for i, d in enumerate(D_diag):
        D[i, i] = d
    return H @ D @ H, H


# ============================================================
# 三种迭代方法
# ============================================================
def power_method(A, x0, tol=1e-12, maxit=2000, record=None):
    """
    幂法：x_{k+1} = A x_k / ||A x_k||，特征值取瑞利商。
    收敛于模最大的特征值，收敛因子约为 |lam2/lam1|。
    """
    x = bm.copy(x0)
    x = x / bm.linalg.norm(x)
    history = []
    lam = rayleigh(x, A)
    for k in range(1, maxit + 1):
        x = A @ x
        x = x / bm.linalg.norm(x)
        lam = rayleigh(x, A)
        res = eig_residual(A, lam, x)
        if record is not None:
            record(history, k, x, lam=lam, residual=res)
        if res < tol:
            return lam, x, 'converged', k, history
    return lam, x, 'maxit', maxit, history


def inverse_power(A, x0, mu=0.0, tol=1e-12, maxit=200, use_cholesky=False,
                  record=None):
    """
    带位移反幂法：每步解 (A - mu I) y = x_k，x_{k+1} = y / ||y||。
    收敛于离 mu 最近的特征值，收敛因子 |lam_near - mu| / |lam_other - mu|。

    不显式求逆：每次迭代通过线性方程组求解完成；
    固定位移时 (A - mu I) 不变，其 Cholesky 分解只需做一次（use_cholesky=True），
    之后每步只做两次三角求解，等价于复用 LU/Cholesky 分解。
    """
    n = A.shape[0]
    B = A - mu * bm.eye(n, dtype=bm.float64)
    L = bm.linalg.cholesky(B) if use_cholesky else None  # 分解只做一次

    def apply_Binv(x):
        if L is not None:
            z = bm.linalg.solve(L, x)       # 下三角回代
            return bm.linalg.solve(L.T, z)  # 上三角回代
        return bm.linalg.solve(B, x)

    x = bm.copy(x0)
    x = x / bm.linalg.norm(x)
    history = []
    lam = rayleigh(x, A)
    for k in range(1, maxit + 1):
        y = apply_Binv(x)
        x = y / bm.linalg.norm(y)
        lam = rayleigh(x, A)
        res = eig_residual(A, lam, x)
        if record is not None:
            record(history, k, x, lam=lam, residual=res)
        if res < tol:
            return lam, x, 'converged', k, history
    return lam, x, 'maxit', maxit, history


def rayleigh_quotient_iteration(A, x0, tol=1e-12, maxit=50, record=None):
    """
    瑞利商迭代：mu_k = rho(x_k)，解 (A - mu_k I) y = x_k，x_{k+1} = y / ||y||。
    对实对称矩阵局部三次收敛：残差大致按 r_{k+1} ~ C r_k^3 下降。
    当 mu_k 逼近特征值时 (A - mu_k I) 越来越接近奇异，
    但 RQI 具有自修正性质，残差仍可降到机器精度。
    """
    n = A.shape[0]
    x = bm.copy(x0)
    x = x / bm.linalg.norm(x)
    history = []
    mu = rayleigh(x, A)
    for k in range(1, maxit + 1):
        try:
            y = bm.linalg.solve(A - mu * bm.eye(n, dtype=bm.float64), x)
        except Exception:
            return mu, x, 'singular', k, history  # (A - mu I) 奇异，无法继续
        if bm.linalg.norm(y) == 0:
            return mu, x, 'singular', k, history
        x = y / bm.linalg.norm(y)
        mu = rayleigh(x, A)
        res = eig_residual(A, mu, x)
        if record is not None:
            record(history, k, x, lam=mu, residual=res)
        if res < tol:
            return mu, x, 'converged', k, history
    return mu, x, 'maxit', maxit, history


def record_lam(history, k, x, lam, residual):
    """记录一步迭代：只存 (k, lam, residual)（本练习只关心特征值与残差）。"""
    history.append({'k': k, 'lam': fnum(lam), 'residual': fnum(residual)})


# ============================================================
# 实验
# ============================================================
def experiment_matrix(bm_name):
    """报告当前后端与测试矩阵。"""
    print(f'[后端] {bm_name}')
    print('测试矩阵：A = H D H^T，H 为 u=(1,2,3,4) 的 Householder 矩阵，D = diag(10,4,2,1)')
    A, H = build_matrix([10.0, 4.0, 2.0, 1.0])
    w, V = bm.linalg.eigh(A)  # 参考解（升序）
    print('bm.linalg.eigh 参考特征值（升序）:', [f'{fnum(t):.12f}' for t in w])
    print('特征向量（H 的列）：')
    for i in range(4):
        print('  v%d ~' % (i + 1), [f'{fnum(t):.6f}' for t in H[:, i]])
    return A, H, w


def exp_power_basic(A):
    print('-' * 72)
    print('实验 1：幂法基本运行（x0 = (1,1,1,1)），应收敛到模最大的 lam = 10')
    print('-' * 72)
    x0 = T([1.0, 1.0, 1.0, 1.0])
    lam, v, status, niter, hist = power_method(A, x0, tol=1e-12, maxit=2000,
                                               record=record_lam)
    print(f'状态: {status_of(status)}，迭代 {niter} 次，lam = {fnum(lam):.14f}')
    print('特征向量 v =', [f'{fnum(t):.6f}' for t in v])
    print('最后 8 步历史（残差按 0.4 = |4/10| 每步下降）：')
    print_history(hist[-8:], cols=['k', 'lam', 'residual'], width=14)
    if len(hist) >= 4:
        r = [e['residual'] for e in hist[-4:]]
        ratio = [r[i + 1] / r[i] for i in range(len(r) - 1)]
        print('末尾相邻残差比:', [f'{t:.6f}' for t in ratio],
              '（理论 |lam2/lam1| = 0.4000）')
    return lam, niter, hist[-1]['residual']


def exp_power_eigenvector_drift(A, H):
    print('-' * 72)
    print('实验 2：初始向量恰为 lam=4 的特征向量 v2 时，漂移现象')
    print('        （理论：精确算术下永不离开 4；浮点舍入误差中 v1 分量')
    print('         按 (10/4)^k 增长，约 40 步后瑞利商跳到 10）')
    print('-' * 72)
    x0 = H[:, 1]  # 精确的特征向量
    lam, v, status, niter, hist = power_method(A, x0, tol=0.0, maxit=120,
                                               record=record_lam)
    # 只打印关键点：起点、跳变附近、终点
    marks = [1, 5, 10, 20, 30, 35, 38, 40, 42, 45, 50, 80, 120]
    rows = []
    for e in hist:
        if e['k'] in marks:
            rows.append([e['k'], f'{e["lam"]:.10f}', f'{e["residual"]:.3e}'])
    print_table(rows, header=['k', '瑞利商', '残差'])
    print('结论：浮点环境里"精确特征向量初值"只把收敛推迟约 40 步，最终仍被模最大特征值俘获。')


def exp_power_repeated(A_rep):
    print('-' * 72)
    print('实验 3：重特征值（D = diag(10,4,4,1)），幂法从不同初值出发')
    print('-' * 72)
    for name, x0 in [('x0 = (1,1,1,1)', T([1.0, 1.0, 1.0, 1.0])),
                     ('x0 = (1,-2,3,-1)', T([1.0, -2.0, 3.0, -1.0]))]:
        lam, v, status, niter, hist = power_method(A_rep, x0, tol=1e-12, maxit=2000,
                                                   record=record_lam)
        print(f'{name}: 状态 {status}，{niter} 步，lam = {fnum(lam):.14f}，'
              f'特征向量 = {[round(fnum(t), 6) for t in v]}')
    print('（重特征值不改变幂法的收敛速度：|lam2/lam1| 仍为 0.4）')

    # 重特征值下特征子空间内组合随初值变化：带位移反幂法（mu=4.1 靠近二重特征值 4）
    print('带位移反幂法 mu=4.1（靠近二重特征值 lam=4），两个不同初值：')
    for name, x0 in [('x0=(1,1,1,1)', T([1.0, 1.0, 1.0, 1.0])),
                     ('x0=(1,-2,3,-1)', T([1.0, -2.0, 3.0, -1.0]))]:
        lam, v, status, niter, hist = inverse_power(A_rep, x0, mu=4.1, tol=1e-12,
                                                    record=record_lam)
        print(f'  {name}: lam = {fnum(lam):.14f}（{niter} 步），'
              f'特征向量 = {[round(fnum(t), 6) for t in v]}')
    print('结论：两个初值都收敛到 lam=4，但收敛向量不同——它们是二重特征子空间内'
          '由 x0 决定的两个不同组合，Av=4v 都成立。重特征值使"特征向量"不再唯一。')


def exp_power_absmax():
    print('-' * 72)
    print('实验 4：绝对值最大 vs 数值最大（D = diag(4,2,1,-10)）')
    print('-' * 72)
    A_neg, _ = build_matrix([4.0, 2.0, 1.0, -10.0])
    lam, v, status, niter, hist = power_method(A_neg, T([1.0, 1.0, 1.0, 1.0]),
                                               tol=1e-12, maxit=2000,
                                               record=record_lam)
    print(f'数值最大的特征值是 4，但幂法收敛到 lam = {fnum(lam):.14f}（{niter} 步，{status}）')
    print('结论：幂法得到的是模（绝对值）最大的特征值，不是数值最大的特征值。')


def exp_inverse_power(A):
    print('-' * 72)
    print('实验 5：带位移反幂法（x0 = (1,1,1,1)）')
    print('-' * 72)
    x0 = T([1.0, 1.0, 1.0, 1.0])
    results = []
    # mu=0：模最小特征值 1；A 对称正定 → 可用 Cholesky 分解复用
    lam, v, status, niter, hist = inverse_power(A, x0, mu=0.0, tol=1e-12,
                                                use_cholesky=True, record=record_lam)
    results.append(('mu=0（Cholesky 复用）', lam, status, niter, hist[-1]['residual']))
    # mu=1.9：靠近 lam=2；理论收敛因子 |2-1.9|/|1-1.9| = 0.1111
    lam, v, status, niter, hist = inverse_power(A, x0, mu=1.9, tol=1e-12,
                                                record=record_lam)
    results.append(('mu=1.9', lam, status, niter, hist[-1]['residual']))
    # mu=3.5：靠近 lam=4；理论收敛因子 |4-3.5|/|2-3.5| = 1/3
    lam, v, status, niter, hist = inverse_power(A, x0, mu=3.5, tol=1e-12,
                                                record=record_lam)
    results.append(('mu=3.5', lam, status, niter, hist[-1]['residual']))

    print_table([[name, f'{fnum(l):.14f}', status_of(st), n, f'{r:.3e}']
                 for name, l, st, n, r in results],
                header=['位移', 'lam', '状态', '迭代', '最终残差'])

    # 收敛因子验证（mu=1.9 的例子）
    print('mu=1.9 的最后几步历史（残差比应 ≈ 0.1111 = |2-1.9|/|1-1.9|）：')
    _, _, _, _, hist19 = inverse_power(A, x0, mu=1.9, tol=1e-12, record=record_lam)
    print_history(hist19[-6:], cols=['k', 'lam', 'residual'], width=14)
    r = [e['residual'] for e in hist19[-4:]]
    ratio = [r[i + 1] / r[i] for i in range(len(r) - 1)]
    print('末尾相邻残差比:', [f'{t:.6f}' for t in ratio], '（理论 0.1111）')

    # 显式求逆 vs 解方程组：数值对比
    print('注：显式求 inv(A-mu I) 再乘向量的方式数值上等价于解一次方程组但更贵更不稳，')
    print('    且丢掉稀疏性/结构；本实现每步只调 bm.linalg.solve，从未显式计算逆矩阵。')
    return results


def exp_rqi(A):
    print('-' * 72)
    print('实验 6：瑞利商迭代（两个不同初值 → 两个不同特征值；三次收敛）')
    print('-' * 72)
    for name, x0 in [('x0=(1,1,1,1)', T([1.0, 1.0, 1.0, 1.0])),
                     ('x0=(1,-3,2,0.5)', T([1.0, -3.0, 2.0, 0.5]))]:
        mu, x, status, niter, hist = rayleigh_quotient_iteration(A, x0, tol=1e-12,
                                                                 maxit=50,
                                                                 record=record_lam)
        print(f'{name}: 状态 {status}，{niter} 步，lam = {fnum(mu):.14f}')
        print_history(hist, cols=['k', 'lam', 'residual'], width=14)
        # 三次收敛验证：r_{k+1} / r_k^3 应有界（比值趋于常数 = 三次收敛；
        # 残差进入机器精度后比值失去意义）
        r = [e['residual'] for e in hist if e['residual'] > 0]
        if len(r) >= 4:
            c1 = r[1] / r[0] ** 3 if r[0] > 0 else float('nan')
            c2 = r[2] / r[1] ** 3 if r[1] > 0 else float('nan')
            print(f'三次收敛检验：r2/r1^3 = {c1:.4f}，r3/r2^3 = {c2:.4f}'
                  '（有限常数则 r ~ C r^3，三次收敛）')
        B = A - mu * bm.eye(A.shape[0], dtype=bm.float64)
        print(f'收敛时 cond(A - mu I) = {fnum(bm.linalg.cond(B)):.3e}'
              '（矩阵近乎奇异，但残差仍到机器精度——RQI 自修正）')
    print('结论：RQI 收敛到哪个特征值由初值的瑞利商所在的吸引域决定；'
          '每步只需一次线性求解，实对称情形下局部三次收敛。')


def exp_compare_eigh(A, w):
    print('-' * 72)
    print('实验 7：三种方法与 bm.linalg.eigh 参考值总对比')
    print('-' * 72)
    x0 = T([1.0, 1.0, 1.0, 1.0])
    lam_p, _, st_p, n_p, h_p = power_method(A, x0, tol=1e-12, maxit=2000, record=record_lam)
    lam_i0, _, st_i0, n_i0, h_i0 = inverse_power(A, x0, mu=0.0, tol=1e-12, use_cholesky=True,
                                                 record=record_lam)
    lam_i19, _, st_i19, n_i19, h_i19 = inverse_power(A, x0, mu=1.9, tol=1e-12, record=record_lam)
    lam_r, _, st_r, n_r, h_r = rayleigh_quotient_iteration(A, x0, tol=1e-12, maxit=50,
                                                           record=record_lam)
    rows = [
        ['幂法', '10（模最大）', f'{fnum(lam_p):.14f}', st_p, n_p, f'{h_p[-1]["residual"]:.3e}'],
        ['反幂法 mu=0', '1（模最小）', f'{fnum(lam_i0):.14f}', st_i0, n_i0, f'{h_i0[-1]["residual"]:.3e}'],
        ['反幂法 mu=1.9', '2（靠近 mu）', f'{fnum(lam_i19):.14f}', st_i19, n_i19, f'{h_i19[-1]["residual"]:.3e}'],
        ['瑞利商迭代', '1（初值吸引域）', f'{fnum(lam_r):.14f}', st_r, n_r, f'{h_r[-1]["residual"]:.3e}'],
    ]
    print_table(rows, header=['方法', '目标特征值', '计算值', '状态', '迭代', '残差'])
    print('eigh 参考值（升序）:', [f'{fnum(t):.12f}' for t in w])


def backend_compare():
    """同一份代码在 numpy 与 pytorch 后端各跑一遍，对比验证误差。"""
    print('#' * 72)
    print('# 双后端一致性检查')
    print('#' * 72)
    saved = {}
    for name in ['numpy', 'pytorch']:
        bm.set_backend(name)
        A, H = build_matrix([10.0, 4.0, 2.0, 1.0])
        x0 = T([1.0, 1.0, 1.0, 1.0])
        lam_p, _, _, n_p, h_p = power_method(A, x0, tol=1e-12, maxit=2000, record=record_lam)
        lam_i, _, _, n_i, h_i = inverse_power(A, x0, mu=1.9, tol=1e-12, record=record_lam)
        lam_r, _, _, n_r, h_r = rayleigh_quotient_iteration(A, x0, tol=1e-12, maxit=50,
                                                            record=record_lam)
        saved[name] = dict(
            lam_p=fnum(lam_p), res_p=h_p[-1]['residual'], n_p=n_p,
            lam_i=fnum(lam_i), res_i=h_i[-1]['residual'], n_i=n_i,
            lam_r=fnum(lam_r), res_r=h_r[-1]['residual'], n_r=n_r,
            eigh=[fnum(t) for t in bm.linalg.eigh(A)[0]],
        )
    rows = []
    for name in ['numpy', 'pytorch']:
        d = saved[name]
        rows.append([name, f'{d["lam_p"]:.12f}', d['n_p'], f'{d["res_p"]:.2e}',
                     f'{d["lam_i"]:.12f}', d['n_i'], f'{d["res_i"]:.2e}',
                     f'{d["lam_r"]:.12f}', d['n_r'], f'{d["res_r"]:.2e}'])
    print_table(rows, header=['后端', '幂法 lam', '迭代', '残差', '反幂 lam(1.9)',
                              '迭代', '残差', 'RQI lam', '迭代', '残差'])
    print('eigh 参考（numpy）:', [f'{t:.12f}' for t in saved['numpy']['eigh']])
    print('eigh 参考（pytorch）:', [f'{t:.12f}' for t in saved['pytorch']['eigh']])
    bm.set_backend('numpy')  # 恢复默认后端
    return saved


def main():
    np.random.seed(20260831)  # 固定随机种子（本练习初值均为确定性向量，种子仅防意外）

    A, H, w = experiment_matrix(bm.backend_name)
    exp_power_basic(A)
    exp_power_eigenvector_drift(A, H)

    A_rep, _ = build_matrix([10.0, 4.0, 4.0, 1.0])
    exp_power_repeated(A_rep)
    exp_power_absmax()
    exp_inverse_power(A)
    exp_rqi(A)
    exp_compare_eigh(A, w)
    backend_compare()


if __name__ == '__main__':
    main()
