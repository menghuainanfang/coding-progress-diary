r"""
练习 6.3：奇异值分解 SVD（实践 C）
====================================
对非方阵（8x5）完成：
  1. bm.linalg.svd 计算 SVD，明确记录 U、奇异值、V^T 因子的形状（full / reduced）
  2. 左、右奇异向量的正交性检查 + 相对重构误差
  3. 不同截断阶数 k 的低秩近似：截断阶数、存储量、近似误差（Eckart-Young 验证）
  4. 2-范数条件数 kappa_2 = s_max / s_min；构造小奇异值矩阵展示
     "最小奇异值接近零 → 条件数爆炸 → 解对扰动极其敏感"
  5. SVD 与 A^T A 特征分解的数学联系，以及"先形成 A^T A"的数值危害

全部运算经 fealpy.backend.backend_manager（bm）完成，同一份代码在
numpy / pytorch 两个后端各跑一次并对比验证误差。

运行方式：
    python exercise_6_3_svd.py
"""

import numpy as np

from fealpy.backend import backend_manager as bm

from common import T, fnum, print_table
from exercise_6_2_qr_decomposition_iteration import householder_qr


def fro_norm(M):
    """Frobenius 范数（后端无关写法）。"""
    return bm.sqrt(bm.sum(M * M))


def rel_recon_err(A, U, s, Vt):
    """相对重构误差 ||A - U diag(s) Vt||_F / ||A||_F（(U * s) @ Vt 用广播实现）。
    U 为 full（m x m）时只取前 k=len(s) 列。"""
    k = s.shape[0]
    Ar = (U[:, :k] * s[None, :]) @ Vt
    return fnum(fro_norm(A - Ar) / fro_norm(A))


def orth_err_full(U, ncols=None):
    """U^T U - I 的 Frobenius 范数（U 为列正交矩阵时）。"""
    k = ncols if ncols is not None else U.shape[1]
    return fnum(fro_norm(U[:, :k].T @ U[:, :k] - bm.eye(k, dtype=U.dtype)))


def low_rank(U, s, Vt, k):
    """截断 SVD 低秩近似 A_k = U[:,:k] diag(s[:k]) Vt[:k,:]。"""
    return (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]


def ill_conditioned_matrix(m=8, n=5, smin=1e-7):
    """
    构造 8x5 病态矩阵 A = U0 diag(10, 3, 1, 0.5, smin) V0^T，
    U0、V0 由固定向量的 Householder QR 给出（无随机性，可复现）。
    kappa_2(A) = 10 / smin。
    """
    # 用确定性矩阵的 QR 得到正交因子（无随机性，可复现）
    M_u = T([[float((i + 1) * (j + 1) + (i == j) * 7.0) for j in range(n)]
             for i in range(m)])
    M_v = T([[float((i + 2) * (j + 1) + (i == j) * 5.0) for j in range(n)]
             for i in range(n)])
    U0, _ = householder_qr(M_u)
    V0, _ = householder_qr(M_v)   # V0: (n, n) 正交
    s = T([10.0, 3.0, 1.0, 0.5, smin])
    A = (U0 * s[None, :]) @ V0.T
    return A, U0, V0, s


# ============================================================
# 实验
# ============================================================
def exp_svd_shapes():
    print('-' * 72)
    print('实验 1：SVD 因子形状与正交性、重构误差（8x5 良态随机矩阵）')
    print('-' * 72)
    rng = np.random.default_rng(20260831)  # 固定随机种子，实验可复现
    A = T(rng.standard_normal((8, 5)))
    print(f'A 形状 = {tuple(A.shape)}（m=8 行 > n=5 列）')

    U, s, Vt = bm.linalg.svd(A, full_matrices=True)
    print(f'full SVD:  U 形状 = {tuple(U.shape)}，s 形状 = {tuple(s.shape)}，'
          f'Vt 形状 = {tuple(Vt.shape)}（U 为 8x8 正交方阵）')
    Ur, sr, Vtr = bm.linalg.svd(A, full_matrices=False)
    print(f'reduced SVD: U 形状 = {tuple(Ur.shape)}，s 形状 = {tuple(sr.shape)}，'
          f'Vt 形状 = {tuple(Vtr.shape)}（U 为 8x5 列正交）')

    print('正交性检查（应接近机器精度 ~1e-15）：')
    rows = [
        ['full U：||U^T U - I||_F', orth_err_full(U)],
        ['reduced U：||U^T U - I||_F', orth_err_full(Ur)],
        ['V：||V V^T - I||_F', orth_err_full(Vt.T)],
    ]
    print_table([[a, f'{b:.2e}'] for a, b in rows], header=['检查项', '值'])
    print(f'相对重构误差 ||A - U diag(s) Vt||_F / ||A||_F = {rel_recon_err(A, U, s, Vt):.2e}')

    # SVD 与 A^T A 特征分解的数学联系（良态情形）
    w = bm.linalg.eigh(A.T @ A)[0]          # A^T A 特征值（升序）
    w = bm.sort(w)[::-1]                     # 降序，与奇异值排序对应
    print('数学联系：sigma_i(A) = sqrt(lam_i(A^T A))（良态情形下逐位吻合）：')
    print_table([[f'{fnum(s[i]):.10f}', f'{fnum(bm.sqrt(w[i])):.10f}',
                  f'{abs(fnum(s[i]) - fnum(bm.sqrt(w[i]))):.2e}']
                 for i in range(5)],
                header=['奇异值 s_i', 'sqrt(lam_i(A^T A))', '差'])
    return A, U, s, Vt


def exp_low_rank(A, U, s, Vt):
    print('-' * 72)
    print('实验 2：截断 SVD 低秩近似（k = 1, 2, 3, 4 与满秩对比）')
    print('-' * 72)
    m, n = A.shape
    afro = fnum(fro_norm(A))
    rows = []
    for k in [1, 2, 3, 4, 5]:
        Ak = low_rank(U, s, Vt, k)
        err = fnum(fro_norm(A - Ak) / afro)
        # Eckart-Young：最优 k 秩近似误差 = sqrt(被丢弃奇异值的平方和)
        theory = fnum(bm.sqrt(bm.sum(s[k:] ** 2)) / afro)
        storage = k * (m + n + 1)
        rows.append([k, f'{storage} ({k}*{m+n+1})', f'{err:.3e}',
                     f'{theory:.3e}', f'{storage / (m * n):.2f}'])
    print_table(rows, header=['截断阶数 k', '存储量', '||A-A_k||_F/||A||_F',
                              '理论 sqrt(sum_{i>k} s_i^2)/||A||_F', '存储/全矩阵'])
    print('Eckart-Young 定理验证：实测误差与理论值一致；误差完全由被丢弃的奇异值决定。')
    print('注意：8x5 这种小矩阵上低秩近似的存储并不划算（k(m+n+1) vs mn），')
    print('      收益来自大矩阵（如 1000x1000 时 k=10 只用 10*(1000+1000+1)≈2 万个')
    print('      数 vs 100 万个）。')
    print('截断损失的信息：小奇异值对应的"细节"方向——数值上损失 ||A-A_k||_F，'
          '意义上丢掉方差/能量最小的那些方向（与主成分分析 PCA 同一个原理）。')


def exp_ill_conditioned():
    print('-' * 72)
    print('实验 3：小奇异值 → 条件数爆炸 → 解对扰动极其敏感')
    print('-' * 72)
    smin = 1e-8
    A, U0, V0, s0 = ill_conditioned_matrix(m=8, n=5, smin=smin)
    print(f'构造 A = U0 diag(10, 3, 1, 0.5, 1e-8) V0^T，奇异值 = '
          f'{[fnum(t) for t in bm.linalg.svd(A, full_matrices=False)[1]]}')
    kappa = fnum(bm.linalg.cond(A))
    print(f'2-范数条件数 kappa_2(A) = s_max/s_min = {kappa:.3e}'
          f'（s_min = 1e-8 接近零 → 条件数 ~1e9）')

    # 解对右端扰动的敏感性：b 与 b + db，db 沿最小奇异值方向
    x_true = T([1.0, -1.0, 2.0, 0.5, 3.0])
    b = A @ x_true
    b = b / bm.linalg.norm(b)                     # ||b|| = 1
    n = A.shape[1]
    U_full = bm.linalg.svd(A, full_matrices=True)[0]
    u_min = U_full[:, n - 1]   # 第 n 列才是最小奇异值的左奇异向量
                               # （m > n 时 full U 的后 m-n 列对应零奇异值，不能用！）
    db = 1e-6 * u_min                             # ||db|| = 1e-6
    x = bm.linalg.lstsq(A, b)[0]
    x_pert = bm.linalg.lstsq(A, b + db)[0]
    dx_norm = fnum(bm.linalg.norm(x_pert - x))
    rel_dx = dx_norm / fnum(bm.linalg.norm(x))
    print(f'右端相对扰动 ||db||/||b|| = 1e-6，解的相对变化 ||dx||/||x|| = {rel_dx:.3e}')
    print(f'放大倍数 ≈ {rel_dx / 1e-6:.3e}（理论上界 kappa_2 = {kappa:.3e}）')
    print('解释：db 沿最小奇异值方向 u_min 时，dx = (1/s_min) * db 方向被放大——')
    print('      小奇异值方向（v_min）是"数据扰动最敏感的方向"。')

    # SVD vs A^T A：病态情形下先形成 A^T A 的危害
    w_ata = bm.linalg.eigh(A.T @ A)[0]            # 升序
    s_asc = sorted(fnum(t) for t in s0)           # 真奇异值升序，与 w_ata 配对
    print('\n"A^T A 特征分解当 SVD"的危害（同一矩阵，s_min = 1e-8）：')
    rows = []
    for i in range(5):
        wv = fnum(w_ata[i])
        rows.append([f'{s_asc[i]:.3e}', f'{wv:.3e}',
                     f'{fnum(bm.sqrt(w_ata[i])):.3e}' if wv >= 0 else '负数→无法开方'])
    print_table(rows, header=['真奇异值 s_i（升序）', 'eig(A^T A) 计算值', 'sqrt(计算值)'])
    print('解释：s_min^2 = 1e-16 而最大特征值 ~100，舍入噪声 ~eps*||A^T A|| ~ 2e-14，')
    print('      最小特征值被噪声吞没（相对误差 O(1)，甚至算成负数）。')
    print('结论：数学上 sigma_i = sqrt(lam_i(A^T A)) 成立，但数值上"先形成 A^T A"')
    print('      把条件数平方（1e9 → 1e18），通用 SVD 算法必须直接在 A 上工作。')


def backend_compare():
    print('#' * 72)
    print('# 双后端一致性检查（SVD）')
    print('#' * 72)
    saved = {}
    for name in ['numpy', 'pytorch']:
        bm.set_backend(name)
        rng = np.random.default_rng(20260831)
        A = T(rng.standard_normal((8, 5)))
        U, s, Vt = bm.linalg.svd(A, full_matrices=True)
        A2, U0, V0, s0 = ill_conditioned_matrix(m=8, n=5, smin=1e-7)
        U2, s2, Vt2 = bm.linalg.svd(A2, full_matrices=True)
        saved[name] = dict(
            recon=rel_recon_err(A, U, s, Vt),
            orthU=orth_err_full(U),
            orthV=orth_err_full(Vt.T),
            kappa=fnum(bm.linalg.cond(A2)),
            smin=fnum(s2[-1]),
            s_ill=[fnum(t) for t in s2],
        )
    rows = []
    for name in ['numpy', 'pytorch']:
        d = saved[name]
        rows.append([name, f'{d["recon"]:.2e}', f'{d["orthU"]:.2e}',
                     f'{d["orthV"]:.2e}', f'{d["kappa"]:.3e}', f'{d["smin"]:.3e}'])
    print_table(rows, header=['后端', '||A-USVt||_F/||A||_F', '||U^T U-I||_F',
                              '||VV^T-I||_F', 'kappa_2(病态A)', 's_min'])
    print('病态矩阵奇异值（numpy）:', [f'{t:.3e}' for t in saved['numpy']['s_ill']])
    print('病态矩阵奇异值（pytorch）:', [f'{t:.3e}' for t in saved['pytorch']['s_ill']])
    bm.set_backend('numpy')


def main():
    print('[后端]', bm.backend_name)
    A, U, s, Vt = exp_svd_shapes()
    exp_low_rank(A, U, s, Vt)
    exp_ill_conditioned()
    backend_compare()


if __name__ == '__main__':
    main()
