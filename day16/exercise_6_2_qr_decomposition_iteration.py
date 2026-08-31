r"""
练习 6.2：QR 分解与 QR 特征值迭代（实践 B）
============================================
所选 QR 分解方法：Householder 变换（约化 QR）。
Householder 正交性达到机器精度（~1e-16），数值稳定；同时实现
经典 Gram-Schmidt（CGS）与改进 Gram-Schmidt（MGS）作对比，
直观展示三者正交性的差异。

全部运算经 fealpy.backend.backend_manager（bm）完成，同一份代码在
numpy / pytorch 两个后端各跑一次并对比验证误差。

实现内容：
  1. Householder QR / CGS / MGS 三种 QR 分解
  2. 矩形矩阵的重构误差 ||A-QR||_F/||A||_F 与正交性误差 ||Q^T Q - I||_F，
     并与 bm.linalg.qr 数值比较（注意：Q 列符号可任意，不能逐元比 Q）
  3. Lauchli 病态矩阵上三种方法正交性的差异
  4. QR 分解解最小二乘问题
  5. 实对称矩阵上的基本 QR 特征值迭代（无位移）+ Wilkinson 位移 QR，
     记录非对角范数，与 bm.linalg.eigh（先排序）对比

运行方式：
    python exercise_6_2_qr_decomposition_iteration.py
"""

import numpy as np

from fealpy.backend import backend_manager as bm

from common import (T, fnum, spectral_norm, off_norm,
                    print_table, status_of)
from exercise_6_1_power_inverse_rayleigh import build_matrix


# ============================================================
# 三种 QR 分解
# ============================================================
def householder_qr(A):
    """
    Householder 约化 QR：对列 j 构造反射 H_j = I - beta v v^T 把
    A[j:, j] 消成 alpha e1，左乘到 A 的尾块，右乘累积到 Q。
    返回 (Q, R)，m >= n 时 Q: (m, n) 列正交，R: (n, n) 上三角。
    """
    A = bm.copy(A)
    m, n = A.shape
    Q = bm.eye(m, dtype=A.dtype)
    for j in range(min(m - 1, n)):
        x = bm.copy(A[j:, j])
        nx = bm.linalg.norm(x)
        if nx == 0:
            continue
        alpha = -bm.sign(x[0]) * nx
        v = bm.copy(x)
        v[0] = v[0] - alpha
        beta = 2.0 / bm.sum(v * v)
        # A := H A，只作用于尾块 A[j:, j:]
        w = A[j:, j:].T @ v                 # (n-j,)，w = A^T v
        A[j:, j:] = A[j:, j:] - beta * v[:, None] * w[None, :]
        # Q := Q H，只作用于 Q[:, j:]
        q = Q[:, j:] @ v                    # (m,)
        Q[:, j:] = Q[:, j:] - beta * q[:, None] * v[None, :]
    R = A[:n, :n]
    Q = Q[:, :n]
    return Q, R


def cgs(A):
    """经典 Gram-Schmidt：r_ij = q_i^T a_j，所有投影系数用旧基一次算完。"""
    m, n = A.shape
    Q = bm.zeros((m, n), dtype=A.dtype)
    R = bm.zeros((n, n), dtype=A.dtype)
    for j in range(n):
        v = bm.copy(A[:, j])
        for i in range(j):
            R[i, j] = bm.sum(Q[:, i] * A[:, j])   # 用原始列 a_j
        v = v - Q[:, :j] @ R[:j, j]
        R[j, j] = bm.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def mgs(A):
    """改进 Gram-Schmidt：每算出一个分量立即从 v 中减去。"""
    m, n = A.shape
    Q = bm.zeros((m, n), dtype=A.dtype)
    R = bm.zeros((n, n), dtype=A.dtype)
    for j in range(n):
        v = bm.copy(A[:, j])
        for i in range(j):
            R[i, j] = bm.sum(Q[:, i] * v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = bm.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def recon_err(A, Q, R):
    """重构误差 ||A - Q R||_F / ||A||_F。"""
    return fnum(bm.sqrt(bm.sum((A - Q @ R) ** 2)) / bm.sqrt(bm.sum(A ** 2)))


def orth_err(Q):
    """正交性误差 ||Q^T Q - I||_F。"""
    m, n = Q.shape
    return fnum(bm.sqrt(bm.sum((Q.T @ Q - bm.eye(n, dtype=Q.dtype)) ** 2)))


# ============================================================
# QR 特征值迭代
# ============================================================
def wilkinson_shift(A):
    """对称矩阵尾部 2x2 块的 Wilkinson 位移：取更靠近 A[n-1,n-1] 的那个特征值。"""
    n = A.shape[0]
    a, b, c = fnum(A[n - 2, n - 2]), fnum(A[n - 1, n - 1]), fnum(A[n - 2, n - 1])
    delta = (a - b) / 2.0
    t = abs(delta) + (delta ** 2 + c ** 2) ** 0.5
    mu = b - (c ** 2 / t) * (1.0 if delta >= 0 else -1.0) if t > 0 else b
    return mu


def qr_iter(A, tol=1e-12, maxit=5000, shift='none', record=None):
    """
    基本 QR 特征值迭代：A_k = Q_k R_k，A_{k+1} = R_k Q_k（+ mu I 若带位移）。
    A_{k+1} = Q_k^T A_k Q_k 与 A_k 相似，特征值不变；
    对实对称矩阵 A_k 趋于对角阵，收敛监视量为非对角范数 off(A_k)。
    shift: 'none'（基本 QR，无位移无减缩）/ 'wilkinson'（Wilkinson 位移）。
    """
    Ak = bm.copy(A)
    n = Ak.shape[0]
    I = bm.eye(n, dtype=bm.float64)
    for k in range(1, maxit + 1):
        mu = wilkinson_shift(Ak) if shift == 'wilkinson' else 0.0
        Q, R = householder_qr(Ak - mu * I)
        Ak = R @ Q + mu * I
        off = off_norm(Ak)
        if record is not None:
            record.append({'k': k, 'off': fnum(off)})
        if off < tol:
            return Ak, 'converged_off', k
    return Ak, 'maxit', maxit


def qr_iter_deflated(A, tol=1e-12, maxit=200, record=None):
    """
    带 Wilkinson 位移 + 减缩（deflation）的 QR 迭代（实用 QR 算法的形态）：
    位移只作用在活动块 A[:m,:m] 上；当 |A[m-1,m-2]| < tol 时把该次对角元置 0
    并收缩活动块（一次"减缩"），刚收敛的对角元就是一个特征值。
    位移加速尾部 2x2 块的收敛（三次收敛），减缩让位移能依次作用于每个块。
    """
    Ak = bm.copy(A)
    n = Ak.shape[0]
    m = n                      # 活动块大小
    deflations = []            # (迭代步, 减缩出的特征值)
    for k in range(1, maxit + 1):
        if m <= 1:
            break
        mu = wilkinson_shift(Ak[:m, :m])
        Q, R = householder_qr(Ak[:m, :m] - mu * bm.eye(m, dtype=bm.float64))
        Ak[:m, :m] = R @ Q + mu * bm.eye(m, dtype=bm.float64)
        off = off_norm(Ak)
        if record is not None:
            record.append({'k': k, 'off': fnum(off)})
        if abs(Ak[m - 1, m - 2]) < tol:      # 尾部次对角元收敛 → 减缩
            Ak[m - 1, m - 2] = 0
            Ak[m - 2, m - 1] = 0
            deflations.append((k, fnum(Ak[m - 1, m - 1])))
            m -= 1
        if off < tol:
            break
    return Ak, deflations, k


# ============================================================
# 实验
# ============================================================
def exp_decompose_well_conditioned():
    print('-' * 72)
    print('实验 1：QR 分解（4x3 良态矩阵），三种自实现方法 vs bm.linalg.qr')
    print('-' * 72)
    A = T([[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0],
           [7.0, 8.0, 10.0],
           [2.0, 1.0, 0.0]])
    print('A = 4x3 矩阵，m=4, n=3：Q 应为 (4,3)，R 应为 (3,3) 上三角')
    Qh, Rh = householder_qr(A)
    Qm, Rm = mgs(A)
    Qc, Rc = cgs(A)
    Ql, Rl = bm.linalg.qr(A)
    rows = [
        ['Householder(自实现)', Qh.shape, recon_err(A, Qh, Rh), orth_err(Qh)],
        ['MGS(自实现)', Qm.shape, recon_err(A, Qm, Rm), orth_err(Qm)],
        ['CGS(自实现)', Qc.shape, recon_err(A, Qc, Rc), orth_err(Qc)],
        ['bm.linalg.qr', tuple(Ql.shape), recon_err(A, Ql, Rl), orth_err(Ql)],
    ]
    print_table(rows, header=['方法', 'Q 形状', '||A-QR||_F/||A||_F', '||Q^T Q-I||_F'])
    print('注：Q 的列符号约定任意（Q,R 相差符号变换仍是合法 QR 分解），'
          '故不与库函数逐元比较 Q，只比较重构误差与正交性这两个不变量。')


def exp_decompose_lauchli():
    print('-' * 72)
    print('实验 2：Lauchli 病态矩阵（列几乎线性相关，eps=1e-8），正交性对比')
    print('-' * 72)
    eps = 1e-8
    A = T([[1.0, 1.0, 1.0],
           [eps, 0.0, 0.0],
           [0.0, eps, 0.0],
           [0.0, 0.0, eps]])
    print(f'A = Lauchli 矩阵（kappa(A) ≈ {fnum(bm.linalg.cond(A)):.2e}）')
    Qh, Rh = householder_qr(A)
    Qm, Rm = mgs(A)
    Qc, Rc = cgs(A)
    Ql, Rl = bm.linalg.qr(A)
    rows = [
        ['Householder(自实现)', recon_err(A, Qh, Rh), orth_err(Qh)],
        ['MGS(自实现)', recon_err(A, Qm, Rm), orth_err(Qm)],
        ['CGS(自实现)', recon_err(A, Qc, Rc), orth_err(Qc)],
        ['bm.linalg.qr', recon_err(A, Ql, Rl), orth_err(Ql)],
    ]
    print_table(rows, header=['方法', '||A-QR||_F/||A||_F', '||Q^T Q-I||_F'])
    print('结论：CGS 正交性完全丢失（~0.5，O(1) 量级）；MGS 丢失 ~1e-8（O(eps*kappa)）；')
    print('      Householder 保持 ~1e-16（O(eps)）。列越接近线性相关差异越悬殊。')


def exp_lstsq():
    print('-' * 72)
    print('实验 3：QR 分解解最小二乘问题 min ||Ax - b||（5x3 超定方程组）')
    print('-' * 72)
    A = T([[1.0, 1.0, 1.0],
           [1.0, 2.0, 4.0],
           [1.0, 3.0, 9.0],
           [1.0, 4.0, 16.0],
           [1.0, 5.0, 25.0]])
    b = T([1.0, 3.0, 8.0, 17.0, 31.0])  # 由 y = 1 + x + x^2 在 5 个点采样（加扰动后）
    b = b + T([0.02, -0.01, 0.03, -0.02, 0.01])  # 加一点噪声使问题成为真正的超定问题
    Q, R = householder_qr(A)
    x_qr = bm.linalg.solve(R, Q.T @ b)      # R x = Q^T b
    x_ls = bm.linalg.lstsq(A, b)[0]
    print('QR 分解解出的 x_qr =', [f'{fnum(t):.10f}' for t in x_qr])
    print('bm.linalg.lstsq 的 x_ls =', [f'{fnum(t):.10f}' for t in x_ls])
    print(f'两解之差的 2-范数 = {fnum(bm.linalg.norm(x_qr - x_ls)):.2e}')
    print('结论：A = QR 后 min||Ax-b|| 等价于解三角方程组 R x = Q^T b，'
          '避免正规方程 A^T A x = A^T b 的平方条件数问题。')


def exp_qr_iter(A):
    print('-' * 72)
    print('实验 4：QR 特征值迭代（A = H D H^T，D = diag(10,4,2,1)）')
    print('-' * 72)
    w0 = [fnum(t) for t in bm.linalg.eigh(A)[0]]

    # 特征值不变性：一步 QR 后 eigh(A1) 应与 eigh(A0) 相同
    Q1, R1 = householder_qr(A)
    A1 = R1 @ Q1
    w1 = [fnum(t) for t in bm.linalg.eigh(A1)[0]]
    print('相似不变性检查：A1 = R1 Q1 = Q1^T A Q1，')
    print('  eigh(A0) =', [f'{t:.12f}' for t in w0])
    print('  eigh(A1) =', [f'{t:.12f}' for t in w1])
    print(f'  ||eigh(A1)-eigh(A0)|| = {max(abs(a-b) for a, b in zip(w0, w1)):.2e}'
          '（QR 迭代不改变特征值）')

    # 无位移基本 QR（无减缩）
    hist = []
    Af, st, niter = qr_iter(A, tol=1e-12, maxit=5000, shift='none', record=hist)
    print(f'\n无位移基本 QR（无减缩）：状态 {status_of(st)}，{niter} 次迭代')
    print('非对角范数 off(A_k) 的部分历史：')
    marks = [1, 5, 10, 20] + list(range(niter - 6, niter + 1))
    rows = [[e['k'], f'{e["off"]:.3e}'] for e in hist if e['k'] in marks]
    print_table(rows, header=['k', 'off(A_k)'])
    if len(hist) >= 4:
        r = [e['off'] for e in hist[-4:]]
        ratio = [r[i + 1] / r[i] for i in range(len(r) - 1)]
        print('末尾 off 比:', [f'{t:.6f}' for t in ratio],
              '（最后一对特征值 1,2 的比值 |1/2| = 0.5 主导线性收敛）')

    # Wilkinson 位移 + 减缩 QR（实用形态）
    hist_s = []
    Af_s, deflations, niter_s = qr_iter_deflated(A, tol=1e-12, maxit=200,
                                                 record=hist_s)
    print(f'\nWilkinson 位移 + 减缩 QR：共 {niter_s} 次迭代，{len(deflations)} 次减缩')
    for k, val in deflations:
        print(f'  第 {k:2d} 步减缩出一个特征值 {val:.12f}')
    print('off(A_k) 历史（位移加速尾部块，减缩后位移依次作用于下一个块）：')
    print_table([[e['k'], f'{e["off"]:.3e}'] for e in hist_s],
                header=['k', 'off(A_k)'])

    # 对角元 vs eigh（先排序再比较！）
    diag = sorted(fnum(t) for t in [Af[i, i] for i in range(Af.shape[0])])
    wref = sorted(w0)
    print('\n与 bm.linalg.eigh 对比（两者均先排序，不假设返回顺序一致）：')
    print_table([[f'{d:.12f}', f'{w:.12f}', f'{abs(d-w):.2e}']
                 for d, w in zip(diag, wref)],
                header=['diag(A_k) 排序后', 'eigh 排序后', '差'])
    print(f'结论：无位移基本 QR 需 {niter} 步（线性收敛，比例 |lam2/lam1|）；')
    print(f'      Wilkinson 位移 + 减缩只需 {niter_s} 步。注意：若只有位移、不减缩，')
    print('      位移只加速尾部 2x2 块，其它非对角元仍按无位移速度衰减——')
    print('      位移必须与减缩配合才能把收益作用到每个块上。')


def backend_compare():
    print('#' * 72)
    print('# 双后端一致性检查（QR 分解 + QR 迭代）')
    print('#' * 72)
    saved = {}
    for name in ['numpy', 'pytorch']:
        bm.set_backend(name)
        A = T([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 10.0],
               [2.0, 1.0, 0.0]])
        Q, R = householder_qr(A)
        Asym, _ = build_matrix([10.0, 4.0, 2.0, 1.0])
        hist = []
        Af, deflations, niter = qr_iter_deflated(Asym, tol=1e-12, maxit=200,
                                                 record=hist)
        diag = sorted(fnum(t) for t in [Af[i, i] for i in range(Af.shape[0])])
        saved[name] = dict(recon=recon_err(A, Q, R), orth=orth_err(Q),
                           niter=niter, off=hist[-1]['off'], diag=diag)
    rows = []
    for name in ['numpy', 'pytorch']:
        d = saved[name]
        rows.append([name, f'{d["recon"]:.2e}', f'{d["orth"]:.2e}', d['niter'],
                     f'{d["off"]:.2e}', [f'{t:.10f}' for t in d['diag']]])
    print_table(rows, header=['后端', '||A-QR||_F/||A||_F', '||Q^T Q-I||_F',
                              'QR迭代次数', '最终 off', 'diag 排序后'])
    bm.set_backend('numpy')


def main():
    np.random.seed(20260831)

    print('[后端]', bm.backend_name)
    exp_decompose_well_conditioned()
    exp_decompose_lauchli()
    exp_lstsq()

    A, H = build_matrix([10.0, 4.0, 2.0, 1.0])
    exp_qr_iter(A)
    backend_compare()


if __name__ == '__main__':
    main()
