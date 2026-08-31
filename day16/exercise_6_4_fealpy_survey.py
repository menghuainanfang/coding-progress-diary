r"""
练习 6.4：FEALPy 中特征值计算用途调研（任务 3 的代码部分）
==========================================================
按任务要求只阅读指定入口/调用点，不展开 PETSc/SLEPc 的内部实现，
并把四处用法以小规模可运行的复现展示出来：

  1. fealpy/fem/linear_elasticity_eigen_lfem_model.py —— 线弹性模态
     S v = lam M v 的组装与求解形态（scipy 路径 eigsh + 边界自由度剔除）
  2. fealpy/cgraph/solver.py 的 EigenSolver —— 通用计算图特征值节点
     （直接调用其 run()，观察 S/M/neigen/which 参数与 eigsh 的对应）
  3. fealpy/mmesh/targetadaptive.py 的 sqrt_M —— 用 bm.linalg.eigh
     对批量小对称矩阵做谱分解并构造矩阵平方根（复现原模式）
  4. app/fracturex/.../phase_fracture_material.py 的
     strain_pm_eig_decomposition —— 相场断裂中应变张量谱分解
     （特征值过 Macaulay 括号构造正/负应变部分）

不安装 petsc4py / slepc4py / MUMPS，不调用任何可选求解器。

运行方式：
    python exercise_6_4_fealpy_survey.py
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from fealpy.backend import backend_manager as bm

from common import T, fnum, print_table


# ============================================================
# 1) 线弹性模态问题：S v = lam M v（eigsh 路径的微型复现）
# ============================================================
def exp_modal_mini():
    print('-' * 72)
    print('调研 1：线弹性模态问题（LinearElasticityEigenLFEMModel 的 scipy 路径）')
    print('-' * 72)
    print('源码要点（fealpy/fem/linear_elasticity_eigen_lfem_model.py）：')
    print('  * linear_system(): BilinearForm + LinearElasticityIntegrator 组装刚度 S，')
    print('    BilinearForm + ScalarMassIntegrator(density) 组装质量 M；')
    print('  * apply_bc(): DirichletBC 求出边界自由度 isFreeDof，S、M 各取自由子块；')
    print('  * solve(): eigsh(S, k=neign, M=M, which="SM", tol=1e-6, maxiter=1000)。')

    # 微型复现：两端固定的一维振动弦（区间 [0,1]，h=1/29），
    # 刚度/质量矩阵与二维线弹性完全同构（都是 BilinearForm 组装 + 自由自由度剔除）
    n = 30
    h = 1.0 / (n - 1)
    d = np.ones(n)
    S = sparse.diags([-d[:-1], 2.0 * d, -d[:-1]], [-1, 0, 1], format='csr') / h
    m = np.ones(n)
    M = sparse.diags([m[:-1] / 6.0, (2.0 / 3.0) * m, m[:-1] / 6.0],
                     [-1, 0, 1], format='csr') * h  # 一致质量矩阵
    # 施加边界条件：剔除两端自由度（与源码 isFreeDof 索引等价）
    isFree = np.zeros(n, dtype=bool)
    isFree[1:-1] = True
    Sf, Mf = S[isFree][:, isFree], M[isFree][:, isFree]

    k = 3
    val, vec = eigsh(Sf, k=k, M=Mf, which='SM', tol=1e-6, maxiter=1000)
    omega = np.sqrt(val)
    theory = np.pi * np.arange(1, k + 1)            # 弦振动频率 omega_k = k*pi
    print(f'两端固定弦 S v = lam M v（n={n} 内部自由度 {Sf.shape[0]}），eigsh 前 {k} 阶：')
    print_table([[i + 1, f'{val[i]:.6f}', f'{omega[i]:.6f}', f'{theory[i]:.6f}',
                  f'{abs(omega[i] - theory[i]):.2e}']
                 for i in range(k)],
                header=['阶', 'lam_k', 'omega_k=sqrt(lam_k)', '理论 k*pi', '差'])
    print('这就是线弹性模态问题的形式：S=刚度（弹性应变能），M=质量（动能），')
    print('omega_k 即结构自振频率（模态）。which="SM" 取最小频率——工程上最关心的低阶模态。')
    print('与 k*pi 的差是有限元离散误差（O((k*pi*h)^2)，随阶数 k 增长），不是求解器误差。')
    return Sf, Mf, val, vec


# ============================================================
# 2) cgraph 的 EigenSolver 节点
# ============================================================
def exp_cgraph_node(Sf, Mf, val_eigsh):
    print('-' * 72)
    print('调研 2：cgraph 计算图特征值节点 EigenSolver')
    print('-' * 72)
    from fealpy.cgraph.solver import EigenSolver
    print('节点定义（fealpy/cgraph/solver.py）：')
    print('  INPUT_SLOTS: S（刚度）, M（质量）, neigen（默认 6）, which（默认 "SM"）')
    print('  OUTPUT_SLOTS: val, vec')
    print('  run() 内部: eigsh(S, k=neigen, M=M, which=which, tol=1e-6, maxiter=1000)')

    val, vec = EigenSolver.run(S=Sf, M=Mf, neigen=3, which='SM')
    print('直接调用 EigenSolver.run(S, M, neigen=3, which="SM")：')
    print('  val =', [f'{fnum(t):.6f}' for t in val])
    print(f'  与上面 eigsh 结果逐位一致：'
          f'{max(abs(fnum(a) - b) for a, b in zip(val, val_eigsh)):.2e}')
    print('意义：特征值求解被封装成计算图节点，可视化编排时可直接连矩阵端口，')
    print('      输入端口 (S, M, neigen, which) 完全对应 eigsh 的参数。')


# ============================================================
# 3) sqrt_M：批量谱分解构造矩阵平方根（mmesh/targetadaptive.py 模式复现）
# ============================================================
def exp_sqrt_m():
    print('-' * 72)
    print('调研 3：度量张量平方根 sqrt_M（fealpy/mmesh/targetadaptive.py 模式复现）')
    print('-' * 72)
    print('源码模式（targetadaptive.py 第 60-75 行）：')
    print('  eigvals, eigvecs = bm.linalg.eigh(M)              # 批量 (NC,GD,GD)')
    print('  sqrt_eigvals = bm.sqrt(bm.maximum(eigvals, 1e-16)) # 特征值后处理：截负')
    print('  sqrt_M = eigvecs @ D_sqrt @ eigvecs^T')

    # 微型复现：3 个单元的 2x2 对称正定度量张量
    M = T([[[2.0, 0.5],
            [0.5, 1.0]],
           [[1.0, 0.0],
            [0.0, 4.0]],
           [[1.5, -0.3],
            [-0.3, 0.8]]])
    eigvals, eigvecs = bm.linalg.eigh(M)                  # (3,2), (3,2,2)
    sqrt_eigvals = bm.sqrt(bm.maximum(eigvals, T(1e-16)))  # 特征值后处理：防负/防零
    D_sqrt = bm.zeros_like(M)
    for i in range(M.shape[-1]):
        D_sqrt = bm.set_at(D_sqrt, (..., i, i), sqrt_eigvals[..., i])
    sqrt_M = eigvecs @ D_sqrt @ bm.swapaxes(eigvecs, -1, -2)

    errs = [fnum(bm.linalg.norm(sqrt_M[i] @ sqrt_M[i] - M[i]))
            for i in range(M.shape[0])]
    print('3 个 2x2 度量张量的谱分解平方根验证 ||sqrt(M) sqrt(M) - M||_F：')
    print(' ', [f'{e:.2e}' for e in errs])
    print('对比 1e-16 截负后处理：eigh 对正定阵也可能给出微小的负特征值（舍入误差），')
    print('开方前用 bm.maximum(eigvals, 1e-16) 截断——这就是任务要求找的')
    print('"对特征值结果进行后处理"的一个实例（另一实例见调研 4 的 Macaulay 括号）。')


# ============================================================
# 4) strain_pm_eig_decomposition：应变谱分解（fracturex 相场断裂模式复现）
# ============================================================
def exp_strain_pm():
    print('-' * 72)
    print('调研 4：应变张量正/负谱分解（fracturex 相场断裂模式复现）')
    print('-' * 72)
    print('源码模式（phase_fracture_material.py 第 281-325 行）：')
    print('  w, v = bm.linalg.eigh(s)                  # 应变张量特征分解')
    print('  p, m = macaulay_operation(w)              # <a>_+=(a+|a|)/2, <a>_-=(a-|a|)/2')
    print('  s_+ = sum_a <w_a>_+ n_a n_a^T,  s_- = sum_a <w_a>_- n_a n_a^T')

    lam, mu = 1.2, 0.8                                  # Lamé 常数
    s = T([[[0.04, 0.02],
            [0.02, -0.01]],
           [[0.03, 0.01],
            [0.01, -0.02]],
           [[-0.02, 0.0],
            [0.0, -0.03]]])                             # 3 个 2x2 应变张量

    w, v = bm.linalg.eigh(s)
    val = bm.abs(w)
    p = (w + val) / 2.0                                 # Macaulay 正部
    m = (w - val) / 2.0                                 # Macaulay 负部

    sp = bm.zeros_like(s)
    sm = bm.zeros_like(s)
    GD = s.shape[-1]
    for i in range(GD):
        n0 = v[..., i]                                  # (NC, GD)
        sp += (p[..., i, None] * n0)[..., None] * n0[..., None, :]
        sm += (m[..., i, None] * n0)[..., None] * n0[..., None, :]

    sum_err = [fnum(bm.linalg.norm(sp[i] + sm[i] - s[i])) for i in range(s.shape[0])]
    print('正/负应变重构验证 ||s_+ + s_- - s||_F：', [f'{e:.2e}' for e in sum_err])

    # 能量可加性：psi = psi_+ + psi_-（Miehe 谱分解的意义）——用残差做验证！
    tr_s = bm.einsum('...ii', s)
    tp = (tr_s + bm.abs(tr_s)) / 2.0
    tm = (tr_s - bm.abs(tr_s)) / 2.0
    psi = lam * tr_s ** 2 / 2.0 + mu * bm.einsum('...ij,...ij', s, s)  # 正确：Σ_ij s_ij^2

    # 方式 A（源码原文）：tsp = einsum('...ii', sp**2) —— 元素平方的对角线之和
    tr2_p_a = bm.einsum('...ii', sp ** 2)
    tr2_m_a = bm.einsum('...ii', sm ** 2)
    psi_p_a = lam * tp ** 2 / 2.0 + mu * tr2_p_a
    psi_m_a = lam * tm ** 2 / 2.0 + mu * tr2_m_a

    # 方式 B（标准 Miehe）：tr(sp @ sp) = sum_ij (sp)_ij^2（对称矩阵的矩阵平方迹）
    tr2_p_b = bm.einsum('...ij,...ji', sp, sp)
    tr2_m_b = bm.einsum('...ij,...ji', sm, sm)
    psi_p_b = lam * tp ** 2 / 2.0 + mu * tr2_p_b
    psi_m_b = lam * tm ** 2 / 2.0 + mu * tr2_m_b

    print('能量可加性验证 |psi_+ + psi_- - psi|（Miehe 分解要求严格等于 0）：')
    print_table([[f'样本{i+1}',
                  f'{fnum(bm.abs(psi_p_a[i] + psi_m_a[i] - psi[i])):.3e}',
                  f'{fnum(bm.abs(psi_p_b[i] + psi_m_b[i] - psi[i])):.3e}']
                 for i in range(3)],
                header=['', '方式A: einsum("...ii", sp**2)（源码写法）',
                        '方式B: tr(sp @ sp)（标准 Miehe）'])
    print('发现：源码写法（方式 A）只累加元素平方的对角线项 Σ_i (sp_ii)^2，')
    print('      缺少非对角贡献 2*(sp_12)^2，因此 psi_+ + psi_- != psi；')
    print('      标准 Miehe 应变能应为 tr(sp @ sp) = Σ_ij (sp)_ij^2，可加性严格成立。')
    print('      —— 这正是"数值结果必须用残差/重构误差/可加性验证"的实例：')
    print('         连库源码里的公式也要经得起验证。')
    print('物理意义：相场断裂中拉应变储存可释放能量（驱动开裂）、压应变不参与，')
    print('因此必须把应变张量沿其特征方向做谱分解（小对称矩阵 eigh 的批量应用）。')


def backend_compare():
    print('#' * 72)
    print('# 双后端一致性检查（sqrt_M 与应变谱分解，均为 bm.linalg.eigh 的批量调用）')
    print('#' * 72)
    saved = {}
    for name in ['numpy', 'pytorch']:
        bm.set_backend(name)
        M = T([[[2.0, 0.5], [0.5, 1.0]],
               [[1.0, 0.0], [0.0, 4.0]],
               [[1.5, -0.3], [-0.3, 0.8]]])
        eigvals, eigvecs = bm.linalg.eigh(M)
        D_sqrt = bm.zeros_like(M)
        for i in range(M.shape[-1]):
            D_sqrt = bm.set_at(D_sqrt, (..., i, i),
                               bm.sqrt(bm.maximum(eigvals, T(1e-16)))[..., i])
        sqrt_M = eigvecs @ D_sqrt @ bm.swapaxes(eigvecs, -1, -2)
        err = fnum(bm.linalg.norm(sqrt_M @ sqrt_M - M))
        s = T([[[0.04, 0.02], [0.02, -0.01]]])
        w, v = bm.linalg.eigh(s)
        p = (w + bm.abs(w)) / 2.0
        m = (w - bm.abs(w)) / 2.0
        sp = bm.zeros_like(s)
        sm = bm.zeros_like(s)
        for i in range(2):
            n0 = v[..., i]
            sp += (p[..., i, None] * n0)[..., None] * n0[..., None, :]
            sm += (m[..., i, None] * n0)[..., None] * n0[..., None, :]
        saved[name] = dict(sqrt_err=err,
                           sp_err=fnum(bm.linalg.norm(sp + sm - s)))
    for name in ['numpy', 'pytorch']:
        d = saved[name]
        print(f'{name}: sqrt_M 验证误差 {d["sqrt_err"]:.2e}，应变谱分解重构误差 {d["sp_err"]:.2e}')
    bm.set_backend('numpy')


def main():
    print('[后端]', bm.backend_name)
    Sf, Mf, val, vec = exp_modal_mini()
    exp_cgraph_node(Sf, Mf, val)
    exp_sqrt_m()
    exp_strain_pm()
    backend_compare()


if __name__ == '__main__':
    main()
