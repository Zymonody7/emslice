"""电磁仿真测试矩阵生成器（Yee 网格旋度-旋度算子）。

离散 ∇×(μ⁻¹∇×E) = k² ε E，化为标准形式 A = Dε^{-1/2} Cᴴ C Dε^{-1/2}：
  * PEC 矩形腔：实对称、半正定，梯度场构成巨大零空间（低频击穿 / 零附近密集谱）；
  * 立方腔：大量重特征值（重数 3、6）；
  * Bloch 周期边界：复厄密矩阵（光子晶体带隙计算场景）；
  * 均匀介质下离散特征值有解析解，可作为与求解器无关的真值。

所有生成都是确定性的（给定参数 → 相同矩阵），便于复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp

EpsFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class YeeGrid:
    n: Sequence[int] = (32, 32, 32)          # 每个方向的网格单元数
    length: Sequence[float] = (1.0, 1.0, 1.0)  # 每个方向的尺寸
    bc: Sequence[str] = ("pec", "pec", "pec")  # "pec" 或 "periodic"
    bloch_phase: Sequence[float] = (0.0, 0.0, 0.0)  # 周期方向的 Bloch 相位 k·L
    eps: EpsFunc | None = None               # 相对介电常数分布 ε(x, y, z)
    meta: dict = field(default_factory=dict)

    @property
    def h(self) -> np.ndarray:
        return np.asarray(self.length, float) / np.asarray(self.n, float)


def _diff(N: int, h: float, bc: str, phase: float) -> sp.csr_matrix:
    """节点 → 单元 的一阶差分 (cells × nodes)。"""
    if bc == "pec":
        D = sp.diags([-np.ones(N), np.ones(N)], [0, 1], shape=(N, N + 1))
        return (D / h).tocsr()
    if bc == "periodic":
        rows = np.r_[np.arange(N), np.arange(N)]
        cols = np.r_[np.arange(N), (np.arange(N) + 1) % N]
        vals = np.r_[-np.ones(N), np.ones(N)].astype(complex if phase else float)
        if phase:
            vals[2 * N - 1] = np.exp(1j * phase)  # 跨越周期边界的项带 Bloch 相位
        return sp.csr_matrix((vals / h, (rows, cols)), shape=(N, N))
    raise ValueError(f"unknown bc {bc!r}")


def _k3(mz, my, mx):
    """x 方向变化最快的三维 Kronecker 积。"""
    return sp.kron(mz, sp.kron(my, mx, format="csr"), format="csr")


def yee_curlcurl(grid: YeeGrid, return_parts: bool = False):
    """生成标准形式的 Yee 旋度-旋度矩阵 A（CSR，实对称或复厄密）。"""
    N = list(map(int, grid.n))
    h = grid.h
    D, Ic, In, nodes = [], [], [], []
    for d in range(3):
        Dd = _diff(N[d], h[d], grid.bc[d], grid.bloch_phase[d])
        D.append(Dd)
        Ic.append(sp.identity(N[d], format="csr"))
        In.append(sp.identity(Dd.shape[1], format="csr"))
        nodes.append(Dd.shape[1])
    Dx, Dy, Dz = D
    # 分量布局 (x, y, z)：c = 单元（半格点），n = 节点
    # Ex:(c,n,n) Ey:(n,c,n) Ez:(n,n,c)；Hx:(n,c,c) Hy:(c,n,c) Hz:(c,c,n)
    dyEz = _k3(Ic[2], Dy, In[0])
    dzEy = _k3(Dz, Ic[1], In[0])
    dzEx = _k3(Dz, In[1], Ic[0])
    dxEz = _k3(Ic[2], In[1], Dx)
    dxEy = _k3(In[2], Ic[1], Dx)
    dyEx = _k3(In[2], Dy, Ic[0])
    nEx, nEy, nEz = dzEx.shape[1], dzEy.shape[1], dyEz.shape[1]
    Z = lambda r, c: sp.csr_matrix((r, c))
    C = sp.bmat(
        [
            [Z(dyEz.shape[0], nEx), -dzEy, dyEz],
            [dzEx, Z(dzEx.shape[0], nEy), -dxEz],
            [-dyEx, dxEy, Z(dyEx.shape[0], nEz)],
        ],
        format="csr",
    )

    # PEC：去掉边界上的切向电场自由度；同时计算每个自由度的坐标（用于采样 ε）
    keep, coords = [], []
    layouts = [("c", "n", "n"), ("n", "c", "n"), ("n", "n", "c")]
    for comp, lay in enumerate(layouts):
        masks, xs = [], []
        for d in range(3):
            if lay[d] == "c":
                m = np.ones(N[d], bool)
                x = (np.arange(N[d]) + 0.5) * h[d]
            else:
                m = np.ones(nodes[d], bool)
                if grid.bc[d] == "pec":
                    m[[0, -1]] = False
                x = np.arange(nodes[d]) * h[d]
            masks.append(m)
            xs.append(x)
        mk = (masks[2][:, None, None] & masks[1][None, :, None] & masks[0][None, None, :]).ravel()
        Z3, Y3, X3 = np.meshgrid(xs[2], xs[1], xs[0], indexing="ij")
        keep.append(mk)
        coords.append(np.stack([X3.ravel()[mk], Y3.ravel()[mk], Z3.ravel()[mk]], 1))
    keep = np.concatenate(keep)
    coords = np.concatenate(coords)
    C = C[:, np.flatnonzero(keep)]

    K = (C.conj().T @ C).tocsr()
    if grid.eps is not None:
        eps = np.asarray(grid.eps(coords[:, 0], coords[:, 1], coords[:, 2]), float)
        if np.any(eps <= 0):
            raise ValueError("ε 必须为正")
        s = sp.diags(1.0 / np.sqrt(eps))
        A = (s @ K @ s).tocsr()
    else:
        A = K
    A.sum_duplicates()
    A.eliminate_zeros()
    if not np.iscomplexobj(A.data) or not np.any(A.data.imag):
        A = A.real.tocsr() if np.iscomplexobj(A.data) else A
    if return_parts:
        return A, {"C": C, "coords": coords}
    return A


def analytic_eigs_homogeneous(grid: YeeGrid, include_null: bool = True) -> np.ndarray:
    """均匀介质 (ε≡1) 下 Yee 腔离散特征值的解析解（含重数）。"""
    if grid.eps is not None:
        raise ValueError("仅适用于均匀介质")
    N = list(map(int, grid.n))
    h = grid.h
    s, zero_ok = [], []
    for d in range(3):
        m = np.arange(N[d])
        if grid.bc[d] == "pec":
            s.append((2.0 / h[d] * np.sin(m * np.pi / (2 * N[d]))) ** 2)
        else:
            s.append((2.0 / h[d] * np.sin((2 * np.pi * m + grid.bloch_phase[d]) / (2 * N[d]))) ** 2)
        zero_ok.append(grid.bc[d] == "pec")
    S = s[0][None, None, :] + s[1][None, :, None] + s[2][:, None, None]
    pec = [b == "pec" for b in grid.bc]
    vals = []
    if all(pec):
        mz, my, mx = np.meshgrid(np.arange(N[2]), np.arange(N[1]), np.arange(N[0]), indexing="ij")
        nz = (mx > 0).astype(int) + (my > 0) + (mz > 0)
        mult = np.where(nz == 3, 2, np.where(nz == 2, 1, 0))
        vals = np.repeat(S.ravel(), mult.ravel())
        null_dim = (N[0] - 1) * (N[1] - 1) * (N[2] - 1)
    elif not any(pec):
        vals = np.repeat(S.ravel(), 2)
        null_dim = N[0] * N[1] * N[2]
        if not any(grid.bloch_phase):
            # Γ 点：常数位势无梯度，另有 3 个常数场谐波模式（特征值 0）
            vals = vals[vals > 1e-12 * vals.max()]
            null_dim = N[0] * N[1] * N[2] - 1 + 3
    else:
        raise NotImplementedError("解析解仅支持全 PEC 或全周期")
    vals = np.sort(vals)
    if include_null:
        vals = np.r_[np.zeros(null_dim), vals]
    return np.sort(vals)


# ---- 预置场景 -------------------------------------------------------------

def puck_eps(center=(0.5, 0.5, 0.5), radius=0.2, height=0.2, eps_r=10.0) -> EpsFunc:
    """圆柱介质谐振器（微波介质滤波器常见结构）。"""
    cx, cy, cz = center

    def f(x, y, z):
        inside = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2) & (np.abs(z - cz) <= height / 2)
        return np.where(inside, eps_r, 1.0)

    return f


def rods_eps(radius=0.2, eps_r=8.9) -> EpsFunc:
    """方格介质柱光子晶体（单胞内一根沿 z 的介质柱）。"""

    def f(x, y, z):
        return np.where((x - 0.5) ** 2 + (y - 0.5) ** 2 <= radius**2, eps_r, 1.0)

    return f


def preset(name: str, N: int = 32) -> tuple[sp.csr_matrix, YeeGrid]:
    if name == "cavity":       # 均匀 PEC 长方体腔（有解析解，少量重特征值）
        g = YeeGrid(n=(N, N + 2, N - 2), length=(1.0, 1.1, 0.9))
    elif name == "cube":       # 均匀 PEC 立方腔（大量重特征值）
        g = YeeGrid(n=(N, N, N))
    elif name == "puck":       # 介质谐振器腔（真实微波器件结构）
        g = YeeGrid(n=(N, N, N), eps=puck_eps())
    elif name == "bloch":      # 光子晶体单胞，Bloch 周期边界 → 复厄密
        g = YeeGrid(n=(N, N, N), bc=("periodic",) * 3, bloch_phase=(0.7, 0.3, 0.1), eps=rods_eps())
    elif name == "bloch-homog":  # 均匀周期介质，Bloch 相位（复厄密，有解析解）
        g = YeeGrid(n=(N, N, N), bc=("periodic",) * 3, bloch_phase=(0.7, 0.3, 0.1))
    else:
        raise ValueError(f"unknown preset {name!r}")
    g.meta["preset"] = name
    return yee_curlcurl(g), g
