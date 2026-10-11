"""有限元电磁谐振测试矩阵（一阶 Nédélec 棱边元）：广义问题 K x = k₀² M x。

  K = ∫ (ν ∇×E)·(∇×v)   旋度-旋度刚度阵，ν = μ_r⁻¹（铁氧体为厄密张量 ⇒ K 复厄密）
  M = ∫ (ε_r E)·v       质量阵（实对称正定）
PEC 外壁：边界棱自由度直接消去。长度单位 mm，特征值 λ = k₀²（mm⁻²），频率 f = c√λ / 2π。

与 Yee 差分矩阵（matgen.py）互补：这是商用/开源电磁 FEM 软件实际产生的矩阵形态——
广义问题、非结构四面体网格、曲面介质、高介电对比度；梯度零空间维数 = 内部节点数（低频击穿）。

  fem-cavity   空矩形腔，结构网格；连续问题有解析解（检验离散与求解链路）
  fem-dr       介质谐振器滤波腔：ε_r=38 陶瓷圆柱 + PTFE 支撑，非结构网格（gmsh）
  fem-ferrite  铁氧体加载腔：磁化旋磁张量 μ（工作点冻结），K 复厄密

依赖：scikit-fem（BSD-3，可选安装）；曲面几何的网格由 gmsh 作为独立外部程序生成（.geo → .msh），
不链接进本项目。网格生成是确定性的，meta 中记录网格 SHA-256。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

C_MM_PER_NS = 299.792458  # 光速，mm/ns ⇒ f[GHz] = C·k₀[mm⁻¹] / 2π


def freq_ghz(lam):
    return C_MM_PER_NS * np.sqrt(np.maximum(np.asarray(lam, float), 0.0)) / (2 * np.pi)


def lam_of_ghz(f):
    return (2 * np.pi * np.asarray(f, float) / C_MM_PER_NS) ** 2


def _skfem():
    try:
        import skfem  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("有限元算例需要 scikit-fem：pip install scikit-fem") from e
    import skfem

    return skfem


# ------------------------------------------------------------------ 网格
def box_mesh(size, h):
    skfem = _skfem()
    xs = [np.linspace(0.0, L, max(1, int(round(L / h))) + 1) for L in size]
    return skfem.MeshTet.init_tensor(*xs)


def _read_msh22(path):
    """读取 gmsh ASCII 2.2 格式中的节点与四面体（类型 4）。"""
    with open(path, encoding="ascii") as f:
        lines = f.read().split("\n")
    i = lines.index("$Nodes")
    nn = int(lines[i + 1])
    nodes = np.array([ln.split() for ln in lines[i + 2: i + 2 + nn]], dtype=float)
    idmap = np.full(int(nodes[:, 0].max()) + 1, -1, dtype=np.int64)
    idmap[nodes[:, 0].astype(np.int64)] = np.arange(nn)
    i = lines.index("$Elements")
    ne = int(lines[i + 1])
    tets = []
    for ln in lines[i + 2: i + 2 + ne]:
        a = ln.split()
        if a[1] == "4":
            nt = int(a[2])
            tets.append([int(v) for v in a[3 + nt: 7 + nt]])
    t = idmap[np.array(tets, dtype=np.int64)]
    return nodes[:, 1:4].T.copy(), t.T.copy()


def _gmsh_cmd():
    exe = shutil.which("gmsh") or os.path.join(os.path.dirname(sys.executable), "gmsh")
    if not os.path.exists(exe):
        raise FileNotFoundError("需要 gmsh 可执行程序（pip install gmsh 或 conda install gmsh）")
    with open(exe, "rb") as f:
        head = f.read(64)
    # pip 版 gmsh 的入口是 Python 脚本：用当前解释器运行，避免 shebang 指向别的 Python
    return [sys.executable, exe] if head.startswith(b"#!") and b"python" in head else [exe]


def gmsh_mesh(geo: str):
    """把 .geo 几何交给 gmsh 独立进程生成四面体网格，返回 (MeshTet, 网格 SHA-256)。"""
    skfem = _skfem()
    with tempfile.TemporaryDirectory() as d:
        g, m = os.path.join(d, "g.geo"), os.path.join(d, "g.msh")
        with open(g, "w", encoding="ascii") as f:
            f.write(geo)
        subprocess.run(_gmsh_cmd() + [g, "-3", "-format", "msh22", "-nt", "1", "-o", m],
                       check=True, capture_output=True, text=True)
        p, t = _read_msh22(m)
    sha = hashlib.sha256(np.ascontiguousarray(p).tobytes() + np.ascontiguousarray(t).tobytes()).hexdigest()
    return skfem.MeshTet(p, t), sha


# ------------------------------------------------------------------ 组装
def assemble(mesh, eps_elem, ferrite=None):
    """eps_elem：每个单元的 ε_r；ferrite：(单元掩码, μ_r, κ)，偏置沿 z 的旋磁张量
    μ = [[μ_r, −jκ, 0], [jκ, μ_r, 0], [0, 0, 1]]，ν = μ⁻¹ 厄密 ⇒ K = K_r + j K_i（K_i 反对称）。"""
    skfem = _skfem()
    from skfem.helpers import curl, dot

    basis = skfem.Basis(mesh, skfem.ElementTetN0())
    nq = basis.X.shape[1]
    eps = np.repeat(np.asarray(eps_elem, float)[:, None], nq, axis=1)

    @skfem.BilinearForm
    def mass(E, v, w):
        return w.eps * dot(E, v)

    M = skfem.asm(mass, basis, eps=eps)
    if ferrite is None:
        @skfem.BilinearForm
        def stiff(E, v, w):
            return dot(curl(E), curl(v))

        K = skfem.asm(stiff, basis)
    else:
        mask, mu, kappa = ferrite
        det = mu * mu - kappa * kappa
        if not (mu - abs(kappa) > 0):
            raise ValueError("旋磁张量须正定：μ_r > |κ|")
        chi = np.repeat(np.asarray(mask, float)[:, None], nq, axis=1)
        nu_t, nu_g = mu / det, kappa / det

        @skfem.BilinearForm
        def stiff_r(E, v, w):
            cE, cv = curl(E), curl(v)
            a = 1.0 + w.chi * (nu_t - 1.0)
            return a * (cE[0] * cv[0] + cE[1] * cv[1]) + cE[2] * cv[2]

        @skfem.BilinearForm
        def stiff_i(E, v, w):
            cE, cv = curl(E), curl(v)
            return w.chi * nu_g * (cE[1] * cv[0] - cE[0] * cv[1])

        K = skfem.asm(stiff_r, basis, chi=chi) + 1j * skfem.asm(stiff_i, basis, chi=chi)
    inner = basis.complement_dofs(basis.get_dofs())        # PEC：消去边界棱
    K = sp.csr_matrix(K[inner][:, inner])
    M = sp.csr_matrix(M[inner][:, inner])
    n_inner_nodes = int(mesh.p.shape[1] - np.unique(mesh.facets[:, mesh.boundary_facets()]).size)
    return K, M, dict(n_tets=int(mesh.t.shape[1]), n_nodes=int(mesh.p.shape[1]), null_dim=n_inner_nodes)


def _centroids(mesh):
    return mesh.p[:, mesh.t].mean(axis=1)


def _in_cyl(c, cx, cy, r, z0, z1):
    return ((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= r * r) & (c[2] >= z0) & (c[2] <= z1)


# ------------------------------------------------------------------ 算例
def analytic_cavity(size, kmax2):
    """矩形 PEC 腔连续问题的 k² 及重数（TE_mnp：(m,n)≠0, p≥1；TM_mnp：m,n≥1）。"""
    a, b, d = size
    out = []
    M = [int(np.sqrt(kmax2) * L / np.pi) + 1 for L in size]
    for m in range(M[0] + 1):
        for n in range(M[1] + 1):
            for p in range(M[2] + 1):
                k2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2 + (p * np.pi / d) ** 2
                mult = int((m or n) and p >= 1) + int(m >= 1 and n >= 1)
                if 0 < k2 <= kmax2 and mult:
                    out += [k2] * mult
    return np.sort(out)


def fem_cavity(h=0.95, size=(30.0, 24.0, 18.0)):
    mesh = box_mesh(size, h)
    K, M, info = assemble(mesh, np.ones(mesh.t.shape[1]))
    return K, M, dict(preset="fem-cavity", size_mm=size, h_mm=h, **info)


def _dr_geo(h):
    hin = 0.6 * h
    return f"""SetFactory("OpenCASCADE");
Box(1) = {{0, 0, 0, 30, 30, 24}};
Cylinder(2) = {{15, 15, 6, 0, 0, 7, 8}};
Cylinder(3) = {{15, 15, 0, 0, 0, 6, 4}};
BooleanFragments{{ Volume{{1}}; Delete; }}{{ Volume{{2, 3}}; Delete; }}
Field[1] = Box;
Field[1].VIn = {hin}; Field[1].VOut = {h};
Field[1].XMin = 6; Field[1].XMax = 24; Field[1].YMin = 6; Field[1].YMax = 24;
Field[1].ZMin = 5; Field[1].ZMax = 14; Field[1].Thickness = 3;
Background Field = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Optimize = 1;
"""


def fem_dr(h=1.25, eps_dr=38.0, eps_support=2.1):
    """介质谐振器滤波腔单腔：30×30×24 mm 铜腔，ε_r=38 陶瓷圆柱（半径 8、高 7 mm）置于 PTFE 支撑上。
    TE01δ 模约 3.5 GHz（Kajfez 近似公式给出孤立谐振器约 3.2 GHz，金属腔使其上移）。"""
    mesh, sha = gmsh_mesh(_dr_geo(h))
    c = _centroids(mesh)
    eps = np.ones(mesh.t.shape[1])
    eps[_in_cyl(c, 15, 15, 4, 0, 6)] = eps_support
    eps[_in_cyl(c, 15, 15, 8, 6, 13)] = eps_dr
    K, M, info = assemble(mesh, eps)
    return K, M, dict(preset="fem-dr", h_mm=h, eps_dr=eps_dr, eps_support=eps_support, mesh_sha256=sha,
                      n_dr_tets=int(np.sum(eps == eps_dr)), **info)


def _ferrite_geo(h):
    hin = 0.6 * h
    return f"""SetFactory("OpenCASCADE");
Box(1) = {{0, 0, 0, 30, 30, 20}};
Cylinder(2) = {{15, 15, 8.5, 0, 0, 3, 7}};
BooleanFragments{{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }}
Field[1] = Box;
Field[1].VIn = {hin}; Field[1].VOut = {h};
Field[1].XMin = 7; Field[1].XMax = 23; Field[1].YMin = 7; Field[1].YMax = 23;
Field[1].ZMin = 7.5; Field[1].ZMax = 12.5; Field[1].Thickness = 3;
Background Field = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Optimize = 1;
"""


def fem_ferrite(h=1.1, eps_f=15.0, mu=0.85, kappa=0.45):
    """铁氧体加载腔：30×30×20 mm 腔中央 YIG 圆片（半径 7、厚 3 mm，ε_r=15），z 向偏置；
    旋磁张量在工作点冻结（μ_r=0.85, κ=0.45），得到复厄密 K 的线性广义问题。"""
    mesh, sha = gmsh_mesh(_ferrite_geo(h))
    c = _centroids(mesh)
    inside = _in_cyl(c, 15, 15, 7, 8.5, 11.5)
    eps = np.where(inside, eps_f, 1.0)
    K, M, info = assemble(mesh, eps, ferrite=(inside, mu, kappa))
    return K, M, dict(preset="fem-ferrite", h_mm=h, eps_f=eps_f, mu_r=mu, kappa=kappa, mesh_sha256=sha,
                      n_ferrite_tets=int(inside.sum()), **info)


PRESETS = {"fem-cavity": fem_cavity, "fem-dr": fem_dr, "fem-ferrite": fem_ferrite}


def preset(name, h=None):
    f = PRESETS[name]
    return f() if h is None else f(h=h)
