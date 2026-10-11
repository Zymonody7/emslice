"""介质谐振器滤波腔（fem-dr）的结构与模场图：用于方案说明文档与演示视频。

python bench/plot_fem_modes.py --out docs/fem_dr_modes.png [--h 1.25]

(a) y=15 mm 剖面上的介电常数分布；(b)(c) DR 中截面 z=9.5 mm 上前两个谐振模的 |E| 与电场方向。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emslice import femgen  # noqa: E402
from emslice.solver import Options, Solver  # noqa: E402


def element_field(basis, inner, x):
    full = np.zeros(basis.N, dtype=x.dtype)
    full[inner] = x
    E = basis.interpolate(full).value          # (3, 单元数, 积分点数)
    return np.real(E.mean(axis=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/fem_dr_modes.png")
    ap.add_argument("--h", type=float, default=1.25)
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import skfem

    for fam in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    mesh, _ = femgen.gmsh_mesh(femgen._dr_geo(a.h))
    c = femgen._centroids(mesh)
    eps = np.ones(mesh.t.shape[1])
    eps[femgen._in_cyl(c, 15, 15, 4, 0, 6)] = 2.1
    eps[femgen._in_cyl(c, 15, 15, 8, 6, 13)] = 38.0
    K, M, info = femgen.assemble(mesh, eps)
    basis = skfem.Basis(mesh, skfem.ElementTetN0())
    inner = basis.complement_dofs(basis.get_dofs())
    lam, X, rep = Solver(K, M, Options(k=2, sigma=float(femgen.lam_of_ghz(0.01)), which="above")).solve()
    f = femgen.freq_ghz(lam)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    sel = np.abs(c[1] - 15) < 0.9 * a.h
    tri = mtri.Triangulation(c[0, sel], c[2, sel])
    im = axs[0].tripcolor(tri, eps[sel], shading="gouraud", cmap="YlOrBr")
    axs[0].set_title(f"(a) 剖面 y=15 mm：ε_r（陶瓷 38 / PTFE 2.1 / 空气）\n{info['n_tets']} 四面体，n={K.shape[0]}", fontsize=9)
    axs[0].set_xlabel("x (mm)")
    axs[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    zsel = np.abs(c[2] - 9.5) < 0.9 * a.h * 0.6
    tri = mtri.Triangulation(c[0, zsel], c[1, zsel])
    for j in range(2):
        E = element_field(basis, inner, X[:, j])
        mag = np.linalg.norm(E, axis=0)
        mag = mag / mag.max()
        ax = axs[j + 1]
        ax.tripcolor(tri, mag[zsel], shading="gouraud", cmap="magma")
        q = zsel & (np.arange(c.shape[1]) % 8 == 0)
        ax.quiver(c[0, q], c[1, q], E[0, q], E[1, q], color="w", scale=None, width=0.003, alpha=0.8)
        ax.add_patch(plt.Circle((15, 15), 8, fill=False, ec="c", lw=1, ls="--"))
        ax.set_aspect("equal")
        ax.set_title(f"({'bc'[j]}) 第 {j + 1} 阶模 {f[j]:.3f} GHz：|E| 与方向（z=9.5 mm）", fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(a.out, np.round(f, 4), "证书", rep.certified)


if __name__ == "__main__":
    main()
