"""基准算例集：生成矩阵 + 查询 + 真值（解析解可得时）。

python bench/cases.py --out data/bench          # 生成全部 ~10 万阶算例
python bench/cases.py --out data/small --N 10   # 小规模冒烟测试

每个算例覆盖任务书要求的一类典型场景：
  mid_band       目标频段（σ 位于谱中部）的常规求解
  degenerate     立方腔高重数特征值（重数 3/6）
  near_zero      零附近密集谱 / 低频击穿（梯度零空间维数 ~3 万）
  near_singular  σ 与谐振点相差 1e-13 量级（A−σI 近奇异）
  complex_herm   Bloch 周期边界，复厄密矩阵
  dielectric     介质谐振器（非均匀，真实器件结构）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emslice.io import save_matrix  # noqa: E402
from emslice.matgen import YeeGrid, analytic_eigs_homogeneous, preset, yee_curlcurl  # noqa: E402


def nearest_ref(ev, sigma, k):
    d = np.abs(ev - sigma)
    idx = np.argsort(d, kind="stable")[:k]
    return np.sort(ev[idx])


def build(out, N):
    os.makedirs(out, exist_ok=True)
    cases = []

    # ---- 1. 立方腔：重特征值 + 零空间 + 近奇异 ----------------------------
    A, g = preset("cube", N)
    f = os.path.join(out, f"cube{N}.npz")
    save_matrix(f, A)
    ev = analytic_eigs_homogeneous(g)
    phys = ev[ev > 1e-9 * ev.max()]
    null_dim = int(np.sum(ev <= 1e-9 * ev.max()))
    mid = float(np.median(phys[: len(phys) // 4]))
    cases += [
        dict(name="cube_mid_band", scenario="mid_band+degenerate", matrix=f, k=20, sigma=mid, which="nearest",
             ref=nearest_ref(ev, mid, 20).tolist()),
        dict(name="cube_low_freq", scenario="near_zero", matrix=f, k=20, sigma=1e-6 * float(phys[0]), which="above",
             ref=np.sort(ev[ev >= 1e-6 * phys[0]])[:20].tolist(), null_dim=null_dim),
        dict(name="cube_near_singular", scenario="near_singular", matrix=f, k=12,
             sigma=float(phys[37] * (1 + 1e-13)), which="nearest", ref=nearest_ref(ev, float(phys[37] * (1 + 1e-13)), 12).tolist()),
        dict(name="cube_null_cluster", scenario="near_zero", matrix=f, k=16, sigma=0.0, which="nearest",
             ref=[0.0] * 16, null_dim=null_dim),
    ]

    # ---- 2. 长方体腔（三边不等，简单谱） --------------------------------------
    A, g = preset("cavity", N)
    f = os.path.join(out, f"cavity{N}.npz")
    save_matrix(f, sp_upper(A), )
    ev = analytic_eigs_homogeneous(g)
    phys = ev[ev > 1e-9 * ev.max()]
    s = float(phys[len(phys) // 10])
    cases.append(dict(name="cavity_upper_storage", scenario="mid_band+triangle_input", matrix=f, k=30, sigma=s,
                      which="nearest", ref=nearest_ref(ev, s, 30).tolist()))

    # ---- 3. Bloch 周期均匀介质（复厄密，有解析解） ----------------------------
    A, g = preset("bloch-homog", N)
    f = os.path.join(out, f"bloch_homog{N}.npz")
    save_matrix(f, A)
    ev = analytic_eigs_homogeneous(g)
    phys = ev[ev > 1e-9 * ev.max()]
    s = float(phys[40])
    cases.append(dict(name="bloch_complex_hermitian", scenario="complex_herm", matrix=f, k=20, sigma=s,
                      which="nearest", ref=nearest_ref(ev, s, 20).tolist()))

    # ---- 4. 介质谐振器腔（无解析解；用独立求解器交叉验证） --------------------
    A, g = preset("puck", N)
    f = os.path.join(out, f"puck{N}.npz")
    save_matrix(f, A)
    cases.append(dict(name="dielectric_puck", scenario="dielectric", matrix=f, k=20, sigma=40.0, which="above", ref=None))

    # ---- 5. 光子晶体单胞（复厄密 + 非均匀） -----------------------------------
    A, g = preset("bloch", N)
    f = os.path.join(out, f"bloch_rods{N}.npz")
    save_matrix(f, A)
    cases.append(dict(name="photonic_crystal", scenario="complex_herm+dielectric", matrix=f, k=16, sigma=60.0,
                      which="nearest", ref=None))

    # ---- 6. 公开电磁矩阵（需先运行 bench/get_public.sh） --------------------
    pub = os.path.join(os.path.dirname(out.rstrip("/")), "suitesparse", "2cubes_sphere.npz")
    if os.path.exists(pub):
        cases += [
            dict(name="suitesparse_2cubes_low", scenario="public_em+wide_dynamic_range", matrix=pub, k=30, sigma=0.0,
                 which="above", ref=None),
            dict(name="suitesparse_2cubes_band", scenario="public_em+mid_band", matrix=pub, k=20, sigma=40.0,
                 which="nearest", ref=None),
        ]

    with open(os.path.join(out, "cases.json"), "w", encoding="utf-8") as fh:
        json.dump(cases, fh, ensure_ascii=False, indent=1)
    for c in cases:
        print(c["name"], c["scenario"], os.path.basename(c["matrix"]))
    return cases


def sp_upper(A):
    import scipy.sparse as sp

    return sp.triu(A, format="csr")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--N", type=int, default=33)
    a = ap.parse_args()
    build(a.out, a.N)
