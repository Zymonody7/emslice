"""随机压力测试：小规模矩阵上与稠密真值逐一比对（稠密求解只用于评测，求解器本身不稠密化）。

覆盖：nearest / above / below / interval 四种查询；实对称 / 复厄密；标准 / 广义（随机正定 B）；
随机不定稀疏矩阵、近简并大簇（6 个值各重复约百次 + 1e-9 扰动）、Yee 旋度-旋度矩阵；
σ 恰为特征值、σ 在谱外、随机位置；k ∈ {1,2,5,10,20}；单精度/双精度计数；串行/进程池。

判定：特征值集合正确（σ 恰为特征值时“含/不含 σ”两种口径都接受）、证书未报失败、后向误差 ≤ tol、
子空间夹角上界不小于真实夹角。证书报告“未通过”而结果不全的算例单独统计（如实报告，不算静默错误）。

python bench/fuzz.py --seeds 1 2 3 4 5 --cases 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emslice.matgen import preset  # noqa: E402
from emslice.solver import Options, Solver  # noqa: E402


def rand_herm(rng, n, cplx, kind):
    if kind == "cluster":
        base = rng.choice(np.round(rng.uniform(-5, 5, 6), 2), size=n)
        E = sp.random(n, n, density=3.0 / n, random_state=rng.integers(1 << 30))
        M = sp.diags(base) + (1e-9 * (E + E.T) if rng.random() < 0.5 else 0 * E)
    else:
        M = sp.random(n, n, density=6.0 / n, random_state=rng.integers(1 << 30), data_rvs=lambda k: rng.standard_normal(k))
        if cplx:
            M = M + 1j * sp.random(n, n, density=6.0 / n, random_state=rng.integers(1 << 30),
                                   data_rvs=lambda k: rng.standard_normal(k))
        M = M + M.conj().T + sp.diags(rng.uniform(-3, 3, n))
    p = rng.permutation(n)
    return sp.csr_matrix(sp.csr_matrix(M)[p][:, p])


def one_case(rng):
    n = int(rng.integers(150, 700))
    cplx = bool(rng.random() < 0.3)
    kind = "cluster" if rng.random() < 0.3 else "rand"
    gen = rng.random() < 0.3
    if rng.random() < 0.2:
        A = preset(rng.choice(["cube", "puck", "bloch"]), int(rng.integers(4, 7)))[0]
        n, cplx, kind = A.shape[0], bool(np.iscomplexobj(A.data)), "yee"
    else:
        A = rand_herm(rng, n, cplx, kind)
    B = None
    if gen:
        R = sp.random(n, n, density=3.0 / n, random_state=rng.integers(1 << 30))
        B = sp.csr_matrix(sp.identity(n) * 2 + 0.3 * (R + R.T) / max(1, abs(R + R.T).sum(1).max()))
    w = la.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    which = str(rng.choice(["nearest", "nearest", "above", "below", "interval"]))
    k = int(rng.choice([1, 2, 5, 10, 20]))
    mode = rng.random()
    if mode < 0.3:
        sigma = float(rng.choice(w))
    elif mode < 0.4:
        sigma = float(w.max() + 1) if rng.random() < 0.5 else float(w.min() - 1)
    else:
        sigma = float(rng.uniform(w.min(), w.max()))
    kw = dict(k=k, sigma=sigma, which=which, precision=str(rng.choice(["auto", "double"])),
              workers=int(rng.choice([1, 1, 2])), seed=int(rng.integers(100)))
    if which == "interval":
        a, b = sorted(rng.uniform(w.min() - 0.5, w.max() + 0.5, 2))
        kw.update(interval=(float(a), float(b)))
    return A, B, w, kw, dict(n=n, cplx=cplx, kind=kind, gen=gen)


def check(A, B, w, kw, lam, X, rep):
    scale = max(abs(w).max(), 1.0)
    which, sigma, k = kw["which"], kw["sigma"], kw["k"]
    if which == "nearest":
        kk = min(k, w.size)
        rk = np.sort(np.abs(w - sigma))[kk - 1]
        ok = lam.size == kk and np.all(np.abs(lam - sigma) <= rk + 1e-8 * scale)
    elif which in ("above", "below"):
        ok = False
        for eps in (-1e-10 * scale, 1e-10 * scale):
            ref = np.sort(w[w >= sigma + eps])[:k] if which == "above" else np.sort(w[w <= sigma + eps])[-k:]
            ok = ok or (lam.size == ref.size and np.allclose(np.sort(lam), ref, atol=1e-8 * scale))
    else:
        a, b = kw["interval"]
        ref = w[(w >= a) & (w < b)]
        ok = lam.size == ref.size and np.allclose(np.sort(lam), ref, atol=1e-8 * scale)
    angle_ok = True
    sb = rep.residuals.get("per_eig_subspace_sin_bound")
    if ok and sb is not None and B is None and lam.size:
        Wv, V = la.eigh(A.toarray())
        for i in range(lam.size):
            if sb[i] is None:
                continue
            E = V[:, np.abs(Wv - lam[i]) <= 1e-7 * scale]
            x = X[:, i] / np.linalg.norm(X[:, i])
            if np.linalg.norm(x - E @ (E.conj().T @ x)) > sb[i] + 1e-12:
                angle_ok = False
    return ok, angle_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--cases", type=int, default=60)
    a = ap.parse_args()
    stats = dict(total=0, ok=0, honest_uncertified=0, residual_only=0, wrong=0, angle_violation=0, error=0)
    t0 = time.time()
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        for it in range(a.cases):
            A, B, w, kw, info = one_case(rng)
            stats["total"] += 1
            try:
                lam, X, rep = Solver(A, B, Options(**kw)).solve()
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                print("ERROR", seed, it, info, kw, type(e).__name__, str(e)[:160], flush=True)
                continue
            ok, angle_ok = check(A, B, w, kw, lam, X, rep)
            res_ok = rep.residuals["all_below_tol"]
            if not angle_ok:
                stats["angle_violation"] += 1
            if ok and res_ok and rep.certified is not False:
                stats["ok"] += 1
                continue
            if rep.certified is False:
                stats["honest_uncertified"] += 1
                tag = "UNCERTIFIED"
            elif ok and not res_ok:
                stats["residual_only"] += 1
                tag = "RESIDUAL"
            else:
                stats["wrong"] += 1
                tag = "WRONG"
            print(tag, seed, it, info, {k_: v for k_, v in kw.items() if k_ != "interval"},
                  "got", lam.size, "back %.1e" % rep.residuals["backward_max"], rep.diagnostics[:2], flush=True)
    print(f"{stats}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
