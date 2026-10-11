import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from emslice.factor import available_backends
from emslice.io import InputError, load_matrix, restore_hermitian, save_matrix
from emslice.krylov import krylov_schur
from emslice.matgen import YeeGrid, analytic_eigs_homogeneous, preset, yee_curlcurl
from emslice.solver import Options, Solver

HAS_INERTIA = "petsc" in available_backends()
needs_inertia = pytest.mark.skipif(not HAS_INERTIA, reason="需要 PETSc+MUMPS 后端")


def _has_fem(gmsh=False):
    try:
        import skfem  # noqa: F401

        if gmsh:
            from emslice.femgen import _gmsh_cmd

            _gmsh_cmd()
        return True
    except (ImportError, FileNotFoundError):
        return False


needs_fem = pytest.mark.skipif(not _has_fem(), reason="需要 scikit-fem")
needs_gmsh = pytest.mark.skipif(not _has_fem(gmsh=True), reason="需要 scikit-fem 与 gmsh")


# ------------------------------------------------------------------ 生成器
@pytest.mark.parametrize(
    "grid",
    [
        YeeGrid(n=(4, 5, 3), length=(1, 1.3, 0.8)),
        YeeGrid(n=(4, 4, 4)),
        YeeGrid(n=(4, 3, 5), bc=("periodic",) * 3, bloch_phase=(0.7, 0.3, 0.1)),
    ],
)
def test_matgen_matches_analytic(grid):
    A = yee_curlcurl(grid)
    ev = np.linalg.eigvalsh(A.toarray())
    ref = analytic_eigs_homogeneous(grid)
    assert len(ev) == len(ref)
    assert np.abs(np.sort(ev) - ref).max() <= 1e-12 * ev.max()


# --------------------------------------------------------------------- I/O
def test_triangle_restore_and_formats(tmp_path):
    A, _ = preset("bloch", 4)
    U = sp.triu(A, format="csr")
    full, how = restore_hermitian(U)
    assert how == "upper" and abs(full - A).max() < 1e-14
    for name, obj in [("a.npz", U), ("b.mtx", U)]:
        save_matrix(tmp_path / name, obj)
        B, info = load_matrix(tmp_path / name)
        assert abs(B - A).max() < 1e-12 and info["storage"] in ("upper", "full")
    scipy.io.savemat(tmp_path / "c.mat", {"K": A})
    C, _ = load_matrix(tmp_path / "c.mat")
    assert abs(C - A).max() < 1e-14
    np.savez(tmp_path / "coo.npz", row=A.tocoo().row, col=A.tocoo().col, data=A.tocoo().data, shape=A.shape)
    D, _ = load_matrix(tmp_path / "coo.npz")
    assert abs(D - A).max() < 1e-14


def test_rejects_non_hermitian_and_pickle(tmp_path):
    M = sp.random(20, 20, density=0.3, random_state=0, format="csr") + sp.identity(20)
    save_matrix(tmp_path / "m.npz", M)
    with pytest.raises(InputError):
        load_matrix(tmp_path / "m.npz", triangle="full")
    np.save(tmp_path / "obj.npy", np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_matrix(tmp_path / "obj.npy")


# ------------------------------------------------------------------ Krylov
def test_krylov_multiplicity_with_block():
    rng = np.random.default_rng(0)
    lam = np.sort(rng.uniform(-5, 5, 800))
    lam[10:14] = 0.01  # 重数 4 的特征值，离位移最近
    d = 1.0 / lam
    r = krylov_schur(lambda X: d[:, None] * X, 800, 6, block=4, tol=1e-12)
    got = np.sort(1.0 / r.theta)
    ref = np.sort(lam[np.argsort(np.abs(lam))[:6]])
    assert r.converged and np.allclose(got, ref, atol=1e-10)


# ------------------------------------------------------------------ 求解器
def _cube(N=6):
    A, g = preset("cube", N)
    return A, analytic_eigs_homogeneous(g)


@needs_inertia
def test_nearest_certified_degenerate():
    A, ev = _cube(6)
    phys = ev[ev > 1e-9]
    sigma = float(phys[30])
    lam, X, rep = Solver(A, None, Options(k=10, sigma=sigma, tol=1e-10)).solve()
    ref = np.sort(ev[np.argsort(np.abs(ev - sigma))[:10]])
    assert rep.certified
    assert np.abs(np.sort(lam) - ref).max() < 1e-8 * ev.max()
    assert rep.residuals["all_below_tol"]


@needs_inertia
@pytest.mark.parametrize("precision", ["auto", "double"])
def test_mixed_precision_certificate_and_angle_bound(precision):
    """单精度计数的外窗口证书（含双精度对照）：结果与稠密真值一致，子空间夹角上界不小于真实夹角。"""
    A, _ = preset("puck", 7)
    w, V = np.linalg.eigh(A.toarray())
    sigma = 150.0
    lam, X, rep = Solver(A, None, Options(k=10, sigma=sigma, precision=precision)).solve()
    ref = np.sort(w[np.argsort(np.abs(w - sigma))[:10]])
    assert rep.certified and np.abs(np.sort(lam) - ref).max() < 1e-8 * w.max()
    if precision == "auto":
        assert "single" in rep.certificate["count_precision"]
        assert any(f.get("precision") == "single" for f in rep.factorizations)
    else:
        assert all(f.get("precision", "double") == "double" for f in rep.factorizations)
    bounds = rep.residuals["per_eig_subspace_sin_bound"]
    for i in range(lam.size):
        if bounds[i] is None:
            continue
        E = V[:, np.abs(w - lam[i]) < 1e-6 * w.max()]
        x = X[:, i] / np.linalg.norm(X[:, i])
        assert np.linalg.norm(x - E @ (E.T @ x)) <= bounds[i] + 1e-14


@needs_inertia
def test_above_skips_null_space():
    A, ev = _cube(6)
    phys = ev[ev > 1e-9]
    lam, X, rep = Solver(A, None, Options(k=8, sigma=1e-3, which="above")).solve()
    assert rep.certified and np.allclose(np.sort(lam), phys[:8], rtol=1e-9)


@needs_inertia
def test_near_singular_shift():
    A, ev = _cube(6)
    phys = ev[ev > 1e-9]
    sigma = float(phys[20]) * (1 + 1e-15)
    lam, X, rep = Solver(A, None, Options(k=6, sigma=sigma)).solve()
    ref = np.sort(ev[np.argsort(np.abs(ev - sigma))[:6]])
    assert np.abs(np.sort(lam) - ref).max() < 1e-8 * ev.max()
    assert rep.residuals["all_below_tol"]


@needs_inertia
def test_complex_hermitian_bloch():
    A, g = preset("bloch-homog", 5)
    ev = analytic_eigs_homogeneous(g)
    phys = ev[ev > 1e-9]
    sigma = float(phys[25])
    lam, X, rep = Solver(A, None, Options(k=8, sigma=sigma)).solve()
    ref = np.sort(ev[np.argsort(np.abs(ev - sigma))[:8]])
    assert rep.certified and np.abs(np.sort(lam) - ref).max() < 1e-8 * ev.max()


@needs_inertia
def test_interval_slicing_matches_count():
    A, ev = _cube(6)
    a, b = 60.0, 400.0
    lam, X, rep = Solver(A, None, Options(which="interval", interval=(a, b), slice_max=12)).solve()
    ref = ev[(ev >= a) & (ev < b)]
    assert rep.certified and lam.size == ref.size
    assert np.abs(np.sort(lam) - ref).max() < 1e-8 * ev.max()


@needs_inertia
def test_generalized_problem():
    A, ev = _cube(5)
    n = A.shape[0]
    d = np.linspace(1.0, 2.0, n)
    Bm = sp.diags(d).tocsr()
    s = sp.diags(1 / np.sqrt(d))
    ref = np.linalg.eigvalsh((s @ A @ s).toarray())
    sigma = float(ref[ref > 1e-9][15])
    lam, X, rep = Solver(A, Bm, Options(k=6, sigma=sigma)).solve()
    want = np.sort(ref[np.argsort(np.abs(ref - sigma))[:6]])
    assert rep.certified and np.abs(np.sort(lam) - want).max() < 1e-8 * ref.max()


@needs_inertia
def test_generalized_angle_bound():
    """广义问题的子空间夹角上界（‖R‖_{B⁻¹} 由共轭梯度求得）在 B-内积下不小于真实夹角。"""
    A, _ = preset("puck", 6)
    n = A.shape[0]
    R = sp.random(n, n, density=3.0 / n, random_state=1)
    Bm = sp.csr_matrix(sp.identity(n) * 2 + 0.3 * (R + R.T) / abs(R + R.T).sum(1).max())
    import scipy.linalg as sla

    w, V = sla.eigh(A.toarray(), Bm.toarray())
    L = np.linalg.cholesky(Bm.toarray())
    lam, X, rep = Solver(A, Bm, Options(k=10, sigma=float(np.median(w)))).solve()
    bounds = rep.residuals["per_eig_subspace_sin_bound"]
    assert rep.certified and any(b is not None for b in bounds)
    for i, b in enumerate(bounds):
        if b is None:
            continue
        E = np.linalg.qr(L.T @ V[:, np.abs(w - lam[i]) < 1e-7 * abs(w).max()])[0]
        y = L.T @ X[:, i]
        y /= np.linalg.norm(y)
        assert np.linalg.norm(y - E @ (E.T @ y)) <= b + 1e-14


def _clustered(n_each, values, seed):
    """近简并簇：若干值各重复多次，再加 1e-9 量级的随机对称扰动（簇宽 ~1e-9）。"""
    rng = np.random.default_rng(seed)
    base = np.repeat(values, n_each)
    n = base.size
    E = sp.random(n, n, density=3.0 / n, random_state=seed)
    A = sp.csr_matrix(sp.diags(base) + 1e-9 * (E + E.T))
    p = rng.permutation(n)
    A = sp.csr_matrix(A[p][:, p])
    return A, np.linalg.eigvalsh(A.toarray())


@needs_inertia
def test_interval_whole_spectrum_of_small_matrix():
    """区间含小矩阵的全部特征值（6 个值各约 30 重）：所需 Krylov 基覆盖整个空间，直接 Rayleigh-Ritz 一次求全
    （此前基维数受 n − 锁定数所限，每轮收缩只多得 1 个，8 轮后仍缺 7 个）。"""
    A, w = _clustered([34, 38, 33, 26, 38, 30], [0.77, 1.27, 1.71, 3.19, 3.32, 3.4], 2)
    lam, _, rep = Solver(A, None, Options(k=20, which="interval", interval=(0.5, 3.8))).solve()
    assert rep.certified and lam.size == A.shape[0]
    assert np.allclose(np.sort(lam), w, atol=1e-12)


@needs_inertia
def test_narrow_slice_keeps_shift_inside():
    """切片窄于标准扰动量（1e-10·‖A‖）且中点贴着特征值：位移扰动须留在片内，否则 Krylov 找的是片外特征值。"""
    A, w = _clustered([30, 30], [1.0, 4.42], 5)
    hi = w[w > 4]
    gaps = np.diff(hi)
    j = 1 + int(np.argmax(np.minimum(gaps[:-1], gaps[1:])))   # 两侧间隙都较大的簇内特征值
    h = 0.5 * min(gaps[j - 1], gaps[j])
    a, b = hi[j] - h, hi[j] + h                                # 中点恰为特征值，片宽 ≪ 1e-10·‖A‖
    assert b - a < 1e-10 * abs(A).sum(0).max()
    lam, _, rep = Solver(A, None, Options(k=1, which="interval", interval=(float(a), float(b)))).solve()
    ref = w[(w >= a) & (w < b)]
    assert rep.certified and lam.size == ref.size and np.allclose(lam, ref, atol=1e-13)


@needs_inertia
def test_nearest_window_reaching_huge_cluster():
    """最近 k 个的窗口跨进超大精确简并簇（如梯度零空间）：块 Krylov 只看到簇的少数方向、用更远的特征值凑满 k 个，
    证书计数发现缺口远大于 k；应核对簇以内无遗漏、簇重数足够，改取簇内向量补足（与第 k 个等距的并列）。"""
    rng = np.random.default_rng(4)
    inner = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.6])                      # 距 σ=1 小于 1
    vals = np.concatenate([inner, np.zeros(300), rng.uniform(2.2, 3.2, 100), rng.uniform(-2.0, -1.5, 50)])
    A = sp.diags(vals[rng.permutation(vals.size)]).tocsr()
    lam, X, rep = Solver(A, None, Options(k=30, sigma=1.0)).solve()
    assert rep.certified and lam.size == 30
    assert np.sum(np.abs(lam) < 1e-12) == 24
    assert np.allclose(np.sort(lam[np.abs(lam) > 1e-12]), inner, atol=1e-12)
    assert np.allclose(A @ X - X * lam, 0, atol=1e-10)


@needs_inertia
def test_generalized_scale_invariance():
    """(A, B) 同乘常数（如 mm 与 m 单位制）特征值不变：单精度计数半径 η、端点精度规划与结果都应与尺度无关。
    η 若按 ‖A‖+|x|‖B‖ 计会随尺度缩放：同乘 1e-6 时声称的 η 比实际误差小百万倍，同乘 1e6 时全程退回双精度。"""
    A, _ = preset("puck", 6)
    n = A.shape[0]
    R = sp.random(n, n, density=3.0 / n, random_state=1)
    Bm = sp.csr_matrix(sp.identity(n) * 2 + 0.3 * (R + R.T) / abs(R + R.T).sum(1).max())
    ref = None
    for c in (1.0, 1e-6, 1e6):
        lam, _, rep = Solver(c * A, c * Bm, Options(k=10, sigma=124.0)).solve()
        cert = rep.certificate
        assert rep.certified and cert["kind"] == "mixed-window"
        got = (np.sort(lam), cert["count_precision"], np.asarray(cert["eta"], float))
        if ref is None:
            ref = got
            continue
        assert np.allclose(got[0], ref[0], rtol=1e-12, atol=0)
        assert got[1] == ref[1]
        assert np.allclose(got[2], ref[2], rtol=1e-9)


@needs_inertia
def test_memory_limit_switches_to_out_of_core(tmp_path, monkeypatch):
    """内存上限过小时 MUMPS 报 INFOG(1)=-19（PETSc 不抛异常，惯性无效）：须检出并自动改用外存重试，结果不变。"""
    monkeypatch.setenv("MUMPS_OOC_TMPDIR", str(tmp_path))
    A, _ = preset("cube", 20)          # n≈2.4 万，在核分解约 80 MB
    r0 = Solver(A, None, Options(k=8, sigma=500.0)).solve()
    r1 = Solver(A, None, Options(k=8, sigma=500.0, mem_limit_gb=0.04)).solve()
    assert r0[2].certified and r1[2].certified
    assert np.allclose(np.sort(r0[0]), np.sort(r1[0]), rtol=1e-10, atol=1e-10)
    assert any("外存" in d for d in r1[2].diagnostics) and any(f.get("ooc") for f in r1[2].factorizations)


@needs_inertia
def test_parallel_pool_matches_serial():
    A, ev = _cube(6)
    phys = ev[ev > 1e-9]
    for which, sigma in (("nearest", float(phys[30])), ("above", 1e-3)):
        r1 = Solver(A, None, Options(k=8, sigma=sigma, which=which, workers=1)).solve()
        r2 = Solver(A, None, Options(k=8, sigma=sigma, which=which, workers=2)).solve()
        assert r1[2].certified and r2[2].certified and r2[2].workers == 2
        assert np.allclose(r1[0], r2[0], rtol=1e-10, atol=1e-10)
        if which == "above":   # 括区计数成批并发（nearest 的 σ 恰在簇上，只需主进程一次计数）
            assert any("worker" in f for f in r2[2].factorizations)


@needs_inertia
def test_parallel_interval_slices():
    A, ev = _cube(7)
    a, b = 50.0, 500.0
    lam, X, rep = Solver(A, None, Options(which="interval", interval=(a, b), slice_max=10, workers=2)).solve()
    ref = ev[(ev >= a) & (ev < b)]
    assert rep.certified and lam.size == ref.size and len(rep.certificate["slices"]) > 1
    assert np.abs(np.sort(lam) - ref).max() < 1e-8 * ev.max()
    assert any(r.get("seconds") is not None for r in rep.krylov_runs)


def test_degraded_backend_without_inertia():
    """仅 SciPy SuperLU（无惯性）时三种模式都能求解，只是不出证书。"""
    A, ev = _cube(5)
    phys = ev[ev > 1e-9]
    sigma = float(phys[20])
    lam, X, rep = Solver(A, None, Options(k=6, sigma=sigma, backend="scipy")).solve()
    ref = np.sort(ev[np.argsort(np.abs(ev - sigma))[:6]])
    assert rep.certified is None and any("[降级]" in d for d in rep.diagnostics)
    assert np.abs(np.sort(lam) - ref).max() < 1e-8 * ev.max() and rep.residuals["all_below_tol"]
    s_gap = 0.5 * (phys[7] + phys[phys > phys[7] * (1 + 1e-9)][0])   # 落在谱间隙中
    lam, X, rep = Solver(A, None, Options(k=6, sigma=s_gap, which="above", backend="scipy")).solve()
    assert np.allclose(lam, phys[phys > s_gap][:6], rtol=1e-9)
    a, b = float(phys[5]) - 1e-6, float(phys[25]) + 1e-6
    lam, X, rep = Solver(A, None, Options(which="interval", interval=(a, b), slice_max=8, backend="scipy")).solve()
    assert lam.size == np.sum((ev >= a) & (ev <= b))


# ------------------------------------------------------------------ 有限元广义问题
def _dense_gen(K, M):
    import scipy.linalg as la

    return la.eigh(K.toarray(), M.toarray(), eigvals_only=True)


@needs_fem
def test_fem_cavity_null_space_and_analytic():
    from emslice.femgen import analytic_cavity, fem_cavity

    K, M, info = fem_cavity(h=4.0)
    w = _dense_gen(K, M)
    assert np.sum(w < 1e-8) == info["null_dim"]                # 梯度零空间维数 = 内部节点数
    phys = w[w > 1e-8]
    ref = analytic_cavity((30.0, 24.0, 18.0), 0.2)
    assert np.abs(phys[:3] / ref[:3] - 1).max() < 0.01         # 粗网格（h=4 mm）下前几阶 k² 与连续解析解相差 < 1%


@needs_fem
@needs_inertia
def test_fem_generalized_certified():
    from emslice.femgen import fem_cavity

    K, M, info = fem_cavity(h=4.0)
    w = _dense_gen(K, M)
    phys = w[w > 1e-8]
    lam, X, rep = Solver(K, M, Options(k=10, sigma=1e-6, which="above")).solve()
    assert rep.certified and np.abs(lam - phys[:10]).max() < 1e-9 * phys[9]
    s = float(0.5 * (phys[20] + phys[21]))
    lam, X, rep = Solver(K, M, Options(k=12, sigma=s)).solve()
    ref = np.sort(w[np.argsort(np.abs(w - s))[:12]])
    assert rep.certified and np.abs(np.sort(lam) - ref).max() < 1e-9 * ref.max()


@needs_gmsh
@needs_inertia
def test_fem_ferrite_complex_hermitian_generalized():
    from emslice.femgen import fem_ferrite

    K, M, info = fem_ferrite(h=6.0)
    assert np.iscomplexobj(K.data) and abs(K - K.conj().T).max() == 0
    w = _dense_gen(K, M)
    phys = w[w > 1e-8]
    s = float(0.5 * (phys[10] + phys[11]))
    lam, X, rep = Solver(K, M, Options(k=8, sigma=s)).solve()
    ref = np.sort(w[np.argsort(np.abs(w - s))[:8]])
    assert rep.certified and np.abs(np.sort(lam) - ref).max() < 1e-9 * ref.max()
