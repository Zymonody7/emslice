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
def test_parallel_pool_matches_serial():
    A, ev = _cube(6)
    phys = ev[ev > 1e-9]
    for which, sigma in (("nearest", float(phys[30])), ("above", 1e-3)):
        r1 = Solver(A, None, Options(k=8, sigma=sigma, which=which, workers=1)).solve()
        r2 = Solver(A, None, Options(k=8, sigma=sigma, which=which, workers=2)).solve()
        assert r1[2].certified and r2[2].certified and r2[2].workers == 2
        assert np.allclose(r1[0], r2[0], rtol=1e-10, atol=1e-10)
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
