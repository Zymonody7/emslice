"""位移矩阵 A − σB 的稀疏直接分解（含惯性）。

后端
  petsc  : PETSc + MUMPS 对称不定 LDLᵀ（sym=2），提供惯性、零主元检测、内存统计、OOC；
  scipy  : SuperLU（仅作无依赖降级模式，不提供惯性）。

复厄密矩阵：MUMPS 的复数 LDLᵀ 是“复对称”而非“厄密”，不能给出厄密惯性。
本实现采用实嵌入  H = Hr + i·Hi  ↦  M = [[Hr, −Hi], [Hi, Hr]]（实对称，2n 阶）：
  * 谱(M) = 谱(H) 每个重数翻倍 ⇒ 惯性(M) = 2 × 惯性(H)；
  * (H − σ)(xr + i·xi) = br + i·bi ⇔ M_σ [xr; xi] = [br; bi]。
其计算量与复数 LU 相当（2n 阶实 LDLᵀ ≈ n 阶复 LU），但同时得到可靠惯性，且只需实数版 PETSc。

固定稀疏结构：ShiftedPattern 预先取 pattern(A) ∪ pattern(B) ∪ diag，对任意 σ 只改数值，
因此 MUMPS 的排序与符号分解（JOB=1）可在多次惯性计数间复用，只做数值分解。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class NearSingularError(RuntimeError):
    pass


def real_embedding(S):
    Sr = sp.csr_matrix(S.real)
    Si = sp.csr_matrix(S.imag)
    return sp.bmat([[Sr, -Si], [Si, Sr]], format="csr")


def _values_on(M, P):
    """把 M 的数值放到超集稀疏结构 P 上（保持 P 的显式零）。"""
    n = P.shape[0]
    rows_p = np.repeat(np.arange(n, dtype=np.int64), np.diff(P.indptr))
    keys_p = rows_p * n + P.indices.astype(np.int64)
    C = sp.coo_matrix(M)
    keys_m = C.row.astype(np.int64) * n + C.col.astype(np.int64)
    pos = np.searchsorted(keys_p, keys_m)
    out = np.zeros(P.nnz, dtype=np.result_type(C.data.dtype, np.float64))
    np.add.at(out, pos, C.data)
    return out


class ShiftedPattern:
    """A − σB 在固定稀疏结构上的数值生成器；复厄密时直接生成实嵌入。"""

    def __init__(self, A, B=None):
        A = sp.csr_matrix(A)
        n = A.shape[0]
        Bm = sp.identity(n, format="csr") if B is None else sp.csr_matrix(B)
        P = (abs(A) + abs(Bm) + sp.identity(n, format="csr")).tocsr()
        P.sum_duplicates()
        P.sort_indices()
        a = _values_on(A, P)
        b = _values_on(Bm, P)
        self.n = n
        self.cplx = bool(np.iscomplexobj(a) and np.any(a.imag)) or bool(np.iscomplexobj(b) and np.any(b.imag))
        self.P = P
        if self.cplx:
            mk = lambda v: sp.csr_matrix((v.astype(complex), P.indices, P.indptr), shape=P.shape)
            EA = real_embedding(mk(a))
            EB = real_embedding(mk(b))
            EA.sort_indices()
            EB.sort_indices()
            assert np.array_equal(EA.indices, EB.indices) and np.array_equal(EA.indptr, EB.indptr)
            self.indptr, self.indices, self.a, self.b = EA.indptr, EA.indices, EA.data, EB.data
            self.ca, self.cb = a, b
        else:
            self.indptr, self.indices = P.indptr, P.indices
            self.a, self.b = np.real(a).astype(np.float64), np.real(b).astype(np.float64)
            self.ca, self.cb = self.a, self.b
        self.order = len(self.indptr) - 1

    def real_data(self, sigma):
        """实对称（或实嵌入）矩阵 A − σB 的 CSR 数值。"""
        return self.a - sigma * self.b

    def shifted(self, sigma):
        """原始（可能复数）的 A − σB，用于探针残差等。"""
        return sp.csr_matrix((self.ca - sigma * self.cb, self.P.indices, self.P.indptr), shape=self.P.shape)


class Factorization:
    inertia: tuple | None = None

    def __init__(self, n, cplx, sigma):
        self.n, self.cplx, self.sigma = n, cplx, sigma
        self.stats: dict = {}

    def solve(self, X):
        X = np.asarray(X)
        one = X.ndim == 1
        X2 = X[:, None] if one else X
        if self.cplx:
            Y = self._solve_real(np.vstack([X2.real, X2.imag]))
            out = Y[: self.n] + 1j * Y[self.n :]
        elif np.iscomplexobj(X2):
            Y = self._solve_real(np.hstack([X2.real, X2.imag]))
            p = X2.shape[1]
            out = Y[:, :p] + 1j * Y[:, p:]
        else:
            out = self._solve_real(X2)
        return out[:, 0] if one else out

    def _solve_real(self, R):  # pragma: no cover
        raise NotImplementedError

    def probe(self, nS, seed=12345):
        """一次随机右端求解估计 ‖S⁻¹‖·‖S‖（条件数下界）。≥1e12 视为位移贴近特征值（近奇异）。"""
        rng = np.random.default_rng(seed)
        b = rng.standard_normal(self.n)
        if self.cplx:
            b = b + 1j * rng.standard_normal(self.n)
        x = self.solve(b)
        growth = float(np.abs(x).max() * nS / np.abs(b).max())
        self.stats["growth"] = growth if np.isfinite(growth) else float("inf")
        return self.stats["growth"]

    def free(self):
        pass


# --------------------------------------------------------------------- PETSc
_PETSC_INIT = False


def _petsc():
    global _PETSC_INIT
    if not _PETSC_INIT:
        if sys.platform == "darwin":
            # macOS 上 MPICH 默认会选中 VPN 虚拟网卡 (utunX)，导致初始化极慢且退出时报错；
            # 单进程运行时改用本地回环上的 UDP 传输即可。
            os.environ.setdefault("FI_PROVIDER", "udp;ofi_rxd")
            os.environ.setdefault("FI_UDP_IFACE", "lo0")
        import petsc4py

        petsc4py.init([])
        _PETSC_INIT = True
    from petsc4py import PETSc

    if np.dtype(PETSc.ScalarType).kind == "c":
        raise RuntimeError("需要实数版 PETSc（复数矩阵通过实嵌入处理）")
    return PETSc


class PetscSlot:
    """一套 PETSc Mat + KSP(Cholesky/MUMPS)。同一稀疏结构上重复分解时复用符号分析。"""

    ORDERINGS = {"auto": 7, "amd": 0, "amf": 2, "scotch": 3, "pord": 4, "metis": 5, "qamd": 6}

    def __init__(self, pattern: ShiftedPattern, ordering="auto", ooc=False, null_tol=None, mem_relax=40, blr=None,
                 count_only=False):
        PETSc = self.PETSc = _petsc()
        it = PETSc.IntType
        self.indptr = pattern.indptr.astype(it)
        self.indices = pattern.indices.astype(it)
        self.mat = PETSc.Mat().createAIJ(size=(pattern.order, pattern.order),
                                         csr=(self.indptr, self.indices, pattern.a.copy()), comm=PETSc.COMM_SELF)
        self.mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.mat.assemble()
        ksp = self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(self.mat)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        F = self.F = pc.getFactorMatrix()
        F.setMumpsIcntl(7, self.ORDERINGS.get(ordering, 7))
        F.setMumpsIcntl(13, 1)        # 根节点不并行化，保证惯性可得
        F.setMumpsIcntl(14, mem_relax)
        F.setMumpsIcntl(24, 1)        # 零主元检测
        if null_tol is not None:
            F.setMumpsCntl(3, null_tol)
        if ooc:
            F.setMumpsIcntl(22, 1)
        if blr:                        # 块低秩压缩（BLR），blr = 截断精度 ε
            F.setMumpsIcntl(35, 2)
            F.setMumpsCntl(7, float(blr))
        if count_only:
            # 只要惯性：分解后丢弃因子（ICNTL(31)=1），内存约降一半；
            # 近奇异改用相对阈值零主元检测（|主元| ≤ 1e-10·‖A‖ 视为零），实测可识别 1e-13 相对距离内的特征值
            F.setMumpsIcntl(31, 1)
            F.setMumpsCntl(3, 1e-10 if null_tol is None else null_tol)
        self.analyzed = False
        self.ooc, self.blr, self.count_only = ooc, blr, count_only

    def factor(self, data):
        t0 = time.perf_counter()
        self.mat.setValuesCSR(self.indptr, self.indices, data)
        self.mat.assemble()
        self.ksp.setOperators(self.mat)
        try:
            self.ksp.setUp()
        except self.PETSc.Error as e:
            raise NearSingularError(f"MUMPS 分解失败: {e}") from e
        dt = time.perf_counter() - t0
        reused = self.analyzed
        self.analyzed = True
        return dt, reused

    def solve_dense(self, R):
        PETSc = self.PETSc
        n2, p = R.shape
        Bm = PETSc.Mat().createDense((n2, p), array=np.asfortranarray(R, dtype=np.float64), comm=PETSc.COMM_SELF)
        Xm = PETSc.Mat().createDense((n2, p), comm=PETSc.COMM_SELF)
        Xm.setUp()
        self.F.matSolve(Bm, Xm)
        Y = np.array(Xm.getDenseArray(), copy=True)
        Bm.destroy()
        Xm.destroy()
        return Y

    def destroy(self):
        if self.ksp is not None:
            self.ksp.destroy()
            self.mat.destroy()
            self.ksp = self.mat = self.F = None


class PetscMumpsFactor(Factorization):
    def __init__(self, pattern: ShiftedPattern, sigma, slot: PetscSlot | None = None, need_solve=True, nS=None, **slot_kw):
        super().__init__(pattern.n, pattern.cplx, sigma)
        self._owned = slot is None
        self._slot = slot or PetscSlot(pattern, **slot_kw)
        dt, reused = self._slot.factor(pattern.real_data(sigma))
        F = self._slot.F
        nneg, nzero, npos = F.getInertia()
        div = 2 if pattern.cplx else 1
        self.inertia = (int(nneg) // div, int(nzero) // div, int(npos) // div)
        infog = F.getMumpsInfog
        nfac = infog(29)
        nfac = -nfac * 1_000_000 if nfac < 0 else nfac
        self.stats = dict(
            backend="petsc-mumps",
            order=int(pattern.order),
            embedded_complex=bool(pattern.cplx),
            seconds_factor=dt,
            symbolic_reused=bool(reused),
            factor_entries=int(nfac),
            flops=float(F.getMumpsRinfog(3)),
            mem_mb_est=int(infog(17)),
            mem_mb_used=int(infog(22)),
            null_pivots=int(infog(28)),
            ooc=bool(self._slot.ooc),
            blr=self._slot.blr,
        )
        self.stats["count_only"] = bool(self._slot.count_only)
        if not self._slot.count_only:  # 因子已丢弃的计数分解无法求解，近奇异由零主元检测负责
            t = time.perf_counter()
            self.probe(nS if nS is not None else 1.0)
            self.stats["seconds_probe"] = time.perf_counter() - t
        if not need_solve:
            self.free()

    def _solve_real(self, R):
        return self._slot.solve_dense(R)

    def free(self):
        if self._owned and self._slot is not None:
            self._slot.destroy()
        self._slot = None


# --------------------------------------------------------------------- SciPy
class SuperLUFactor(Factorization):
    def __init__(self, pattern: ShiftedPattern, sigma, need_solve=True, nS=None, **_):
        super().__init__(pattern.n, pattern.cplx, sigma)
        M = sp.csr_matrix((pattern.real_data(sigma), pattern.indices, pattern.indptr), shape=(pattern.order,) * 2)
        t0 = time.perf_counter()
        try:
            self._lu = spla.splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        except RuntimeError as e:
            raise NearSingularError(str(e)) from e
        self.stats = dict(backend="scipy-superlu", order=int(M.shape[0]), embedded_complex=bool(pattern.cplx),
                          seconds_factor=time.perf_counter() - t0,
                          factor_entries=int(self._lu.L.nnz + self._lu.U.nnz))
        self.inertia = None
        self.probe(nS if nS is not None else 1.0)

    def _solve_real(self, R):
        return self._lu.solve(R)

    def free(self):
        self._lu = None


# ------------------------------------------------------------------ 调度入口
def available_backends():
    out = []
    try:
        _petsc()
        out.append("petsc")
    except Exception:
        pass
    out.append("scipy")
    return out


def factorize(pattern: ShiftedPattern, sigma, backend="auto", ordering="auto", ooc="auto", need_solve=True,
              slot: PetscSlot | None = None, nS=None, blr=None, **_):
    if backend == "auto":
        backend = available_backends()[0]
    use_ooc = bool(ooc) if ooc != "auto" else False
    if backend == "petsc":
        return PetscMumpsFactor(pattern, sigma, slot=slot, need_solve=need_solve, nS=nS, ordering=ordering,
                                ooc=use_ooc, blr=blr)
    if backend == "scipy":
        return SuperLUFactor(pattern, sigma, need_solve=need_solve, nS=nS)
    raise ValueError(f"未知后端 {backend}")
