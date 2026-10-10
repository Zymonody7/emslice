"""对比基线：
  scipy  — scipy.sparse.linalg.eigsh 位移-求逆（ARPACK 隐式重启 Lanczos + SuperLU），最常用的开源做法；
  slepc  — SLEPc Krylov-Schur + ST(sinvert) + MUMPS，业界标准开源方案（默认参数）。
           复厄密矩阵在实数版 SLEPc 上以实嵌入求 2k 个再还原。
两者均不提供“目标频段无遗漏”的保证，这正是 emslice 惯性证书要解决的问题。
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _scipy(A, k, sigma, which):
    w = {"nearest": "LM", "above": "LA", "below": "SA"}[which]
    lam, X = spla.eigsh(A, k=k, sigma=sigma, which=w, tol=1e-12)
    return lam, X, dict(method="scipy.eigsh(shift-invert, SuperLU)")


def _complex_from_embedding(A, lam2, U):
    n = A.shape[0]
    Xc = U[:n] + 1j * U[n:]
    Q, s, _ = la.svd(Xc, full_matrices=False)
    r = int(np.sum(s > 1e-8 * s[0]))
    Q = Q[:, :r]
    H = Q.conj().T @ (A @ Q)
    w, S = la.eigh(0.5 * (H + H.conj().T))
    return w, Q @ S


def _slepc(A, k, sigma, which):
    import os
    import sys

    if sys.platform == "darwin":
        os.environ.setdefault("FI_PROVIDER", "udp;ofi_rxd")
        os.environ.setdefault("FI_UDP_IFACE", "lo0")
    import petsc4py
    import slepc4py

    petsc4py.init([])
    slepc4py.init([])
    from petsc4py import PETSc
    from slepc4py import SLEPc

    cplx = np.iscomplexobj(A.data) and np.any(A.data.imag)
    M = A
    # SLEPc 无“σ 以上/以下 k 个”的原生查询；按常见用法求 3k 个最近者再筛选
    k_req = k if which == "nearest" else 3 * k
    nev = k_req
    if cplx:
        Ar, Ai = sp.csr_matrix(A.real), sp.csr_matrix(A.imag)
        M = sp.bmat([[Ar, -Ai], [Ai, Ar]], format="csr")
        nev = 2 * k_req
    M = sp.csr_matrix(M, dtype=np.float64)
    M.sort_indices()
    mat = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr.astype(PETSc.IntType), M.indices.astype(PETSc.IntType), M.data),
                                comm=PETSc.COMM_SELF)
    mat.assemble()
    E = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
    E.setOperators(mat)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setDimensions(nev=nev)
    E.setTarget(sigma)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setTolerances(1e-12, 1000)
    st = E.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")
    E.setFromOptions()
    E.solve()
    nconv = E.getConverged()
    vr = mat.createVecs()[0]
    lam, U = [], []
    for i in range(min(nconv, nev)):
        lam.append(E.getEigenpair(i, vr).real)
        U.append(vr.getArray().copy())
    lam = np.array(lam)
    U = np.array(U).T if U else np.empty((M.shape[0], 0))
    meta = dict(method="SLEPc Krylov-Schur + sinvert + MUMPS", nconv=int(nconv), its=int(E.getIterationNumber()))
    X = U
    if cplx:
        lam, X = _complex_from_embedding(A, lam, U)
        order = np.argsort(np.abs(lam - sigma))[:k_req]
        lam, X = lam[order], X[:, order]
    if which != "nearest":
        keep = lam >= sigma if which == "above" else lam <= sigma
        lam, X = lam[keep], X[:, keep]
        order = np.argsort(lam if which == "above" else -lam)[:k]
        lam, X = lam[order], X[:, order]
        if lam.size < k:
            meta["note"] = f"3k 个最近特征值中仅 {lam.size} 个满足 {which} σ（被零空间/另一侧簇占满）"
    return lam, X, meta


def run_baseline(name, A, k, sigma, which, **_):
    if name == "scipy":
        return _scipy(A, k, sigma, which)
    if name == "slepc":
        return _slepc(A, k, sigma, which)
    raise ValueError(name)
