"""厚重启块 Krylov-Schur（自伴算子版，即厚重启块 Lanczos）。

求自伴算子 T（关于 B 内积 ⟨x,y⟩ = xᴴBy）模最大 / 代数最大 / 代数最小的若干特征对。
在位移-求逆中 T = (A − σB)⁻¹B，θ = 1/(λ − σ)。

要点
  * 块大小 p：一次 Krylov 扩展可捕获重数 ≤ p 的特征值；
  * 完全再正交化（两遍经典 Gram-Schmidt，DGKS）——数十万阶以内开销可忽略，换取数值稳健；
  * 锁定集 X_L：与已知特征向量 B-正交地收缩（用于找回漏掉的重特征值）；
  * 块失效（不变子空间）时以随机方向补齐并将对应耦合系数置零。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as la


@dataclass
class KSResult:
    theta: np.ndarray          # 期望的 nev 个 Ritz 值（按 which 排序）
    X: np.ndarray              # 对应 Ritz 向量（B-正交归一）
    resid: np.ndarray          # 算子残差 ‖T x − θ x‖_B
    converged: bool
    restarts: int
    n_apply: int               # 算子作用的向量次数
    theta_extra: np.ndarray = field(default_factory=lambda: np.empty(0))  # 其余 Ritz 值（估计谱间隙用）
    conv_mask: np.ndarray = field(default_factory=lambda: np.empty(0, bool))  # 各期望 Ritz 对是否收敛


class _Inner:
    def __init__(self, B):
        self.B = B

    def __call__(self, X, Y):
        return X.conj().T @ (Y if self.B is None else self.B @ Y)

    def norms(self, X):
        if self.B is None:
            return np.linalg.norm(X, axis=0)
        return np.sqrt(np.maximum(np.real(np.einsum("ij,ij->j", X.conj(), self.B @ X)), 0.0))


def _project_out(W, V, inner, passes=2):
    """W ← (I − V Vᴴ B) W，两遍。返回累计系数。"""
    if V is None or V.shape[1] == 0:
        return np.zeros((0, W.shape[1]), W.dtype)
    coef = inner(V, W)
    W -= V @ coef
    for _ in range(passes - 1):
        c2 = inner(V, W)
        W -= V @ c2
        coef += c2
    return coef


def _block_orth(W, inner, against, rng, tol=1e-10):
    """B-正交化块 W（已与 against 正交），返回 Q, R 使 W ≈ Q R。秩亏列以随机向量补齐。"""
    p = W.shape[1]
    norms0 = inner.norms(W)
    Q = np.array(W, order="F", copy=True)
    R = np.zeros((p, p), dtype=W.dtype)
    for j in range(p):  # 改进 Gram-Schmidt + 再正交化，对秩亏稳健
        for _ in range(2):
            if j:
                c = inner(Q[:, :j], Q[:, j : j + 1]).ravel()
                Q[:, j] -= Q[:, :j] @ c
                R[:j, j] += c
        nrm = inner.norms(Q[:, j : j + 1])[0]
        if nrm <= tol * max(norms0[j], np.finfo(float).tiny):
            # 失效：不变子空间已到达，补一个随机正交方向，耦合系数为 0
            R[j, j] = 0.0
            for _ in range(3):
                z = rng.standard_normal(W.shape[0]).astype(W.dtype)
                if np.iscomplexobj(z):
                    z = z + 1j * rng.standard_normal(W.shape[0])
                z = z[:, None]
                _project_out(z, against, inner)
                if j:
                    _project_out(z, Q[:, :j], inner)
                zn = inner.norms(z)[0]
                if zn > 1e-8:
                    Q[:, j] = z[:, 0] / zn
                    break
        else:
            R[j, j] += nrm
            Q[:, j] /= nrm
    return Q, R


def _select(theta, which):
    if which == "LM":
        return np.argsort(-np.abs(theta), kind="stable")
    if which == "LA":
        return np.argsort(-theta, kind="stable")
    if which == "SA":
        return np.argsort(theta, kind="stable")
    raise ValueError(which)


def krylov_schur(
    apply_op: Callable[[np.ndarray], np.ndarray],
    n: int,
    nev: int,
    *,
    dtype=np.float64,
    which: str = "LM",
    block: int = 4,
    ncv: int | None = None,
    tol: float = 1e-12,
    max_restarts: int = 200,
    stall: int = 12,
    noise_rel: float = 0.0,
    B=None,
    locked: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    seed: int = 0,
    monitor: Callable | None = None,
) -> KSResult:
    """monitor(theta, resid, nev)：每次重启前回调（期望的 Ritz 值在前），供上层提前启动与迭代无关的工作。"""
    rng = np.random.default_rng(seed)
    inner = _Inner(B)
    p = max(1, int(block))
    if ncv is None:
        ncv = max(2 * nev + 2 * p, nev + 4 * p, 24)
    m = int(np.ceil(ncv / p) * p)
    m = min(m, n - (0 if locked is None else locked.shape[1]))
    m = max(m - m % p, p)
    nev_eff = min(nev, max(m - p, 1))
    XL = locked if (locked is not None and locked.shape[1]) else None

    V = np.zeros((n, m + p), dtype=dtype, order="F")  # 列主序：列切片连续，避免 BLAS 前的整块拷贝
    H = np.zeros((m + p, m), dtype=dtype)

    def deflate(W):
        if XL is not None:
            _project_out(W, XL, inner)
        return W

    if v0 is None:
        W0 = rng.standard_normal((n, p))
        if np.dtype(dtype).kind == "c":
            W0 = W0 + 1j * rng.standard_normal((n, p))
        W0 = W0.astype(dtype)
    else:
        W0 = np.array(v0, dtype=dtype).reshape(n, -1)
        if W0.shape[1] < p:
            extra = rng.standard_normal((n, p - W0.shape[1])).astype(dtype)
            W0 = np.hstack([W0, extra])
        W0 = W0[:, :p]
    deflate(W0)
    Q, _ = _block_orth(W0, inner, XL, rng)
    V[:, :p] = Q

    k = 0           # 已作用算子的基向量数
    n_apply = 0
    restarts = 0
    best_nconv, last_gain = -1, 0
    while True:
        while k < m:
            W = apply_op(V[:, k : k + p])
            n_apply += p
            W = np.asarray(W, dtype=dtype)
            deflate(W)
            h = _project_out(W, V[:, : k + p], inner)
            H[: k + p, k : k + p] = h
            against = V[:, : k + p] if XL is None else np.hstack([XL, V[:, : k + p]])
            Q, R = _block_orth(W, inner, against, rng)
            V[:, k + p : k + 2 * p] = Q
            H[k + p : k + 2 * p, k : k + p] = R
            k += p

        Hm = H[:m, :m]
        Hm = 0.5 * (Hm + Hm.conj().T)
        theta, S = la.eigh(Hm)
        order = _select(theta, which)
        theta, S = theta[order], S[:, order]
        E = H[m : m + p, :m]
        resid = np.linalg.norm(E @ S, axis=0)
        scale = np.maximum(np.abs(theta), np.finfo(float).tiny)
        # 噪声地板。(1) 通用：算子作用的舍入误差 ~ε·max|θ|（预偏移保证 max|θ| 不致过大，见 solver.PRE_OFFSET）；
        # (2) 近奇异位移：向后稳定求解的误差 ≈ S⁻¹·δS·x，集中在 |θ| 最大的特征方向上，
        #     量级 noise_rel·max|θ|（noise_rel = ε·cond(S)，由探针增长因子给出）。
        #     只对主导簇（|θ| ≥ ½max|θ|）的 Ritz 对放宽——例如目标恰是零空间簇时；
        #     小 |θ| 的 Ritz 对几乎不受此噪声影响，仍按严格阈值。
        tmax = np.abs(theta).max()
        floor = np.full(theta.shape, 100.0 * np.finfo(float).eps * np.sqrt(m) * tmax)
        if noise_rel > 0:
            dom = np.abs(theta) >= 0.5 * tmax
            floor[dom] = np.maximum(floor[dom], 10.0 * np.sqrt(m) * noise_rel * tmax)
        conv = resid <= np.maximum(tol * scale, floor)
        if monitor is not None:
            monitor(theta, resid, nev_eff)
        done = bool(np.all(conv[:nev_eff]))
        nconv_now = int(np.sum(conv[:nev_eff]))
        if nconv_now > best_nconv:
            best_nconv, last_gain = nconv_now, restarts
        # 停滞：重数超过块尺寸的特征值在块 Krylov 空间中只能缓慢出现，
        # 此时返回已收敛子集，由上层惯性证书发现缺口并以收缩+新起始块补齐（比硬迭代快得多）
        stalled = restarts - last_gain >= (stall if nconv_now > 0 else 3 * stall)
        if done or stalled or restarts >= max_restarts:
            X = V[:, :m] @ S[:, :nev_eff]
            return KSResult(
                theta=theta[:nev_eff],
                X=X,
                resid=resid[:nev_eff],
                converged=done,
                restarts=restarts,
                n_apply=n_apply,
                theta_extra=theta[nev_eff:],
                conv_mask=conv[:nev_eff].copy(),
            )

        # 厚重启：保留 l 个 Ritz 向量（期望的 + 若干缓冲），l ≡ m (mod p)
        nconv = int(np.sum(conv[:nev_eff]))
        l_target = min(m - p, max(nev_eff + (m - nev_eff) // 2, nev_eff + nconv // 2))
        l = m - p * int((m - l_target) // p)   # 向上取整到 ≡ m (mod p)，保证 l ≥ l_target
        Vk = V[:, :m] @ S[:, :l]
        Vr = V[:, m : m + p].copy(order="F")
        V[:, :l] = Vk
        V[:, l : l + p] = Vr
        V[:, l + p :] = 0
        Hn = np.zeros_like(H)
        Hn[:l, :l] = np.diag(theta[:l])
        Hn[l : l + p, :l] = E @ S[:, :l]
        H = Hn
        k = l
        restarts += 1
