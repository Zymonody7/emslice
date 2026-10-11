"""惯性引导的位移-求逆谱切片求解器（主流程）。

问题：A x = λ B x，A 厄密（可非正定），B 厄密正定（缺省为 I）。
模式：
  nearest  —— 距目标位移 σ 最近的 k 个特征值（目标频段）
  above    —— ≥ σ 的最小 k 个
  below    —— ≤ σ 的最大 k 个
  interval —— [a, b] 内全部特征值（自动切片）

完备性证书（核心创新）：
  由 Sylvester 惯性定理，LDLᵀ(A − xB) 的负主元数 ν(x) = #{λ < x}。
  求得特征值后在谱间隙处做惯性计数，若计数 = 已求得个数，
  则证明窗口内**没有遗漏的特征值**（重特征值、密集谱均适用）；
  否则自动收缩已得特征向量、加大块尺寸继续求解，直到计数吻合。

并行：所有“只要惯性”的分解（证书、括区、切片二分）彼此独立，由常驻进程池并发执行；
同一稀疏结构上的重复分解复用 MUMPS 符号分析。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .factor import NearSingularError, PetscSlot, ShiftedPattern, available_backends, factorize
from .krylov import krylov_schur

log = logging.getLogger("emslice")
INF = float("inf")


@dataclass
class Options:
    k: int = 10
    sigma: float = 0.0
    which: str = "nearest"          # nearest | above | below | interval
    interval: tuple | None = None
    tol: float = 1e-10              # 目标后向误差 ‖Ax−λBx‖ / ((‖A‖+|λ|‖B‖)‖x‖)
    block: int | None = None        # Krylov 块尺寸（None = 自动）
    ncv: int | None = None
    verify: bool = True             # 惯性完备性证书
    max_rounds: int = 8
    slice_max: int = 48             # 区间模式下每个切片的最大特征值数
    backend: str = "auto"
    precision: str = "double"       # double | mixed（预留）
    ordering: str = "auto"
    mem_limit_gb: float | None = None
    ooc: bool | str = "auto"
    threads: int | None = None
    seed: int = 0
    refine_steps: int = 2
    workers: int | str = "auto"     # 惯性计数并发进程数（auto = min(3, 线程预算/2)，每进程至少 2 线程）
    blr: float | None = None        # MUMPS 块低秩压缩精度（None = 关闭）


@dataclass
class Report:
    mode: str
    n: int
    dtype: str
    sigma: float
    k_requested: int
    k_returned: int = 0
    certified: bool | None = None
    certificate: dict = field(default_factory=dict)
    shifts: list = field(default_factory=list)
    factorizations: list = field(default_factory=list)
    krylov_runs: list = field(default_factory=list)
    residuals: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    norm_A: float = 0.0
    norm_B: float = 1.0
    workers: int = 1


def cpu_count() -> int:
    """可用逻辑核数：优先取进程 CPU 亲和集（尊重 numactl / taskset / 容器绑定），否则取全部逻辑核。"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS 等无亲和集接口
        return os.cpu_count() or 2


class _Threads:
    """线程控制（threadpoolctl）：限制 MUMPS/OpenBLAS 的 OpenMP 线程数，缺失时不做限制。"""

    def __init__(self, n):
        self.n = n
        try:
            from threadpoolctl import threadpool_limits

            self._lim = threadpool_limits
        except Exception:  # pragma: no cover
            self._lim = None

    def __call__(self, n=None):
        import contextlib

        if self._lim is None:
            return contextlib.nullcontext()
        return self._lim(limits=n or self.n)


def onenorm(M) -> float:
    if M is None:
        return 1.0
    return float(abs(M).sum(axis=0).max())


class Solver:
    NEAR_SINGULAR_GROWTH = 1e12
    # 算子位移的预偏移（相对 ‖A‖）：σ 恰为谐振点时免去一次“检测→重分解”。
    # 偏移太小 ⇒ cond(A−sI) 过大，求解噪声污染远处特征对（实测 1e-11 时后向误差 1e-10，1e-9 时 1e-12）；
    # 偏移太大 ⇒ 在谱跨度极大的矩阵上（如 ‖A‖=4e10、局部谱间距 0.3）改变“距 σ 最近”的集合。
    # 因此先用 1e-9‖A‖，求得局部半径 r_k 后若偏移 > 1% r_k，再按局部尺度 1e-4·r_k 重分解一次（见 _solve_nearest）；
    # 切片的局部尺度（片宽）已知，直接取 min(1e-9‖A‖, 1e-4·片宽)。
    # 惯性计数点不做预偏移（否则会改变“≥σ/≤σ”的语义），只在真正近奇异时自适应扰动。
    PRE_OFFSET = 1e-9 * 0.6180339887

    def __init__(self, A, B=None, opts: Options | None = None):
        self.A = sp.csr_matrix(A)
        self.B = None if B is None else sp.csr_matrix(B)
        self.o = opts or Options()
        self.n = self.A.shape[0]
        self.cplx = np.iscomplexobj(self.A.data) or (self.B is not None and np.iscomplexobj(self.B.data))
        self.dtype = np.complex128 if self.cplx else np.float64
        self.nA = onenorm(self.A)
        self.nB = onenorm(self.B)
        self.pattern = ShiftedPattern(self.A, self.B)
        self.backend = self.o.backend if self.o.backend != "auto" else available_backends()[0]
        self._slot = None          # 惯性计数复用的 MUMPS 槽
        self.pool = None           # AsyncCountPool：后台创建，提交即返回
        self._parallel = False
        self._workers_warm = False  # 子进程计数槽已完成符号分析
        self.max_conc = 1
        self.conc = {"count": 1, "slice": 1}
        # 线程预算：默认取物理核数估计（逻辑核/2，上限 8）；主进程与并发子任务按“预算 ÷ 并发数”分配
        self.t_budget = int(self.o.threads or max(1, min(8, cpu_count() // 2)))
        t_fac = min(self.t_budget, 4) if not self.o.threads else self.t_budget
        # MUMPS 默认 L0-OpenMP 要求求解与分解线程数一致（否则 INFOG(1)=-58），两阶段用同一线程数；
        # 进程池子进程各 1 线程，主进程 ≤4 线程，总线程数不超过核数（实测 8 线程会落到能效核反而变慢）
        self._thr_fac = _Threads(t_fac)
        self._thr_solve = self._thr_fac
        self._thr_cnt = self._thr_fac   # 主进程计数槽的线程数（并发模式下固定为每进程份额，以复用符号分析）
        self.rep = Report(mode=self.o.which, n=self.n, dtype=str(np.dtype(self.dtype)), sigma=float(self.o.sigma),
                          k_requested=int(self.o.k), norm_A=self.nA, norm_B=self.nB)

    # ------------------------------------------------------------------ 分解
    def _factor(self, shift: float, purpose: str, keep: bool, direction: int = 0, max_delta: float = INF):
        """带自适应位移的分解。

        位移贴近（或恰为）特征值时 A − σB 近奇异：惯性符号不可靠、求解放大舍入误差。
        检测手段：MUMPS 零主元 + 随机探针求解的增长因子 ‖x‖‖S‖/‖b‖。
        处理：按 ‖A‖ 尺度逐级扰动（1e-10, 1e-8, …），direction 指定扰动方向
        （above 模式向下、below 模式向上，保证贴近 σ 的特征值仍落在所求一侧）。
        """
        scale = max(self.nA, np.finfo(float).tiny)
        delta = 0.0
        reasons = []
        for attempt in range(8):
            x = shift + delta
            slot = None
            if not keep and self.backend == "petsc":
                if self._slot is None:
                    self._slot = PetscSlot(self.pattern, ordering=self.o.ordering, blr=self.o.blr, count_only=True)
                slot = self._slot
            try:
                with (self._thr_fac if keep else self._thr_cnt)():
                    F = factorize(self.pattern, x, backend=self.backend, ordering=self.o.ordering, ooc=self.o.ooc,
                                  slot=slot, nS=self.nA + abs(x) * self.nB, blr=self.o.blr)
            except NearSingularError as e:
                F, reason = None, f"分解失败({e})"
            else:
                reason = None
                if F.inertia is not None and F.inertia[1] > 0:
                    reason = f"{F.inertia[1]} 个零主元"
                elif F.stats.get("growth", 0.0) > self.NEAR_SINGULAR_GROWTH:
                    reason = f"探针增长因子 {F.stats['growth']:.1e}"
            if reason is None:
                self.rep.factorizations.append(dict(F.stats, shift=x, purpose=purpose, inertia=F.inertia))
                if delta:
                    self.rep.diagnostics.append(
                        f"[近奇异位移] {purpose} 位移 {shift:.15g} 贴近特征值（{'; '.join(reasons)}），"
                        f"自适应扰动 {delta:+.2e} → {x:.15g}")
                if not keep:
                    F.free()
                return F, x
            if F is not None:
                F.free()
            reasons.append(reason)
            mag = 1e-10 * 100 ** (attempt // 2) * scale
            sgn = direction if direction else (1 if attempt % 2 == 0 else -1)
            delta = sgn * mag * (1 + 0.1 * attempt)
            if abs(delta) > max_delta:
                break
        raise NearSingularError(f"在 {shift} 附近多次扰动后仍奇异: {reasons}")

    def count_below(self, x: float, max_delta: float = INF, direction: int = 0) -> tuple[int, float]:
        """ν(x) = #{λ < x}（惯性计数），返回 (计数, 实际使用的位移)。"""
        F, xu = self._factor(x, "inertia", keep=False, max_delta=max_delta, direction=direction)
        if F.inertia is None:
            raise RuntimeError("当前后端不提供惯性，无法做完备性验证（请安装 MUMPS 后端或 --no-verify）")
        return F.inertia[0], xu

    def counts(self, reqs, safe=False):
        """批量惯性计数 [(x, max_delta, direction)] → [(ν, x_used)]；有进程池时并发执行。
        safe=True 时近奇异失败返回 None 而不抛异常。"""
        def local(req):
            try:
                return self.count_below(*req)
            except NearSingularError:
                if not safe:
                    raise
                return None

        if not self._parallel:
            return [local(r) for r in reqs]
        out = []
        i = 0
        while i < len(reqs):
            rest = len(reqs) - i
            w = self.conc["count"]
            # 子进程已完成符号分析且一批放得下：全部交给子进程；否则主进程也承担一个（不空等），
            # 内存受限（并发已被下调）时主进程不参与
            if (self._workers_warm and rest <= w) or w < self.max_conc:
                n_work, n_main = min(rest, w), 0
            else:
                n_work, n_main = min(rest - 1, w), 1
            batch = reqs[i : i + n_work + n_main]
            i += len(batch)
            futs = [self.pool.submit(x, md, d) for (x, md, d) in batch[:n_work]]
            mine = [local(r) for r in batch[n_work:]]
            for f in futs:
                try:
                    n, xu, facs, diags, pid = f.result()
                except NearSingularError:
                    if not safe:
                        raise
                    out.append(None)
                    continue
                for fa in facs:
                    fa["worker"] = pid
                self.rep.factorizations.extend(facs)
                self.rep.diagnostics.extend(diags)
                out.append((n, xu))
                self._memory_guard(max((fa.get("mem_mb_used") or 0) for fa in facs) if facs else None, "count")
            if futs:
                self._workers_warm = True
            out.extend(mine)
        return out

    def _start_pool(self):
        w = self.o.workers
        if w == "auto":
            if self.n < 20000:  # 小规模分解只需毫秒级，进程池启动开销得不偿失
                return
            w = max(1, min(3, self.t_budget // 2))   # 每个子进程至少 2 线程；8 核机器上为 2 个子进程
        w = int(w)
        if w <= 1 or self.backend != "petsc" or (not self.o.verify and self.o.which == "nearest"):
            return
        from .parallel import AsyncCountPool

        # 每个子进程的线程数在池创建时固定（复用符号分析）；主进程参与计数时用同样的份额
        thr = max(1, self.t_budget // w)
        # nearest：子进程可在主进程分解 + Krylov 期间先做一次计数分解完成符号分析（petsc4py 不支持只做符号分析）。
        # 只在核数富余时开启——8 核机器上它会与主进程争抢核与内存带宽，得不偿失
        t_main = self._thr_fac.n
        warm = bool(self.o.verify and self.o.which == "nearest" and cpu_count() >= t_main + 2 * w * thr)
        self.pool = AsyncCountPool(self.A, self.B, asdict(self.o), w, threads=thr, warm=warm)
        self._parallel = True
        self._workers_warm = warm
        self._thr_cnt = _Threads(thr)
        self.max_conc = w
        self.conc = {"count": w, "slice": w}
        self.rep.workers = w

    def _memory_guard(self, mem_mb_each, kind="slice"):
        """按单次分解内存与可用内存限制并发数（计数分解丢弃因子，内存约为保留因子时的一半，分开统计）。"""
        if not self._parallel or not mem_mb_each:
            return
        try:
            import psutil

            avail = psutil.virtual_memory().available / 2**20
        except Exception:  # pragma: no cover
            return
        if self.o.mem_limit_gb:
            avail = min(avail, self.o.mem_limit_gb * 1024)
        allowed = max(1, int((0.8 * avail - mem_mb_each) // max(mem_mb_each, 1)))
        if allowed < self.conc[kind]:
            self.conc[kind] = allowed
            if kind == "slice":  # 保留因子的分解更大，计数并发也不应超过它的估计
                self.conc["count"] = min(self.conc["count"], max(allowed, 1) * 2)
            self.rep.diagnostics.append(f"[内存] 单次{'计数' if kind == 'count' else '切片'}分解约 {mem_mb_each} MB，"
                                        f"并发降为 {self.conc[kind]}")

    # --------------------------------------------------------------- 残差等
    def residuals(self, lam, X):
        AX = self.A @ X
        BX = X if self.B is None else self.B @ X
        R = AX - BX * lam[None, :]
        rn = np.linalg.norm(R, axis=0)
        xn = np.linalg.norm(X, axis=0)
        back = rn / ((self.nA + np.abs(lam) * self.nB) * xn)
        rel = rn / np.maximum(np.abs(lam) * xn, np.finfo(float).tiny)
        return rn / xn, back, rel

    def rayleigh_ritz(self, X):
        """在子空间 span(X) 上做 Rayleigh-Ritz：对重/密集特征值给出最优基。"""
        BX = X if self.B is None else self.B @ X
        G = X.conj().T @ BX
        Hm = X.conj().T @ (self.A @ X)
        Hm = 0.5 * (Hm + Hm.conj().T)
        G = 0.5 * (G + G.conj().T)
        w, S = la.eigh(Hm, G)
        return w, X @ S

    # ------------------------------------------------------- 一次位移-求逆求解
    def _shift_invert(self, F, shift, nev, which, locked=None, block=None, v0=None):
        B = self.B

        def op(X):
            return F.solve(X if B is None else B @ X)

        # 默认块尺寸 8：MUMPS 求解 8 个右端与 4 个耗时相当，且可一次容纳重数 ≤ 8 的简并特征值
        p = block or self.o.block or min(8, max(1, nev))
        if self.o.ncv:
            ncv = self.o.ncv
        elif locked is None:
            ncv = max(2 * nev + 2 * p, nev + 6 * p, 40)
        else:  # 收缩补解：只缺少量特征值，用小基即可
            ncv = max(2 * nev + 2 * p, 3 * p)
        ks_which = {"nearest": "LM", "above": "LA", "below": "SA"}[which]
        t = time.perf_counter()
        with self._thr_solve():
            r = self._ks(op, nev, ks_which, p, ncv, locked, F, v0)
        dt = time.perf_counter() - t
        nconv = int(r.conv_mask.sum())
        self.rep.krylov_runs.append(dict(shift=shift, nev=nev, which=ks_which, block=p, ncv=ncv, restarts=r.restarts,
                                         n_apply=r.n_apply, converged=r.converged, nconv=nconv, seconds=dt,
                                         locked=0 if locked is None else int(locked.shape[1])))
        # 只保留已收敛的 Ritz 向量；若一个都没收敛则全部保留（交给 Rayleigh-Ritz/逆迭代精化）
        return r.X[:, r.conv_mask] if nconv else r.X

    def _ks(self, op, nev, ks_which, p, ncv, locked, F, v0=None):
        B = self.B
        if v0 is not None and v0.shape[1] > p:   # 热启动：把已得近似子空间压成 p 个起始向量
            v0 = v0 @ np.random.default_rng(self.o.seed).standard_normal((v0.shape[1], p))
        return krylov_schur(op, self.n, nev, dtype=self.dtype, which=ks_which, block=p, ncv=ncv, v0=v0,
                            tol=max(self.o.tol * 0.1, 1e-14), B=B, locked=locked,
                            seed=self.o.seed + len(self.rep.krylov_runs),
                            noise_rel=np.finfo(float).eps * F.stats.get("growth", 1.0))

    def _refine(self, F, X):
        """Rayleigh-Ritz + 至多 refine_steps 次块逆迭代，把残差压到 tol 以下。"""
        if self.B is None:
            X = np.linalg.qr(X)[0]
        lam, X = self.rayleigh_ritz(X)
        for _ in range(self.o.refine_steps):
            _, back, _ = self.residuals(lam, X)
            if back.max() <= self.o.tol:
                break
            with self._thr_solve():
                Y = F.solve(X if self.B is None else self.B @ X)
            lam, X = self.rayleigh_ritz(np.linalg.qr(Y)[0])
        return lam, X

    # ------------------------------------------------------------------ 主入口
    def solve(self):
        o = self.o
        t0 = time.perf_counter()
        try:
            self._start_pool()
            if self.backend != "petsc":
                lam, X = self._solve_degraded()
            elif o.which == "interval":
                lam, X = self._solve_interval(*o.interval)
            elif o.which in ("above", "below"):
                lam, X = self._solve_one_sided(float(o.sigma), int(o.k), o.which)
            else:
                lam, X = self._solve_nearest(float(o.sigma), int(o.k))
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool = None
                self._parallel = False
            if self._slot is not None:
                self._slot.destroy()
                self._slot = None
        idx = np.argsort(lam, kind="stable")  # 输出统一按特征值升序
        lam, X = lam[idx], X[:, idx]
        rn, back, rel = self.residuals(lam, X)
        X = X / np.linalg.norm(X, axis=0)[None, :]
        self.rep.k_returned = int(lam.size)
        self.rep.residuals = dict(abs_max=float(rn.max(initial=0)), backward_max=float(back.max(initial=0)),
                                  rel_max=float(rel.max(initial=0)), per_eig_backward=back.tolist(),
                                  tol=o.tol, all_below_tol=bool(np.all(back <= o.tol)))
        self.rep.timings["total"] = time.perf_counter() - t0
        fz = self.rep.factorizations
        self.rep.timings["factor_sum"] = float(sum(f.get("seconds_factor", 0) for f in fz))
        self.rep.timings["krylov_sum"] = float(sum(r["seconds"] for r in self.rep.krylov_runs))
        return lam, X, self.rep

    # ---------------------------------------------------------- 降级模式
    def _solve_degraded(self):
        """后端不提供惯性（如仅 SciPy SuperLU）时的最小化降级：可求解，但不出完备性证书。

        nearest 照常求解；above/below/interval 无法用惯性括区，改为“扩大最近集合再筛选”：
        在 σ（区间则取中点）处求最近 m 个，直到所得集合越出目标范围或个数足够，m 逐次加倍。
        Krylov 按距位移由近到远收敛，这一做法通常可靠，但重特征值/密集谱下不能保证无遗漏。
        """
        o = self.o
        self.rep.diagnostics.append(f"[降级] 后端 {self.backend} 不提供惯性：不做完备性证书"
                                    + ("" if o.which == "nearest" else "；单侧/区间查询改为扩大最近集合再筛选，可能漏解"))
        self.o = o = replace(o, verify=False)
        if o.which == "nearest":
            return self._solve_nearest(float(o.sigma), int(o.k))
        if o.which == "interval":
            a, b = (float(v) for v in o.interval)
            if not a < b:
                raise ValueError("区间需满足 a < b")
            c, m = 0.5 * (a + b), max(int(o.slice_max), 1)
        else:
            c, m = float(o.sigma), 2 * int(o.k)
        F, s_used = self._factor(c + self.PRE_OFFSET * (min(self.nA, 1e3 * abs(c)) if c else self.nA), "shift", keep=True)
        self.rep.shifts.append(s_used)
        try:
            for _ in range(o.max_rounds):
                m = min(m, self.n)
                lam, X = self._refine(F, self._shift_invert(F, s_used, m, "nearest"))
                if o.which == "interval":
                    keep = (lam >= a) & (lam <= b)
                    if np.any(~keep) or m >= self.n:   # 最近集合已越出区间：区间内的都已取到
                        return lam[keep], X[:, keep]
                else:
                    lam, X = self._order(lam, X, c, o.which)
                    ok = int(np.sum(lam >= c) if o.which == "above" else np.sum(lam <= c))
                    if ok >= o.k or m >= self.n:
                        return lam[: min(ok, o.k)], X[:, : min(ok, o.k)]
                m *= 2
            self.rep.diagnostics.append(f"[降级] {o.max_rounds} 轮扩大后仍未覆盖目标范围（m={m}），结果可能不完整")
            if o.which == "interval":
                keep = (lam >= a) & (lam <= b)
                return lam[keep], X[:, keep]
            ok = int(np.sum(lam >= c) if o.which == "above" else np.sum(lam <= c))
            return lam[:ok], X[:, :ok]
        finally:
            F.free()

    # ---------------------------------------------------------- 最近 k 个
    def _solve_nearest(self, sigma, k):
        """距 σ 最近的 k 个特征值 + 内窗口惯性证书。

        证书：取 ρ 略小于第 k 个距离 r_k，若 ν(σ+ρ) − ν(σ−ρ) 等于已求得且距离 < ρ 的个数，
        则不存在被遗漏的更近特征值（与第 k 个等距的重特征值视为并列，任取其一均正确）。
        """
        o = self.o
        if k <= 0:
            raise ValueError("k 必须为正")
        # 预偏移尺度：‖A‖，但 σ≠0 时不超过 1e3·|σ|（谱跨度极大的矩阵上避免偏移超过局部谱间距）
        scale = min(self.nA, 1e3 * abs(sigma)) if sigma != 0 else self.nA
        F, s_used = self._factor(sigma + self.PRE_OFFSET * scale, "shift", keep=True)
        self.rep.shifts.append(s_used)
        self._memory_guard(F.stats.get("mem_mb_used"))
        lam, X = self._refine(F, self._shift_invert(F, s_used, k, "nearest"))
        lam, X = self._order(lam, X, sigma, "nearest")
        rk = float(np.abs(lam[:k] - sigma).max()) if lam.size else 0.0
        if lam.size and abs(s_used - sigma) > 1e-2 * rk and rk > 1e-11 * self.nA:
            # 预偏移相对局部谱尺度过大（谱跨度极大的矩阵）：按局部尺度重分解，并以已得向量热启动
            s_new = sigma + 1e-4 * rk * 0.6180339887
            self.rep.diagnostics.append(f"[局部尺度] 预偏移 {s_used - sigma:.2e} 超过局部半径 r_k={rk:.3g} 的 1%，"
                                        f"改用 σ+{s_new - sigma:.2e} 重分解")
            F.free()
            F, s_used = self._factor(s_new, "shift", keep=True)
            self.rep.shifts.append(s_used)
            X0 = X[:, : min(k, X.shape[1])]
            lam, X = self._refine(F, self._shift_invert(F, s_used, k, "nearest", v0=X0))
            lam, X = self._order(lam, X, sigma, "nearest")
        block = o.block or 8
        for _ in range(o.max_rounds):
            if lam.size >= k:
                break
            # Krylov 因停滞只返回了部分收敛对：收缩已得向量，补足 k 个（证书之前）
            need = k - lam.size
            block = min(max(block, need), 16)
            X2 = self._shift_invert(F, s_used, need, "nearest", locked=X if lam.size else None, block=block)
            lam, X = self._refine(F, np.hstack([X, X2]) if lam.size else X2)
            lam, X = self._order(lam, X, sigma, "nearest")
        if not o.verify:
            F.free()
            return lam[:k], X[:, :k]
        zero = 1e-11 * self.nA
        for rnd in range(o.max_rounds):
            rk = float(np.abs(lam[:k] - sigma).max())
            rho = rk - max(1e-7 * rk, zero)
            if rho <= zero:
                self.rep.certified = True
                self.rep.certificate = dict(round=rnd, kind="cluster-at-sigma", r_k=rk)
                self._cluster_note(sigma, rk)
                break
            (n_lo, x_lo), (n_hi, x_hi) = self.counts([(sigma - rho, INF, 0), (sigma + rho, INF, 0)])
            inside = int(n_hi - n_lo)
            found_in = int(np.sum((lam > x_lo) & (lam < x_hi)))
            self.rep.certificate = dict(round=rnd, kind="inner-window", window=[x_lo, x_hi], r_k=rk,
                                        inertia_count=inside, found_in_window=found_in)
            if inside == found_in:
                self.rep.certified = True
                break
            if inside < found_in:
                self.rep.certified = False
                self.rep.diagnostics.append(f"惯性计数 {inside} < 窗口内已得 {found_in}：窗口边界贴近特征值，结果仍按残差有效")
                break
            missing = inside - found_in
            block = min(max(block, missing + 1), 16)
            self.rep.diagnostics.append(f"[完备性] 第 {rnd} 轮：距 σ 小于 r_k 的惯性计数 {inside} > 已得 {found_in}，"
                                        f"漏解 {missing} 个（重特征值/密集谱）→ 收缩已得向量，块尺寸 {block} 继续")
            X2 = self._shift_invert(F, s_used, missing, "nearest", locked=X, block=block)
            lam, X = self._refine(F, np.hstack([X, X2]))
            lam, X = self._order(lam, X, sigma, "nearest")
        else:
            self.rep.certified = False
        F.free()
        return lam[:k], X[:, :k]

    def _cluster_note(self, sigma, rk):
        """σ 恰为高重数特征值（如旋度-旋度算子的梯度零空间）：报告其重数。"""
        w = max(1e-8 * self.nA, 10 * rk)
        try:
            (n_lo, _), (n_hi, _) = self.counts([(sigma - w, INF, 0), (sigma + w, INF, 0)])
            self.rep.diagnostics.append(
                f"[低频/零空间] σ={sigma:.6g} 处为 {n_hi - n_lo} 重特征值簇（|λ−σ|<{w:.1e}），返回其中 {self.o.k} 个正交基向量")
            self.rep.certificate["cluster_multiplicity"] = int(n_hi - n_lo)
        except Exception as e:  # 仅诊断用途
            self.rep.diagnostics.append(f"零空间重数统计失败: {e}")

    @staticmethod
    def _order(lam, X, sigma, which):
        lam = np.real(lam)
        if which == "nearest":
            idx = np.argsort(np.abs(lam - sigma), kind="stable")
        elif which == "above":
            idx = np.argsort(np.where(lam >= sigma, lam, np.inf), kind="stable")
        else:
            idx = np.argsort(np.where(lam <= sigma, -lam, np.inf), kind="stable")
        return lam[idx], X[:, idx]

    # ------------------------------------------------------ σ 以上/以下 k 个
    def _solve_one_sided(self, sigma, k, which):
        """惯性引导的目标重定位：先用惯性计数括出含 [k, 2k+8] 个特征值的区间 [σ, b]，
        再把位移放在区间中部做位移-求逆。避免 σ 紧邻巨大零空间/密集簇时 Krylov 被另一侧主导而停滞。
        括区计数按批并发（每批候选点数 = 并发进程数）。
        """
        if k <= 0:
            raise ValueError("k 必须为正")
        sgn = 1 if which == "above" else -1
        x0_req = sigma
        d0 = max(k * 2.0 * self.nA / self.n, 1e-6 * max(abs(sigma), self.nA * 1e-3))
        w = max(1, self.conc["count"])
        # 第一批：d0 附近 + 低三个数量级的点（应对谱跨度极大的矩阵，如对角元 1e2~1e10）
        first = [d0, d0 * 16, d0 * 1e-3, d0 * 4][: max(1, w)]
        res = self.counts([(x0_req, INF, -sgn)] + [(x0_req + sgn * d, INF, 0) for d in first])
        n0, x0 = res[0]
        avail = (self.n - n0) if sgn > 0 else n0
        if avail < k:
            self.rep.diagnostics.append(f"σ 的{'上' if sgn > 0 else '下'}方仅有 {avail} 个特征值")
            k = avail
            if k == 0:
                self.rep.certified = True
                return np.empty(0), np.empty((self.n, 0), self.dtype)
        pts = [(abs(xb - x0), (nb - n0) * sgn, nb, xb) for (nb, xb) in res[1:]]
        n_batches = 1
        for _ in range(16):
            if any(k <= p[1] <= 2 * k + 8 for p in pts):
                break
            unders = sorted([p for p in pts if p[1] < k], key=lambda p: p[0])
            overs = sorted([p for p in pts if p[1] >= k], key=lambda p: p[0])
            if not overs:                           # 还没括住：向外外推
                dl, cl = unders[-1][0], unders[-1][1]
                pos = [p for p in unders if p[1] > 0]
                if cl == 0:
                    cands = [dl * 16.0 ** j for j in range(1, w + 1)]
                else:
                    alpha = 1.0
                    if len(pos) >= 2 and pos[-1][1] > pos[-2][1]:
                        alpha = np.clip(np.log(pos[-1][1] / pos[-2][1]) / np.log(pos[-1][0] / pos[-2][0]), 0.5, 3.0)
                    dstar = dl * (1.3 * k / cl) ** (1.0 / alpha)
                    cands = [dstar * f for f in (1.0, 1.6, 2.5, 4.0)[:w]]
            else:
                dh, ch = overs[0][0], overs[0][1]
                lower = [p for p in unders if p[0] < dh]
                dl, cl = (lower[-1][0], lower[-1][1]) if lower else (0.0, 0)
                if cl > 0:                          # 幂律插值
                    dstar = self._interp(dl, cl, dh, ch, 1.3 * k)
                    cands = sorted({dstar, 0.5 * (dl + dstar), 0.5 * (dstar + dh)})[:w] if w > 1 else [dstar]
                else:                               # 下端计数为 0：对数尺度等分，每批把对数区间缩小 (w+1) 倍
                    lo = dl if dl > 0 else dh * 1e-9
                    cands = [lo * (dh / lo) ** (j / (w + 1)) for j in range(1, w + 1)]
            res = self.counts([(x0 + sgn * d, INF, 0) for d in cands])
            pts += [(abs(xb - x0), (nb - n0) * sgn, nb, xb) for (nb, xb) in res]
            n_batches += 1
        d, c, nb, xb = min((p for p in pts if p[1] >= k), key=lambda p: p[1])
        self.rep.diagnostics.append(f"[目标重定位] 惯性括区：[{min(x0, xb):.6g}, {max(x0, xb):.6g}] 内含 {c} 个特征值"
                                    f"（{len(pts) + 1} 次惯性计数，{n_batches} 批并发），位移移至区间中部求解")
        a, b, na, nbb = (x0, xb, n0, nb) if sgn > 0 else (xb, x0, nb, n0)
        lam, X = self._solve_range(a, b, na, nbb)
        lam, X = self._order(lam, X, sigma, which)
        return lam[:k], X[:, :k]

    @staticmethod
    def _interp(d1, c1, d2, c2, target):
        if c1 <= 0 or d1 <= 0:
            return 0.5 * (d1 + d2)
        alpha = max(np.log(c2 / c1) / np.log(d2 / d1), 0.3)
        d = d1 * (target / c1) ** (1.0 / alpha)
        return float(np.clip(d, d1 + 0.1 * (d2 - d1), d2 - 0.1 * (d2 - d1)))

    # -------------------------------------------------------------- 区间模式
    def _solve_interval(self, a, b):
        a, b = float(a), float(b)
        if not a < b:
            raise ValueError("区间需满足 a < b")
        (na, xa), (nb, xb) = self.counts([(a, INF, 0), (b, INF, 0)])
        self.rep.k_requested = int(nb - na)
        return self._solve_range(xa, xb, na, nb)

    def _solve_range(self, xa, xb, na, nb):
        total = nb - na
        self.rep.certificate = dict(kind="interval", interval=[xa, xb], count=int(total))
        if total == 0:
            self.rep.certified = True
            return np.empty(0), np.empty((self.n, 0), self.dtype)
        slices = self._split(xa, xb, na, nb)
        self.rep.certificate["slices"] = [dict(a=s[0], b=s[1], count=s[3] - s[2]) for s in slices]
        lams, Xs = [], []
        if self._parallel and len(slices) > 1:
            # 切片彼此独立：分发到进程池并发求解（每片：分解 + 位移-求逆 + 证书补解）
            i = 0
            while i < len(slices):  # 按内存允许的并发数分批提交
                wave = slices[i : i + self.conc["slice"]]
                i += len(wave)
                for f in [self.pool.submit_slice(*sl) for sl in wave]:
                    lam, X, facs, runs, diags, pid = f.result()
                    for fa in facs:
                        fa["worker"] = pid
                    self.rep.factorizations.extend(facs)
                    self.rep.krylov_runs.extend(runs)
                    self.rep.diagnostics.extend(diags)
                    lams.append(lam)
                    Xs.append(X)
                    self._memory_guard(max((fa.get("mem_mb_used") or 0) for fa in facs) if facs else None, "slice")
        else:
            for sl in slices:
                lam, X = self._solve_one_slice(*sl)
                lams.append(lam)
                Xs.append(X)
        lam = np.concatenate(lams)
        X = np.hstack(Xs)
        idx = np.argsort(lam)
        self.rep.certified = bool(lam.size == total)
        return lam[idx], X[:, idx]

    def _solve_one_slice(self, sa, sb, ca, cb):
        """单个切片：在中点分解，求切片内全部 cnt 个特征值；少于惯性计数时收缩补解。"""
        cnt = cb - ca
        off = min(self.PRE_OFFSET * self.nA, 1e-4 * (sb - sa) * 0.6180339887)
        F, s_used = self._factor(0.5 * (sa + sb) + off, "slice", keep=True)
        self.rep.shifts.append(s_used)
        self._memory_guard(F.stats.get("mem_mb_used"))
        lam, X = self._refine(F, self._shift_invert(F, s_used, cnt, "nearest"))
        inside = (lam > sa) & (lam < sb)
        lam, X = lam[inside], X[:, inside]
        rounds = 0
        while lam.size < cnt and rounds < self.o.max_rounds:
            missing = cnt - lam.size
            blk = min(max(self.o.block or 8, missing + 1), 16)
            self.rep.diagnostics.append(f"[完备性] 切片 [{sa:.6g},{sb:.6g}) 惯性计数 {cnt}，已得 {lam.size}，收缩后继续（块 {blk}）")
            X2 = self._shift_invert(F, s_used, missing, "nearest", locked=X if lam.size else None, block=blk)
            lam, X = self._refine(F, np.hstack([X, X2]) if lam.size else X2)
            inside = (lam > sa) & (lam < sb)
            lam, X = lam[inside], X[:, inside]
            rounds += 1
        F.free()
        if lam.size != cnt:
            self.rep.diagnostics.append(f"切片 [{sa:.6g},{sb:.6g}) 惯性计数 {cnt} 但仅求得 {lam.size}")
        return lam, X

    def _split(self, a, b, na, nb):
        """按惯性二分，使每个切片内特征值数 ≤ slice_max。同一层的分点并发计数。

        分点若贴近特征值（近奇异），在区间内改取黄金分割等备选点；扰动量限制在区间宽度内，
        保证区间严格收缩。无法再分（同值簇超过 slice_max）时整体作为一个切片。
        """
        out, todo = [], [(a, b, na, nb, 0)]
        fracs = (0.5, 0.382, 0.618, 0.27, 0.73)
        while todo:
            need = []
            for (sa, sb, ca, cb, depth) in todo:
                width = sb - sa
                if cb - ca <= self.o.slice_max or depth >= 60 or width <= 1e-9 * max(abs(sa), abs(sb), self.nA * 1e-6):
                    if cb > ca:
                        if cb - ca > self.o.slice_max:
                            self.rep.diagnostics.append(f"切片 [{sa:.10g},{sb:.10g}) 含 {cb - ca} 个近同值特征值，作为整体求解")
                        out.append((sa, sb, ca, cb))
                else:
                    need.append((sa, sb, ca, cb, depth))
            todo = []
            if not need:
                break
            res = self.counts([(sa + 0.5 * (sb - sa), 0.1 * (sb - sa), 0) for (sa, sb, *_r) in need], safe=True)
            for (sa, sb, ca, cb, depth), r in zip(need, res):
                if r is None or not (sa < r[1] < sb):
                    r = None
                    for f in fracs[1:]:
                        try:
                            cm, xm = self.count_below(sa + f * (sb - sa), max_delta=0.1 * (sb - sa))
                        except NearSingularError:
                            continue
                        if sa < xm < sb:
                            r = (cm, xm)
                            break
                if r is None:
                    out.append((sa, sb, ca, cb))
                    continue
                cm, xm = r
                todo += [(sa, xm, ca, cm, depth + 1), (xm, sb, cm, cb, depth + 1)]
        return sorted(out)


def solve(A, B=None, **kw):
    o = Options(**kw)
    lam, X, rep = Solver(A, B, o).solve()
    return lam, X, asdict(rep)
