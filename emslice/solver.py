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
import sys
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .factor import (FactorMemoryError, NearSingularError, PetscSlot, ShiftedPattern, available_backends, count_eta,
                     factorize)
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
    slice_max: int = 200            # 区间模式下每个切片的最大特征值数（鲲鹏 8 核：192 个特征值的区间 48→200 时 56.9→39.0 s）
    backend: str = "auto"
    precision: str = "auto"         # 惯性计数精度：auto/mixed = 单精度计数 + η 裕量证书 + 双精度回退；double = 全双精度
    ordering: str = "auto"
    mem_limit_gb: float | None = None
    ooc: bool | str = "auto"        # auto：内存超限（--mem-limit-gb）或分配失败时自动改用外存；True/False 强制
    threads: int | None = None
    seed: int = 0
    refine_steps: int = 6
    workers: int | str = "auto"     # 计数/切片子进程数（auto：线程预算 ≥16 时 min(3, 预算/8)，否则不开进程池、主进程串行）
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


def physical_cores() -> int:
    """可用物理核数：亲和集 ÷ 每核硬件线程数（SMT）。macOS 取性能核数——MUMPS 线程落到能效核上反而拖慢整体。"""
    n = cpu_count()
    if sys.platform == "darwin":
        try:
            import subprocess

            p = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True,
                                   timeout=2).stdout)
            return max(1, min(p, n))
        except Exception:
            return max(1, n // 2)
    try:
        import psutil

        lg, ph = psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)
        smt = max(1, round(lg / ph)) if lg and ph else 2
    except Exception:  # pragma: no cover
        smt = 2
    return max(1, n // smt)


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
        # 计数精度：单精度 LDLᵀ 只给出“邻近矩阵”的精确惯性，凡进入证书的计数都检查 η 裕量，不足时回退双精度
        self.count_prec = "single" if (self.backend == "petsc" and self.o.precision in ("auto", "mixed")) else "double"
        self._slots = {}           # 常驻 MUMPS 槽：keep（保留因子，双精度计数在其空闲时也复用它）、single（单精度计数）
        self._keep_busy = False    # keep 槽上是否有活的保留因子分解
        self.pool = None           # AsyncCountPool：后台创建，提交即返回
        self._parallel = False
        self._workers_warm = False  # 子进程计数槽已完成符号分析
        self.max_conc = 1
        self.conc = {"count": 1, "slice": 1}
        self._ritz = np.empty(0)   # 最近一次 Krylov 的额外 Ritz 值（λ 估计），用于判断证书端点外侧是否有干净间隙
        self._ritz_vecs = None     # 对应的下一批 Ritz 向量（按需补求时作初始块）
        self._known = None         # 证书窗口 (lo, hi, 窗口内全部已得特征值)：窗口内无未知特征值，用于子空间夹角上界
        # 线程预算：默认取可用物理核数（尊重 numactl/taskset 绑定；鲲鹏无超线程，x86 超线程按 2 折算）
        self.t_budget = int(self.o.threads or min(physical_cores(), 64))
        t_fac = self.t_budget
        # MUMPS 默认 L0-OpenMP 要求求解与分解线程数一致（否则 INFOG(1)=-58），两阶段用同一线程数
        self._thr_fac = _Threads(t_fac)
        self._thr_solve = self._thr_fac
        self._thr_cnt = self._thr_fac   # 主进程计数槽的线程数（并发模式下固定为每进程份额，以复用符号分析）
        self.rep = Report(mode=self.o.which, n=self.n, dtype=str(np.dtype(self.dtype)), sigma=float(self.o.sigma),
                          k_requested=int(self.o.k), norm_A=self.nA, norm_B=self.nB)

    # ------------------------------------------------------------------ 分解
    def eta(self, x: float, prec: str) -> float:
        """位移 x 处惯性计数的不确定半径（见 factor.count_eta）。"""
        return count_eta(prec, self.nA + abs(x) * self.nB)

    def _factor(self, shift: float, purpose: str, keep: bool, direction: int = 0, max_delta: float = INF,
                prec: str = "double"):
        """带自适应位移的分解。

        位移贴近（或恰为）特征值时 A − σB 近奇异：惯性符号不可靠、求解放大舍入误差。
        检测手段：MUMPS 零主元 + 随机探针求解的增长因子 ‖x‖‖S‖/‖b‖。
        处理：按 ‖A‖ 尺度逐级扰动（1e-10, 1e-8, …），direction 指定扰动方向
        （above 模式向下、below 模式向上，保证贴近 σ 的特征值仍落在所求一侧）。
        """
        scale = max(self.nA, np.finfo(float).tiny)
        prec = "double" if keep else prec
        eta0 = self.eta(shift, prec) if prec == "single" else 0.0
        delta = 0.0
        reasons = []
        for attempt in range(8):
            x = shift + delta
            slot = None
            if self.backend == "petsc":
                # 同一稀疏结构的槽按用途常驻复用（只做数值分解）：保留因子的位移/切片分解一个槽，计数按精度各一个槽。
                # 保留槽上同一时刻只有一个活的分解（调用方在下一次保留分解前已 free）
                # 双精度计数在 keep 槽空闲且线程数相同时复用它（与位移/切片分解共享符号分析；
                # MUMPS 的 L0-OpenMP 结构按分析时的线程数确定，线程数不同的复用会得到错误结果）
                share = prec == "double" and not self._keep_busy and self._thr_cnt.n == self._thr_fac.n
                key = "keep" if (keep or share) else prec
                if key not in self._slots:
                    self._slots[key] = self._new_slot(key, prec, ooc=self.o.ooc is True)
                slot = self._slots[key]
            try:
                try:
                    with (self._thr_fac if keep else self._thr_cnt)():
                        F = factorize(self.pattern, x, backend=self.backend, ordering=self.o.ordering, ooc=self.o.ooc,
                                      slot=slot, nS=self.nA + abs(x) * self.nB, blr=self.o.blr, precision=prec)
                except FactorMemoryError as e:
                    if slot is None or slot.ooc or self.o.ooc is False:
                        raise MemoryError(f"{e}；可设置更大的 --mem-limit-gb、开启 --ooc on 或减少并发") from e
                    # 内存超限：该槽改用外存（因子写盘，内存约减半，求解变慢）后重试同一位移
                    self.rep.diagnostics.append(f"[内存] {purpose} 分解超出内存上限（{e}），改用外存（OOC）")
                    slot.destroy()
                    slot = self._slots[key] = self._new_slot(key, prec, ooc=True)
                    try:
                        with (self._thr_fac if keep else self._thr_cnt)():
                            F = factorize(self.pattern, x, backend=self.backend, ordering=self.o.ordering, ooc=True,
                                          slot=slot, nS=self.nA + abs(x) * self.nB, blr=self.o.blr, precision=prec)
                    except FactorMemoryError as e2:
                        raise MemoryError(f"改用外存后 {purpose} 分解仍超出内存上限 {self.o.mem_limit_gb} GB（{e2}）；"
                                          f"请增大 --mem-limit-gb，或减小问题规模") from e2
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
                if keep and slot is not None:
                    self._keep_busy = True
                    F.on_free = self._release_keep
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
            # 单精度计数的扰动从 2η 起步（更小的扰动在单精度下无意义）
            mag = max(1e-10 * 100 ** (attempt // 2) * scale, 2 * eta0 * 3 ** attempt)
            sgn = direction if direction else (1 if attempt % 2 == 0 else -1)
            delta = sgn * mag * (1 + 0.1 * attempt)
            if abs(delta) > max_delta:
                break
        raise NearSingularError(f"在 {shift} 附近多次扰动后仍奇异: {reasons}")

    def _new_slot(self, key, prec, ooc=False):
        """新建常驻槽；--mem-limit-gb 时把每进程工作内存上限交给 MUMPS（ICNTL(23)）。"""
        lim = None
        if self.o.mem_limit_gb:
            procs = 1 + (self.max_conc if self._parallel else 0)
            lim = max(64, int(0.8 * self.o.mem_limit_gb * 1024 / procs))
        return PetscSlot(self.pattern, ordering=self.o.ordering, blr=self.o.blr, count_only=key != "keep", precision=prec,
                         ooc=ooc, mem_limit_mb=lim)

    def _release_keep(self):
        self._keep_busy = False

    def _ensure_factor(self, F, shift):
        """保留槽被双精度计数复用过：补解前按原位移重新数值分解（复用符号分析）。"""
        if F.freed:
            F, _ = self._factor(shift, "shift(重分解)", keep=True)
        return F

    def count_below(self, x: float, max_delta: float = INF, direction: int = 0, prec: str = "double") -> tuple[int, float]:
        """ν(x) = #{λ < x}（惯性计数），返回 (计数, 实际使用的位移)。
        prec="single" 时为邻近矩阵的精确计数：只对 |λ − x| > eta(x) 的特征值保证正确。"""
        F, xu = self._factor(x, "inertia", keep=False, max_delta=max_delta, direction=direction, prec=prec)
        if F.inertia is None:
            raise RuntimeError("当前后端不提供惯性，无法做完备性验证（请安装 MUMPS 后端或 --no-verify）")
        return F.inertia[0], xu

    def counts(self, reqs, safe=False):
        """批量惯性计数 [(x, max_delta, direction[, prec])] → [(ν, x_used)]；有进程池时并发执行。
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
            futs = [self.pool.submit(*r) for r in batch[:n_work]]
            mine = [local(r) for r in batch[n_work:]]
            out.extend(self._collect(f, safe) for f in futs)
            out.extend(mine)
        return out

    def _collect(self, f, safe=False, purpose=None):
        """取回子进程计数结果并记账（分解统计、诊断、内存守护）。"""
        try:
            n, xu, facs, diags, pid = f.result()
        except NearSingularError:
            if not safe:
                raise
            return None
        for fa in facs:
            fa["worker"] = pid
            if purpose:
                fa["purpose"] = purpose
        self.rep.factorizations.extend(facs)
        self.rep.diagnostics.extend(diags)
        self._workers_warm = True
        self._memory_guard(max((fa.get("mem_mb_used") or 0) for fa in facs) if facs else None, "count")
        return n, xu

    def _start_pool(self):
        w = self.o.workers
        if w == "auto":
            # 核数 < 16 时不开进程池：计数在主进程串行、用满全部线程并复用同一计数槽的符号分析，
            # 比分给多个子进程（各自重做分析、每个只分到 2 线程）更快——鲲鹏 8 核实测两个子进程各 3.6–5.3 s，
            # 主进程单精度数值分解 0.4 s + 一次分析 0.45 s
            if self.n < 20000 or self.t_budget < 16:
                return
            w = min(3, self.t_budget // 8)
        w = int(w)
        if w <= 1 or self.backend != "petsc" or (not self.o.verify and self.o.which == "nearest"):
            return
        from .parallel import AsyncCountPool

        # 每个子进程的线程数在池创建时固定（复用符号分析）；主进程参与计数时用同样的份额
        thr = max(1, self.t_budget // (w + 1))
        # nearest 模式的证书计数在 Krylov 迭代中途推测式提交（见 _solve_nearest），无需预热
        warm = False
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

    def _binv_norms(self, R):
        """各列的 ‖r‖_{B⁻¹} = sqrt(rᴴB⁻¹r)：Jacobi 预条件共轭梯度解 BW = R（B 对称正定，质量阵条件数不大，
        十万阶有限元质量阵几十步收敛）。B 缺省时即 2-范数。"""
        if self.B is None:
            return np.linalg.norm(R, axis=0)
        d = np.real(self.B.diagonal())
        W = np.zeros_like(R)
        Rr = R.copy()
        Z = Rr / d[:, None]
        P = Z.copy()
        rz = np.real(np.einsum("ij,ij->j", Rr.conj(), Z))
        r0 = np.maximum(np.linalg.norm(R, axis=0), np.finfo(float).tiny)
        for _ in range(2000):
            BP = self.B @ P
            pbp = np.real(np.einsum("ij,ij->j", P.conj(), BP))
            alpha = np.where(pbp > 0, rz / np.where(pbp > 0, pbp, 1.0), 0.0)
            W += P * alpha
            Rr -= BP * alpha
            if np.all(np.linalg.norm(Rr, axis=0) <= 1e-4 * r0):   # 二次型 rᴴB⁻¹r 的相对误差约为其平方（~1e-8）
                break
            Z = Rr / d[:, None]
            rz_new = np.real(np.einsum("ij,ij->j", Rr.conj(), Z))
            P = Z + P * np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
            rz = rz_new
        return np.sqrt(np.maximum(np.real(np.einsum("ij,ij->j", R.conj(), W)), 0.0))

    def _angle_bounds(self, lam, X):
        """特征向量子空间夹角的后验上界（Davis–Kahan sinΘ 定理）。

        把返回的特征值按 1e-8 相对容差分簇 C；对簇的 Ritz 向量 X_C（B-正交化）与残差 R_C = A X_C − B X_C Λ_C，
          ‖sin Θ(X_C, 真实谱子空间)‖_F ≤ ‖B^{-1/2} R_C‖_F / δ_C，
        δ_C 为簇到“簇外真实特征值”的距离。完备性证书保证窗口内的真实特征值都已求得（各在其 Ritz 值的残差半径内），
        窗口外的特征值离簇至少到窗口边界的距离，因此 δ_C 有可靠下界——没有证书时这一下界无从得知。
        落在证书窗口外的特征值（如与第 k 个并列、按内窗口证书接受的副本）不给上界。
        广义问题的 ‖·‖_{B⁻¹} 由共轭梯度求得（见 _binv_norms）。"""
        lo, hi, known = self._known
        known = np.sort(np.asarray(known, dtype=float))
        tol_c = 1e-8 * max(np.abs(lam).max(), 1e-3 * self.nA)
        cuts = np.flatnonzero(np.diff(lam) > tol_c) + 1
        # Rayleigh-Ritz 给出的 Ritz 向量彼此 B-正交：按 B-范数归一后各列残差的 ‖·‖_{B⁻¹} 只需一次共轭梯度
        BX = X if self.B is None else self.B @ X
        xb = np.sqrt(np.maximum(np.real(np.einsum("ij,ij->j", X.conj(), BX)), np.finfo(float).tiny))
        nb = self._binv_norms((self.A @ X - BX * lam[None, :]) / xb[None, :])
        rmax = float(nb.max())
        out = np.full(lam.size, np.nan)
        for idx in np.split(np.arange(lam.size), cuts):
            c_lo, c_hi = lam[idx[0]], lam[idx[-1]]
            if not (lo < c_lo and c_hi < hi):
                continue
            others = known[(known < c_lo - tol_c) | (known > c_hi + tol_c)]
            gaps = [c_lo - lo, hi - c_hi]
            if others.size:
                gaps.append(float(np.min(np.maximum(others - c_hi, c_lo - others))) - rmax)
            delta = min(gaps)
            if delta <= 0:
                continue
            out[idx] = min(1.0, float(np.sqrt(np.sum(nb[idx] ** 2))) / delta)
        ok = out[np.isfinite(out)]
        return dict(subspace_sin_bound_max=float(ok.max()) if ok.size else None,
                    per_eig_subspace_sin_bound=[None if not np.isfinite(v) else float(v) for v in out])

    # ------------------------------------------------------- 一次位移-求逆求解
    def _shift_invert(self, F, shift, nev, which, locked=None, block=None, v0=None, monitor=None):
        B = self.B

        def op(X):
            return F.solve(X if B is None else B @ X)

        # 默认块尺寸 8：MUMPS 求解 8 个右端与 4 个耗时相当，且可一次容纳重数 ≤ 8 的简并特征值
        p = block or self.o.block or min(8, max(1, nev))
        if self.o.ncv:
            ncv = self.o.ncv
        elif locked is None:
            # 鲲鹏 8 核实测（k=20/30，四个十万阶矩阵）：nev+8p 比 nev+6p 少 5–22% 的 Krylov 时间（重启 3–5 次 → 2–3 次）
            ncv = max(2 * nev + 2 * p, nev + 8 * p, 48)
        else:  # 收缩补解：只缺少量特征值，但缺的往往是窗口边缘的简并副本，紧邻其它特征值，基太小会反复重启
            # （鲲鹏复厄密算例：缺 4 个、ncv=24 时重启 36 次、作用 312 次）
            ncv = max(2 * nev + 2 * p, nev + 6 * p, 40)
        ks_which = {"nearest": "LM", "above": "LA", "below": "SA"}[which]
        t = time.perf_counter()
        with self._thr_solve():
            r = self._ks(op, nev, ks_which, p, ncv, locked, F, v0, monitor)
        dt = time.perf_counter() - t
        nconv = int(r.conv_mask.sum())
        th = np.concatenate([r.theta_extra, r.theta[~r.conv_mask]]) if r.conv_mask.size else r.theta_extra
        th = np.real(th[np.abs(th) > 0])
        self._ritz = shift + 1.0 / th
        self._ritz_vecs = r.X_extra
        self.rep.krylov_runs.append(dict(shift=shift, nev=nev, which=ks_which, block=p, ncv=ncv, restarts=r.restarts,
                                         n_apply=r.n_apply, converged=r.converged, nconv=nconv, seconds=dt,
                                         locked=0 if locked is None else int(locked.shape[1])))
        # 只保留已收敛的 Ritz 向量；若一个都没收敛则全部保留（交给 Rayleigh-Ritz/逆迭代精化）
        return r.X[:, r.conv_mask] if nconv else r.X

    def _ks(self, op, nev, ks_which, p, ncv, locked, F, v0=None, monitor=None):
        B = self.B
        if v0 is not None and v0.shape[1] > p:   # 热启动：把已得近似子空间压成 p 个起始向量
            v0 = v0 @ np.random.default_rng(self.o.seed).standard_normal((v0.shape[1], p))
        return krylov_schur(op, self.n, nev, dtype=self.dtype, which=ks_which, block=p, ncv=ncv, v0=v0,
                            tol=max(self.o.tol * 0.1, 1e-14), B=B, locked=locked,
                            seed=self.o.seed + len(self.rep.krylov_runs),
                            noise_rel=np.finfo(float).eps * F.stats.get("growth", 1.0), monitor=monitor)

    def _refine(self, F, X):
        """Rayleigh-Ritz + 至多 refine_steps 次块逆迭代，把后向误差压到 tol 以下（达标即停，不再改善也停）。

        位移贴近特征值时 Krylov 的收敛判据受噪声地板 ~ε·max|θ| 限制，远离位移的边界特征对可能只到 1e-8 量级；
        块逆迭代（含块内 Rayleigh-Ritz）每步把它们再压低数倍（随机压力测试：σ 恰为特征值、k=20 时
        8e-8 → 1e-8 → 3e-9 → 7e-10 → …），故最多做 refine_steps（默认 6）步。
        （试过在 span[X, S⁻¹R] 上做残差校正：已收敛特征对的校正量只剩噪声，会引入伪 Ritz 值，不采用。）"""
        if self.B is None:
            X = np.linalg.qr(X)[0]
        lam, X = self.rayleigh_ritz(X)
        _, back, _ = self.residuals(lam, X)
        for _ in range(self.o.refine_steps):
            if back.max() <= self.o.tol:
                break
            with self._thr_solve():
                Y = F.solve(X if self.B is None else self.B @ X)
            lam2, X2 = self.rayleigh_ritz(np.linalg.qr(Y)[0])
            _, back2, _ = self.residuals(lam2, X2)
            if back2.max() >= back.max():
                break
            lam, X, back = lam2, X2, back2
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
            for slot in self._slots.values():
                slot.destroy()
            self._slots.clear()
        idx = np.argsort(lam, kind="stable")  # 输出统一按特征值升序
        lam, X = lam[idx], X[:, idx]
        rn, back, rel = self.residuals(lam, X)
        X = X / np.linalg.norm(X, axis=0)[None, :]
        self.rep.k_returned = int(lam.size)
        self.rep.residuals = dict(abs_max=float(rn.max(initial=0)), backward_max=float(back.max(initial=0)),
                                  rel_max=float(rel.max(initial=0)), per_eig_backward=back.tolist(),
                                  tol=o.tol, all_below_tol=bool(np.all(back <= o.tol)))
        if self.rep.certified and self._known is not None and lam.size:
            self.rep.residuals.update(self._angle_bounds(lam, X))
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
        spec = {}
        monitor = None
        if o.verify and self.count_prec == "single" and self._parallel:
            def monitor(theta, resid, nev):
                """推测式证书计数：期望的 Ritz 值稳定（相对残差 ≤ 1e-6）后按估计窗口提前提交单精度计数，
                与剩余 Krylov 迭代重叠；收敛后核对端点仍有效才采用（见 _certify_nearest_mixed）。"""
                if spec or not np.all(resid[:nev] <= 1e-6 * np.abs(theta[:nev])):
                    return
                th, te = np.real(theta[:nev]), np.real(theta[nev:])
                plan = self._plan_mixed(sigma, k, s_used + 1.0 / th[th != 0], s_used + 1.0 / te[te != 0])
                spec["plan"] = plan
                if plan is not None and plan[0][2] == plan[1][2] == "single":
                    spec["futs"] = [self.pool.submit(x, INF, d, pr) for (x, d, pr) in plan]
        lam, X = self._refine(F, self._shift_invert(F, s_used, k, "nearest", monitor=monitor))
        lam, X = self._order(lam, X, sigma, "nearest")
        rk = float(np.abs(lam[:k] - sigma).max()) if lam.size else 0.0
        too_far = abs(s_used - sigma) > 1e-2 * rk
        # 位移紧贴特征值（σ 恰为谐振点）时，远处特征对受噪声地板 ~ε·max|θ| 所限，逆迭代精化后仍可能不达标：
        # 改在 σ + 1e-3·r_k 处重分解（max|θ|/|θ_k| ≈ 1e3），只在确实未达标时付出这次分解
        too_close = (abs(s_used - sigma) < 1e-3 * rk and lam.size
                     and self.residuals(lam[:k], X[:, :k])[1].max() > o.tol)
        if lam.size and (too_far or too_close) and rk > 1e-11 * self.nA:
            # 预偏移相对局部谱尺度过大（谱跨度极大的矩阵）：按局部尺度重分解，并以已得向量热启动
            s_new = sigma + (1e-4 if too_far else 1e-3) * rk * 0.6180339887
            self.rep.diagnostics.append(
                f"[局部尺度] 预偏移 {s_used - sigma:.2e} " + ("超过局部半径 r_k={:.3g} 的 1%".format(rk) if too_far else
                                                         "使远处特征对精度受限（σ 贴近谐振点）") + f"，改用 σ+{s_new - sigma:.2e} 重分解")
            F.free()
            F, s_used = self._factor(s_new, "shift", keep=True)
            self.rep.shifts.append(s_used)
            X0 = X[:, : min(k, X.shape[1])]
            old = (lam, X, self.residuals(lam[:k], X[:, :k])[1].max())
            lam, X = self._refine(F, self._shift_invert(F, s_used, k, "nearest", v0=X0))
            lam, X = self._order(lam, X, sigma, "nearest")
            if too_close and lam.size >= k and self.residuals(lam[:k], X[:, :k])[1].max() > old[2]:
                lam, X = old[0], old[1]   # 未改善（如 k 截断了近简并簇，残差下限由簇宽决定）：保留原结果
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
        if self.count_prec == "single":
            done, lam, X, F = self._certify_nearest_mixed(F, s_used, sigma, k, lam, X, block, spec)
            if done:
                F.free()
                return lam[:k], X[:, :k]
        zero = 1e-11 * self.nA
        for rnd in range(o.max_rounds):
            rk = float(np.abs(lam[:k] - sigma).max())
            rho = rk - max(1e-7 * rk, zero)
            if rho <= zero:
                self.rep.certified = True
                self.rep.certificate = dict(round=rnd, kind="cluster-at-sigma", r_k=rk)
                if not self._parallel:
                    F.free()     # 计数复用位移分解的槽（F.inertia 仍可用）
                self._cluster_note(sigma, rk, F, s_used)
                break
            if not self._parallel:
                F.free()         # 串行计数复用位移分解的槽，省一次符号分析；补解时再数值重分解
            # 内端点近奇异时只向 σ 一侧扰动：端点紧挨着 r_k 处的并列特征值（可能是上万重的梯度零空间），
            # 向外扰动会把整个簇算进窗口
            (n_lo, x_lo), (n_hi, x_hi) = self.counts([(sigma - rho, INF, 1), (sigma + rho, INF, -1)])
            inside = int(n_hi - n_lo)
            found_in = int(np.sum((lam > x_lo) & (lam < x_hi)))
            self.rep.certificate = dict(round=rnd, kind="inner-window", window=[x_lo, x_hi], r_k=rk,
                                        inertia_count=inside, found_in_window=found_in)
            if inside == found_in:
                self.rep.certified = True
                self._known = (x_lo, x_hi, lam[(lam > x_lo) & (lam < x_hi)])
                break
            if inside < found_in:
                self.rep.certified = False
                self.rep.diagnostics.append(f"惯性计数 {inside} < 窗口内已得 {found_in}：窗口边界贴近特征值，结果仍按残差有效")
                break
            missing = inside - found_in
            if missing > 4 * k + 64:
                # 兜底：缺口远大于所求个数，说明窗口边界压在大簇上（扰动已限定向内，正常不会出现）；不盲目补解
                self.rep.certified = False
                self.rep.diagnostics.append(f"[证书] 窗口计数 {inside} 远大于已得 {found_in}（窗口边界贴近大簇），未出证书")
                break
            block = min(max(block, missing + 1), 16)
            self.rep.diagnostics.append(f"[完备性] 第 {rnd} 轮：距 σ 小于 r_k 的惯性计数 {inside} > 已得 {found_in}，"
                                        f"漏解 {missing} 个（重特征值/密集谱）→ 收缩已得向量，块尺寸 {block} 继续")
            F = self._ensure_factor(F, s_used)
            X2 = self._shift_invert(F, s_used, missing, "nearest", locked=X, block=block)
            lam, X = self._refine(F, np.hstack([X, X2]))
            lam, X = self._order(lam, X, sigma, "nearest")
        else:
            self.rep.certified = False
        F.free()
        return lam[:k], X[:, :k]

    def _first_factor_seconds(self):
        """首次保留因子分解的实测耗时（含符号分析），用于在“多一次分析/多一轮补解”与“省一半数值分解”之间取舍。"""
        return next((f.get("seconds_factor", 0.0) for f in self.rep.factorizations if not f.get("count_only")), 0.0)

    def _absorb_budget(self):
        """把未求出的特征值估计并入证书窗口、交给收缩补解，以换取该侧用单精度计数，是否划算：返回 (副本额度, 近并列额度)。
        省下的约是半次数值分解（≈0.4×首次分解耗时）。付出的是一轮补解 Krylov：
          * 已得特征值的精确简并副本：收缩掉已得的那个后它就是剩余谱中最靠近位移的，补解快（约半次主 Krylov）；
          * r_k 外侧数值不同的近并列特征值：与窗口外紧邻者离位移几乎一样远，补解与主 Krylov 相当
            （鲲鹏复厄密算例实测补解 160–310 次算子作用，主 Krylov 152 次）。
        只影响耗时，不影响正确性。"""
        ks = self.rep.krylov_runs[0]["seconds"] if self.rep.krylov_runs else np.inf
        save = 0.4 * self._first_factor_seconds()
        return (8 if save > 0.5 * ks else 0), (8 if save > 2.0 * ks else 0)

    def _plan_mixed(self, sigma, k, lam, ritz, budgets=(0, 0)):
        """混合精度证书的端点规划：两侧各返回 (x, 扰动方向, 精度)；单精度不适用时返回 None。

        单精度计数 ν̃(x) 是邻近矩阵的精确惯性：ν(x−η) ≤ ν̃(x) ≤ ν(x+η)（η 见 factor.count_eta）。
        每一侧沿 |λ−σ| 由 r_k 向外扫描已得特征值与“未求出特征值的估计”（Krylov 额外 Ritz 值——它们与已收敛的
        Ritz 向量正交，数值与已得特征值相同时正是未求出的简并副本），在第一个宽度 ≥ 4η 的间隙里放单精度外端点
        （距两侧各 ≥ 2η，故 ρ − η > r_k）；已得特征值可以被划入窗口，未求出的估计值挡在间隙之前则该侧
        改用双精度内端点 x = σ ± (r_k − τ)，τ = 1e-7·r_k（与第 k 个并列的特征值任取其一）。
        估计不准只会造成计数多出（按“补解”处理）或回退双精度，不影响证书的正确性。
        budgets = (副本额度, 近并列额度)（见 _absorb_budget）：额度内的未求出估计值不再迫使回退双精度，端点越过它们，
        计数多出的部分由收缩补解求出。与某个已得特征值相对差 ≤1e-8 的估计值算“副本”（精确简并），其余算“近并列”。
        """
        lam = np.asarray(lam, dtype=float)
        if lam.size < k:
            return None
        rk = float(np.sort(np.abs(lam - sigma))[k - 1])
        eta = self.eta(abs(sigma) + 2 * rk, "single")
        tau = max(1e-7 * rk, 1e-11 * self.nA)
        if rk < 10 * eta:   # 目标簇贴近 σ（如 σ 落在零空间簇上）：单精度分辨率不够
            return None
        ritz = np.asarray(ritz, dtype=float)
        ritz = ritz[np.isfinite(ritz)]

        ctol = 1e-8 * max(abs(sigma) + rk, 1e-3 * self.nA)

        def plan(side):
            d = np.abs(lam - sigma)[np.sign(lam - sigma) == side]
            rd = np.abs(ritz - sigma)[np.sign(ritz - sigma) == side]
            copy = (np.min(np.abs(rd[:, None] - d[None, :]), axis=1) <= ctol) if d.size and rd.size else np.zeros(rd.size, bool)
            budget = list(budgets)                  # [副本额度, 近并列额度]

            def take(is_copy):
                i = 0 if is_copy else 1
                if budget[i] <= 0:
                    return False
                budget[i] -= 1
                return True

            ties = (rd > rk - 2 * eta) & (rd <= rk + 2 * eta)
            for c in copy[ties]:
                if not take(bool(c)):   # 双精度内端点，近奇异时只向 σ 一侧扰动（见 _solve_nearest）
                    return sigma + side * (rk - tau), -side, "double"
            d = np.concatenate([d, rd[ties]])       # 当作“将被补解求出”的特征值划入窗口
            rd, copy = rd[~ties], copy[~ties]
            pts = sorted([(v, True, False) for v in d[d > rk]] + [(v, False, bool(c)) for v, c in zip(rd, copy) if v > rk])
            a = rk
            for v, found, is_copy in pts:
                if v - a >= 4 * eta:
                    break
                if not found and not take(is_copy):
                    return sigma + side * (rk - tau), -side, "double"
                a = v
            else:
                v = np.inf
            rho = a + max(2 * eta, min(4 * eta, 0.5 * (v - a)))
            return sigma + side * rho, side, "single"

        return plan(-1), plan(1)

    def _certify_nearest_mixed(self, F, s_used, sigma, k, lam, X, block, spec=None):
        """混合精度完备性证书。返回 (是否已出结论, lam, X, F)；未出结论时由双精度内窗口证书接手。

        端点规划见 _plan_mixed。计数 = 窗口内已得个数且单精度端点 η 邻域内无已得特征值
        ⇒ 收缩窗口 (x_lo+η_lo, x_hi−η_hi) 内无遗漏，它覆盖 |λ−σ| < r_k − τ。
        计数 > 已得：缺的特征值都在窗口内，收缩已得向量补解后与同一计数比较，无需重新分解。
        spec：Krylov 迭代中途按 Ritz 值提前提交的计数；最终结果下端点仍满足规划条件时直接采用。
        """
        o = self.o
        plan = self._plan_mixed(sigma, k, lam, self._ritz, self._absorb_budget())
        ks0 = self.rep.krylov_runs[0]["seconds"] if self.rep.krylov_runs else np.inf
        if (plan is not None and "double" in (plan[0][2], plan[1][2]) and not self._parallel
                and 0.4 * self._first_factor_seconds() > ks0):
            # 按需扩展：一侧因 r_k 附近未求出的估计值（多为第 k 个的简并副本）被迫双精度，而该侧改用单精度省下的时间
            # （≈半次数值分解）超过整轮主 Krylov 时，收缩已得向量再求一个块，让它们收敛为“已得”后重新规划
            # （鲲鹏 Bloch 复厄密：两侧由 d/s 变为 s/s，149 → 128 s）
            p = o.block or 8
            # 以主 Krylov 子空间里的下一批 Ritz 向量作初始块：补求几乎从已收敛状态起步（随机初始块时实测与主 Krylov 一样贵）
            X2 = self._shift_invert(F, s_used, p, "nearest", locked=X, block=p, v0=self._ritz_vecs)
            lam, X = self._refine(F, np.hstack([X, X2]))
            lam, X = self._order(lam, X, sigma, "nearest")
            self.rep.diagnostics.append(f"[证书] 第 k 个附近有未求出的简并副本，补求 {p} 个后重新规划单精度计数端点")
            plan = self._plan_mixed(sigma, k, lam, self._ritz, self._absorb_budget())
        if plan is None or plan[0][2] == plan[1][2] == "double":
            return False, lam, X, F
        lo, hi = plan
        if (not self._parallel and (lo[2] == "double") != (hi[2] == "double") and "single" not in self._slots
                and self._first_factor_seconds() < 4.0):
            # 一侧必须双精度时，另一侧用单精度还要为单精度槽另做一次符号分析；分解本身不贵时（鲲鹏 8 核十万阶实矩阵：
            # 首次分解 1.9 s，其中分析 0.75 s、单/双精度数值 0.7/1.4 s）两侧都在位移槽上做双精度更快；
            # 分解贵时（有限元铁氧体腔首次 12 s，数值约 8.5 s）单精度省下的一半数值分解远超一次分析
            lo, hi = [(x, d, "double") for (x, d, _p) in (lo, hi)]
        res = None
        if spec and spec.get("futs"):
            (xs_lo, _, _), (xs_hi, _, _) = spec["plan"]
            rk = float(np.abs(lam[:k] - sigma).max())
            el, eh = self.eta(xs_lo, "single"), self.eta(xs_hi, "single")
            ok = (lo[2] == hi[2] == "single" and xs_lo + el < sigma - rk and xs_hi - eh > sigma + rk
                  and not np.any(np.abs(lam - xs_lo) <= 2 * el) and not np.any(np.abs(lam - xs_hi) <= 2 * eh))
            got = [self._collect(f, purpose="inertia(推测)" if ok else "inertia(推测,未采用)") for f in spec["futs"]]
            if ok:
                res, lo, hi = got, (xs_lo, -1, "single"), (xs_hi, 1, "single")
                self.rep.diagnostics.append("[并行] 证书计数在 Krylov 迭代中途按 Ritz 值提前启动，端点经核对有效")
        if res is None:
            if not self._parallel and "double" in (lo[2], hi[2]):
                F.free()     # 双精度端点复用位移分解的槽；补解时再数值重分解
            res = self.counts([(x, INF, d, pr) for (x, d, pr) in (lo, hi)])
        (n_lo, x_lo), (n_hi, x_hi) = res
        e_lo = self.eta(x_lo, "single") if lo[2] == "single" else 0.0
        e_hi = self.eta(x_hi, "single") if hi[2] == "single" else 0.0
        inside = int(n_hi - n_lo)
        for rnd in range(o.max_rounds):
            if (e_lo and np.any(np.abs(lam - x_lo) <= e_lo)) or (e_hi and np.any(np.abs(lam - x_hi) <= e_hi)):
                self.rep.diagnostics.append("[证书] 单精度计数端点的 η 邻域内有特征值，改用双精度计数")
                return False, lam, X, F
            found_in = int(np.sum((lam > x_lo) & (lam < x_hi)))
            rk = float(np.abs(lam[:k] - sigma).max())
            self.rep.certificate = dict(round=rnd, kind="mixed-window", count_precision=[lo[2], hi[2]], window=[x_lo, x_hi],
                                        eta=[e_lo, e_hi], r_k=rk, inertia_count=inside, found_in_window=found_in)
            if inside == found_in:
                lam, X = self._order(lam, X, sigma, "nearest")
                rk = float(np.abs(lam[:k] - sigma).max())
                need = rk - max(1e-7 * rk, 1e-11 * self.nA)
                if not (x_lo + e_lo <= sigma - need and x_hi - e_hi >= sigma + need):
                    return False, lam, X, F
                self.rep.certified = True
                self.rep.certificate["r_k"] = rk
                self._known = (x_lo + e_lo, x_hi - e_hi, lam[(lam > x_lo) & (lam < x_hi)])
                self._tie_note(lam, k)
                return True, lam, X, F
            if inside < found_in:
                self.rep.diagnostics.append(f"[证书] 计数 {inside} < 窗口内已得 {found_in}，改用双精度内窗口")
                return False, lam, X, F
            missing = inside - found_in
            if missing > max(k, 16):    # 窗口外缘吞入了大簇：交给双精度内窗口证书
                self.rep.diagnostics.append(f"[证书] 窗口比已得多 {missing} 个（大簇贴近窗口边缘），改用双精度内窗口")
                return False, lam, X, F
            block = min(max(block, missing + 1), 16)
            self.rep.diagnostics.append(f"[完备性] 第 {rnd} 轮：窗口惯性计数 {inside} > 已得 {found_in}，"
                                        f"补解 {missing} 个（重特征值/密集谱）→ 收缩已得向量，块尺寸 {block} 继续")
            F = self._ensure_factor(F, s_used)
            X2 = self._shift_invert(F, s_used, missing, "nearest", locked=X, block=block)
            lam, X = self._refine(F, np.hstack([X, X2]))
            lam, X = self._order(lam, X, sigma, "nearest")
        self.rep.certified = False
        return True, lam, X, F

    def _tie_note(self, lam, k):
        """第 k 个特征值是重特征值且部分副本落在前 k 个之外时说明（并列者任取其一均正确）。"""
        if lam.size <= k:
            return
        tol = 1e-8 * max(abs(lam[k - 1]), 1e-3 * self.nA)
        mult = int(np.sum(np.abs(lam - lam[k - 1]) <= tol))
        inc = int(np.sum(np.abs(lam[:k] - lam[k - 1]) <= tol))
        if mult > inc:
            self.rep.diagnostics.append(f"[重特征值] 第 {k} 个特征值 {lam[k - 1]:.12g} 为 {mult} 重（已全部求出并计入证书），"
                                        f"按并列返回其中 {inc} 个")
            self.rep.certificate["kth_multiplicity"] = mult

    def _cluster_note(self, sigma, rk, F=None, s_used=None):
        """σ 恰为高重数特征值（如旋度-旋度算子的梯度零空间）：报告其重数。
        簇上端直接用位移分解自带的惯性（位移 s 在 (σ+r_k, σ+w) 内，簇全部在 s 之下），只需在 σ−w 处计数一次。"""
        w = max(1e-8 * self.nA, 10 * rk)
        try:
            if F is not None and F.inertia is not None and sigma + rk < s_used < sigma + w:
                (n_lo, _), = self.counts([(sigma - w, INF, 0)])
                n_hi = F.inertia[0]
            else:
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
        # σ 处的计数决定“σ 以上/以下”的语义，必须双精度；括区搜索点只起定位作用，用单精度
        res = self.counts([(x0_req, INF, -sgn)] + [(x0_req + sgn * d, INF, 0, self.count_prec) for d in first])
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
            res = self.counts([(x0 + sgn * d, INF, 0, self.count_prec) for d in cands])
            pts += [(abs(xb - x0), (nb - n0) * sgn, nb, xb) for (nb, xb) in res]
            n_batches += 1
        d, c, nb, xb = min((p for p in pts if p[1] >= k), key=lambda p: p[1])
        self.rep.diagnostics.append(f"[目标重定位] 惯性括区：[{min(x0, xb):.6g}, {max(x0, xb):.6g}] 内含 {c} 个特征值"
                                    f"（{len(pts) + 1} 次惯性计数，{n_batches} 批并发），位移移至区间中部求解")
        eb = self.eta(xb, self.count_prec) if self.count_prec == "single" else 0.0
        a, b, na, nbb, ea_, eb_ = (x0, xb, n0, nb, 0.0, eb) if sgn > 0 else (xb, x0, nb, n0, eb, 0.0)
        lam, X = self._solve_range(a, b, na, nbb, ea_, eb_)
        # 以 σ 处实际计数点 x0 为界（σ 贴近特征值时 x0 已向“包含 σ”的一侧扰动）：与括区计数口径一致，
        # 恰在 σ 上的特征值即使数值上落在 σ 另一侧 1e-16 处也计入
        lam, X = self._order(lam, X, x0, which)
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

    def _solve_range(self, xa, xb, na, nb, ea=0.0, eb=0.0):
        """[xa, xb) 内全部特征值。ea/eb > 0 表示端点计数为单精度（η 半径），结束时按 η 裕量核对，必要时双精度重数。"""
        total = nb - na
        self.rep.certificate = dict(kind="interval", interval=[xa, xb], count=int(total))
        if total == 0 and not (ea or eb):
            self.rep.certified = True
            return np.empty(0), np.empty((self.n, 0), self.dtype)
        slices = self._split(xa, xb, na, nb, ea, eb) if total > 0 else []
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
        lam = np.concatenate(lams) if lams else np.empty(0)
        X = np.hstack(Xs) if Xs else np.empty((self.n, 0), self.dtype)
        idx = np.argsort(lam)
        lam, X = lam[idx], X[:, idx]
        if ea or eb:
            # 单精度端点：计数吻合且端点 η 邻域内无已得特征值 ⇒ 收缩窗口 (xa+ea, xb−eb) 内无遗漏；否则双精度重数端点
            near = (ea and np.any(np.abs(lam - xa) <= ea)) or (eb and np.any(np.abs(lam - xb) <= eb))
            if lam.size != total or near:
                redo = [(x, INF, 0) for x, e in ((xa, ea), (xb, eb)) if e]
                res = iter(self.counts(redo))
                na = next(res)[0] if ea else na
                nb = next(res)[0] if eb else nb
                self.rep.diagnostics.append(f"[证书] 单精度端点计数 {total} 与已得 {lam.size} 不符或端点贴近特征值，"
                                            f"双精度重数得 {nb - na}")
                total = nb - na
                self.rep.certificate["count"] = int(total)
                ea = eb = 0.0
            self.rep.certificate.update(count_precision="single" if (ea or eb) else "double", eta=[ea, eb])
        self.rep.certified = bool(lam.size == total)
        if self.rep.certified:
            self._known = (xa + ea, xb - eb, lam)
        return lam, X

    def _solve_one_slice(self, sa, sb, ca, cb, ea=0.0, eb=0.0):
        """单个切片：在中点分解，求切片内全部 cnt 个特征值；少于惯性计数时收缩补解。
        ea/eb：边界计数的单精度不确定半径。边界外 η 内求得的特征值可能被该边界计数算入本片，计作“边界含糊”，
        不再为它们补解（总数由区间两端的计数核对）。"""
        cnt = cb - ca
        if cnt > max(8 * self.o.slice_max, 512):
            # 无法再分的近同值簇（如区间含上万重的梯度零空间）：整体求解需要 cnt 维 Krylov 基，内存不可承受；
            # 如实报告其重数（惯性计数已精确给出），不求向量
            self.rep.diagnostics.append(f"[簇] 切片 [{sa:.10g},{sb:.10g}) 含 {cnt} 个近同值特征值（如梯度零空间），"
                                        f"超出可整体求解的规模，未求特征向量；其重数由惯性计数给出")
            return np.empty(0), np.empty((self.n, 0), self.dtype)
        off = min(self.PRE_OFFSET * self.nA, 1e-4 * (sb - sa) * 0.6180339887)
        F, s_used = self._factor(0.5 * (sa + sb) + off, "slice", keep=True)
        self.rep.shifts.append(s_used)
        self._memory_guard(F.stats.get("mem_mb_used"))
        lam, X = self._refine(F, self._shift_invert(F, s_used, cnt, "nearest"))

        def ambiguous(lv):
            return int(np.sum(((lv <= sa) & (lv >= sa - ea)) | ((lv >= sb) & (lv <= sb + eb))))

        amb = ambiguous(lam)
        inside = (lam > sa) & (lam < sb)
        lam, X = lam[inside], X[:, inside]
        rounds = 0
        while lam.size + amb < cnt and rounds < self.o.max_rounds:
            missing = cnt - lam.size
            blk = min(max(self.o.block or 8, missing + 1), 16)
            self.rep.diagnostics.append(f"[完备性] 切片 [{sa:.6g},{sb:.6g}) 惯性计数 {cnt}，已得 {lam.size}，收缩后继续（块 {blk}）")
            X2 = self._shift_invert(F, s_used, missing, "nearest", locked=X if lam.size else None, block=blk)
            lam, X = self._refine(F, np.hstack([X, X2]) if lam.size else X2)
            amb = max(amb, ambiguous(lam))
            inside = (lam > sa) & (lam < sb)
            lam, X = lam[inside], X[:, inside]
            rounds += 1
        F.free()
        if lam.size + amb < cnt:
            self.rep.diagnostics.append(f"切片 [{sa:.6g},{sb:.6g}) 惯性计数 {cnt} 但仅求得 {lam.size}")
        return lam, X

    def _split(self, a, b, na, nb, ea=0.0, eb=0.0):
        """按惯性二分，使每个切片内特征值数 ≤ slice_max。同一层的分点并发计数。

        分点若贴近特征值（近奇异），在区间内改取黄金分割等备选点；扰动量限制在区间宽度内，
        保证区间严格收缩。无法再分（同值簇超过 slice_max）时整体作为一个切片。
        分点计数只决定各片的目标个数，用单精度；区间两端的计数另行核对（见 _solve_range）。
        区间宽度不足 20η 时单精度无法分辨（重特征值簇宽度为 0），改用双精度分点，近同值簇由近奇异检测识别。
        返回 [(sa, sb, ca, cb, ea, eb)]，ea/eb 为两侧边界计数的 η。
        """
        def prec_for(sa, sb):
            if self.count_prec == "single" and sb - sa > 20 * self.eta(max(abs(sa), abs(sb)), "single"):
                return "single"
            return "double"

        etas = {a: ea, b: eb}
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
            res = self.counts([(sa + 0.5 * (sb - sa), 0.1 * (sb - sa), 0, prec_for(sa, sb)) for (sa, sb, *_r) in need],
                              safe=True)
            for (sa, sb, ca, cb, depth), r in zip(need, res):
                prec = prec_for(sa, sb)
                if r is None or not (sa < r[1] < sb):
                    r = None
                    for f in fracs[1:]:
                        try:
                            cm, xm = self.count_below(sa + f * (sb - sa), max_delta=0.1 * (sb - sa), prec=prec)
                        except NearSingularError:
                            continue
                        if sa < xm < sb:
                            r = (cm, xm)
                            break
                if r is None:
                    out.append((sa, sb, ca, cb))
                    continue
                cm, xm = r
                etas[xm] = self.eta(xm, prec) if prec == "single" else 0.0
                todo += [(sa, xm, ca, cm, depth + 1), (xm, sb, cm, cb, depth + 1)]
        return sorted((sa, sb, ca, cb, etas.get(sa, 0.0), etas.get(sb, 0.0)) for (sa, sb, ca, cb) in out)


def solve(A, B=None, **kw):
    o = Options(**kw)
    lam, X, rep = Solver(A, B, o).solve()
    return lam, X, asdict(rep)
