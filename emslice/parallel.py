"""惯性计数进程池。

完备性证书、惯性括区、谱切片二分都需要在多个位移处做“只要惯性”的分解，彼此独立。
这里用常驻的 spawn 子进程池并发执行：每个子进程单线程、持有一份矩阵与一个可复用的
MUMPS 槽（符号分析只做一次）。MUMPS 自身的 OpenMP 加速在 10 万阶上收益有限（4 线程 ≈1.3×），
进程级并发对独立分解几乎线性。
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

_W: dict = {}


def _init(A, B, opts: dict):
    try:
        from threadpoolctl import threadpool_limits

        _W["tp"] = threadpool_limits(1)
    except Exception:  # pragma: no cover
        pass
    from .solver import Options, Solver

    opts = dict(opts, workers=1)
    _W["solver"] = Solver(A, B, Options(**opts))


def _set_threads(threads):
    """设置本任务线程数。MUMPS 的 L0-OpenMP 结构在符号分析时按线程数确定，
    线程数变化时必须丢弃复用槽、重新分析（否则数值分解/回代结果错误）。"""
    from .solver import _Threads

    s = _W["solver"]
    threads = max(1, int(threads))
    if _W.get("threads") != threads:
        if s._slot is not None:
            s._slot.destroy()
            s._slot = None
        s._thr_fac = s._thr_solve = _Threads(threads)
        _W["threads"] = threads
    return s


def _count(x, max_delta, direction, threads=1):
    s = _set_threads(threads)
    nf, nd = len(s.rep.factorizations), len(s.rep.diagnostics)
    n, xu = s.count_below(x, max_delta=max_delta, direction=direction)
    return n, xu, s.rep.factorizations[nf:], s.rep.diagnostics[nd:], os.getpid()


def _slice(sa, sb, ca, cb, threads=1):
    s = _set_threads(threads)
    nf, nk, nd = len(s.rep.factorizations), len(s.rep.krylov_runs), len(s.rep.diagnostics)
    lam, X = s._solve_one_slice(sa, sb, ca, cb)
    return lam, X, s.rep.factorizations[nf:], s.rep.krylov_runs[nk:], s.rep.diagnostics[nd:], os.getpid()


def _ping():
    return os.getpid()


@contextlib.contextmanager
def _hide_main():
    """spawn 子进程启动时默认会重新导入主模块（脚本 / python -m 包），
    对 stdin 脚本会失败、对无 __name__ 保护的脚本会重复执行。子进程只需 emslice，故临时隐藏。"""
    main = sys.modules.get("__main__")
    saved = {k: getattr(main, k) for k in ("__file__", "__spec__") if main is not None and hasattr(main, k)}
    try:
        for k in saved:
            if k == "__spec__":
                main.__spec__ = None
            else:
                delattr(main, k)
        yield
    finally:
        for k, v in saved.items():
            setattr(main, k, v)


class CountPool:
    def __init__(self, A, B, opts: dict, workers: int):
        self.workers = workers
        ctx = mp.get_context("spawn")
        with _hide_main():
            self.ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init, initargs=(A, B, opts))
            # 预热：子进程在主进程做第一次分解的同时完成启动与矩阵加载（进程在 submit 时同步创建）
            self._warm = [self.ex.submit(_ping) for _ in range(workers)]

    def submit(self, x, max_delta=float("inf"), direction=0, threads=1):
        return self.ex.submit(_count, x, max_delta, direction, threads)

    def submit_slice(self, sa, sb, ca, cb, threads=1):
        return self.ex.submit(_slice, sa, sb, ca, cb, threads)

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)
