"""惯性计数进程池。

完备性证书、惯性括区、谱切片二分都需要在多个位移处做“只要惯性”的分解，彼此独立。
这里用常驻的 spawn 子进程池并发执行：每个子进程持有一份矩阵与一个可复用的 MUMPS 槽。

* 线程数在池创建时固定：MUMPS 的 L0-OpenMP 结构随线程数而变，线程数不变才能复用符号分析
  （10 万阶上符号分析约占一次计数分解的 45%）；
* warm=True 时子进程启动后立即做一次计数分解完成符号分析（nearest 模式下主进程分解与 Krylov
  期间子进程本来空闲），之后证书计数只需数值分解；
* AsyncCountPool 在后台线程创建进程池（spawn + 传矩阵约 1 s，MUMPS 分解期间释放 GIL），
  创建完成前提交的任务排队，主进程不必等待。
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
import threading
from concurrent.futures import Future, ProcessPoolExecutor

_W: dict = {}


def _init(A, B, opts: dict, threads=1, warm=False):
    try:
        from threadpoolctl import threadpool_limits

        _W["tp"] = threadpool_limits(1)
    except Exception:  # pragma: no cover
        pass
    from .solver import Options, Solver

    opts = dict(opts, workers=1)
    _W["solver"] = s = Solver(A, B, Options(**opts))
    _set_threads(threads)
    if warm:  # 预做符号分析：在远离谱的位移处做一次计数分解（结果丢弃）
        try:
            s.count_below(-1.1 * s.nA / max(s.nB, 1e-300), prec=s.count_prec)
        except Exception:  # pragma: no cover - 预热失败不影响后续计数
            pass
        s.rep.factorizations.clear()
        s.rep.diagnostics.clear()


def _set_threads(threads):
    """设置本任务线程数。MUMPS 的 L0-OpenMP 结构在符号分析时按线程数确定，
    线程数变化时必须丢弃复用槽、重新分析（否则数值分解/回代结果错误）。"""
    from .solver import _Threads

    s = _W["solver"]
    threads = max(1, int(threads))
    if _W.get("threads") != threads:
        for slot in s._slots.values():
            slot.destroy()
        s._slots.clear()
        s._thr_fac = s._thr_solve = _Threads(threads)
        _W["threads"] = threads
    return s


def _count(x, max_delta, direction, prec="double", threads=1):
    s = _set_threads(threads)
    nf, nd = len(s.rep.factorizations), len(s.rep.diagnostics)
    n, xu = s.count_below(x, max_delta=max_delta, direction=direction, prec=prec)
    return n, xu, s.rep.factorizations[nf:], s.rep.diagnostics[nd:], os.getpid()


def _slice(sa, sb, ca, cb, ea=0.0, eb=0.0, threads=1):
    s = _set_threads(threads)
    nf, nk, nd = len(s.rep.factorizations), len(s.rep.krylov_runs), len(s.rep.diagnostics)
    lam, X = s._solve_one_slice(sa, sb, ca, cb, ea, eb)
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
    def __init__(self, A, B, opts: dict, workers: int, threads: int = 1, warm: bool = False):
        self.workers = workers
        self.threads = threads
        ctx = mp.get_context("spawn")
        with _hide_main():
            self.ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init,
                                          initargs=(A, B, opts, threads, warm))
            # 预热：子进程在主进程做第一次分解的同时完成启动与矩阵加载（进程在 submit 时同步创建）
            self._warm = [self.ex.submit(_ping) for _ in range(workers)]

    def submit(self, x, max_delta=float("inf"), direction=0, prec="double"):
        return self.ex.submit(_count, x, max_delta, direction, prec, self.threads)

    def submit_slice(self, sa, sb, ca, cb, ea=0.0, eb=0.0):
        return self.ex.submit(_slice, sa, sb, ca, cb, ea, eb, self.threads)

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)


def _chain(inner: Future, outer: Future):
    if inner.cancelled():
        outer.cancel()
    elif inner.exception() is not None:
        outer.set_exception(inner.exception())
    else:
        outer.set_result(inner.result())


class AsyncCountPool:
    """在后台线程创建 CountPool；创建完成前提交的任务排队，完成后立即分发。接口与 CountPool 相同。"""

    def __init__(self, *args, **kw):
        self._lock = threading.Lock()
        self._pool = None
        self._err = None
        self._queue = []
        self._t = threading.Thread(target=self._create, args=args, kwargs=kw, daemon=True)
        self._t.start()

    def _create(self, *args, **kw):
        try:
            pool = CountPool(*args, **kw)
        except BaseException as e:  # pragma: no cover
            pool, err = None, e
        else:
            err = None
        with self._lock:
            self._pool, self._err = pool, err
            q, self._queue = self._queue, None
        for item in q:
            self._dispatch(*item)

    def _dispatch(self, meth, args, outer):
        if self._err is not None:
            outer.set_exception(self._err)
            return
        getattr(self._pool, meth)(*args).add_done_callback(lambda f: _chain(f, outer))

    def _enqueue(self, meth, args):
        outer = Future()
        with self._lock:
            if self._queue is not None:
                self._queue.append((meth, args, outer))
                return outer
        self._dispatch(meth, args, outer)
        return outer

    def submit(self, x, max_delta=float("inf"), direction=0, prec="double"):
        return self._enqueue("submit", (x, max_delta, direction, prec))

    def submit_slice(self, sa, sb, ca, cb, ea=0.0, eb=0.0):
        return self._enqueue("submit_slice", (sa, sb, ca, cb, ea, eb))

    def close(self):
        self._t.join()
        if self._pool is not None:
            self._pool.close()
