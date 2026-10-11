"""运行基准算例：每个算例在独立子进程中执行（隔离峰值内存），输出 JSON 与 Markdown 汇总。

python bench/run.py --cases data/bench/cases.json --out results/bench [--only cube_mid_band] [--solver emslice|slepc|scipy]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WORKER = r"""
import json, sys, time, resource, os
sys.path.insert(0, {root!r})
import numpy as np
case = json.loads(sys.argv[1]); solver = sys.argv[2]; extra = json.loads(sys.argv[3])
from emslice.io import load_matrix
A, info = load_matrix(case["matrix"])
B = load_matrix(case["B"])[0] if case.get("B") else None
t0 = time.perf_counter()
out = dict(name=case["name"], solver=solver, n=A.shape[0], dtype=str(A.dtype))
try:
    if solver == "emslice":
        from emslice.solver import Options, Solver
        lam, X, rep = Solver(A, B, Options(k=case["k"], sigma=case["sigma"], which=case["which"], **extra)).solve()
        out.update(certified=rep.certified, n_factor=len(rep.factorizations),
                   n_apply=sum(r["n_apply"] for r in rep.krylov_runs), diagnostics=rep.diagnostics,
                   timings=rep.timings, workers=rep.workers,
                   factor_log=[(f["purpose"], round(f.get("seconds_factor", 0), 2), f.get("worker", 0), f.get("symbolic_reused"), f.get("precision", "double")[0]) for f in rep.factorizations],
                   krylov_log=[(r["nev"], r["block"], r["ncv"], r["n_apply"], r["restarts"], r["nconv"], round(r["seconds"], 2)) for r in rep.krylov_runs])
    else:
        from bench.baselines import run_baseline
        lam, X, meta = run_baseline(solver, A, case["k"], case["sigma"], case["which"], B=B, **extra)
        out.update(meta)
    out["seconds"] = time.perf_counter() - t0
    from emslice.solver import Solver as _S
    s = _S(A, B)
    if lam.size:
        rn, back, rel = s.residuals(np.real(lam), X)
        out.update(k=int(lam.size), backward_max=float(back.max()), abs_res_max=float(rn.max()))
    else:
        out.update(k=0, backward_max=None, abs_res_max=None)
    lam = np.sort(np.real(lam))
    out["eigenvalues"] = lam.tolist()
    if case.get("ref") is not None:
        ref = np.asarray(case["ref"])
        if lam.size == ref.size:
            pass
        scale = s.nA  # 以 ‖A‖ 为尺度（参考值可能全为 0，如零空间算例）
        if lam.size == ref.size:
            out["max_eig_err_abs"] = float(np.abs(lam - ref).max())
            out["max_eig_err_rel"] = out["max_eig_err_abs"] / scale
        out["missing_vs_ref"] = int(ref.size - (np.sum(np.isclose(ref[:, None], lam[None, :], rtol=0, atol=1e-8 * scale).any(1)) if lam.size else 0))
except Exception as e:
    out["error"] = f"{{type(e).__name__}}: {{e}}"
    out["seconds"] = time.perf_counter() - t0
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)
print("@@RESULT" + json.dumps(out, default=str))
"""


def run_case(case, solver, extra, threads, timeout):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        if threads > 0:          # threads=0：不强制环境变量，交给求解器分阶段控制
            env[v] = str(threads)
        else:
            env.pop(v, None)
    code = WORKER.format(root=ROOT)
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", code, json.dumps(case), solver, json.dumps(extra)], capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return dict(name=case["name"], solver=solver, error=f"timeout {timeout}s", seconds=float(timeout))  # 墙钟可能含系统休眠，记为时限
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT"):
            return json.loads(line[8:])
    return dict(name=case["name"], solver=solver, error=(p.stderr or p.stdout)[-800:], seconds=time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(ROOT, "data/bench/cases.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/bench"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--solver", nargs="*", default=["emslice"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--extra", default="{}", help="传给求解器的额外参数（JSON）")
    ap.add_argument("--label", help="结果中的求解器名称（默认同 --solver），用于区分同一求解器的不同配置")
    a = ap.parse_args()
    cases = json.load(open(a.cases, encoding="utf-8"))
    if a.only:
        cases = [c for c in cases if c["name"] in a.only]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for c in cases:
        for s in a.solver:
            r = run_case(c, s, json.loads(a.extra), a.threads, a.timeout)
            r["scenario"] = c["scenario"]
            if a.label:
                r["solver"] = a.label
            rows.append(r)
            brief = {k: r.get(k) for k in ("name", "solver", "seconds", "k", "certified", "backward_max", "max_eig_err_rel",
                                           "missing_vs_ref", "peak_rss_mb", "error")}
            print(json.dumps(brief, ensure_ascii=False, default=str), flush=True)
            with open(os.path.join(a.out, "results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return rows


if __name__ == "__main__":
    main()
