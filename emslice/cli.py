"""命令行入口：emslice solve / gen / check / info。"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time


def _set_threads(n):
    if n:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = str(n)


def _load_config(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            cfg = yaml.safe_load(f) or {}
        else:
            cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("配置文件顶层必须是映射")
    return cfg


SOLVE_KEYS = ("k", "sigma", "which", "interval", "tol", "block", "ncv", "verify", "max_rounds", "slice_max",
              "backend", "precision", "ordering", "mem_limit_gb", "ooc", "threads", "seed", "refine_steps",
              "workers", "blr")


def cmd_solve(a):
    cfg = _load_config(a.config)
    cli = {k: v for k, v in vars(a).items() if v is not None}
    merged = {**cfg, **{k: cli[k] for k in cli if k in SOLVE_KEYS}}
    if a.interval is not None:
        merged["which"] = "interval"
        merged["interval"] = tuple(a.interval)
    if a.no_verify:
        merged["verify"] = False
    unknown = set(cfg) - set(SOLVE_KEYS) - {"matrix", "B", "out", "format", "triangle", "key", "symmetrize", "secure", "save_vectors"}
    if unknown:
        raise SystemExit(f"配置文件含未知键: {sorted(unknown)}")
    _set_threads(merged.get("threads"))

    from .io import load_matrix, save_results
    from .solver import Options, Solver
    from dataclasses import asdict

    log = logging.getLogger("emslice")
    t0 = time.perf_counter()
    A, ainfo = load_matrix(a.matrix, key=a.key, triangle=a.triangle or cfg.get("triangle", "auto"),
                           symmetrize=a.symmetrize or cfg.get("symmetrize", False))
    B = None
    binfo = None
    if a.B or cfg.get("B"):
        B, binfo = load_matrix(a.B or cfg["B"], triangle=a.triangle or "auto")
    # 日志仅含元数据（不输出矩阵数值）
    log.info("矩阵 n=%d nnz=%d dtype=%s storage=%s sha256=%s…", ainfo["shape"][0], ainfo["nnz"], ainfo["dtype"],
             ainfo["storage"], ainfo["sha256"][:12])
    t_read = time.perf_counter() - t0
    opts = Options(**merged)
    lam, X, rep = Solver(A, B, opts).solve()
    report = asdict(rep)
    report["input"] = {"A": {k: v for k, v in ainfo.items() if k != "path" or not a.secure},
                       "B": binfo}
    report["timings"]["read"] = t_read
    report["options"] = asdict(opts)
    report["eigenvalues"] = lam.tolist()
    out = a.out or cfg.get("out") or "result"
    files = save_results(out, lam, X, report, fmt=a.format or cfg.get("format", "npy"), secure=a.secure,
                         save_vectors=not a.no_vectors)
    r = report["residuals"]
    print(f"[emslice] {lam.size} 个特征值  证书={'通过' if rep.certified else rep.certified}  "
          f"最大后向误差={r['backward_max']:.2e}  耗时={report['timings']['total']:.2f}s")
    for d in rep.diagnostics:
        print("  诊断:", d)
    print("  输出:", ", ".join(files))
    return 0 if (r["all_below_tol"] and rep.certified is not False) else 2


def cmd_gen(a):
    from .io import save_matrix

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.preset.startswith("fem-"):
        from .femgen import preset as fem_preset

        K, M, meta = fem_preset(a.preset, a.h)
        root, ext = os.path.splitext(a.out)
        out_b = f"{root}_B{ext or '.npz'}"
        save_matrix(a.out, sp_triu(K) if a.upper else K)
        save_matrix(out_b, sp_triu(M) if a.upper else M)
        with open(f"{root}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"{a.preset} h={meta['h_mm']}mm: n={K.shape[0]} nnz(K)={K.nnz} dtype={K.dtype} "
              f"梯度零空间={meta['null_dim']} → K: {a.out}  M: {out_b}")
        print(f"求解示例：emslice solve {a.out} --B {out_b} --sigma 1e-6 --which above -k 10")
        return 0
    from .matgen import preset

    A, g = preset(a.preset, a.N)
    save_matrix(a.out, sp_triu(A) if a.upper else A)
    print(f"{a.preset} N={a.N}: n={A.shape[0]} nnz={A.nnz} dtype={A.dtype} → {a.out}")
    return 0


def sp_triu(A):
    import scipy.sparse as sp

    return sp.triu(A, format="csr")


def cmd_check(a):
    import numpy as np

    from .io import load_matrix
    from .solver import Solver

    A, _ = load_matrix(a.matrix)
    B = load_matrix(a.B)[0] if a.B else None
    lam = np.load(os.path.join(a.result, "eigenvalues.npy"))
    X = np.load(os.path.join(a.result, "eigenvectors.npy"))
    s = Solver(A, B)
    rn, back, rel = s.residuals(lam, X)
    G = X.conj().T @ (X if B is None else B @ X)
    orth = float(np.abs(G - np.diag(np.diag(G))).max()) if lam.size > 1 else 0.0
    print(json.dumps(dict(n=A.shape[0], k=int(lam.size), backward_max=float(back.max()), abs_res_max=float(rn.max()),
                          orth_offdiag_max=orth), ensure_ascii=False, indent=2))
    return 0


def cmd_info(a):
    from .factor import available_backends

    import numpy as np
    import scipy

    print(json.dumps(dict(backends=available_backends(), numpy=np.__version__, scipy=scipy.__version__,
                          cpu_count=os.cpu_count()), ensure_ascii=False, indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="emslice", description="面向电磁谐振的大规模非正定厄密稀疏特征值求解器")
    p.add_argument("--log-level", default="WARNING")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="求解特征值")
    s.add_argument("matrix", nargs="?")
    s.add_argument("--B", help="质量矩阵（广义问题 Ax=λBx）")
    s.add_argument("--config", help="YAML/JSON 配置文件，命令行参数优先")
    s.add_argument("-k", "--k", type=int)
    s.add_argument("--sigma", type=float, help="目标位移（对应目标频率）")
    s.add_argument("--which", choices=["nearest", "above", "below"])
    s.add_argument("--interval", type=float, nargs=2, metavar=("A", "B"), help="求 [A,B] 内全部特征值")
    s.add_argument("--tol", type=float)
    s.add_argument("--block", type=int)
    s.add_argument("--ncv", type=int)
    s.add_argument("--no-verify", action="store_true", help="跳过惯性完备性证书")
    s.add_argument("--backend", choices=["auto", "petsc", "scipy"])
    s.add_argument("--precision", choices=["double", "mixed", "auto"])
    s.add_argument("--ordering", choices=["auto", "metis", "amd", "scotch", "pord"])
    s.add_argument("--mem-limit-gb", type=float, dest="mem_limit_gb")
    s.add_argument("--threads", type=int)
    s.add_argument("--workers", type=int, help="惯性计数并发进程数（默认自动）")
    s.add_argument("--blr", type=float, help="启用 MUMPS 块低秩压缩，参数为截断精度（如 1e-10）")
    s.add_argument("--seed", type=int)
    s.add_argument("--triangle", choices=["auto", "full", "upper", "lower"])
    s.add_argument("--key", help="npz/mat 文件中的变量名")
    s.add_argument("--symmetrize", action="store_true")
    s.add_argument("--out", help="输出目录")
    s.add_argument("--format", choices=["npy", "npz", "mat"])
    s.add_argument("--no-vectors", action="store_true")
    s.add_argument("--secure", action="store_true", help="结果文件 0600 权限，报告中不记录输入路径")
    s.set_defaults(func=cmd_solve)

    g = sub.add_parser("gen", help="生成电磁测试矩阵")
    g.add_argument("preset", choices=["cavity", "cube", "puck", "bloch", "bloch-homog", "fem-cavity", "fem-dr", "fem-ferrite"],
                   help="Yee 差分算子（标准问题），或 fem-*：Nédélec 有限元广义问题 Kx=λMx（需 scikit-fem；fem-dr/fem-ferrite 另需 gmsh）")
    g.add_argument("--N", type=int, default=32, help="Yee 网格每方向单元数")
    g.add_argument("--h", type=float, help="fem-* 网格尺寸（mm），默认约 10 万阶")
    g.add_argument("--out", required=True)
    g.add_argument("--upper", action="store_true", help="仅保存上三角（测试三角恢复）")
    g.set_defaults(func=cmd_gen)

    c = sub.add_parser("check", help="独立校验结果残差与正交性")
    c.add_argument("matrix")
    c.add_argument("result")
    c.add_argument("--B")
    c.set_defaults(func=cmd_check)

    i = sub.add_parser("info", help="显示可用后端")
    i.set_defaults(func=cmd_info)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.WARNING), format="%(asctime)s %(message)s")
    if a.cmd == "solve" and not a.matrix and not a.config:
        raise SystemExit("需要矩阵文件或 --config")
    if a.cmd == "solve" and not a.matrix:
        a.matrix = _load_config(a.config).get("matrix")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
