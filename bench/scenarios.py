"""任务书“故障场景与验证”：按场景汇总精度、性能、迭代次数（Markdown 表）。

python bench/scenarios.py results/kunpeng8-final results/explore > results/scenarios.md
每行取一个结果目录中的一次运行（emslice 含证书）；迭代次数 = Krylov 算子作用次数（重启次数）与分解次数（d 双精度 / s 单精度）。
"""
from __future__ import annotations

import json
import os
import sys

ROWS = [  # (场景, 算例, 求解器标签, 说明)
    ("零附近密集谱 / 低频击穿", "cube_low_freq", "emslice", "σ≈0 以上 20 个，3.2 万重零空间"),
    ("零附近密集谱 / 低频击穿", "fem_cavity_low", "emslice", "有限元，1.3 万维零空间"),
    ("零附近密集谱 / 低频击穿", "fem_dr_above50", "emslice", "有限元介质谐振器，σ≈0 以上 50 个"),
    ("零附近密集谱 / 低频击穿", "cube_null_cluster", "emslice", "σ 恰在零空间上"),
    ("重特征值", "cube_mid_band", "emslice", "σ 恰为 18 重谐振点，第 k 个为 12 重"),
    ("重特征值", "bloch_complex_hermitian", "emslice", "复厄密，全谱二重简并"),
    ("近奇异位移", "cube_near_singular", "emslice", "σ 与谐振点相对差 1e-13"),
    ("近奇异位移", "cavity_upper_storage", "emslice", "σ 恰为二重谐振点 + 仅存上三角"),
    ("大 k", "cube_k100", "emslice", "最近 100 个"),
    ("大 k", "cube_k200", "emslice", "最近 200 个"),
    ("大 k", "fem_dr_k60", "emslice", "有限元最近 60 个"),
    ("区间", "cube_interval", "emslice", "[2990, 3040] 内全部"),
    ("区间", "fem_dr_interval", "emslice", "有限元 [0.0105, 0.016] 内全部"),
    ("内存受限", "bloch_mem", "emslice-default", "复厄密 21.6 万实嵌入，默认"),
    ("内存受限", "bloch_mem", "emslice-limit5", "--mem-limit-gb 5（自动外存）"),
]


def load(dirs):
    rows = {}
    for d in dirs:
        p = os.path.join(d, "results.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                rows[(r["name"], r["solver"])] = r
    return rows


def main(dirs):
    rows = load(dirs)
    out = ["| 场景 | 算例 | 返回 / 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 耗时 | 峰值内存 | 分解（d/s） | Krylov 作用次数（重启） |",
           "|---|---|---|---|---|---|---|---|---|"]
    for scen, name, solver, note in ROWS:
        r = rows.get((name, solver))
        if not r or r.get("error"):
            continue
        prec = "".join(f[4] if len(f) > 4 else "d" for f in r.get("factor_log", []))
        nd, ns = prec.count("d"), prec.count("s")
        ks = r.get("krylov_log", [])
        napply = sum(x[3] for x in ks)
        rst = sum(x[4] for x in ks)
        mem = max(v for v in (r.get("peak_tree_mb"), r.get("peak_rss_mb")) if v) / 1024
        err = r.get("max_eig_err_rel")
        out.append(f"| {scen} | {note} | {r.get('k')} / {'√' if r.get('certified') else '✗'} | {r['backward_max']:.0e} | "
                   f"{'—' if err is None else f'{err:.0e}'} | {r['seconds']:.1f} s | {mem:.1f} GB | {nd}/{ns} | "
                   f"{napply}（{rst}）{'，' + str(len(ks)) + ' 轮' if len(ks) > 1 else ''} |")
    print("\n".join(out).replace("e-0", "e-"))


if __name__ == "__main__":
    main(sys.argv[1:])
