"""合并多个 results 目录并生成文档用的对比表（Markdown）。

python bench/summary.py --out results/kunpeng8-final results/kunpeng8 results/kunpeng8-scipy \
    --override results/kunpeng8-cplx3 --baseline results/kunpeng8-v01

* 依次读入各目录的 results.jsonl，同一 (算例, 求解器) 以后读入的为准；--override 目录最后读入；
* --baseline 目录里的 emslice 结果作为“v0.1”列；
* 输出合并后的 results.jsonl（供 bench/report.py 出图）与 summary.md（文档主表）。
"""
from __future__ import annotations

import argparse
import json
import os

TITLE = {
    "cube_mid_band": "立方腔目标频段（σ 恰为 18 重谐振点）",
    "cube_low_freq": "低频：σ≈0 以上 20 个物理模式（3.2 万重零空间）",
    "cube_near_singular": "近奇异：σ 与谐振点相对差 1e-13",
    "cube_null_cluster": "σ 落在零空间上",
    "cavity_upper_storage": "仅存上三角输入",
    "dielectric_puck": "介质谐振器（σ 以上 20 个）",
    "bloch_complex_hermitian": "复厄密（Bloch 周期）",
    "photonic_crystal": "光子晶体（复厄密 + 介质柱）",
    "suitesparse_2cubes_low": "公开矩阵 2cubes_sphere 最低 30 个",
    "suitesparse_2cubes_band": "公开矩阵 2cubes_sphere σ=40 附近 20 个",
    "fem_cavity_low": "FEM 空腔低频（1.3 万维零空间）",
    "fem_dr_low": "FEM 介质谐振器低频前 10 个",
    "fem_dr_band": "FEM 介质谐振器 5 GHz 附近 20 个",
    "fem_ferrite_band": "FEM 铁氧体腔（复厄密）8.5 GHz 附近 8 个",
}


def load(d):
    out = {}
    p = os.path.join(d, "results.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[(r["name"], r["solver"])] = r
    return out


def cell(r, ref_k=None, emslice=False):
    if r is None:
        return "—"
    err = r.get("error")
    if err:
        return "**超时**" if str(err).startswith("timeout") else "**失败**"
    t = f"{r['seconds']:.1f} s"
    back = r.get("backward_max")
    k = r.get("k") or 0
    if ref_k is not None and k < ref_k:
        return f"{t} **返回 {k} 个**" if k == 0 else f"{t} **漏 {ref_k - k} 个**"
    if (r.get("missing_vs_ref") or 0) > 0:
        return f"{t} **漏 {r['missing_vs_ref']} 个**"
    if back is None:
        return t
    s = f"{back:.0e}".replace("e-0", "e-")
    if back > 1e-8:
        return f"{t} **后向误差 {s}**"
    if emslice:
        return f"{t} √ {s}" if r.get("certified") else f"{t} {s}"
    return f"{t} {s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--baseline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for d in a.dirs + a.override:
        rows.update(load(d))
    base = load(a.baseline) if a.baseline else {}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    names = [n for n in TITLE if any(k[0] == n for k in rows)]
    md = ""
    for title, group in (("标准问题与公开矩阵", [n for n in names if not n.startswith("fem_")]),
                         ("有限元广义问题", [n for n in names if n.startswith("fem_")])):
        if group:
            md += f"### {title}\n\n" + table(group, rows, base) + "\n"
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


def table(names, rows, base):
    lines = ["| 算例（n≈10 万） | emslice 含证书 | emslice 快速 | v0.1 含证书 | SLEPc | SciPy | 峰值内存 emslice / SLEPc |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        e = rows.get((n, "emslice"))
        ref_k = e.get("k") if e else None
        b = base.get((n, "emslice"))
        sl = rows.get((n, "slepc"))
        mem = lambda r: (max(v for v in (r.get("peak_tree_mb"), r.get("peak_rss_mb")) if v) if r else None)
        me, ms = mem(e), mem(sl)
        lines.append("| " + " | ".join([
            TITLE[n], cell(e, emslice=True), cell(rows.get((n, "emslice-noverify")), ref_k),
            f"{b['seconds']:.1f} s" if b and not b.get("error") else "—",
            cell(sl, ref_k), cell(rows.get((n, "scipy")), ref_k),
            f"{me / 1024:.1f} / {ms / 1024:.1f} GB" if me and ms else "—"]) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
