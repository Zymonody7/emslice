"""并行扩展性汇总：读取按线程数打标签的结果（label = emslice-t{T} / slepc-t{T}），输出表格与加速比图。

python bench/run.py --solver emslice --threads T --extra '{"threads": T}' --label emslice-tT --out results/scaling
python bench/scaling_report.py results/scaling            # → results/scaling/scaling.md, scaling.png
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


def load(d):
    by = defaultdict(dict)        # name -> {(solver, T): row}
    with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = re.fullmatch(r"(.+)-t(\d+)", r["solver"])
            if m:
                by[r["name"]][(m.group(1), int(m.group(2)))] = r
    return by


def main(d, title=""):
    by = load(d)
    lines = [f"# 并行扩展性{(' — ' + title) if title else ''}", "",
             "| 算例 | 求解器 | 线程 | 耗时 (s) | 加速比 | 分解耗时合计 (s) | 证书 | 最大后向误差 |", "|---|---|---|---|---|---|---|---|"]
    curves = {}
    for name, rows in by.items():
        for solver in sorted({s for s, _ in rows}):
            ts = sorted(t for s, t in rows if s == solver)
            base = rows[(solver, ts[0])].get("seconds")
            pts = []
            for t in ts:
                r = rows[(solver, t)]
                sec = r.get("seconds") if not r.get("error") else None
                fsum = (r.get("timings") or {}).get("factor_sum")
                sp = (base * ts[0] / sec) if (sec and base) else None   # 相对 1 线程（或最少线程）的加速比
                pts.append((t, sec, sp))
                if not sec:
                    lines.append(f"| {name} | {solver} | {t} | 失败 | — | — | — | {str(r.get('error'))[:40]} |")
                    continue
                fs = "—" if fsum is None else f"{fsum:.1f}"
                bw = "—" if r.get("backward_max") is None else f"{r['backward_max']:.1e}"
                lines.append(f"| {name} | {solver} | {t} | {sec:.1f} | {sp:.2f} | {fs} | "
                             f"{'✅' if r.get('certified') else '—'} | {bw} |")
            curves[(name, solver)] = pts
    out = os.path.join(d, "scaling.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out)
    plot(curves, d, title)


def plot(curves, d, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fam in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    tmax = 1
    for (name, solver), pts in curves.items():
        pts = [p for p in pts if p[1]]
        if not pts:
            continue
        t = np.array([p[0] for p in pts])
        tmax = max(tmax, t.max())
        ls = "-" if solver == "emslice" else "--"
        lab = f"{name}（{solver}）"
        axs[0].plot(t, [p[1] for p in pts], ls, marker="o", ms=4, label=lab)
        axs[1].plot(t, [p[2] for p in pts], ls, marker="o", ms=4, label=lab)
    ideal = np.array([1, tmax])
    axs[1].plot(ideal, ideal, ":", color="gray", label="理想线性")
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("线程数")
        ax.grid(alpha=0.3)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("耗时 (s)")
    axs[1].set_ylabel("加速比（相对 1 线程）")
    axs[1].legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "scaling.png"), dpi=150)
    print(os.path.join(d, "scaling.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/scaling", sys.argv[2] if len(sys.argv) > 2 else "")
