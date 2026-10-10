"""把 bench/run.py 的 results.jsonl 汇总为 Markdown 表格与对比图。

python bench/report.py results/full            # → results/full/report.md, *.png
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

SOLVERS = ("emslice", "emslice-noverify", "slepc", "scipy")
LABEL = {"emslice": "emslice（含证书）", "emslice-noverify": "emslice（无证书快速模式）",
         "slepc": "SLEPc Krylov-Schur + MUMPS", "scipy": "SciPy eigsh + SuperLU"}


def load(d):
    rows = []
    with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    by = defaultdict(dict)
    for r in rows:
        by[r["name"]][r["solver"]] = r
    return by


def agree(a, b, nA_rel=1e-8):
    """两组特征值是否一致（以最大值为尺度）。"""
    if not a or not b or len(a) != len(b):
        return None
    a, b = np.sort(a), np.sort(b)
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))


def fmt(x, f="{:.2e}"):
    return "—" if x is None else f.format(x)


def main(d):
    by = load(d)
    lines = ["# 基准测试结果", "", f"数据：`{d}/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离）。", ""]
    lines += ["| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sv in by.items():
        for s in SOLVERS:
            r = sv.get(s)
            if not r:
                continue
            ref_err = r.get("max_eig_err_rel")
            if ref_err is None and s != "emslice" and "emslice" in sv:
                ref_err = agree(r.get("eigenvalues"), sv["emslice"].get("eigenvalues"))
            cert = {True: "✅", False: "❌", None: "无"}[r.get("certified")] if s.startswith("emslice") else "无"
            err = r.get("error")
            lines.append(
                f"| {name} | {r.get('scenario', '')} | {r.get('n', '—')} | {LABEL[s]} | "
                f"{fmt(r.get('seconds'), '{:.1f}')} | {fmt(r.get('peak_rss_mb'), '{:.0f}')} | "
                f"{r.get('k', '—') if not err else '失败'} | {cert} | {fmt(r.get('backward_max'))} | {fmt(ref_err)} | "
                f"{r.get('missing_vs_ref', '—') if not err else err[:40]} |")
    out = os.path.join(d, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out)
    try:
        plot(by, d)
    except Exception as e:  # pragma: no cover
        print("绘图失败:", e)


def plot(by, d):
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
    names = list(by)
    x = np.arange(len(names))
    wdt = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(names)), 4.2))
    colors = {"emslice": "#2a6fdb", "emslice-noverify": "#8fb4ee", "slepc": "#9aa3ad", "scipy": "#d9a441"}
    for i, s in enumerate(SOLVERS):
        vals, bad = [], []
        for n in names:
            r = by[n].get(s, {})
            v = r.get("seconds") if r and not r.get("error") else None
            vals.append(np.nan if v is None else v)
            # 结果错误：后向误差 > 1e-8，或返回个数不足（漏解/空结果）
            req = by[n].get("emslice", {}).get("k")
            bad.append(bool(r) and ((r.get("backward_max") or 0) > 1e-8 or (req is not None and (r.get("k") or 0) < req)))
        xs = x + (i - 1.5) * wdt
        ax.bar(xs, vals, wdt, label=LABEL[s], color=colors[s])
        for xi, v, b in zip(xs, vals, bad):
            if b:
                hv = v if np.isfinite(v) else 1.0
                ax.bar(xi, hv, wdt, color="none", edgecolor="#c0392b", hatch="////", linewidth=0)
                ax.text(xi, hv * 1.08, "×", ha="center", va="bottom", color="#c0392b", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("耗时 (s，对数)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("红色斜线 + × = 结果错误（后向误差 > 1e-8 或漏解/空结果）", fontsize=8, color="#c0392b", pad=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "time_compare.png"), dpi=160)
    print(os.path.join(d, "time_compare.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/full")
