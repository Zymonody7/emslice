"""把 emslice 的 report.json 打印成便于录屏/阅读的摘要。

python scripts/show_report.py result/mid            # 证书、残差、诊断、特征值
python scripts/show_report.py result/dr --ghz       # 有限元算例：特征值换算为频率（GHz）
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

C_MM_PER_NS = 299.792458


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--ghz", action="store_true", help="λ=k₀²（mm⁻²）换算为频率 f=c√λ/2π")
    ap.add_argument("--max", type=int, default=20, help="最多显示的特征值个数")
    a = ap.parse_args()
    r = json.load(open(os.path.join(a.result, "report.json"), encoding="utf-8"))
    lam = np.load(os.path.join(a.result, "eigenvalues.npy"))

    cert = r.get("certificate") or {}
    ok = {True: "通过", False: "未通过", None: "未做（无惯性后端或 --no-verify）"}[r.get("certified")]
    print(f"■ 问题：n={r['n']}  {r['dtype']}  模式={r['mode']}  σ={r['sigma']:.6g}  请求 k={r['k_requested']}  返回 {r['k_returned']}")
    print(f"■ 完备性证书：{ok}")
    kind = cert.get("kind")
    if kind == "inner-window":
        lo, hi = cert["window"]
        print(f"    窗口 ({lo:.10g}, {hi:.10g}) 内惯性计数 = {cert['inertia_count']}，已求得 = {cert['found_in_window']}")
    elif kind == "interval":
        print(f"    区间 [{cert['interval'][0]:.6g}, {cert['interval'][1]:.6g}] 内惯性计数 = {cert['count']}，"
              f"切片 {len(cert.get('slices', []))} 个")
    elif kind:
        print(f"    {kind}: " + ", ".join(f"{k}={v}" for k, v in cert.items() if k != "kind"))
    res = r["residuals"]
    print(f"■ 精度：最大后向误差 {res['backward_max']:.2e}（门槛 {res['tol']:.0e}，{'全部达标' if res['all_below_tol'] else '未全部达标'}）")
    fz, kr = r["factorizations"], r["krylov_runs"]
    print(f"■ 代价：总耗时 {r['timings']['total']:.2f} s；分解 {len(fz)} 次（{r['timings']['factor_sum']:.2f} s），"
          f"Krylov 算子作用 {sum(k['n_apply'] for k in kr)} 次")
    for d in r.get("diagnostics", []):
        print(f"■ 诊断：{d}")
    show = lam[: a.max]
    if a.ghz:
        f = C_MM_PER_NS * np.sqrt(np.maximum(show, 0)) / (2 * np.pi)
        print("■ 谐振频率 (GHz)：" + "  ".join(f"{v:.4f}" for v in f))
    else:
        print("■ 特征值：" + "  ".join(f"{v:.10g}" for v in show))


if __name__ == "__main__":
    main()
