#!/usr/bin/env bash
# 一键复现：生成测试矩阵 → 下载公开矩阵 → 单元测试 → 基准（emslice / SLEPc / SciPy）→ 汇总报告
# 用法：conda activate emslice && bash scripts/reproduce.sh [N=33]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-33}"
python -m pytest -q
python bench/cases.py --out data/bench --N "$N"
bash bench/get_public.sh || echo "（跳过公开矩阵：网络不可用）"
python bench/cases.py --out data/bench --N "$N"      # 重新生成 cases.json 以纳入公开矩阵
rm -rf results/full && mkdir -p results/full
python bench/run.py --solver emslice --out results/full
python bench/run.py --solver slepc   --out results/full
python bench/run.py --solver scipy   --out results/full \
  --only cube_mid_band cube_low_freq cube_near_singular cube_null_cluster cavity_upper_storage dielectric_puck \
         suitesparse_2cubes_low suitesparse_2cubes_band
python bench/report.py results/full
echo "报告：results/full/report.md"
