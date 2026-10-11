#!/usr/bin/env bash
# 一键复现：生成测试矩阵（Yee + 有限元）→ 下载公开矩阵 → 单元测试 → 基准（emslice / SLEPc / SciPy）→ 汇总报告
# 有限元算例需 scikit-fem 与 gmsh（environment.yml 已包含），缺失时自动跳过
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
python bench/run.py --solver emslice --out results/full --label emslice-noverify --extra '{"verify": false}'
python bench/run.py --solver slepc   --out results/full
python bench/run.py --solver scipy   --out results/full --timeout 1500 \
  --only cube_mid_band cube_low_freq cube_near_singular cube_null_cluster cavity_upper_storage dielectric_puck \
         suitesparse_2cubes_low suitesparse_2cubes_band fem_cavity_low fem_dr_low fem_dr_band
python bench/report.py results/full
echo "报告：results/full/report.md"
