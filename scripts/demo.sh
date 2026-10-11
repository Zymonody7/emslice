#!/usr/bin/env bash
# 演示视频录屏脚本：按 docs/演示视频脚本.md 的分段依次运行，每段先显示字幕再执行命令。
#
#   conda activate emslice
#   bash scripts/demo.sh            # 每段结束按回车继续（适合边讲边录）
#   DEMO_AUTO=3 bash scripts/demo.sh   # 每段自动停 3 秒（适合配后期旁白）
#   DEMO_ONLY="2 3" bash scripts/demo.sh  # 只录指定分段
#
# 复厄密（光子晶体）分段约需 4 分钟，默认展示基准中已跑好的结果；DEMO_SLOW=1 时现场求解。
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
D=data/demo
R=result/demo
mkdir -p "$D" "$R"

C0=$'\033[0m'; CT=$'\033[1;36m'; CC=$'\033[1;33m'; CN=$'\033[0;37m'

title() { printf '\n%s━━ %s ━━%s\n' "$CT" "$1" "$C0"; }
say()   { printf '%s  %s%s\n' "$CN" "$1" "$C0"; }
run()   { printf '%s$ %s%s\n' "$CC" "$*" "$C0"; "$@"; }
pause() {
  if [[ -n "${DEMO_AUTO:-}" ]]; then sleep "$DEMO_AUTO"; else read -r -p "  ⏎ 继续" _; fi
}
want() { [[ -z "${DEMO_ONLY:-}" ]] || [[ " $DEMO_ONLY " == *" $1 "* ]]; }

if want 0; then
  title "emslice：电磁谐振模式求解，一个不漏，并给出证据"
  say "十万阶非正定厄密稀疏特征值问题 · 惯性证书 · 位移-求逆块 Krylov-Schur"
  pause
fi

if want 1; then
  title "1. 问题与数据"
  run emslice info
  run emslice gen cube --N 33 --out "$D/cube33.npz"
  say "PEC 立方腔 Yee 旋度-旋度算子：大量 6/12 重简并模式 + 3.2 万维梯度零空间（低频击穿）"
  pause
fi

if want 2; then
  title "2. 目标频率恰为 6 重谐振点（近奇异位移 + 重特征值）"
  [[ -f "$D/cube33.npz" ]] || emslice gen cube --N 33 --out "$D/cube33.npz" >/dev/null
  run emslice solve "$D/cube33.npz" --sigma 3306.3592973618242 -k 20 --out "$R/mid"
  run python scripts/show_report.py "$R/mid"
  say "对比（results/full/report.md）：同一算例 SLEPc 常规用法后向误差 0.12、漏 2 个简并模式"
  run grep -E "cube_mid_band .*SLEPc" results/full/report.md
  pause
fi

if want 3; then
  title "3. 低频击穿：σ≈0 以上的前 20 个物理模式"
  [[ -f "$D/cube33.npz" ]] || emslice gen cube --N 33 --out "$D/cube33.npz" >/dev/null
  run emslice solve "$D/cube33.npz" --sigma 1.97e-5 --which above -k 20 --out "$R/low"
  run python scripts/show_report.py "$R/low"
  say "SLEPc 常规做法（求 3k 个最近者再筛选）在同一问题上返回 0 个有效结果"
  run grep -E "cube_low_freq .*SLEPc" results/full/report.md
  pause
fi

if want 4; then
  title "4. 有限元工程场景：介质谐振器滤波腔（广义问题 Kx = k₀²Mx）"
  run emslice gen fem-dr --out "$D/dr.npz"
  say "gmsh 非结构四面体网格 + Nédélec 棱边元；ε_r=38 陶瓷圆柱置于 PTFE 支撑上，30×30×24 mm 铜腔"
  run emslice solve "$D/dr.npz" --B "$D/dr_B.npz" --sigma 1e-6 --which above -k 10 --out "$R/dr"
  run python scripts/show_report.py "$R/dr" --ghz
  say "TE01δ 主模 3.36 GHz，其后为 HEM 简并对——滤波器设计关心的主模与寄生模；结构与模场图见 docs/fem_dr_modes.png"
  pause
fi

if want 5; then
  title "5. 复厄密：光子晶体（Bloch 周期边界）与铁氧体加载腔"
  if [[ -n "${DEMO_SLOW:-}" ]]; then
    run emslice gen bloch --N 33 --out "$D/bloch_rods33.npz"
    run emslice solve "$D/bloch_rods33.npz" --sigma 60 -k 16 --out "$R/pc"
    run python scripts/show_report.py "$R/pc"
  else
    say "（约 4 分钟，展示基准中已跑好的结果；DEMO_SLOW=1 可现场求解）"
    run grep -E "photonic_crystal|bloch_complex" results/full/report.md
  fi
  run grep -E "fem_ferrite_band" results/fem/report.md
  pause
fi

if want 6; then
  title "6. 独立复核、测试与可复现"
  [[ -d "$R/dr" ]] && run emslice check "$D/dr.npz" "$R/dr" --B "$D/dr_B.npz"
  run python -m pytest -q
  say "CI：Linux/macOS 完整后端 + Python 3.9/3.12 降级模式；Docker 一键复现；已在鲲鹏 + openEuler 验证"
  say "结果报告：results/full/report.md、results/fem/report.md；方案说明：docs/方案说明文档.pdf"
  pause
fi

title "emslice：目标频段谐振模式一个不漏，并给出可审计的证据 · Apache-2.0"
