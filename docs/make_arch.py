"""生成作品介绍中的架构图：python docs/make_arch.py → docs/arch.png"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

for fam in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110)
ax.set_ylim(0, 46)
ax.axis("off")

BLUE, GREEN, ORANGE, GRAY = "#dbe8fb", "#dff3e4", "#fde9d2", "#eceff3"


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#5b6b7f", lw=1))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=7.6, linespacing=1.45)


def arrow(x0, y0, x1, y1, text="", color="#5b6b7f", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, text, ha="center", fontsize=7.2, color=color)


box(1, 24, 19, 19, "输入与预处理", "npz / npy / mtx / mat\n上/下三角自动恢复\n厄密性、有限值校验\n复厄密 → 实嵌入\n固定稀疏结构", GRAY)
box(24, 24, 19, 19, "查询与目标重定位", "nearest：σ 最近 k 个\nabove / below：σ 一侧 k 个\ninterval：区间内全部\n惯性括区避开零空间\n谱切片（并发二分）", BLUE)
box(47, 24, 21, 19, "位移-求逆块 Krylov-Schur", "MUMPS 对称不定 LDL^T 分解\n近奇异检测与方向扰动\n块 8 · 厚重启 · B-内积\n锁定收缩补解\nRayleigh-Ritz 精化", BLUE)
box(72, 24, 17, 19, "惯性完备性证书", "ν(x) = #{λ < x}\n窗口计数 = 已求得？\n独立于 Krylov 过程\n不吻合 → 自动补解", GREEN)
box(93, 24, 16, 19, "输出与报告", "特征值 / 特征向量\nreport.json：证书、\n残差、诊断、分解统计\nemslice check 复核", GRAY)
arrow(20.6, 33.5, 23.4, 33.5)
arrow(43.6, 33.5, 46.4, 33.5)
arrow(68.6, 33.5, 71.4, 33.5)
arrow(89.6, 33.5, 92.4, 33.5, "通过")
arrow(80.5, 23.2, 57.5, 23.2, color="#c0392b", rad=-0.35)
ax.text(69, 16.2, "证书不吻合：缺口 = 计数 − 已得 → 收缩已得向量继续求解", ha="center", fontsize=7.4, color="#c0392b")

box(24, 1.0, 44, 12.0, "并行与资源", "常驻进程池并发惯性计数 / 切片 · 符号分析复用 · 计数分解丢弃因子（内存减半）\n线程预算按并发分配 · 按可用内存自动降并发 · 无 PETSc 时 SciPy 降级", ORANGE)
box(72, 1.0, 37, 12.0, "工程场景（14 个十万阶算例）", "Yee 腔 · 介质谐振器 · 光子晶体 · 公开矩阵\n有限元 Kx=k₀²Mx：滤波腔、铁氧体腔（复厄密）", ORANGE)
box(1, 1.0, 19, 12.0, "开源底座", "PETSc · MUMPS · METIS\nNumPy / SciPy\nApache-2.0", GRAY)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch.png")
fig.savefig(out, dpi=170)
print(out)
