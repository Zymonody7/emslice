# emslice：面向电磁谐振与微波器件仿真的非正定厄密稀疏特征值求解器

> 2026 上海开源软件应用创新大赛 · AI+工业软件赛道 · 中国商飞赛题
> 「面向电磁谐振与微波器件仿真的高效、稳定求解方案」

求解十万阶及以上**非正定厄密稀疏矩阵** `A x = λ x`（可选广义 `A x = λ B x`）中目标频段的特征对。
核心目标：**目标频段的谐振模式一个不漏**，并给出可审计的证据。

## 核心技术

| 能力 | 做法 | 对应评审点 |
|---|---|---|
| **完备性证书** | 基于 Sylvester 惯性定理：LDLᵀ(A−xI) 的负主元数 = #{λ < x}。求得特征值后在谱间隙处做两次惯性计数，计数吻合即**证明窗口内无遗漏**；不吻合则自动收缩已得向量继续求解 | 正确性、重特征值、目标频段不丢失 |
| **位移-求逆块 Krylov-Schur** | 自研厚重启块 Lanczos（B-内积自伴），块尺寸 8，完全再正交化；停滞时返回已收敛子集，由证书驱动补齐 | 核心算法 |
| **近奇异位移自适应** | 随机探针估计 ‖S⁻¹‖‖S‖ + MUMPS 零主元检测；σ 贴近谐振点时按谱尺度逐级扰动，并按求解方向选择扰动符号 | 近奇异位移鲁棒性 |
| **低频击穿 / 零空间** | `above` 模式用惯性括区把位移重定位到目标簇中部，避免被数万重梯度零空间主导；σ 落在零空间上时报告其维数 | 零附近密集谱 |
| **复厄密矩阵** | 实嵌入 `[[Hr,−Hi],[Hi,Hr]]`：MUMPS 复数 LDLᵀ 只支持复对称，实嵌入同时给出可靠惯性与求解，代价与复数 LU 相当 | 光子晶体/铁氧体等复厄密场景 |
| **谱切片** | 区间模式按惯性二分成若干切片，每片独立位移求解，切片数量与内存可控 | 大区间、并行 |
| **并行与资源** | 惯性计数/括区/切片由常驻进程池并发；同一稀疏结构复用符号分析；计数分解丢弃因子（内存减半）；按可用内存自动降并发；线程预算按并发分配 | 性能与资源效率 |
| **安全设计** | 禁用 pickle 反序列化；日志/报告只含形状、nnz、SHA-256；`--secure` 输出 0600 且不记录路径；无硬编码、不读参考答案 | 安全要求 |

## 安装

```bash
conda env create -f environment.yml      # conda-forge：PETSc(real) + MUMPS + SLEPc
conda activate emslice
pip install -e .
emslice info                              # 查看可用后端
```

无 PETSc 时可只用 `pip install -e .`：自动降级为 SciPy SuperLU 后端，三种查询模式都能求解，但不提供惯性证书（σ 单侧/区间查询改为“扩大最近集合再筛选”，报告中会注明）。

## 快速开始

```bash
# 生成测试矩阵（Yee 网格旋度-旋度算子；--upper 仅存上三角）
emslice gen cube  --N 20 --out data/small/cube20.npz
emslice gen bloch --N 16 --out data/small/bloch16.npz --upper

# 距目标位移最近的 k 个
emslice solve data/small/cube20.npz --sigma 500 -k 20 --out result
# σ 以上最小的 k 个（低频模式，自动跳过零空间）
emslice solve data/small/cube20.npz --sigma 1e-3 --which above -k 20
# 区间内全部
emslice solve data/small/cube20.npz --interval 100 300
# 用配置文件
emslice solve --config configs/default.yaml

# 有限元广义问题 Kx=λMx（需 scikit-fem；fem-dr/fem-ferrite 另需 gmsh）：介质谐振器滤波腔，约 10 万阶
emslice gen fem-dr --out data/fem/dr.npz            # 写出 dr.npz (K)、dr_B.npz (M)、dr.json（网格信息）
emslice solve data/fem/dr.npz --B data/fem/dr_B.npz --sigma 1e-6 --which above -k 10 --out result/dr

# 独立复核残差与正交性
emslice check data/small/cube20.npz result
```

输入：`.npz`（scipy.sparse.save_npz，或含 data/indices/indptr、row/col/data 的 numpy npz）、`.npy`、`.mtx`、`.mat`；
仅存上/下三角时自动恢复（`--triangle` 可强制）。
输出：`eigenvalues.npy`、`eigenvectors.npy`（或 `--format npz/mat`）与 `report.json`（残差、证书、分解统计、诊断）。

## 测试矩阵与真值

`emslice/matgen.py` 生成 Yee 网格旋度-旋度算子 `A = Dε^{-1/2} Cᴴ C Dε^{-1/2}`：

* **PEC 腔**：实对称半正定，梯度零空间维数 (N−1)³（低频击穿）；均匀介质时离散特征值有**解析解**；
* **立方腔**：重数 3/6/12 的重特征值；
* **Bloch 周期**：复厄密（光子晶体带隙计算）；
* **介质谐振器**：非均匀 ε（微波介质滤波器结构）。

解析解与稠密求解对比误差 ~1e-15（见 `tests/`）。

`emslice/femgen.py` 生成**有限元**（一阶 Nédélec 棱边元）广义问题 `K x = k₀² M x`，即电磁 FEM 软件实际产生的矩阵形态
（非结构四面体网格、曲面介质、高介电对比度、梯度零空间维数 = 内部节点数），长度单位 mm，频率 f = c√λ/2π：

* **fem-cavity**：空矩形腔，连续问题有解析解；
* **fem-dr**：介质谐振器滤波腔（ε_r=38 陶瓷圆柱 + PTFE 支撑，30×30×24 mm 铜腔），TE01δ 模约 3.36 GHz；
* **fem-ferrite**：铁氧体加载腔，z 向偏置的旋磁张量 ⇒ **复厄密** K，可见反向旋转模的劈裂（环行器原理）。

网格由 gmsh 作为独立外部程序生成（.geo → .msh），生成过程确定，矩阵逐字节可复现。

## 目录

```
emslice/   matgen.py Yee 测试矩阵 | femgen.py 有限元测试矩阵 | io.py 读写与校验 | factor.py 分解与惯性 | krylov.py 块 Krylov-Schur | solver.py 主流程 | parallel.py 进程池 | cli.py
bench/     cases.py 基准算例 | run.py 运行器（子进程隔离内存）| baselines.py SciPy/SLEPc 对比 | report.py 汇总 | get_public.sh 公开矩阵
tests/     单元与场景测试（pytest）
.github/   CI（完整后端 + 降级模式）、Issue/PR 模板
configs/   默认配置
docs/      赛题任务书、方案说明文档、演示视频脚本
scripts/   reproduce.sh 一键复现
```

## 当前进展（2026-10-10）

- [x] 求解器全部核心能力；15 项测试通过（CI：Linux/macOS 完整后端 + Python 3.9/3.12 降级模式）
- [x] 10 万阶基准：10 个 Yee/公开矩阵算例 + 4 个有限元广义问题算例（含复厄密铁氧体），全部证书通过
- [x] 与 SLEPc / SciPy 对比（`results/full/report.md`）
- [x] 方案说明文档初稿（`docs/方案说明文档.md`）、演示视频脚本、Dockerfile、一键复现脚本
- [x] 方案说明文档 PDF
- [x] 有限元（Nédélec）广义问题场景（`results/fem/report.md`）
- [x] Linux 验证：Docker（Linux aarch64，conda-forge 全量环境）18/18 测试通过；华为鲲鹏（HiSilicon，640 核）+ openEuler 22.03 上 17 项通过（1 项因未装 gmsh 跳过）
- [ ] 多核扩展性实测、录制演示视频（见 [ROADMAP.md](ROADMAP.md)）

## 参与贡献

[CONTRIBUTING.md](CONTRIBUTING.md) · [ROADMAP.md](ROADMAP.md) · [CHANGELOG.md](CHANGELOG.md) · [SECURITY.md](SECURITY.md) · [行为准则](CODE_OF_CONDUCT.md) · [维护者](MAINTAINERS.md)

## 许可证

Apache-2.0
