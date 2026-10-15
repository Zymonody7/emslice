# emslice：面向电磁谐振与微波器件仿真的非正定厄密稀疏特征值求解器

> 2026 上海开源软件应用创新大赛 · AI+工业软件赛道 · 中国商飞赛题
> 「面向电磁谐振与微波器件仿真的高效、稳定求解方案」
>
> [演示视频](https://zymonody7.github.io/emslice/) · [作品介绍（PDF）](docs/作品介绍.pdf) · [方案说明文档（PDF）](docs/方案说明文档.pdf) · [v0.2.0 发布](https://github.com/Zymonody7/emslice/releases/tag/v0.2.0)

求解十万阶及以上**非正定厄密稀疏矩阵** `A x = λ x`（可选广义 `A x = λ B x`）中目标频段的特征对。
核心目标：**目标频段的谐振模式一个不漏**，并给出可审计的证据。

## 核心技术

| 能力 | 做法 | 对应评审点 |
|---|---|---|
| **完备性证书** | 基于 Sylvester 惯性定理：LDLᵀ(A−xI) 的负主元数 = #{λ < x}。求得特征值后在谱间隙处做两次惯性计数，计数吻合即**证明窗口内无遗漏**；不吻合则自动收缩已得向量继续求解 | 正确性、重特征值、目标频段不丢失 |
| **混合精度证书** | 计数只要惯性：用单精度 LDLᵀ（快 2 倍、内存减半）。单精度惯性是邻近矩阵的精确惯性，证书只对收缩 η 后的窗口下结论；端点放在由已得特征值与 Ritz 值估计出的谱间隙里，无干净间隙时自动用双精度 | 混合精度、性能与资源 |
| **子空间夹角上界** | Davis–Kahan sinΘ ≤ ‖R‖/δ，谱间隙 δ 由证书窗口保证——没有证书时无从得知 | 特征向量子空间夹角 |
| **位移-求逆块 Krylov-Schur** | 自研厚重启块 Lanczos（B-内积自伴），块尺寸 8，完全再正交化；停滞时返回已收敛子集，由证书驱动补齐 | 核心算法 |
| **近奇异位移自适应** | 随机探针估计 ‖S⁻¹‖‖S‖ + MUMPS 零主元检测；σ 贴近谐振点时按谱尺度逐级扰动，并按求解方向选择扰动符号 | 近奇异位移鲁棒性 |
| **低频击穿 / 零空间** | `above` 模式用惯性括区把位移重定位到目标簇中部，避免被数万重梯度零空间主导；σ 落在零空间上时报告其维数 | 零附近密集谱 |
| **复厄密矩阵** | 实嵌入 `[[Hr,−Hi],[Hi,Hr]]`：MUMPS 复数 LDLᵀ 只支持复对称，实嵌入同时给出可靠惯性与求解，代价与复数 LU 相当 | 光子晶体/铁氧体等复厄密场景 |
| **谱切片** | 区间模式按惯性二分成若干切片，每片独立位移求解，切片数量与内存可控 | 大区间、并行 |
| **并行与资源** | 线程预算 = 可用物理核；符号分析（不随线程扩展）每个 MUMPS 槽只做一次，实矩阵用 SCOTCH 排序；核数 ≥16 时计数/括区/切片由常驻进程池并发，证书计数在 Krylov 中途推测式提前提交；计数分解丢弃因子；按可用内存自动降并发 | 性能与资源效率 |
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
# 惯性计数默认用单精度 LDLᵀ（η 裕量证书，必要时自动双精度）；--precision double 全程双精度，--no-verify 跳过证书
emslice solve data/small/cube20.npz --sigma 500 -k 20 --precision double

# 有限元广义问题 Kx=λMx（需 scikit-fem；fem-dr/fem-ferrite 另需 gmsh）：介质谐振器滤波腔，约 10 万阶
emslice gen fem-dr --out data/fem/dr.npz            # 写出 dr.npz (K)、dr_B.npz (M)、dr.json（网格信息）
emslice solve data/fem/dr.npz --B data/fem/dr_B.npz --sigma 1e-6 --which above -k 10 --out result/dr

# 独立复核残差与正交性
emslice check data/small/cube20.npz result
```

输入：`.npz`（scipy.sparse.save_npz，或含 data/indices/indptr、row/col/data 的 numpy npz）、`.npy`、`.mtx`、`.mat`；
仅存上/下三角时自动恢复（`--triangle` 可强制）。
输出：`eigenvalues.npy`、`eigenvectors.npy`（或 `--format npz/mat`）与 `report.json`（残差、证书、特征向量子空间夹角上界、分解统计、诊断）。

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

## 使用的接口与权限

按任务书“实现约束”要求，声明本项目实际使用的接口与所需权限：

| 类别 | 内容 |
|---|---|
| 运行身份 | 普通用户即可，不需要 root / 管理员权限，不安装系统服务 |
| 网络 | 求解、复核与测试**不访问网络**，不调用任何外部或闭源服务；只有安装依赖（conda/pip）与 `bench/get_public.sh` 下载 SuiteSparse 公开矩阵时联网 |
| 读文件 | 命令行给出的矩阵文件（含 `--B` 质量阵）与 `--config` 配置文件；numpy 文件一律 `allow_pickle=False` 读取 |
| 写文件 | 只写 `--out` 指定的目录（特征值、特征向量、`report.json`）；外存模式下 MUMPS 因子临时写入 `MUMPS_OOC_TMPDIR`（缺省 `/tmp`），结束即删除；`gen fem-*` 的网格在系统临时目录生成后删除 |
| 进程 | 本机 `spawn` 子进程（惯性计数进程池，矩阵只在本进程树内传递）；`gen fem-*` 调用外部程序 `gmsh`；macOS 上调用 `sysctl` 读取性能核数 |
| 库接口 | petsc4py：PETSc 稀疏矩阵、MUMPS 对称不定 LDLᵀ（`getInertia` 取惯性，ICNTL 7/13/14/22/23/24/31/35 控制排序、零主元检测、外存、内存上限、丢弃因子、BLR）；SciPy：稀疏矩阵与 SuperLU 降级后端；slepc4py 只在 `bench/` 中作对比基线 |
| 环境变量 | `--threads` 设置 `OMP_NUM_THREADS` 等线程数变量；单进程 MPI 传输 Linux 设 `UCX_TLS=self,sm`、macOS 设 `FI_PROVIDER`/`FI_UDP_IFACE`（均不覆盖用户已设的值） |
| 命题方资源 | 未使用命题方的测试环境或基础镜像；测试矩阵由本项目生成器确定性生成，另用 SuiteSparse 公开电磁矩阵 |
| 数据保护 | 日志与报告只记录形状、非零元数、dtype、SHA-256 摘要，不输出矩阵数值；`--secure` 使结果文件为 0600 且报告不记录输入路径 |

## 目录

```
emslice/   matgen.py Yee 测试矩阵 | femgen.py 有限元测试矩阵 | io.py 读写与校验 | factor.py 分解与惯性 | krylov.py 块 Krylov-Schur | solver.py 主流程 | parallel.py 进程池 | cli.py
bench/     cases.py 基准算例 | run.py 运行器（子进程隔离内存）| baselines.py SciPy/SLEPc 对比 | report.py 汇总 | get_public.sh 公开矩阵
tests/     单元与场景测试（pytest）
.github/   CI（完整后端 + 降级模式）、Issue/PR 模板
configs/   默认配置
docs/      方案说明文档、作品介绍、演示视频脚本（赛题任务书以大赛官网为准，不入库）
site/      演示视频页（GitHub Pages，视频取自最新 Release）
scripts/   reproduce.sh 一键复现
```

## 当前进展（2026-10-15）

- [x] 求解器全部核心能力；26 项测试通过（CI：Linux/macOS 完整后端 + Python 3.9/3.12 降级模式）
- [x] 10 万阶基准：10 个 Yee/公开矩阵算例 + 4 个有限元广义问题算例（含复厄密铁氧体），全部证书通过
- [x] 与 SLEPc / SciPy 对比（`results/full/report.md`）
- [x] 方案说明文档初稿（`docs/方案说明文档.md`）、演示视频脚本、Dockerfile、一键复现脚本
- [x] 方案说明文档 PDF
- [x] 有限元（Nédélec）广义问题场景（`results/fem/report.md`）
- [x] Linux 验证：Docker（Linux aarch64，conda-forge 全量环境）18/18 测试通过；华为鲲鹏（HiSilicon，320 物理核 / 640 硬件线程）+ openEuler 22.03 上 25 项通过（1 项因未装 gmsh 跳过）
- [x] 多核扩展性（鲲鹏单 NUMA 节点 1–32 线程）：复厄密算例加速 15.6 倍，24 次运行全部证书通过（`results/scaling-kunpeng/`）
- [x] 演示录屏脚本 `scripts/demo.sh`、作品介绍 `docs/作品介绍.pdf`
- [x] 演示视频（`video/emslice演示视频.mp4`，脚本 `scripts/make_video.py`）
- [x] v0.2 算法：混合精度惯性证书、子空间夹角上界、推测式证书计数、SCOTCH 排序与符号分析复用、精化与正交化稳健性；
      鲲鹏 8 个物理核正式基准（`results/kunpeng8-final/`，与 SLEPc、v0.1 同一时段交替运行；含证书在 SLEPc 结果正确的算例上为其 0.13–0.96 倍耗时）
      与随机压力测试（`bench/fuzz.py`，600 例 0 错误；目录说明见 `results/README.md`）
- [x] 任务书故障场景逐项实测（`bench/scenarios.py`）：低频击穿、重特征值、近奇异、大 k、区间、内存受限（`--mem-limit-gb` 自动外存）
- [x] 稳健性：检出 MUMPS 静默失败；k 跨进梯度零空间时证书端点只向内扰动、窗口跨进超大簇时核对后补取簇内向量；区间内超大近同值簇只报告重数；
      阈值以 ‖A‖/‖B‖ 为特征值单位（广义问题与单位制无关）
- [x] v0.2.0 发布（GitHub Release 附方案说明文档、作品介绍 PDF 与演示视频）

## 参与贡献

[CONTRIBUTING.md](CONTRIBUTING.md) · [ROADMAP.md](ROADMAP.md) · [CHANGELOG.md](CHANGELOG.md) · [SECURITY.md](SECURITY.md) · [行为准则](CODE_OF_CONDUCT.md) · [维护者](MAINTAINERS.md)

## 许可证

Apache-2.0
