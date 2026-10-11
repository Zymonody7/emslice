# 更新日志

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循语义化版本。

## [未发布]

### 新增
- 有限元算例生成器 `emslice/femgen.py`（Nédélec 棱边元，广义问题 Kx=λMx）：空腔（解析解）、介质谐振器滤波腔、
  铁氧体加载腔（复厄密）；`emslice gen fem-*`；4 个十万阶有限元基准算例，SLEPc/SciPy 基线支持广义问题。
- 多核扩展性实测（华为鲲鹏 + openEuler，单 NUMA 节点 1–32 线程）：复厄密算例加速 15.6 倍；`bench/scaling_report.py`。
- 演示录屏脚本 `scripts/demo.sh` 与报告摘要 `scripts/show_report.py`；作品介绍（按大赛四个评审维度组织）与架构图。
- 降级模式：后端不提供惯性（仅 SciPy SuperLU）时三种查询模式均可求解，报告中注明未出证书。
- CI：Linux/macOS 完整后端 + Python 3.9/3.12 降级模式；端到端冒烟（生成 → 求解 → 复核 → 证书）。
- 贡献指南、安全策略、行为准则、路线图。

### 修复
- 无 PETSc 时默认 `emslice solve` 因要求证书而报错退出。

## [0.1.0] - 2026-10-10

### 新增
- 位移-求逆厚重启块 Krylov-Schur（B-内积自伴，DGKS 再正交化，锁定/收缩）。
- Sylvester 惯性完备性证书：nearest 内窗口证书、above/below 惯性括区、interval 谱切片。
- 近奇异位移检测（探针增长因子 + MUMPS 零主元）与方向性自适应扰动。
- 低频零空间：惯性括区目标重定位；σ 落在零空间时报告其维数。
- 复厄密矩阵实嵌入；广义问题 $Ax=\lambda Bx$。
- 进程池并行惯性计数与切片；符号分析复用；计数分解丢弃因子。
- 基准：10 个十万阶算例，对比 SLEPc / SciPy；方案说明文档、演示视频脚本、Dockerfile。
