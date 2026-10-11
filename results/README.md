# 基准结果目录

每个目录含 `results.jsonl`（每行一次运行：耗时、证书、后向误差、峰值内存、分解与 Krylov 日志），由 `bench/run.py` 生成；
汇总表由 `bench/summary.py`、分场景表由 `bench/scenarios.py`、图由 `bench/report.py` 生成。

## 正式数据（华为鲲鹏 + openEuler 22.03，`numactl` 绑定）

鲲鹏服务器为 SMT2（320 物理核 / 640 硬件线程，相邻编号为同一物理核）。下列数据均按**物理核**绑定。

| 目录 | 内容 | 核数 |
|---|---|---|
| `p8/main` | 主表：emslice 含证书 / 快速模式 / SLEPc，按算例交替运行 | 8 物理核（16 硬件线程） |
| `p8/fem3` | 有限元算例在 9c5ac93（η 以 ‖A‖/‖B‖ 为单位）之后重跑，覆盖 `p8/main` 的同名行 | 8 |
| `p8/pair`, `p8/pair-fem3` | v0.1 与 v0.2 同一时段交替运行（“相对 v0.1”列） | 8 |
| `p8/explore`, `p8/explore-fem3` | 大 k（50/100/200/60）、区间、σ 以上 50 个，及 SLEPc 对照 | 8 |
| `p8/memory` | 内存受限：默认 / 全双精度 / 上限 8、5、4 GB / 强制外存 / SLEPc | 8 |
| `p32/main`, `p32/fem3` | 32 物理核（同一 NUMA 节点）：进程池 + 推测式证书计数 | 32 |
| `kunpeng8-scipy` | SciPy 基线（SuperLU 串行，与核数无关） | — |
| `kunpeng8-final` | 上述合并结果（文档第 9 节主表与图） | — |
| `fuzz/` | 随机压力测试日志（600 例与稠密真值比对） | 单线程 |

## 历史数据

`kunpeng8`、`kunpeng8-fem2`、`kunpeng8-v01`、`explore-v2`、`kunpeng32*` 为 2026-10-11 的运行：当时绑定 8 / 32 个 **cpu 编号**，
在 SMT2 机器上实为 4 / 16 个物理核，保留作对照。`full/`、`fem/` 为 v0.1 在 MacBook Air（M2）上的结果，
`scaling-kunpeng/` 为 v0.1 的 1–32 线程扩展性。
