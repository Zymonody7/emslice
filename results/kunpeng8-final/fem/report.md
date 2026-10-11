# 基准测试结果

数据：`results/kunpeng8-final/fem/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离，含惯性计数子进程）。

| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |
|---|---|---|---|---|---|---|---|---|---|---|
| fem_cavity_low | fem_generalized+near_zero | 100827 | emslice（含证书） | 12.4 | 998 | 20 | ✅ | 3.23e-13 | — | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | emslice（无证书快速模式） | 11.9 | 996 | 20 | ✅ | 1.91e-13 | 2.13e-15 | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | SLEPc Krylov-Schur + MUMPS | 103.0 | 815 | 0 | 无 | — | — | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | SciPy eigsh + SuperLU | 756.2 | 4303 | 20 | 无 | 2.22e-06 | 3.63e-06 | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | emslice（含证书） | 12.8 | 1049 | 10 | ✅ | 1.22e-12 | — | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | emslice（无证书快速模式） | 13.1 | 1008 | 10 | ✅ | 1.21e-12 | 1.66e-15 | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | SLEPc Krylov-Schur + MUMPS | 73.4 | 844 | 0 | 无 | — | — | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | SciPy eigsh + SuperLU | 1213.1 | 5001 | 10 | 无 | 1.02e-08 | 1.17e-07 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | emslice（含证书） | 12.5 | 967 | 20 | ✅ | 1.71e-14 | — | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | emslice（无证书快速模式） | 8.8 | 948 | 20 | 无 | 3.59e-14 | 2.73e-15 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | SLEPc Krylov-Schur + MUMPS | 12.4 | 821 | 20 | 无 | 1.05e-13 | 7.25e-13 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | SciPy eigsh + SuperLU | 1156.9 | 5001 | 20 | 无 | 9.79e-16 | 2.00e-14 | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | emslice（含证书） | 46.3 | 2568 | 8 | ✅ | 7.35e-13 | — | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | emslice（无证书快速模式） | 22.7 | 2569 | 8 | 无 | 7.34e-13 | 9.19e-16 | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | SLEPc Krylov-Schur + MUMPS | 41.2 | 2727 | 8 | 无 | 6.51e-15 | 1.38e-15 | — |
