# 基准测试结果

数据：`results/kunpeng8-final/fem/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离，含惯性计数子进程）。

| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |
|---|---|---|---|---|---|---|---|---|---|---|
| fem_cavity_low | fem_generalized+near_zero | 100827 | emslice（含证书） | 9.5 | 1057 | 20 | ✅ | 6.96e-14 | — | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | emslice（无证书快速模式） | 8.9 | 1042 | 20 | ✅ | 4.52e-13 | 1.16e-15 | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | SLEPc Krylov-Schur + MUMPS | 96.4 | 1090 | 0 | 无 | — | — | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | SciPy eigsh + SuperLU | 756.2 | 4303 | 20 | 无 | 2.22e-06 | 3.63e-06 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | emslice（含证书） | 9.9 | 974 | 20 | ✅ | 7.24e-14 | — | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | emslice（无证书快速模式） | 7.6 | 972 | 20 | 无 | 1.80e-13 | 3.64e-15 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | SLEPc Krylov-Schur + MUMPS | 10.8 | 1147 | 20 | 无 | 4.53e-14 | 1.37e-12 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | SciPy eigsh + SuperLU | 1156.9 | 5001 | 20 | 无 | 9.79e-16 | 2.08e-14 | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | emslice（含证书） | 10.5 | 1116 | 10 | ✅ | 7.24e-13 | — | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | emslice（无证书快速模式） | 10.9 | 1063 | 10 | ✅ | 6.91e-13 | 1.53e-15 | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | SLEPc Krylov-Schur + MUMPS | 67.4 | 1170 | 0 | 无 | — | — | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | SciPy eigsh + SuperLU | 1213.1 | 5001 | 10 | 无 | 1.02e-08 | 1.17e-07 | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | emslice（含证书） | 28.8 | 4581 | 8 | ✅ | 7.35e-13 | — | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | emslice（无证书快速模式） | 17.3 | 2836 | 8 | 无 | 7.35e-13 | 1.99e-15 | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | SLEPc Krylov-Schur + MUMPS | 34.9 | 2784 | 8 | 无 | 9.40e-15 | 2.15e-15 | — |
