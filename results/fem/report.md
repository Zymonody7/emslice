# 基准测试结果

数据：`results/fem/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离）。

| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |
|---|---|---|---|---|---|---|---|---|---|---|
| fem_cavity_low | fem_generalized+near_zero | 100827 | emslice（含证书） | 15.9 | 1162 | 20 | ✅ | 1.17e-12 | — | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | emslice（无证书快速模式） | 17.2 | 1235 | 20 | ✅ | 4.93e-13 | 1.55e-15 | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | SLEPc Krylov-Schur + MUMPS | 196.4 | 1013 | 0 | 无 | — | — | — |
| fem_cavity_low | fem_generalized+near_zero | 100827 | SciPy eigsh + SuperLU | 377.7 | 3313 | 20 | 无 | 2.16e-06 | 1.04e-05 | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | emslice（含证书） | 18.0 | 1401 | 10 | ✅ | 2.02e-13 | — | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | emslice（无证书快速模式） | 19.3 | 1449 | 10 | ✅ | 1.66e-13 | 1.66e-15 | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | SLEPc Krylov-Schur + MUMPS | 151.0 | 1029 | 0 | 无 | — | — | — |
| fem_dr_low | fem_generalized+dielectric+near_zero | 102396 | SciPy eigsh + SuperLU | 438.4 | 4128 | 10 | 无 | 7.76e-09 | 5.18e-08 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | emslice（含证书） | 15.6 | 1451 | 20 | ✅ | 6.13e-13 | — | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | emslice（无证书快速模式） | 10.7 | 1445 | 20 | 无 | 8.55e-13 | 3.46e-15 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | SLEPc Krylov-Schur + MUMPS | 20.4 | 1064 | 20 | 无 | 1.41e-13 | 3.08e-12 | — |
| fem_dr_band | fem_generalized+dielectric+mid_band | 102396 | SciPy eigsh + SuperLU | 405.5 | 4149 | 20 | 无 | 7.52e-16 | 2.22e-14 | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | emslice（含证书） | 51.0 | 2820 | 8 | ✅ | 2.36e-12 | — | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | emslice（无证书快速模式） | 26.5 | 2975 | 8 | 无 | 2.36e-12 | 2.30e-15 | — |
| fem_ferrite_band | fem_generalized+complex_herm | 103707 | SLEPc Krylov-Schur + MUMPS | 58.7 | 2254 | 8 | 无 | 9.11e-15 | 1.38e-15 | — |
| fem_ferrite_band | fem_generalized+complex_herm | — | SciPy eigsh + SuperLU | 1500.0 | — | 失败 | 无 | — | — | timeout 1500s |
