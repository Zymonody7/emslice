# 基准测试结果

数据：`results/kunpeng8-final/std/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离，含惯性计数子进程）。

| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |
|---|---|---|---|---|---|---|---|---|---|---|
| cube_mid_band | mid_band+degenerate | 101376 | emslice（含证书） | 11.5 | 968 | 20 | ✅ | 3.99e-11 | 3.39e-16 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | emslice（无证书快速模式） | 6.2 | 986 | 20 | 无 | 3.70e-11 | 3.91e-16 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | SLEPc Krylov-Schur + MUMPS | 7.1 | 835 | 20 | 无 | 2.59e-03 | 4.57e-08 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | SciPy eigsh + SuperLU | 733.0 | 5156 | 20 | 无 | 1.30e-07 | 8.82e-11 | 0 |
| cube_low_freq | near_zero | 101376 | emslice（含证书） | 11.2 | 1142 | 20 | ✅ | 4.47e-14 | 8.56e-18 | 0 |
| cube_low_freq | near_zero | 101376 | emslice（无证书快速模式） | 11.8 | 1114 | 20 | ✅ | 1.03e-13 | 8.56e-18 | 0 |
| cube_low_freq | near_zero | 101376 | SLEPc Krylov-Schur + MUMPS | 58.1 | 949 | 0 | 无 | — | — | 20 |
| cube_low_freq | near_zero | 101376 | SciPy eigsh + SuperLU | 677.8 | 3898 | 20 | 无 | 2.32e-07 | 4.26e-09 | 0 |
| cube_near_singular | near_singular | 101376 | emslice（含证书） | 8.5 | 940 | 12 | ✅ | 3.39e-13 | 9.79e-18 | 0 |
| cube_near_singular | near_singular | 101376 | emslice（无证书快速模式） | 5.1 | 927 | 12 | 无 | 6.84e-12 | 1.14e-17 | 0 |
| cube_near_singular | near_singular | 101376 | SLEPc Krylov-Schur + MUMPS | 7.2 | 808 | 12 | 无 | 1.23e-03 | 4.56e-06 | 0 |
| cube_near_singular | near_singular | 101376 | SciPy eigsh + SuperLU | 583.2 | 3898 | 12 | 无 | 1.05e-02 | 5.95e-04 | 0 |
| cube_null_cluster | near_zero | 101376 | emslice（含证书） | 6.3 | 877 | 16 | ✅ | 3.85e-16 | 5.23e-18 | 0 |
| cube_null_cluster | near_zero | 101376 | emslice（无证书快速模式） | 4.8 | 902 | 16 | 无 | 3.13e-16 | 3.47e-18 | 0 |
| cube_null_cluster | near_zero | 101376 | SLEPc Krylov-Schur + MUMPS | 6.2 | 810 | 16 | 无 | 3.25e-15 | 1.50e-20 | 0 |
| cube_null_cluster | near_zero | 101376 | SciPy eigsh + SuperLU | 662.2 | 3853 | 16 | 无 | 2.26e-16 | 1.09e-26 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | emslice（含证书） | 8.8 | 1033 | 30 | ✅ | 5.79e-12 | 6.43e-16 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | emslice（无证书快速模式） | 6.2 | 985 | 30 | 无 | 5.79e-12 | 3.09e-16 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | SLEPc Krylov-Schur + MUMPS | 6.8 | 863 | 30 | 无 | 3.14e-05 | 6.49e-11 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | SciPy eigsh + SuperLU | 818.3 | 5209 | 30 | 无 | 1.01e-12 | 2.11e-15 | 0 |
| dielectric_puck | dielectric | 101376 | emslice（含证书） | 9.9 | 1097 | 20 | ✅ | 2.78e-13 | — | — |
| dielectric_puck | dielectric | 101376 | emslice（无证书快速模式） | 9.6 | 1133 | 20 | ✅ | 5.59e-13 | 1.80e-15 | — |
| dielectric_puck | dielectric | 101376 | SLEPc Krylov-Schur + MUMPS | 15.9 | 977 | 20 | 无 | 3.80e-13 | 1.10e-12 | — |
| dielectric_puck | dielectric | 101376 | SciPy eigsh + SuperLU | 641.3 | 3896 | 20 | 无 | 3.37e-15 | 6.59e-15 | — |
| bloch_complex_hermitian | complex_herm | 107811 | emslice（含证书） | 131.2 | 6882 | 20 | ✅ | 1.08e-11 | 1.88e-17 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | emslice（无证书快速模式） | 62.2 | 5657 | 20 | 无 | 1.61e-11 | 2.28e-17 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | SLEPc Krylov-Schur + MUMPS | 91.8 | 5779 | 20 | 无 | 4.92e-03 | 4.30e-04 | 0 |
| photonic_crystal | complex_herm+dielectric | 107811 | emslice（含证书） | 157.9 | 5620 | 16 | ✅ | 1.26e-13 | — | — |
| photonic_crystal | complex_herm+dielectric | 107811 | emslice（无证书快速模式） | 76.1 | 5614 | 16 | 无 | 1.07e-13 | 1.93e-15 | — |
| photonic_crystal | complex_herm+dielectric | 107811 | SLEPc Krylov-Schur + MUMPS | 144.0 | 5581 | 16 | 无 | 2.16e-13 | 4.49e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | emslice（含证书） | 25.4 | 1119 | 30 | ✅ | 3.98e-17 | — | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | emslice（无证书快速模式） | 26.1 | 1100 | 30 | ✅ | 3.32e-17 | 1.63e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | SLEPc Krylov-Schur + MUMPS | 59.3 | 1038 | 30 | 无 | 4.22e-16 | 2.61e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | SciPy eigsh + SuperLU | 1231.7 | 4974 | 30 | 无 | 4.45e-20 | 3.42e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | emslice（含证书） | 9.9 | 926 | 20 | ✅ | 3.50e-14 | — | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | emslice（无证书快速模式） | 6.8 | 930 | 20 | 无 | 3.51e-14 | 1.66e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | SLEPc Krylov-Schur + MUMPS | 10.3 | 779 | 20 | 无 | 1.96e-15 | 2.99e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | SciPy eigsh + SuperLU | 1144.2 | 4994 | 20 | 无 | 2.92e-22 | 2.99e-15 | — |
