# 基准测试结果

数据：`results/kunpeng8-final/std/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离，含惯性计数子进程）。

| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |
|---|---|---|---|---|---|---|---|---|---|---|
| cube_mid_band | mid_band+degenerate | 101376 | emslice（含证书） | 10.3 | 1014 | 20 | ✅ | 2.59e-11 | 4.18e-16 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | emslice（无证书快速模式） | 4.8 | 1024 | 20 | 无 | 4.55e-11 | 2.87e-16 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | SLEPc Krylov-Schur + MUMPS | 11.2 | 828 | 20 | 无 | 3.24e-03 | 7.71e-08 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | SciPy eigsh + SuperLU | 733.0 | 5156 | 20 | 无 | 1.30e-07 | 8.82e-11 | 0 |
| cube_low_freq | near_zero | 101376 | emslice（含证书） | 8.0 | 1135 | 20 | ✅ | 6.36e-14 | 4.89e-18 | 0 |
| cube_low_freq | near_zero | 101376 | emslice（无证书快速模式） | 9.5 | 1164 | 20 | ✅ | 6.45e-13 | 7.34e-18 | 0 |
| cube_low_freq | near_zero | 101376 | SLEPc Krylov-Schur + MUMPS | 93.9 | 950 | 0 | 无 | — | — | 20 |
| cube_low_freq | near_zero | 101376 | SciPy eigsh + SuperLU | 677.8 | 3898 | 20 | 无 | 2.32e-07 | 4.26e-09 | 0 |
| cube_near_singular | near_singular | 101376 | emslice（含证书） | 6.9 | 953 | 12 | ✅ | 5.92e-13 | 1.06e-17 | 0 |
| cube_near_singular | near_singular | 101376 | emslice（无证书快速模式） | 3.8 | 965 | 12 | 无 | 8.15e-13 | 6.52e-18 | 0 |
| cube_near_singular | near_singular | 101376 | SLEPc Krylov-Schur + MUMPS | 11.6 | 811 | 12 | 无 | 7.22e-04 | 2.95e-06 | 0 |
| cube_near_singular | near_singular | 101376 | SciPy eigsh + SuperLU | 583.2 | 3898 | 12 | 无 | 1.05e-02 | 5.95e-04 | 0 |
| cube_null_cluster | near_zero | 101376 | emslice（含证书） | 4.5 | 914 | 16 | ✅ | 2.95e-16 | 3.29e-18 | 0 |
| cube_null_cluster | near_zero | 101376 | emslice（无证书快速模式） | 4.2 | 900 | 16 | 无 | 5.50e-16 | 3.75e-18 | 0 |
| cube_null_cluster | near_zero | 101376 | SLEPc Krylov-Schur + MUMPS | 10.2 | 802 | 16 | 无 | 5.62e-15 | 1.50e-20 | 0 |
| cube_null_cluster | near_zero | 101376 | SciPy eigsh + SuperLU | 662.2 | 3853 | 16 | 无 | 2.26e-16 | 1.09e-26 | 0 |
| dielectric_puck | dielectric | 101376 | emslice（含证书） | 8.0 | 1143 | 20 | ✅ | 4.32e-13 | — | — |
| dielectric_puck | dielectric | 101376 | emslice（无证书快速模式） | 7.6 | 1098 | 20 | ✅ | 5.69e-13 | 1.80e-15 | — |
| dielectric_puck | dielectric | 101376 | SLEPc Krylov-Schur + MUMPS | 59.7 | 954 | 20 | 无 | 3.99e-13 | 5.52e-13 | — |
| dielectric_puck | dielectric | 101376 | SciPy eigsh + SuperLU | 641.3 | 3896 | 20 | 无 | 3.37e-15 | 6.89e-15 | — |
| cavity_upper_storage | mid_band+triangle_input | 100988 | emslice（含证书） | 6.9 | 1100 | 30 | ✅ | 5.79e-12 | 2.57e-16 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | emslice（无证书快速模式） | 5.1 | 1012 | 30 | 无 | 5.79e-12 | 4.89e-16 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | SLEPc Krylov-Schur + MUMPS | 10.6 | 863 | 30 | 无 | 1.93e-05 | 4.51e-11 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | SciPy eigsh + SuperLU | 818.3 | 5209 | 30 | 无 | 1.01e-12 | 2.11e-15 | 0 |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | emslice（含证书） | 6.6 | 936 | 20 | ✅ | 3.51e-14 | — | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | emslice（无证书快速模式） | 4.4 | 944 | 20 | 无 | 3.50e-14 | 1.50e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | SLEPc Krylov-Schur + MUMPS | 9.5 | 999 | 20 | 无 | 1.99e-15 | 2.16e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | SciPy eigsh + SuperLU | 1144.2 | 4994 | 20 | 无 | 2.92e-22 | 1.83e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | emslice（含证书） | 16.2 | 1133 | 30 | ✅ | 3.57e-17 | — | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | emslice（无证书快速模式） | 17.8 | 1132 | 30 | ✅ | 3.47e-17 | 1.63e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | SLEPc Krylov-Schur + MUMPS | 52.0 | 1261 | 30 | 无 | 4.25e-16 | 1.79e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | SciPy eigsh + SuperLU | 1231.7 | 4974 | 30 | 无 | 4.45e-20 | 3.26e-15 | — |
| bloch_complex_hermitian | complex_herm | 107811 | emslice（含证书） | 95.5 | 11628 | 20 | ✅ | 1.21e-11 | 2.28e-17 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | emslice（无证书快速模式） | 41.1 | 5734 | 20 | 无 | 1.56e-11 | 2.12e-17 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | SLEPc Krylov-Schur + MUMPS | 102.6 | 6255 | 20 | 无 | 7.01e-03 | 5.91e-04 | 2 |
| photonic_crystal | complex_herm+dielectric | 107811 | emslice（含证书） | 101.2 | 13086 | 16 | ✅ | 1.47e-13 | — | — |
| photonic_crystal | complex_herm+dielectric | 107811 | emslice（无证书快速模式） | 41.3 | 5692 | 16 | 无 | 1.33e-13 | 3.42e-15 | — |
| photonic_crystal | complex_herm+dielectric | 107811 | SLEPc Krylov-Schur + MUMPS | 105.5 | 6057 | 16 | 无 | 1.83e-13 | 2.57e-15 | — |
