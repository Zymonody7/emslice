# 基准测试结果

数据：`results/full/results.jsonl`；每个算例在独立子进程中运行（峰值内存隔离）。

| 算例 | 场景 | n | 求解器 | 耗时 (s) | 峰值内存 (MB) | 返回 | 证书 | 最大后向误差 | 特征值误差（相对‖A‖） | 漏解 |
|---|---|---|---|---|---|---|---|---|---|---|
| cube_mid_band | mid_band+degenerate | 101376 | emslice（含证书） | 13.9 | 1127 | 20 | ✅ | 3.19e-11 | 3.91e-16 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | emslice（无证书快速模式） | 6.9 | 1155 | 20 | 无 | 3.09e-11 | 4.18e-16 | 0 |
| cube_mid_band | mid_band+degenerate | 101376 | SLEPc Krylov-Schur + MUMPS | 5.3 | 1204 | 20 | 无 | 1.19e-01 | 1.90e-04 | 2 |
| cube_mid_band | mid_band+degenerate | 101376 | SciPy eigsh + SuperLU | 385.9 | 5147 | 20 | 无 | 4.10e-09 | 2.84e-13 | 0 |
| cube_low_freq | near_zero | 101376 | emslice（含证书） | 21.4 | 1182 | 20 | ✅ | 1.06e-13 | 8.16e-18 | 0 |
| cube_low_freq | near_zero | 101376 | emslice（无证书快速模式） | 20.9 | 1168 | 20 | ✅ | 1.21e-13 | 6.93e-18 | 0 |
| cube_low_freq | near_zero | 101376 | SLEPc Krylov-Schur + MUMPS | 62.7 | 1241 | 0 | 无 | — | — | 20 |
| cube_low_freq | near_zero | 101376 | SciPy eigsh + SuperLU | 327.4 | 3974 | 20 | 无 | 3.38e-07 | 8.43e-09 | 0 |
| cube_near_singular | near_singular | 101376 | emslice（含证书） | 13.3 | 1027 | 12 | ✅ | 8.45e-13 | 8.16e-18 | 0 |
| cube_near_singular | near_singular | 101376 | emslice（无证书快速模式） | 5.0 | 1037 | 12 | 无 | 9.33e-13 | 1.39e-17 | 0 |
| cube_near_singular | near_singular | 101376 | SLEPc Krylov-Schur + MUMPS | 6.6 | 1153 | 12 | 无 | 6.71e-03 | 5.81e-04 | 0 |
| cube_near_singular | near_singular | 101376 | SciPy eigsh + SuperLU | 283.6 | 3952 | 12 | 无 | 1.05e-02 | 5.83e-04 | 0 |
| cube_null_cluster | near_zero | 101376 | emslice（含证书） | 12.1 | 1028 | 16 | ✅ | 4.25e-16 | 3.95e-18 | 0 |
| cube_null_cluster | near_zero | 101376 | emslice（无证书快速模式） | 4.5 | 1035 | 16 | 无 | 3.90e-16 | 7.07e-18 | 0 |
| cube_null_cluster | near_zero | 101376 | SLEPc Krylov-Schur + MUMPS | 8.6 | 1158 | 16 | 无 | 2.20e-14 | 1.55e-20 | 0 |
| cube_null_cluster | near_zero | 101376 | SciPy eigsh + SuperLU | 313.8 | 3899 | 16 | 无 | 1.73e-16 | 3.31e-27 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | emslice（含证书） | 14.6 | 1142 | 30 | ✅ | 3.46e-13 | 3.60e-16 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | emslice（无证书快速模式） | 6.1 | 1159 | 30 | 无 | 3.57e-13 | 3.34e-16 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | SLEPc Krylov-Schur + MUMPS | 7.9 | 1277 | 30 | 无 | 1.41e-04 | 3.99e-10 | 0 |
| cavity_upper_storage | mid_band+triangle_input | 100988 | SciPy eigsh + SuperLU | 392.2 | 4304 | 30 | 无 | 5.19e-12 | 3.37e-15 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | emslice（含证书） | 262.1 | 6119 | 20 | ✅ | 3.64e-11 | 2.37e-17 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | emslice（无证书快速模式） | 93.8 | 5625 | 20 | 无 | 3.64e-11 | 1.63e-17 | 0 |
| bloch_complex_hermitian | complex_herm | 107811 | SLEPc Krylov-Schur + MUMPS | 155.5 | 6400 | 20 | 无 | 6.05e-03 | 3.80e-04 | 0 |
| dielectric_puck | dielectric | 101376 | emslice（含证书） | 32.3 | 1142 | 20 | ✅ | 3.23e-12 | — | — |
| dielectric_puck | dielectric | 101376 | emslice（无证书快速模式） | 29.1 | 1101 | 20 | ✅ | 3.58e-12 | 1.40e-15 | — |
| dielectric_puck | dielectric | 101376 | SLEPc Krylov-Schur + MUMPS | 69.0 | 1319 | 20 | 无 | 5.98e-13 | 5.31e-13 | — |
| dielectric_puck | dielectric | 101376 | SciPy eigsh + SuperLU | 295.9 | 3961 | 20 | 无 | 4.19e-15 | 6.19e-15 | — |
| photonic_crystal | complex_herm+dielectric | 107811 | emslice（含证书） | 265.3 | 6109 | 16 | ✅ | 1.97e-13 | — | — |
| photonic_crystal | complex_herm+dielectric | 107811 | emslice（无证书快速模式） | 108.6 | 5824 | 16 | 无 | 2.14e-13 | 3.00e-15 | — |
| photonic_crystal | complex_herm+dielectric | 107811 | SLEPc Krylov-Schur + MUMPS | 177.5 | 6223 | 16 | 无 | 5.47e-13 | 1.71e-14 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | emslice（含证书） | 47.2 | 426 | 30 | ✅ | 1.11e-15 | — | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | emslice（无证书快速模式） | 49.1 | 437 | 30 | ✅ | 1.11e-15 | 2.77e-15 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | SLEPc Krylov-Schur + MUMPS | 50.4 | 1369 | 30 | 无 | 4.24e-16 | 3.88e-12 | — |
| suitesparse_2cubes_low | public_em+wide_dynamic_range | 101492 | SciPy eigsh + SuperLU | 436.4 | 4870 | 30 | 无 | 5.59e-20 | 3.89e-12 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | emslice（含证书） | 12.3 | 1308 | 20 | ✅ | 2.38e-15 | — | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | emslice（无证书快速模式） | 6.6 | 1327 | 20 | 无 | 2.38e-15 | 2.33e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | SLEPc Krylov-Schur + MUMPS | 6.0 | 1075 | 20 | 无 | 1.70e-15 | 3.16e-15 | — |
| suitesparse_2cubes_band | public_em+mid_band | 101492 | SciPy eigsh + SuperLU | 382.6 | 5054 | 20 | 无 | 3.21e-22 | 2.16e-15 | — |
