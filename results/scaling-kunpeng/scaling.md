# 并行扩展性 — 华为鲲鹏（HiSilicon，单 NUMA 节点）+ openEuler 22.03，n≈10 万

| 算例 | 求解器 | 线程 | 耗时 (s) | 加速比 | 分解耗时合计 (s) | 证书 | 最大后向误差 |
|---|---|---|---|---|---|---|---|
| cube_mid_band | emslice | 1 | 24.2 | 1.00 | 16.0 | ✅ | 1.6e-11 |
| cube_mid_band | emslice | 2 | 16.1 | 1.50 | 14.8 | ✅ | 1.5e-11 |
| cube_mid_band | emslice | 4 | 11.3 | 2.14 | 9.0 | ✅ | 1.8e-11 |
| cube_mid_band | emslice | 8 | 9.4 | 2.59 | 6.4 | ✅ | 2.4e-11 |
| cube_mid_band | emslice | 16 | 8.3 | 2.92 | 5.0 | ✅ | 2.5e-11 |
| cube_mid_band | emslice | 32 | 8.1 | 3.00 | 4.4 | ✅ | 1.8e-11 |
| cube_mid_band | slepc | 1 | 12.5 | 1.00 | — | — | 6.6e-03 |
| cube_mid_band | slepc | 8 | 5.5 | 2.26 | — | — | 1.0e-02 |
| cube_mid_band | slepc | 32 | 5.5 | 2.29 | — | — | 6.0e-03 |
| cube_low_freq | emslice | 1 | 31.5 | 1.00 | 20.8 | ✅ | 7.6e-14 |
| cube_low_freq | emslice | 2 | 22.9 | 1.38 | 18.0 | ✅ | 5.7e-14 |
| cube_low_freq | emslice | 4 | 19.4 | 1.63 | 21.0 | ✅ | 5.2e-14 |
| cube_low_freq | emslice | 8 | 14.3 | 2.21 | 13.0 | ✅ | 5.9e-14 |
| cube_low_freq | emslice | 16 | 12.8 | 2.47 | 9.1 | ✅ | 4.4e-14 |
| cube_low_freq | emslice | 32 | 12.1 | 2.62 | 7.4 | ✅ | 4.6e-14 |
| cube_low_freq | slepc | 1 | 91.4 | 1.00 | — | — | — |
| cube_low_freq | slepc | 8 | 31.2 | 2.93 | — | — | — |
| cube_low_freq | slepc | 32 | 29.3 | 3.12 | — | — | — |
| bloch_complex_hermitian | emslice | 1 | 525.5 | 1.00 | 486.3 | ✅ | 3.5e-11 |
| bloch_complex_hermitian | emslice | 2 | 273.0 | 1.93 | 389.9 | ✅ | 3.5e-11 |
| bloch_complex_hermitian | emslice | 4 | 142.2 | 3.70 | 205.4 | ✅ | 2.6e-11 |
| bloch_complex_hermitian | emslice | 8 | 81.3 | 6.47 | 113.2 | ✅ | 3.4e-11 |
| bloch_complex_hermitian | emslice | 16 | 48.6 | 10.81 | 61.3 | ✅ | 3.4e-11 |
| bloch_complex_hermitian | emslice | 32 | 33.7 | 15.61 | 37.3 | ✅ | 3.5e-11 |
| bloch_complex_hermitian | slepc | 1 | 222.7 | 1.00 | — | — | 9.2e-03 |
| bloch_complex_hermitian | slepc | 8 | 43.8 | 5.09 | — | — | 6.3e-03 |
| bloch_complex_hermitian | slepc | 32 | 26.2 | 8.49 | — | — | 2.3e-02 |
| fem_cavity_low | emslice | 1 | 22.5 | 1.00 | 13.3 | ✅ | 7.8e-13 |
| fem_cavity_low | emslice | 2 | 18.3 | 1.23 | 11.8 | ✅ | 7.1e-13 |
| fem_cavity_low | emslice | 4 | 16.7 | 1.34 | 14.3 | ✅ | 9.9e-13 |
| fem_cavity_low | emslice | 8 | 13.9 | 1.62 | 10.0 | ✅ | 6.2e-13 |
| fem_cavity_low | emslice | 16 | 12.7 | 1.77 | 7.1 | ✅ | 5.4e-13 |
| fem_cavity_low | emslice | 32 | 12.9 | 1.75 | 6.2 | ✅ | 6.8e-13 |
| fem_cavity_low | slepc | 1 | 70.2 | 1.00 | — | — | — |
| fem_cavity_low | slepc | 8 | 33.9 | 2.07 | — | — | — |
| fem_cavity_low | slepc | 32 | 36.8 | 1.91 | — | — | — |
