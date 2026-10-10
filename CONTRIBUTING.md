# 参与贡献

欢迎提交 Issue 与 Pull Request。emslice 的首要目标是**目标频段的谐振模式一个不漏**，因此任何改动都必须保住惯性证书与残差门槛。

## 开发环境

```bash
conda env create -f environment.yml && conda activate emslice
pip install --no-deps -e .
emslice info                      # 确认 petsc 后端可用
OMP_NUM_THREADS=2 python -m pytest -q
```

没有 PETSc 时可只装 `pip install -e ".[test]"`：求解走 SciPy SuperLU 降级路径，需惯性的测试会自动跳过。

## 提交前检查

1. `python -m pytest -q` 全部通过（CI 会在 Linux/macOS 完整后端与 Python 3.9/3.12 降级模式下各跑一遍）。
2. 改动求解路径（`solver.py`、`krylov.py`、`factor.py`）时，在 `data/bench` 上重跑基准并附结果：
   ```bash
   python bench/cases.py --out data/bench --N 33
   python bench/run.py --solver emslice --out results/dev-<分支名>
   python bench/report.py results/dev-<分支名>
   ```
   PR 描述里写明：证书是否全部通过、最大后向误差、各算例耗时相对 `results/full` 的变化。
3. 新增场景（新的矩阵族、故障类型）请同时加测试，参考 `tests/test_emslice.py` 中用解析解或稠密求解作对照的写法。

## 约定

* 代码风格：与周边代码一致；注释写“为什么”，数值上的取舍写清实测依据（参见 `Solver.PRE_OFFSET` 的注释）。
* 不在仓库中提交矩阵数据或求解结果（`data/`、`result*/` 已忽略）；基准矩阵由脚本生成或下载。
* 禁止将稀疏矩阵稠密化（测试中的小规模对照除外），禁止硬编码答案或读取参考结果。
* 提交信息：一行摘要 + 必要的正文，说明动机与验证方式。

## 报告问题

* 求解结果疑似错误：请附 `report.json`（只含形状、nnz、SHA-256 摘要，不含矩阵内容）和运行命令。
* 安全问题请勿公开提 Issue，见 [SECURITY.md](SECURITY.md)。

## 决策方式

维护者见 [MAINTAINERS.md](MAINTAINERS.md)。普通改动需一名维护者审核后合并；影响证书正确性或公开接口（CLI 参数、`report.json` 字段、Python API）的改动需先开 Issue 讨论。路线图见 [ROADMAP.md](ROADMAP.md)。

参与本项目即表示同意遵守 [行为准则](CODE_OF_CONDUCT.md)。
