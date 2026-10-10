#!/usr/bin/env bash
# 下载公开电磁仿真矩阵（SuiteSparse Matrix Collection）并转为 npz
#   Um/2cubes_sphere：FEM 电磁问题（两立方体置于球内，Stanford 地球物理），n = 101,492
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/suitesparse/dl
if [ ! -f data/suitesparse/2cubes_sphere.npz ]; then
  curl -sSL -o data/suitesparse/dl/2cubes_sphere.tar.gz \
    https://suitesparse-collection-website.herokuapp.com/MM/Um/2cubes_sphere.tar.gz
  tar -xzf data/suitesparse/dl/2cubes_sphere.tar.gz -C data/suitesparse/dl
  python -c "import scipy.io, scipy.sparse as sp; sp.save_npz('data/suitesparse/2cubes_sphere.npz', scipy.io.mmread('data/suitesparse/dl/2cubes_sphere/2cubes_sphere.mtx').tocsr(), compressed=False)"
fi
echo "data/suitesparse/2cubes_sphere.npz ready"
