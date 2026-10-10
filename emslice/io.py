"""矩阵读取 / 结果输出。

安全约定：
  * 一律以 allow_pickle=False 读取 numpy 文件，不执行任何反序列化代码；
  * 日志与报告只记录形状、非零元数、SHA-256 摘要等元数据，不输出矩阵数值；
  * 结果文件可选 0600 权限（--secure）。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


class InputError(ValueError):
    pass


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _from_npz(path: Path, key: str | None) -> sp.spmatrix | np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if "format" in keys and {"data", "shape"} <= keys:  # scipy.sparse.save_npz
            fmt = z["format"].item()
            fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
            shape = tuple(int(s) for s in z["shape"])
            if fmt in ("csr", "csc"):
                cls = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
                return cls((z["data"], z["indices"], z["indptr"]), shape=shape)
            if fmt == "coo":
                r = z["row"] if "row" in keys else z["coords"][0]
                c = z["col"] if "col" in keys else z["coords"][1]
                return sp.coo_matrix((z["data"], (r, c)), shape=shape)
            raise InputError(f"不支持的稀疏格式 {fmt}")
        if {"data", "indices", "indptr"} <= keys:
            shape = tuple(int(s) for s in z["shape"]) if "shape" in keys else None
            return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
        for rk, ck, vk in (("row", "col", "data"), ("i", "j", "v"), ("rows", "cols", "vals")):
            if {rk, ck, vk} <= keys:
                shape = tuple(int(s) for s in z["shape"]) if "shape" in keys else None
                return sp.coo_matrix((z[vk], (z[rk], z[ck])), shape=shape)
        if key is not None:
            return z[key]
        dense = [k for k in z.files if z[k].ndim == 2]
        if len(dense) == 1:
            return z[dense[0]]
        raise InputError(f"无法识别 npz 内容，键为 {sorted(keys)}；请用 --key 指定")


def _from_mat(path: Path, key: str | None):
    m = scipy.io.loadmat(path)
    cands = {k: v for k, v in m.items() if not k.startswith("__")}
    if key is not None:
        return cands[key]
    mats = [k for k, v in cands.items() if sp.issparse(v) or (isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == v.shape[1] and v.shape[0] > 1)]
    if len(mats) != 1:
        raise InputError(f".mat 中候选矩阵 {mats}，请用 --key 指定")
    return cands[mats[0]]


def read_raw(path: str | os.PathLike, key: str | None = None):
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npz":
        return _from_npz(path, key)
    if suf == ".npy":
        return np.load(path, allow_pickle=False)
    if suf in (".mtx", ".mm"):
        return scipy.io.mmread(str(path))
    if suf == ".mat":
        return _from_mat(path, key)
    raise InputError(f"不支持的文件类型 {suf}（支持 .npz .npy .mtx .mat）")


def restore_hermitian(A: sp.spmatrix, triangle: str = "auto") -> tuple[sp.csr_matrix, str]:
    """按需由上/下三角恢复完整厄密矩阵。返回 (A, 实际使用的存储方式)。"""
    A = sp.csr_matrix(A)
    if triangle == "full":
        return A, "full"
    U = sp.triu(A, 1, format="csr")
    L = sp.tril(A, -1, format="csr")
    if triangle == "auto":
        if U.nnz and L.nnz:
            return A, "full"
        if U.nnz == 0 and L.nnz == 0:
            return A, "diagonal"
        triangle = "upper" if U.nnz else "lower"
    if triangle not in ("upper", "lower"):
        raise InputError(f"triangle 取值 auto/full/upper/lower，收到 {triangle!r}")
    T = U if triangle == "upper" else L
    d = A.diagonal()
    full = (T + T.conj().T + sp.diags(d)).tocsr()
    return full, triangle


def check_matrix(A: sp.csr_matrix, herm_tol: float = 1e-10, symmetrize: bool = False) -> dict:
    if A.shape[0] != A.shape[1]:
        raise InputError(f"矩阵非方阵: {A.shape}")
    if not np.all(np.isfinite(A.data)):
        raise InputError("矩阵含 NaN/Inf")
    amax = np.abs(A.data).max() if A.nnz else 0.0
    D = (A - A.conj().T).tocsr()
    asym = (np.abs(D.data).max() if D.nnz else 0.0) / (amax or 1.0)
    info = {"hermitian_defect": float(asym)}
    if asym > herm_tol:
        if not symmetrize:
            raise InputError(f"矩阵非厄密：相对偏差 {asym:.3e} > {herm_tol:.1e}（可用 --symmetrize 强制取 (A+Aᴴ)/2）")
        info["symmetrized"] = True
    return info


def load_matrix(path, key=None, triangle="auto", herm_tol=1e-10, symmetrize=False, dtype=None):
    """读取矩阵 → (CSR 厄密矩阵, 元信息)。"""
    raw = read_raw(path, key)
    if isinstance(raw, np.ndarray):
        if raw.ndim != 2:
            raise InputError(f"数组维度应为 2，收到 {raw.ndim}")
        raw = sp.csr_matrix(raw)  # 稠密输入仅转换为稀疏存储；求解全程不做稠密化
    A, layout = restore_hermitian(raw, triangle)
    A.sum_duplicates()
    A.eliminate_zeros()
    if dtype is not None:
        A = A.astype(dtype)
    elif A.dtype not in (np.float64, np.complex128):
        A = A.astype(np.complex128 if np.iscomplexobj(A.data) else np.float64)
    if np.iscomplexobj(A.data) and A.nnz and not np.any(A.data.imag):
        A = A.real.tocsr()
    info = check_matrix(A, herm_tol, symmetrize)
    if info.get("symmetrized"):
        A = ((A + A.conj().T) * 0.5).tocsr()
    info.update(
        path=str(path),
        sha256=sha256_file(path),
        shape=list(A.shape),
        nnz=int(A.nnz),
        dtype=str(A.dtype),
        storage=layout,
    )
    return A, info


def save_matrix(path, A, fmt=None):
    """保存为 scipy.sparse npz（默认，可由 load_matrix 直接读取）或 .mtx。"""
    path = Path(path)
    if (fmt or path.suffix.lstrip(".")) == "mtx":
        scipy.io.mmwrite(str(path), A)
    else:
        sp.save_npz(path, sp.csr_matrix(A), compressed=False)


def save_results(outdir, evals, evecs, report: dict, fmt: str = "npy", secure: bool = False, save_vectors: bool = True):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    if fmt == "npy":
        np.save(out / "eigenvalues.npy", evals)
        files.append("eigenvalues.npy")
        if save_vectors:
            np.save(out / "eigenvectors.npy", evecs)
            files.append("eigenvectors.npy")
    elif fmt == "npz":
        payload = {"eigenvalues": evals}
        if save_vectors:
            payload["eigenvectors"] = evecs
        np.savez(out / "eigenpairs.npz", **payload)
        files.append("eigenpairs.npz")
    elif fmt == "mat":
        payload = {"eigenvalues": evals}
        if save_vectors:
            payload["eigenvectors"] = evecs
        scipy.io.savemat(out / "eigenpairs.mat", payload)
        files.append("eigenpairs.mat")
    else:
        raise ValueError(f"未知输出格式 {fmt}")
    report = dict(report, output_files=files)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    files.append("report.json")
    if secure:
        for fn in files:
            os.chmod(out / fn, 0o600)
    return [str(out / f) for f in files]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)
