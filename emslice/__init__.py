"""emslice —— 面向电磁谐振与微波器件仿真的大规模非正定厄密稀疏特征值求解器。"""
__version__ = "0.1.0"


def solve(*args, **kwargs):
    from .solver import solve as _solve

    return _solve(*args, **kwargs)
