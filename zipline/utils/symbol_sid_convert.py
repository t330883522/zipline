from collections import Iterable

def to_sid(x):
    """输入股票代码转换为sid"""
    if isinstance(x, str):
        return int(x)
    elif isinstance(x, Iterable):
        return [int(n) for n in x]
    return int(x)

def to_symbol(x):
    """输入sid转换为股票代码"""
    if isinstance(x, int):
        return str(x).zfill(6)
    elif isinstance(x, Iterable):
        return [str(n).zfill(6) for n in x]
    return str(x).zfill(6)