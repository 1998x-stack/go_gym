from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

# 0=EMPTY, 1=BLACK, 2=WHITE
_ZCACHE: Dict[int, Tuple[np.ndarray, int]] = {}  # size -> (table[size,size,2], side_key)

def _init(size: int) -> Tuple[np.ndarray, int]:
    if size in _ZCACHE:
        return _ZCACHE[size]
    rng = np.random.default_rng(20240917)  # 固定种子，便于复现
    # 仅为“落子点上的颜色”分配随机数：0不需要，1->idx0, 2->idx1
    table = rng.integers(1, (1 << 63) - 1, size=(size, size, 2), dtype=np.int64)
    side_key = int(rng.integers(1, (1 << 63) - 1, dtype=np.int64))  # 行棋方额外key
    _ZCACHE[size] = (table, side_key)
    return table, side_key

def hash_grid(grid: np.ndarray, to_play: int) -> int:
    """64-bit Zobrist 哈希：根据棋盘与行棋方返回键值。"""
    size = grid.shape[0]
    table, side_key = _init(size)
    h = np.int64(0)
    # 黑子
    br, bc = np.where(grid == 1)
    if br.size:
        h ^= np.bitwise_xor.reduce(table[br, bc, 0].astype(np.int64))
    # 白子
    wr, wc = np.where(grid == 2)
    if wr.size:
        h ^= np.bitwise_xor.reduce(table[wr, wc, 1].astype(np.int64))
    if int(to_play) == 2:  # 用 WHITE 标记行棋方也参与哈希
        h ^= np.int64(side_key)
    return int(h)