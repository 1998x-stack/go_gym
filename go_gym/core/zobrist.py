from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

# Cache per board size: (table[size,size,2], side_key)
_ZCACHE: Dict[int, Tuple[np.ndarray, int]] = {}

def _init(size: int) -> Tuple[np.ndarray, int]:
    if size in _ZCACHE:
        return _ZCACHE[size]
    rng = np.random.default_rng(20240917)
    table = rng.integers(1, (1 << 63) - 1, size=(size, size, 2), dtype=np.int64)
    side_key = int(rng.integers(1, (1 << 63) - 1, dtype=np.int64))
    _ZCACHE[size] = (table, side_key)
    return table, side_key

def hash_grid(grid: np.ndarray, to_play: int) -> int:
    """64-bit Zobrist hash for (grid, to_play). 0=empty, 1=black, 2=white."""
    size = grid.shape[0]
    table, side_key = _init(size)
    h = np.int64(0)
    br, bc = np.where(grid == 1)
    if br.size:
        h ^= np.bitwise_xor.reduce(table[br, bc, 0].astype(np.int64))
    wr, wc = np.where(grid == 2)
    if wr.size:
        h ^= np.bitwise_xor.reduce(table[wr, wc, 1].astype(np.int64))
    if int(to_play) == 2:  # include side-to-move
        h ^= np.int64(side_key)
    return int(h)
