from __future__ import annotations
from typing import Optional
from . import __package__ as _pkg  # silence linters
from ..core.board import Board, EMPTY, BLACK, WHITE


def render_ascii(board: Board, last_move: Optional[tuple] = None) -> str:
    """简单 ASCII 渲染。last_move 用 '()' 标注。"""
    size = board.size
    rows = []
    header = "   " + " ".join(f"{c:2d}" for c in range(size))
    rows.append(header)
    for r in range(size):
        line = [f"{r:2d} "]
        for c in range(size):
            v = int(board.grid[r, c])
            if v == EMPTY:
                ch = "."
            elif v == BLACK:
                ch = "X"
            else:
                ch = "O"
            if last_move == (r, c):
                ch = f"({ch})"
            else:
                ch = f" {ch} "
            line.append(ch)
        rows.append("".join(line))
    return "\n".join(rows)
