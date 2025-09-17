# go_gym/utils/sgf.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple

Point = Tuple[int, int]

def _sgf_coord(p: Optional[Point], size: int) -> str:
    # SGF pass 用空 []；坐标 a.. 对应 0..25
    if p is None:
        return ""
    r, c = p
    return f"{chr(ord('a')+c)}{chr(ord('a')+r)}"

def write_sgf(moves: Iterable[Tuple[str, Optional[Point]]], size: int = 19, komi: float = 7.5,
              result: Optional[str] = None,
              pb: str = "Black", pw: str = "White") -> str:
    """moves: 序列如 [("B",(r,c)), ("W",None), ...]"""
    header = f"(;GM[1]FF[4]SZ[{size}]KM[{komi}]PB[{pb}]PW[{pw}]"
    if result:
        header += f"RE[{result}]"
    body = []
    for color, p in moves:
        body.append(f";{color}[{_sgf_coord(p, size)}]")
    return header + "".join(body) + ")"
