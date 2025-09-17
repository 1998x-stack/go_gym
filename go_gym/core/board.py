# go_gym/core/board.py
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Iterable, Set

Color = int
EMPTY: Color = 0
BLACK: Color = 1
WHITE: Color = 2
Point = Tuple[int, int]  # (row, col)


class Board:
    """围棋棋盘与基本落子/提子/合法性判定（中国规则：禁自杀 + 普通劫禁）。

    网格：int8，0=空, 1=黑, 2=白。
    劫：普通劫禁（simple-ko）对比“上一手落子前”的局面（含下一手行棋方）。
    超级劫：可选，禁止重复任意历史局面（含行棋方）。
    """

    def __init__(self, size: int = 19, allow_suicide: bool = False, use_superko: bool = False) -> None:
        self.size = int(size)
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.allow_suicide = bool(allow_suicide)
        self.use_superko = bool(use_superko)
        # 历史签名：(to_play, grid_bytes) —— 追加于“执行每一步落子/停着之前”
        self.history: List[Tuple[int, bytes]] = []

    # ---------- 基础 ----------
    def in_bounds(self, p: Point) -> bool:
        r, c = p
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, p: Point) -> Iterable[Point]:
        r, c = p
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield (nr, nc)

    def get(self, p: Point) -> Color:
        return int(self.grid[p[0], p[1]])

    def set(self, p: Point, color: Color) -> None:
        self.grid[p[0], p[1]] = color

    def copy(self) -> "Board":
        """深拷贝棋盘与历史（grid bytes 可共享；bytes 本身不可变）。"""
        b = Board(self.size, allow_suicide=self.allow_suicide, use_superko=self.use_superko)
        b.grid = self.grid.copy()
        b.history = list(self.history)
        return b

    # ---------- 连通块与气 ----------
    def _collect_group(self, start: Point) -> Tuple[Set[Point], Set[Point]]:
        color = self.get(start)
        assert color in (BLACK, WHITE)
        visited: Set[Point] = set([start])
        liberties: Set[Point] = set()
        stack = [start]
        while stack:
            p = stack.pop()
            for q in self.neighbors(p):
                cq = self.get(q)
                if cq == EMPTY:
                    liberties.add(q)
                elif cq == color and q not in visited:
                    visited.add(q)
                    stack.append(q)
        return visited, liberties

    def _remove_group(self, group: Iterable[Point]) -> int:
        cnt = 0
        for p in group:
            if self.get(p) != EMPTY:
                self.set(p, EMPTY)
                cnt += 1
        return cnt

    # ---------- 在拷贝上模拟一手 ----------
    def _collect_group_on_grid(self, grid: np.ndarray, start: Point) -> Tuple[Set[Point], Set[Point]]:
        color = int(grid[start[0], start[1]])
        assert color in (BLACK, WHITE)
        visited: Set[Point] = set([start])
        liberties: Set[Point] = set()
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    v = int(grid[nr, nc])
                    if v == EMPTY:
                        liberties.add((nr, nc))
                    elif v == color:
                        q = (nr, nc)
                        if q not in visited:
                            visited.add(q)
                            stack.append(q)
        return visited, liberties

    def _simulate(self, color: Color, p: Optional[Point]) -> Tuple[np.ndarray, int]:
        """在副本上模拟一手（含提子），返回（新网格、被提子数）。p=None 表示 pass。"""
        new_grid = self.grid.copy()
        captured_total = 0
        if p is None:
            return new_grid, 0

        r, c = p
        if new_grid[r, c] != EMPTY:
            raise ValueError("simulate: position not empty")

        new_grid[r, c] = color
        opp = BLACK if color == WHITE else WHITE

        # 先提对方无气块
        to_capture: Set[Point] = set()
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            nr, nc = q
            if 0 <= nr < self.size and 0 <= nc < self.size and new_grid[nr, nc] == opp:
                group, libs = self._collect_group_on_grid(new_grid, (nr, nc))
                if not libs:
                    to_capture |= group
        for q in to_capture:
            new_grid[q[0], q[1]] = EMPTY
            captured_total += 1

        # 再检查己方是否自杀
        group, libs = self._collect_group_on_grid(new_grid, p)
        if not libs and not self.allow_suicide:
            raise ValueError("simulate: suicide not allowed")

        return new_grid, captured_total

    # ---------- 合法性与执行 ----------
    def is_legal(self, color: Color, p: Optional[Point], to_play: Color) -> bool:
        """合法性：空点、禁自杀、普通劫禁（或可选超级劫）。"""
        try:
            # pass 永远合法
            if p is None:
                return True
            if not self.in_bounds(p) or self.get(p) != EMPTY:
                return False

            new_grid, _ = self._simulate(color, p)
            next_to_play = BLACK if to_play == WHITE else WHITE

            # 普通劫禁：禁止“新局面(下一手行棋方) == 上一手落子前的局面签名”
            if len(self.history) >= 1 and not self.use_superko:
                prev_to_play, prev_grid_bytes = self.history[-1]
                if prev_to_play == int(next_to_play) and new_grid.tobytes() == prev_grid_bytes:
                    return False

            # 超级劫禁：禁止与任意历史签名相同
            if self.use_superko:
                new_sig = (int(next_to_play), new_grid.tobytes())
                if new_sig in self.history:
                    return False

            return True
        except ValueError:
            return False

    def play(self, color: Color, p: Optional[Point], to_play: Color) -> int:
        """执行一手。返回被提子数。"""
        if not self.is_legal(color, p, to_play):
            raise ValueError("Illegal move")
        # 记录“落子前”的签名（用于劫禁/超级劫）
        self.history.append((int(to_play), self.grid.tobytes()))

        if p is None:
            return 0

        new_grid, captured = self._simulate(color, p)
        self.grid[:, :] = new_grid
        return captured

    # ---------- 统计与枚举 ----------
    def stones_count(self, color: Color) -> int:
        return int((self.grid == color).sum())

    def empty_points(self) -> Iterable[Point]:
        for r in range(self.size):
            for c in range(self.size):
                if int(self.grid[r, c]) == EMPTY:
                    yield (r, c)

    def legal_moves(self, color: Color, to_play: Color) -> List[Optional[Point]]:
        moves: List[Optional[Point]] = [None]  # pass 永远合法
        for r, c in self.empty_points():
            if self.is_legal(color, (r, c), to_play):
                moves.append((r, c))
        return moves