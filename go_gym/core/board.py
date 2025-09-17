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

    网格用 int8：0=空, 1=黑, 2=白。
    劫：用“上一上一手棋局面”比较实现普通劫禁（不允许立即劫回）。
    """

    def __init__(self, size: int = 19, allow_suicide: bool = False, use_superko: bool = False) -> None:
        self.size = int(size)
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.allow_suicide = bool(allow_suicide)
        self.use_superko = bool(use_superko)
        # 历史局面（仅在 simple-ko 用于对比上上手；在 superko 时用于全局去重）
        self.history: List[Tuple[int, bytes]] = []  # (to_play, grid.tobytes())

    # ---------- 基础工具 ----------
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

    # ---------- 连通块与气 ----------
    def _collect_group(self, start: Point) -> Tuple[Set[Point], Set[Point]]:
        """返回（同色连通块、该块的气集合）"""
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
        """提子：移除给定点集；返回被提子数。"""
        cnt = 0
        for p in group:
            if self.get(p) != EMPTY:
                self.set(p, EMPTY)
                cnt += 1
        return cnt

    # ---------- 落子模拟（用于合法性/劫禁判定） ----------
    def _simulate(self, color: Color, p: Optional[Point]) -> Tuple[np.ndarray, int]:
        """拷贝棋盘并在其上模拟一手（含提子），返回（新网格、总提子数）。

        若 p 为 None 表示停着（pass），直接返回拷贝。
        """
        new_grid = self.grid.copy()
        captured_total = 0
        if p is None:
            return new_grid, 0

        r, c = p
        if new_grid[r, c] != EMPTY:
            raise ValueError("simulate: position not empty")

        new_grid[r, c] = color

        # 先检查并提掉对方无气块
        opp = BLACK if color == WHITE else WHITE
        to_capture: Set[Point] = set()
        for q in self.neighbors(p):
            if new_grid[q[0], q[1]] == opp:
                # 收集对方块的气
                group, libs = self._collect_group_on_grid(new_grid, q)
                if not libs:
                    to_capture |= group
        for q in to_capture:
            new_grid[q[0], q[1]] = EMPTY
            captured_total += 1

        # 再检查己方块是否自杀（无气且未通过提子获得气）
        group, libs = self._collect_group_on_grid(new_grid, p)
        if not libs:
            # 自杀：若禁自杀，则模拟非法
            if not self.allow_suicide:
                raise ValueError("simulate: suicide not allowed")
            # 允许自杀则保留（极少使用）
        return new_grid, captured_total

    def _collect_group_on_grid(self, grid: np.ndarray, start: Point) -> Tuple[Set[Point], Set[Point]]:
        color = int(grid[start[0], start[1]])
        assert color in (BLACK, WHITE)
        visited: Set[Point] = set([start])
        liberties: Set[Point] = set()
        stack = [start]
        while stack:
            p = stack.pop()
            r, c = p
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

    # ---------- 合法性与落子执行 ----------
    def is_legal(self, color: Color, p: Optional[Point], to_play: Color) -> bool:
        """合法性：空点、禁自杀、劫禁（普通劫；可选超级劫）"""
        try:
            # 停着永远合法（不受劫禁）
            if p is None:
                return True

            r, c = p
            if not self.in_bounds(p) or self.get(p) != EMPTY:
                return False

            new_grid, _ = self._simulate(color, p)

            # 普通劫禁：不能令局面等于“上上手”的位置（含棋盘+行棋方）
            if len(self.history) >= 2 and not self.use_superko:
                prev2_to_play, prev2_grid_bytes = self.history[-2]
                if (prev2_to_play == int(to_play)) and (new_grid.tobytes() == prev2_grid_bytes):
                    return False

            # 超级劫禁：不得重复任何已出现过的局面（含行棋方）
            if self.use_superko:
                new_sig = (int(to_play), new_grid.tobytes())
                if new_sig in self.history:
                    return False

            return True
        except ValueError:
            return False

    def play(self, color: Color, p: Optional[Point], to_play: Color) -> int:
        """执行一手。返回本手提掉对方子的数量。"""
        if not self.is_legal(color, p, to_play):
            raise ValueError("Illegal move")
        # 保存历史（当前局面签名，用于劫禁）
        self.history.append((int(to_play), self.grid.tobytes()))

        if p is None:
            return 0

        new_grid, captured = self._simulate(color, p)
        self.grid[:, :] = new_grid
        return captured

    # ---------- 查询 ----------
    def stones_count(self, color: Color) -> int:
        return int((self.grid == color).sum())

    def empty_points(self) -> Iterable[Point]:
        for r in range(self.size):
            for c in range(self.size):
                if int(self.grid[r, c]) == EMPTY:
                    yield (r, c)
