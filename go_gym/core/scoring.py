from __future__ import annotations
import numpy as np
from typing import Tuple, Set
from .board import Board, BLACK, WHITE, EMPTY, Point


def chinese_area_score(board: Board, komi: float = 7.5) -> Tuple[float, float]:
    """中国规则面积数子：得分 = 盘上活子数 + 所围空点数（中立点/共活点不计）；
    白方再加贴目 komi。
    返回：(black_score, white_score_with_komi)。
    """
    g = board.grid
    size = board.size

    # 盘上子
    black_stones = int((g == BLACK).sum())
    white_stones = int((g == WHITE).sum())

    # 领地：对每个空连通域做 flood-fill，若其邻接仅黑或仅白，则记为该方地
    visited = np.zeros_like(g, dtype=bool)
    black_terr = 0
    white_terr = 0

    def neighbors(p: Point):
        r, c = p
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < size and 0 <= nc < size:
                yield (nr, nc)

    for r in range(size):
        for c in range(size):
            if g[r, c] != EMPTY or visited[r, c]:
                continue
            # flood 空域
            stack = [(r, c)]
            region: Set[Point] = set()
            adj_colors: Set[int] = set()
            visited[r, c] = True
            while stack:
                pr, pc = stack.pop()
                region.add((pr, pc))
                for nr, nc in neighbors((pr, pc)):
                    v = int(g[nr, nc])
                    if v == EMPTY and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                    elif v in (BLACK, WHITE):
                        adj_colors.add(v)
            # 归属
            if adj_colors == {BLACK}:
                black_terr += len(region)
            elif adj_colors == {WHITE}:
                white_terr += len(region)
            # 否则是中立点/共活（不计）

    black_score = black_stones + black_terr
    white_score = white_stones + white_terr + float(komi)
    return float(black_score), float(white_score)
