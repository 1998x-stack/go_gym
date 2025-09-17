# go_gym/agents/uct.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import math
import random

from ..core.board import Board, BLACK, WHITE, EMPTY, Color, Point
from ..core.scoring import chinese_area_score


def action_to_point(action: int, size: int) -> Optional[Point]:
    """将离散动作映射为点坐标；末位表示 pass。"""
    if action == size * size:
        return None
    return (action // size, action % size)


def point_to_action(p: Optional[Point], size: int) -> int:
    if p is None:
        return size * size
    r, c = p
    return r * size + c


@dataclass(frozen=True)
class GameState:
    board: Board
    to_play: Color
    pass_count: int
    komi: float
    last_move: Optional[Point] = None

    def is_terminal(self) -> bool:
        return self.pass_count >= 2

    @property
    def size(self) -> int:
        return self.board.size

    def legal_actions(self) -> List[int]:
        moves = self.board.legal_moves(self.to_play, self.to_play)
        return [point_to_action(m, self.size) for m in moves]

    def next_state(self, action: int) -> "GameState":
        p = action_to_point(action, self.size)
        b2 = self.board.copy()
        if p is None:
            b2.play(self.to_play, None, to_play=self.to_play)
            next_to_play = BLACK if self.to_play == WHITE else WHITE
            return GameState(
                board=b2, to_play=next_to_play,
                pass_count=self.pass_count + 1, komi=self.komi, last_move=None
            )
        else:
            b2.play(self.to_play, p, to_play=self.to_play)
            next_to_play = BLACK if self.to_play == WHITE else WHITE
            return GameState(
                board=b2, to_play=next_to_play,
                pass_count=0, komi=self.komi, last_move=p
            )

    def terminal_value_black(self) -> float:
        """终局返回黑视角价值：黑胜 +1，白胜 -1，和 0。"""
        bs, ws = chinese_area_score(self.board, self.komi)
        if bs > ws:
            return +1.0
        elif ws > bs:
            return -1.0
        return 0.0


class Node:
    """UCT 节点。存储黑方视角累计价值 total_value 与访问次数 visits。"""
    __slots__ = ("state", "parent", "children", "untried", "visits", "total_value", "action_from_parent")

    def __init__(self, state: GameState, parent: Optional["Node"] = None, action_from_parent: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[int, Node] = {}
        self.untried: List[int] = state.legal_actions()
        self.visits = 0
        self.total_value = 0.0  # 从黑方视角累计
        self.action_from_parent = action_from_parent

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def best_child(self, c_puct: float) -> Tuple[int, "Node"]:
        """UCB1：在“当前行棋方”视角最大化。注意 total_value 存的是黑视角，需要对白方取反。"""
        best_a = None
        best_node = None
        best_score = -1e18

        # 避免除零：确保调用前所有子节点至少访问 1 次
        logN = math.log(max(1, self.visits))
        for a, ch in self.children.items():
            mean_black = ch.total_value / max(1, ch.visits)
            # 当前行棋方为白时，需要最小化黑视角 => 等价于最大化 (-mean_black)
            mean_for_player = mean_black if self.state.to_play == BLACK else -mean_black
            ucb = mean_for_player + c_puct * math.sqrt(logN / max(1, ch.visits))
            if ucb > best_score:
                best_score = ucb
                best_a = a
                best_node = ch
        assert best_a is not None and best_node is not None
        return best_a, best_node

    def expand_one(self) -> "Node":
        a = self.untried.pop(random.randrange(len(self.untried)))
        child_state = self.state.next_state(a)
        child = Node(child_state, parent=self, action_from_parent=a)
        self.children[a] = child
        return child

    def backprop(self, z_black: float) -> None:
        node: Optional[Node] = self
        while node is not None:
            node.visits += 1
            node.total_value += z_black
            node = node.parent


def default_rollout_policy(state: GameState, max_steps: int = 512) -> float:
    """随机走子直到两次 pass 或达到步数上限，返回黑视角价值。"""
    s = state
    steps = 0
    while not s.is_terminal() and steps < max_steps:
        acts = s.legal_actions()
        a = random.choice(acts)
        s = s.next_state(a)
        steps += 1
    return s.terminal_value_black()


def uct_search(root_state: GameState, n_simulations: int = 800, c_puct: float = 1.4,
               rollout_limit: int = 512) -> int:
    """在 root_state 上做 UCT 搜索，返回选择的动作（按访问次数最大）。"""
    root = Node(root_state)
    # 至少展开一个节点，避免 best_child 的除零
    if not root.state.is_terminal():
        leaf = root.expand_one()
        z = default_rollout_policy(leaf.state, max_steps=rollout_limit)
        leaf.backprop(z)

    for _ in range(n_simulations):
        node = root
        # Selection
        while not node.state.is_terminal() and node.is_fully_expanded():
            _, node = node.best_child(c_puct)
        # Expansion
        if not node.state.is_terminal() and not node.is_fully_expanded():
            node = node.expand_one()
        # Simulation
        z = default_rollout_policy(node.state, max_steps=rollout_limit)
        # Backprop
        node.backprop(z)

    # 选择访问次数最多的子节点动作
    best_a, best_vis = None, -1
    for a, ch in root.children.items():
        if ch.visits > best_vis:
            best_vis = ch.visits
            best_a = a
    # 如果还没有孩子（root 已终局），则只能 pass
    return best_a if best_a is not None else point_to_action(None, root_state.size)