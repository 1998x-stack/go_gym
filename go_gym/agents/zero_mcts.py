from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math
import threading
import numpy as np
import torch
from loguru import logger

from ..core.board import Board, BLACK, WHITE, EMPTY, Color, Point
from ..core.scoring import chinese_area_score
from ..core.zobrist import hash_grid
from ..nn.infer_server import AsyncBatcher   # 新增


# ----------------- 状态与观测 -----------------
@dataclass(frozen=True)
class GameState:
    board: Board
    to_play: Color
    pass_count: int
    komi: float
    last_move: Optional[Point] = None

    @property
    def size(self) -> int:
        return self.board.size

    def is_terminal(self) -> bool:
        return self.pass_count >= 2

    def legal_moves(self) -> List[Optional[Point]]:
        return self.board.legal_moves(self.to_play, self.to_play)

    def next_state(self, move: Optional[Point]) -> "GameState":
        b2 = self.board.copy()
        if move is None:
            b2.play(self.to_play, None, to_play=self.to_play)
            next_to_play = BLACK if self.to_play == WHITE else WHITE
            return GameState(b2, next_to_play, self.pass_count + 1, self.komi, last_move=None)
        else:
            b2.play(self.to_play, move, to_play=self.to_play)
            next_to_play = BLACK if self.to_play == WHITE else WHITE
            return GameState(b2, next_to_play, 0, self.komi, last_move=move)

    def terminal_value_black(self) -> float:
        b, w = chinese_area_score(self.board, self.komi)
        return 1.0 if b > w else (-1.0 if w > b else 0.0)


def planes_from_state(s: GameState, use_planes: int = 8) -> np.ndarray:
    """把棋盘转为网络输入平面（C,H,W）。默认 8 通道（可裁剪为前 4 个）"""
    g = s.board.grid
    H, W = g.shape
    own  = (g == (BLACK if s.to_play == BLACK else WHITE)).astype(np.float32)
    opp  = (g == (WHITE if s.to_play == BLACK else BLACK)).astype(np.float32)
    to_play_plane = np.ones_like(g, dtype=np.float32) if s.to_play == BLACK else np.zeros_like(g, dtype=np.float32)
    last = np.zeros_like(g, dtype=np.float32)
    if s.last_move is not None:
        last[s.last_move] = 1.0

    # 可选辅助平面（示意：合法动作平面）
    legal = np.zeros_like(g, dtype=np.float32)
    for mv in s.legal_moves():
        if mv is not None:
            legal[mv] = 1.0

    planes = [own, opp, to_play_plane, last, legal]
    # 占位：更多启发式平面（眼位/打吃/星位等），这里简化为零平面
    while len(planes) < use_planes:
        planes.append(np.zeros_like(g, dtype=np.float32))
    return np.stack(planes[:use_planes], axis=0)  # [C,H,W]


def action_space_size(board_size: int) -> int:
    return board_size * board_size + 1


def point_to_action(p: Optional[Point], size: int) -> int:
    return size * size if p is None else (p[0] * size + p[1])


def action_to_point(a: int, size: int) -> Optional[Point]:
    return None if a == size * size else (a // size, a % size)


# ----------------- MCTS 节点 -----------------
class Node:
    __slots__ = ("P", "N", "W", "Q", "to_play", "children", "is_expanded", "legal_actions")

    def __init__(self, to_play: Color):
        self.P: Dict[int, float] = {}     # 先验
        self.N: Dict[int, int]   = {}     # 访问次数
        self.W: Dict[int, float] = {}     # 累计价值(黑视角)
        self.Q: Dict[int, float] = {}     # 平均价值(黑视角)
        self.to_play = to_play
        self.children: Dict[int, Node] = {}
        self.is_expanded = False
        self.legal_actions: List[int] = []


class ZeroMCTS:
    """神经网络引导的 PUCT 搜索（单进程版，附带根噪声与温度采样）。"""
    def __init__(self, net: torch.nn.Module, device: torch.device,
                 board_size: int, c_puct: float = 1.25, n_simulations: int = 400,
                 dirichlet_epsilon: float = 0.25, dirichlet_alpha: float = 0.10,
                 use_planes: int = 8, cache_capacity: int = 100000,
                 eval_batcher: Optional[AsyncBatcher] = None):   # 新增
        self.net = net
        self.device = device
        self.size = int(board_size)
        self.A = action_space_size(self.size)
        self.c_puct = float(c_puct)
        self.n_sim = int(n_simulations)
        self.dir_eps = float(dirichlet_epsilon)
        self.dir_alpha = float(dirichlet_alpha)
        self.use_planes = int(use_planes)
        self.cache: Dict[int, Tuple[np.ndarray, float]] = {}   # <--- Zobrist key
        self.cache_capacity = int(cache_capacity)
        self.eval_batcher = eval_batcher

    # ---------- 搜索入口 ----------
    def run(self, root_state: GameState, temperature: float = 1.0, add_root_noise: bool = True) -> Tuple[np.ndarray, int]:
        root = Node(root_state.to_play)
        self._expand(root, root_state)

        if add_root_noise:
            self._inject_root_dirichlet_noise(root)

        for _ in range(self.n_sim):
            self._simulate(root_state, root)

        # --- 根节点 N -> π，并只在合法动作上均匀 ---
        N_arr = np.zeros(self.A, dtype=np.float32)
        for a in root.legal_actions:
            N_arr[a] = root.N.get(a, 0)

        if temperature > 1e-8:
            pi = (N_arr ** (1.0 / temperature))
            s = pi.sum()
            if s > 0:
                pi /= s
            else:
                pi = np.zeros_like(N_arr, dtype=np.float32)
                if root.legal_actions:
                    pi[root.legal_actions] = 1.0 / float(len(root.legal_actions))
        else:
            pi = np.zeros_like(N_arr, dtype=np.float32)
            if root.legal_actions:
                a_star = int(max(root.legal_actions, key=lambda a: root.N.get(a, 0)))
                pi[a_star] = 1.0

        action = int(np.random.choice(self.A, p=pi))
        return pi, action

    # ---------- 单次模拟 ----------
    def _simulate(self, state: GameState, node: Node) -> float:
        path = []
        cur_state = state
        cur_node = node

        # Selection
        while True:
            if cur_node.is_expanded is False:
                break
            if cur_state.is_terminal():
                break
            a = self._select_puct(cur_node)
            path.append((cur_state, cur_node, a))
            mv = action_to_point(a, self.size)
            cur_state = cur_state.next_state(mv)
            if a not in cur_node.children:
                cur_node.children[a] = Node(cur_state.to_play)
            cur_node = cur_node.children[a]

        # Expansion / Evaluation
        if not cur_state.is_terminal():
            self._expand(cur_node, cur_state)

        # Value (黑视角) for current state
        if cur_state.is_terminal():
            v_black = cur_state.terminal_value_black()
        else:
            P, v_black = self._eval_state(cur_state)
            # 将先验写入节点（已在 _expand 配好）
            # 此处 v_black 即叶子估计

        # Backup（沿路径回传）
        for s, n, a in reversed(path):
            # 当前节点 n.to_play 的价值定义：Q 为黑视角
            # 当轮到白走时，叶子价值应取反以保持“从黑视角看当前节点收益”
            z = v_black if n.to_play == BLACK else -v_black
            n.N[a] = n.N.get(a, 0) + 1
            n.W[a] = n.W.get(a, 0.0) + z
            n.Q[a] = n.W[a] / n.N[a]
        return v_black

    # ---------- 展开 ----------
    def _expand(self, node: Node, state: GameState):
        legal_points = state.legal_moves()
        legal_actions = [point_to_action(p, self.size) for p in legal_points]
        node.legal_actions = legal_actions

        # Eval policy & value
        P, v_black = self._eval_state(state, legal_actions)  # 传入合法动作，便于批处理mask

        # 只保留合法动作的先验，并做归一化（避免全部为零的退化）
        priors = np.zeros(self.A, dtype=np.float32)
        s = float(P[legal_actions].sum()) if legal_actions else 0.0
        if s > 0:
            priors[legal_actions] = P[legal_actions] / s
        elif legal_actions:
            priors[legal_actions] = 1.0 / len(legal_actions)

        node.P = {a: float(priors[a]) for a in legal_actions}
        node.is_expanded = True
        # 初始化 N/W/Q
        for a in legal_actions:
            if a not in node.N:
                node.N[a] = 0
                node.W[a] = 0.0
                node.Q[a] = 0.0

    # ---------- 根噪声 ----------
    def _inject_root_dirichlet_noise(self, node: Node):
        actions = node.legal_actions
        if not actions:
            return
        alpha = self.dir_alpha
        eps = self.dir_eps
        noise = np.random.dirichlet([alpha] * len(actions)).astype(np.float32)
        for i, a in enumerate(actions):
            node.P[a] = (1 - eps) * node.P.get(a, 0.0) + eps * float(noise[i])

    # ---------- 选择 ----------
    def _select_puct(self, node: Node) -> int:
        # U(s,a)=c_puct * P(s,a) * sqrt(sum_b N) / (1+N(s,a))
        sumN = max(1, sum(node.N.get(a, 0) for a in node.legal_actions))
        best, best_score = None, -1e18
        for a in node.legal_actions:
            Q = node.Q.get(a, 0.0)
            P = node.P.get(a, 0.0)
            U = self.c_puct * P * math.sqrt(sumN) / (1.0 + node.N.get(a, 0))
            # 当前节点价值从谁视角定义？这里 node.Q 按“黑视角”
            # 若轮到白走，要“最小化黑视角”，等价于选择 -Q + U
            score = (Q if node.to_play == BLACK else -Q) + U
            if score > best_score:
                best_score = score
                best = a
        assert best is not None
        return best

    # ---------- 评估（批量可扩展） ----------
    def _eval_state(self, s: GameState, legal_actions: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        # --- Zobrist 缓存 ---
        key = hash_grid(s.board.grid, s.to_play)
        if key in self.cache:
            return self.cache[key]

        # --- 异步批量：若提供 batcher，就走 submit/回调 ---
        if self.eval_batcher is not None:
            done = threading.Event()
            out: Tuple[np.ndarray, float] = (None, 0.0)  # type: ignore

            def _cb(P: np.ndarray, v: float):
                nonlocal out
                out = (P, v)
                done.set()

            planes = planes_from_state(s, self.use_planes)
            la = legal_actions if legal_actions is not None else [point_to_action(p, self.size) for p in s.legal_moves()]
            self.eval_batcher.submit(planes, la, _cb)
            done.wait()  # 阻塞等待
            P, v = out
        else:
            # --- 单样本前向（老路径） ---
            x = planes_from_state(s, self.use_planes)
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.net(x)
                logits = logits.float()
                value = value.float()
            legal_mask_1d = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            la = legal_actions if legal_actions is not None else [point_to_action(p, self.size) for p in s.legal_moves()]
            for a in la:
                legal_mask_1d[a] = True
            logits = logits.masked_fill(~legal_mask_1d.unsqueeze(0), -1e9)
            P = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
            v = float(value.squeeze(0).item())

        # 缓存（简单近似FIFO）
        if len(self.cache) >= self.cache_capacity:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (P, v)
        return P, v
