from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math, random, threading
import numpy as np
import torch
from loguru import logger

from ..core.board import Board, BLACK, WHITE, Color, Point
from ..core.scoring import chinese_area_score
from ..core.zobrist import hash_grid
from ..nn.infer_server import AsyncBatcher

# ---------- helpers ----------
@dataclass(frozen=True)
class GameState:
    board: Board
    to_play: Color
    pass_count: int
    komi: float
    last_move: Optional[Point] = None
    @property
    def size(self) -> int: return self.board.size
    def is_terminal(self) -> bool: return self.pass_count >= 2
    def legal_moves(self) -> List[Optional[Point]]: return self.board.legal_moves(self.to_play, self.to_play)
    def next_state(self, move: Optional[Point]) -> "GameState":
        b2 = self.board.copy()
        if move is None:
            b2.play(self.to_play, None, to_play=self.to_play)
            nxt = BLACK if self.to_play == WHITE else WHITE
            return GameState(b2, nxt, self.pass_count + 1, self.komi, last_move=None)
        b2.play(self.to_play, move, to_play=self.to_play)
        nxt = BLACK if self.to_play == WHITE else WHITE
        return GameState(b2, nxt, 0, self.komi, last_move=move)
    def terminal_value_black(self) -> float:
        b, w = chinese_area_score(self.board, self.komi)
        return 1.0 if b > w else (-1.0 if w > b else 0.0)

def planes_from_state(s: GameState, use_planes: int = 8) -> np.ndarray:
    g = s.board.grid
    H, W = g.shape
    own = (g == (BLACK if s.to_play == BLACK else WHITE)).astype(np.float32)
    opp = (g == (WHITE if s.to_play == BLACK else BLACK)).astype(np.float32)
    to_play_plane = np.ones_like(g, dtype=np.float32) if s.to_play == BLACK else np.zeros_like(g, dtype=np.float32)
    last = np.zeros_like(g, dtype=np.float32)
    if s.last_move is not None: last[s.last_move] = 1.0
    legal = np.zeros_like(g, dtype=np.float32)
    for mv in s.legal_moves():
        if mv is not None: legal[mv] = 1.0
    planes = [own, opp, to_play_plane, last, legal]
    while len(planes) < use_planes:
        planes.append(np.zeros_like(g, dtype=np.float32))
    return np.stack(planes[:use_planes], axis=0)

def action_space_size(sz: int) -> int: return sz * sz + 1
def point_to_action(p: Optional[Point], size: int) -> int: return size*size if p is None else (p[0]*size + p[1])
def action_to_point(a: int, size: int) -> Optional[Point]: return None if a == size*size else (a//size, a%size)

class Node:
    __slots__ = ("P","N","W","Q","to_play","children","is_expanded","legal_actions")
    def __init__(self, to_play: Color):
        self.P: Dict[int,float] = {}
        self.N: Dict[int,int]   = {}
        self.W: Dict[int,float] = {}
        self.Q: Dict[int,float] = {}
        self.to_play = to_play
        self.children: Dict[int, Node] = {}
        self.is_expanded = False
        self.legal_actions: List[int] = []

class ZeroMCTS:
    """PUCT with NN prior/value, async-batched eval, Zobrist cache."""
    def __init__(self, net: torch.nn.Module, device: torch.device, board_size: int,
                 c_puct: float = 1.25, n_simulations: int = 400,
                 dirichlet_epsilon: float = 0.25, dirichlet_alpha: float = 0.10,
                 use_planes: int = 8, cache_capacity: int = 100000,
                 eval_batcher: Optional[AsyncBatcher] = None):
        self.net = net
        self.device = device
        self.size = int(board_size)
        self.A = action_space_size(self.size)
        self.c_puct = float(c_puct)
        self.n_sim = int(n_simulations)
        self.dir_eps = float(dirichlet_epsilon)
        self.dir_alpha = float(dirichlet_alpha)
        self.use_planes = int(use_planes)
        self.cache: Dict[int, Tuple[np.ndarray, float]] = {}
        self.cache_capacity = int(cache_capacity)
        self.eval_batcher = eval_batcher

    # ---------- public ----------
    def run(self, root_state: GameState, temperature: float = 1.0, add_root_noise: bool = True) -> Tuple[np.ndarray, int]:
        root = Node(root_state.to_play)
        self._expand(root, root_state)
        if add_root_noise: self._inject_root_dirichlet_noise(root)
        for _ in range(self.n_sim):
            self._simulate(root_state, root)

        N_arr = np.zeros(self.A, dtype=np.float32)
        for a in root.legal_actions:
            N_arr[a] = root.N.get(a, 0)

        if temperature > 1e-8:
            pi = (N_arr ** (1.0 / temperature))
            s = pi.sum()
            if s > 0: pi /= s
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

    # ---------- internals ----------
    def _simulate(self, state: GameState, node: Node) -> float:
        path = []
        cur_state, cur_node = state, node
        while True:
            if not cur_node.is_expanded or cur_state.is_terminal():
                break
            a = self._select_puct(cur_node)
            path.append((cur_node, a))
            mv = action_to_point(a, self.size)
            cur_state = cur_state.next_state(mv)
            if a not in cur_node.children:
                cur_node.children[a] = Node(cur_state.to_play)
            cur_node = cur_node.children[a]

        if not cur_state.is_terminal():
            self._expand(cur_node, cur_state)

        v_black = cur_state.terminal_value_black() if cur_state.is_terminal() else self._eval_value(cur_state)

        # backup along path
        for n, a in reversed(path):
            z = v_black if n.to_play == BLACK else -v_black
            n.N[a] = n.N.get(a, 0) + 1
            n.W[a] = n.W.get(a, 0.0) + z
            n.Q[a] = n.W[a] / n.N[a]
        return v_black

    def _expand(self, node: Node, state: GameState):
        legal_points = state.legal_moves()
        legal_actions = [point_to_action(p, self.size) for p in legal_points]
        node.legal_actions = legal_actions
        P, _ = self._eval_policy_value(state, legal_actions)  # prior only for expansion
        priors = np.zeros(self.A, dtype=np.float32)
        s = float(P[legal_actions].sum()) if legal_actions else 0.0
        if s > 0: priors[legal_actions] = P[legal_actions] / s
        elif legal_actions: priors[legal_actions] = 1.0 / len(legal_actions)
        node.P = {a: float(priors[a]) for a in legal_actions}
        node.is_expanded = True
        for a in legal_actions:
            node.N.setdefault(a, 0); node.W.setdefault(a, 0.0); node.Q.setdefault(a, 0.0)

    def _inject_root_dirichlet_noise(self, node: Node):
        A = node.legal_actions
        if not A: return
        noise = np.random.dirichlet([self.dir_alpha] * len(A)).astype(np.float32)
        for i, a in enumerate(A):
            node.P[a] = (1 - self.dir_eps) * node.P.get(a, 0.0) + self.dir_eps * float(noise[i])

    def _select_puct(self, node: Node) -> int:
        sumN = max(1, sum(node.N.get(a, 0) for a in node.legal_actions))
        best, best_score = None, -1e18
        for a in node.legal_actions:
            Q = node.Q.get(a, 0.0)
            P = node.P.get(a, 0.0)
            U = self.c_puct * P * math.sqrt(sumN) / (1.0 + node.N.get(a, 0))
            score = (Q if node.to_play == BLACK else -Q) + U
            if score > best_score:
                best_score, best = score, a
        assert best is not None
        return best

    # -- eval path (policy+value, with async batcher and Zobrist cache) --
    def _eval_policy_value(self, s: GameState, legal_actions: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        key = hash_grid(s.board.grid, s.to_play)
        if key in self.cache:
            return self.cache[key]

        if self.eval_batcher is not None:
            done = threading.Event()
            out: Tuple[np.ndarray, float] = (None, 0.0)  # type: ignore
            la = legal_actions if legal_actions is not None else [point_to_action(p, self.size) for p in s.legal_moves()]
            def _cb(P: np.ndarray, v: float):
                nonlocal out
                out = (P, v); done.set()
            self.eval_batcher.submit(planes_from_state(s, self.use_planes), la, _cb)
            done.wait()
            P, v = out
        else:
            x = planes_from_state(s, self.use_planes)
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.net(x)
                logits = logits.float(); value = value.float()
            la = legal_actions if legal_actions is not None else [point_to_action(p, self.size) for p in s.legal_moves()]
            mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            for a in la: mask[a] = True
            logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
            P = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
            v = float(value.squeeze(0).item())

        if len(self.cache) >= self.cache_capacity:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (P, v)
        return P, v

    def _eval_value(self, s: GameState) -> float:
        return self._eval_policy_value(s)[1]
