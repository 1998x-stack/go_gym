from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from loguru import logger

from ..agents.zero_mcts import ZeroMCTS, GameState
from ..core.board import Board, BLACK, WHITE
from ..core.scoring import chinese_area_score


@dataclass
class EvalConfig:
    games: int = 50
    sims: int = 600
    c_puct: float = 1.25
    temperature: float = 0.0  # 评测用 τ=0
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 0.03
    use_planes: int = 8


def play_eval_game(net_a: torch.nn.Module, net_b: torch.nn.Module, device: torch.device,
                   board_size: int, komi: float, sims: int, c_puct: float, use_planes: int) -> int:
    """A 执黑，B 执白；返回 +1(黑胜/A胜), -1(白胜/B胜), 0(和)"""
    mcts_a = ZeroMCTS(net_a, device, board_size, c_puct, sims, dirichlet_epsilon=0.0, dirichlet_alpha=0.03, use_planes=use_planes)
    mcts_b = ZeroMCTS(net_b, device, board_size, c_puct, sims, dirichlet_epsilon=0.0, dirichlet_alpha=0.03, use_planes=use_planes)

    board = Board(board_size, allow_suicide=False, use_superko=False)
    to_play = BLACK
    pass_count = 0
    last_move = None

    while pass_count < 2:
        state = GameState(board=board.copy(), to_play=to_play, pass_count=pass_count, komi=komi, last_move=last_move)
        mcts = mcts_a if to_play == BLACK else mcts_b
        pi, action = mcts.run(state, temperature=0.0, add_root_noise=False)
        move = None if action == board_size * board_size else (action // board_size, action % board_size)
        if move is None:
            board.play(to_play, None, to_play=to_play)
            pass_count += 1
            last_move = None
        else:
            board.play(to_play, move, to_play=to_play)
            pass_count = 0
            last_move = move
        to_play = BLACK if to_play == WHITE else WHITE

    b, w = chinese_area_score(board, komi)
    return 1 if b > w else (-1 if w > b else 0)


def arena(net_new: torch.nn.Module, net_best: torch.nn.Module, device: torch.device,
          board_size: int, komi: float, cfg: EvalConfig) -> float:
    wins = 0
    games = max(2, cfg.games)
    for i in range(games):
        # 轮换先后
        if i % 2 == 0:
            r = play_eval_game(net_new, net_best, device, board_size, komi, cfg.sims, cfg.c_puct, cfg.use_planes)
            if r == 1: wins += 1
        else:
            r = play_eval_game(net_best, net_new, device, board_size, komi, cfg.sims, cfg.c_puct, cfg.use_planes)
            if r == -1: wins += 1
    winrate = wins / games
    logger.info(f"Arena finished: new vs best = {wins}/{games} ({winrate*100:.1f}%)")
    return winrate
