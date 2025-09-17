from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import torch
from loguru import logger

from ..core.board import Board, BLACK, WHITE, Color, Point
from ..agents.zero_mcts import ZeroMCTS, GameState, planes_from_state
from ..nn.infer_server import AsyncBatcher
from ..core.scoring import chinese_area_score

@dataclass
class SelfPlayConfig:
    board_size: int = 9
    komi: float = 7.5
    sims: int = 400
    c_puct: float = 1.25
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.10
    use_planes: int = 8
    temperature_moves: int = 10
    temperature_after: float = 0.3
    resign_threshold: float = -0.95
    resign_check_steps: int = 3
    max_moves: int = 300

def temperature_for_move(t: int, cfg: SelfPlayConfig) -> float:
    return 1.0 if t < cfg.temperature_moves else cfg.temperature_after

def play_one_game(net: torch.nn.Module, device: torch.device, cfg: SelfPlayConfig,
                  dump_sgf: bool = False,
                  eval_batcher: Optional[AsyncBatcher] = None
                  ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    board = Board(cfg.board_size, allow_suicide=False, use_superko=False)
    to_play: Color = BLACK
    pass_count = 0
    last_move: Optional[Point] = None

    mcts = ZeroMCTS(net, device, board_size=cfg.board_size,
                    c_puct=cfg.c_puct, n_simulations=cfg.sims,
                    dirichlet_epsilon=cfg.dirichlet_epsilon, dirichlet_alpha=cfg.dirichlet_alpha,
                    use_planes=cfg.use_planes, eval_batcher=eval_batcher)

    planes_list, pi_list = [], []
    value_track = []

    for t in range(cfg.max_moves):
        state = GameState(board=board.copy(), to_play=to_play, pass_count=pass_count, komi=cfg.komi, last_move=last_move)
        temp = temperature_for_move(t, cfg)
        pi, action = mcts.run(state, temperature=temp, add_root_noise=True)

        planes = planes_from_state(state, cfg.use_planes)
        planes_list.append(planes); pi_list.append(pi)

        _, v_black = mcts._eval_policy_value(state)
        value_track.append(v_black)
        if len(value_track) >= cfg.resign_check_steps and np.mean(value_track[-cfg.resign_check_steps:]) < cfg.resign_threshold:
            z = -1.0 if to_play == BLACK else 1.0
            logger.info(f"Resign triggered at move {t}: z={z:.1f}, v_mean={np.mean(value_track[-cfg.resign_check_steps:]):.3f}")
            return planes_list, pi_list, z

        move = None if action == cfg.board_size * cfg.board_size else (action // cfg.board_size, action % cfg.board_size)
        if move is None:
            board.play(to_play, None, to_play=to_play); pass_count += 1; last_move = None
        else:
            board.play(to_play, move, to_play=to_play); pass_count = 0; last_move = move
        to_play = BLACK if to_play == WHITE else WHITE
        if pass_count >= 2: break

    b, w = chinese_area_score(board, cfg.komi)
    z = 1.0 if b > w else (-1.0 if w > b else 0.0)
    logger.info(f"Selfplay finished: B={b:.1f}, W={w:.1f}, z={z:+.1f}")
    return planes_list, pi_list, z
