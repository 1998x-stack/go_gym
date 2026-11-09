from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from ..core.board import Board, BLACK, WHITE, EMPTY, Point, Color
from ..core.scoring import chinese_area_score
from ..utils.render import render_ascii


class GoChineseEnv(gym.Env):
    """Gymnasium 环境 —— 中国围棋规则（面积数子、禁自杀、普通劫禁）。

    观察(obs)：(C, H, W) float32，通道：
        0: 当前执子方的己方棋子平面（1/0）
        1: 对手棋子平面（1/0）
        2: to_play 平面（全 1 表示当前是“黑走”，全 0 表示“白走”）
        3: 上一步落子点平面（1/0）
    行动(action)：Discrete(size*size + 1)，0..N-1 表示在对应交叉点落子；最后一个索引表示 pass。
    结局：连续两次 pass 正常终局，max_steps 为时间截断；正常终局奖励为黑方视角（黑胜 +1、白胜 -1、和棋 0）。
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        board_size: int = 19,
        komi: float = 7.5,
        allow_suicide: bool = False,
        use_superko: bool = False,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.size = int(board_size)
        self.komi = float(komi)
        self.allow_suicide = bool(allow_suicide)
        self.use_superko = bool(use_superko)
        self.max_steps = int(max_steps) if max_steps is not None else self.size * self.size * 2
        if self.size < 2 or self.max_steps < 1:
            raise ValueError("board_size must be >= 2 and max_steps >= 1")
        if render_mode not in (None, "ansi"):
            raise ValueError("unsupported render_mode")
        self.render_mode = render_mode

        self.board = Board(self.size, allow_suicide=self.allow_suicide, use_superko=self.use_superko)
        self.to_play: Color = BLACK
        self.pass_count = 0
        self.steps = 0
        self.last_move: Optional[Point] = None
        self._finished = False

        # Observation / Action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4, self.size, self.size), dtype=np.float32)
        self.action_space = spaces.Discrete(self.size * self.size + 1)  # +1 for pass

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.board = Board(self.size, allow_suicide=self.allow_suicide, use_superko=self.use_superko)
        self.to_play = BLACK
        self.pass_count = 0
        self.steps = 0
        self.last_move = None
        self._finished = False
        return self._obs(), {"action_mask": self.legal_action_mask()}

    def step(self, action: int):
        if self._finished:
            raise RuntimeError("episode finished; call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("invalid action index")

        reward = 0.0
        if action == self.size * self.size:
            self.board.play(self.to_play, None, to_play=self.to_play)
            self.pass_count += 1
            self.last_move = None
        else:
            r, c = divmod(int(action), self.size)
            p = (r, c)
            if not self.board.is_legal(self.to_play, p, to_play=self.to_play):
                # Invalid moves lose immediately; the board and turn are unchanged.
                self._finished = True
                return self._obs(), -1.0, True, False, {
                    "illegal_move": True,
                    "terminated_reason": "illegal_move",
                    "action_mask": self.legal_action_mask(),
                }
            self.board.play(self.to_play, p, to_play=self.to_play)
            self.pass_count = 0
            self.last_move = p

        self.steps += 1
        self.to_play = WHITE if self.to_play == BLACK else BLACK
        terminated = self.pass_count >= 2
        truncated = not terminated and self.steps >= self.max_steps

        if terminated:
            self._finished = True
            black_score, white_score = chinese_area_score(self.board, komi=self.komi)
            if black_score > white_score:
                reward, winner = 1.0, "black"
            elif white_score > black_score:
                reward, winner = -1.0, "white"
            else:
                reward, winner = 0.0, "draw"
            return self._obs(), reward, True, False, {
                "terminated_reason": "two_passes",
                "score_black": black_score,
                "score_white": white_score,
                "winner": winner,
                "action_mask": self.legal_action_mask(),
            }

        if truncated:
            # The move budget is a time limit, not a scored Go resignation.
            self._finished = True
            return self._obs(), 0.0, False, True, {
                "terminated_reason": "max_steps",
                "action_mask": self.legal_action_mask(),
            }

        return self._obs(), 0.0, False, False, {
            "action_mask": self.legal_action_mask(),
        }

    def render(self):
        if self.render_mode == "ansi":
            return render_ascii(self.board, last_move=self.last_move)
        # 可扩展 RGB 渲染（matplotlib）

    def close(self):
        pass

    # ---------- 观察与动作掩码 ----------
    def _obs(self) -> np.ndarray:
        """返回 (4, H, W) 观测：己方、对方、to_play、last_move。"""
        g = self.board.grid
        if self.to_play == BLACK:
            own = (g == BLACK)
            opp = (g == WHITE)
            to_play_plane = np.ones_like(g, dtype=bool)
        else:
            own = (g == WHITE)
            opp = (g == BLACK)
            to_play_plane = np.zeros_like(g, dtype=bool)

        last = np.zeros_like(g, dtype=bool)
        if self.last_move is not None:
            last[self.last_move] = True

        planes = np.stack([
            own.astype(np.float32),
            opp.astype(np.float32),
            to_play_plane.astype(np.float32),
            last.astype(np.float32),
        ], axis=0)
        return planes

    def legal_action_mask(self) -> np.ndarray:
        """返回长度 N+1 的 0/1 掩码（含 pass）。"""
        mask = np.zeros(self.size * self.size + 1, dtype=np.int8)
        if self._finished:
            return mask  # no actions are legal after a terminal/truncated step
        # pass is legal only during an active episode
        mask[-1] = 1
        for r in range(self.size):
            for c in range(self.size):
                if self.board.grid[r, c] != EMPTY:
                    continue
                if self.board.is_legal(self.to_play, (r, c), to_play=self.to_play):
                    idx = r * self.size + c
                    mask[idx] = 1
        return mask
