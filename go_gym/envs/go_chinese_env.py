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
    结局：连续两次 pass 终局；胜负以中国面积数子 + komi（白加）裁定；奖励 winner 得 +1，其余 -1（和棋得 0）。
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
        self.render_mode = render_mode

        self.board = Board(self.size, allow_suicide=self.allow_suicide, use_superko=self.use_superko)
        self.to_play: Color = BLACK
        self.pass_count = 0
        self.steps = 0
        self.last_move: Optional[Point] = None

        # Observation / Action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4, self.size, self.size), dtype=np.float32)
        self.action_space = spaces.Discrete(self.size * self.size + 1)  # +1 for pass

        # 随机数
        self.np_random, _ = gym.utils.seeding.np_random(None)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.board = Board(self.size, allow_suicide=self.allow_suicide, use_superko=self.use_superko)
        self.to_play = BLACK
        self.pass_count = 0
        self.steps = 0
        self.last_move = None
        obs = self._obs()
        info = {"action_mask": self.legal_action_mask()}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action index"
        done = False
        reward = 0.0

        # 行动解码
        if action == self.size * self.size:
            # pass
            self.board.play(self.to_play, None, to_play=self.to_play)
            self.pass_count += 1
            self.last_move = None
        else:
            r = action // self.size
            c = action % self.size
            p = (int(r), int(c))
            if not self.board.is_legal(self.to_play, p, to_play=self.to_play):
                # 非法手：给一个强信号（可按需求改为直接抛错）
                # 这里选择：非法手等价于“无效动作”：给予 -1 奖励并终局
                done = True
                reward = -1.0
                obs = self._obs()  # 不改变局面
                info = {"illegal_move": True, "action_mask": self.legal_action_mask()}
                return obs, reward, done, False, info

            captured = self.board.play(self.to_play, p, to_play=self.to_play)
            self.pass_count = 0
            self.last_move = p

        self.steps += 1

        # 切换行棋方
        self.to_play = BLACK if self.to_play == WHITE else WHITE

        # 终局判断：两次连续 pass 或步数上限
        if self.pass_count >= 2 or self.steps >= self.max_steps:
            done = True
            b_score, w_score = chinese_area_score(self.board, komi=self.komi)
            # 胜负与奖励：当前定义为“黑胜 +1，白胜 -1”；若你做自方视角训练，可在外层转换
            if b_score > w_score:
                reward = +1.0  # 黑胜
                winner = "black"
            elif w_score > b_score:
                reward = -1.0  # 白胜
                winner = "white"
            else:
                reward = 0.0
                winner = "draw"
            info = {
                "terminated_reason": "two_passes" if self.pass_count >= 2 else "max_steps",
                "score_black": b_score,
                "score_white": w_score,
                "winner": winner,
                "action_mask": self.legal_action_mask(),
            }
            obs = self._obs()
            return obs, reward, done, False, info

        obs = self._obs()
        info = {"action_mask": self.legal_action_mask()}
        return obs, reward, done, False, info

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
        # pass 永远合法
        mask[-1] = 1
        for r in range(self.size):
            for c in range(self.size):
                if self.board.grid[r, c] != EMPTY:
                    continue
                if self.board.is_legal(self.to_play, (r, c), to_play=self.to_play):
                    idx = r * self.size + c
                    mask[idx] = 1
        return mask
