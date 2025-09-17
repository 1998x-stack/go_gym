from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ResidualBlock(nn.Module):
    """标准 3x3 残差块：Conv-BN-ReLU-Conv-BN + Skip + ReLU"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class PolicyValueNet(nn.Module):
    """策略-价值网络骨架（AlphaZero 风格）

    输入: [B, C, H, W]  (C=4~8)，输出:
      - policy_logits: [B, H*W+1] （含 PASS）
      - value:         [B, 1]     （tanh 到 [-1,1]，黑视角）
    训练时可在外层对 policy_logits 做“非法动作屏蔽后再 CrossEntropy”。
    """
    def __init__(self, planes: int = 8, channels: int = 128, blocks: int = 6, board_size: int = 19):
        super().__init__()
        self.board_size = int(board_size)
        self.action_dim = self.board_size * self.board_size + 1

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Tower
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * self.board_size * self.board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),  # 黑视角 [-1, 1]
        )

        self._init_weights()
        logger.info(f"PolicyValueNet initialized: planes={planes}, channels={channels}, blocks={blocks}, size={board_size}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.tower(h)
        policy_logits = self.policy_head(h)          # [B, A]
        value         = self.value_head(h)           # [B, 1]  (-1..1)
        return policy_logits, value
