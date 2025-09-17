from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from pathlib import Path
import time


def create_writer(logdir: str) -> SummaryWriter:
    Path(logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)
    logger.info(f"TensorBoard SummaryWriter created at: {logdir}")
    return writer


class StepTimer:
    """统计/打印迭代速度的小工具"""
    def __init__(self, window: int = 100):
        self.window = window
        self.t0 = time.time()
        self.n  = 0

    def tick(self, k: int = 1) -> float:
        self.n += k
        if self.n % self.window == 0:
            t1 = time.time()
            dt = t1 - self.t0
            ips = self.window / max(1e-6, dt)
            self.t0 = t1
            return ips
        return -1.0
