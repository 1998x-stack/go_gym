from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import threading
import queue
import time
import numpy as np
import torch
from loguru import logger

@dataclass
class _Req:
    planes: np.ndarray          # [C,H,W] float32
    legal_actions: List[int]    # 合法动作索引
    cb: Callable[[np.ndarray, float], None]  # 回调 (P, v) → None

class AsyncBatcher:
    """单进程/多线程异步批量推理器（把自博弈线程的评估请求合批到 GPU）。
    - 线程安全：submit() 非阻塞，返回值通过回调传回。
    - 批量屏蔽：每个样本按其合法动作动态mask再softmax。
    """
    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device,
        planes: int,
        board_size: int,
        max_batch: int = 64,
        timeout_ms: int = 5,
    ):
        self.net = net.eval()
        self.device = device
        self.planes = int(planes)
        self.size = int(board_size)
        self.A = self.size * self.size + 1
        self.max_batch = int(max_batch)
        self.timeout_ms = int(timeout_ms)

        self.q: "queue.Queue[_Req]" = queue.Queue(maxsize=8192)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        self._batch_counter = 0
        logger.info(f"AsyncBatcher started: max_batch={self.max_batch}, timeout_ms={self.timeout_ms}")

    def update_net(self, net: torch.nn.Module):
        """热更新底层模型（例如EMA推送后）"""
        self.net = net.eval()
        logger.info("AsyncBatcher: net updated (EMA)")

    def stop(self):
        self._stop.set()
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass

    def submit(self, planes: np.ndarray, legal_actions: List[int], cb: Callable[[np.ndarray, float], None]):
        self.q.put(_Req(planes.astype(np.float32), list(legal_actions), cb))

    # ---------------- internal ----------------
    @torch.no_grad()
    def _loop(self):
        pend: List[_Req] = []
        last_log = time.time()
        while not self._stop.is_set():
            try:
                # 至少取一个，阻塞等待
                req = self.q.get(timeout=0.01)
            except queue.Empty:
                continue
            pend.append(req)
            t0 = time.time()
            # 继续收集直到batch满或超时
            while len(pend) < self.max_batch:
                dt = (time.time() - t0) * 1000.0
                if dt >= self.timeout_ms:
                    break
                try:
                    pend.append(self.q.get(timeout=(self.timeout_ms - dt) / 1000.0))
                except queue.Empty:
                    break

            # 组批
            B = len(pend)
            xs = np.stack([r.planes for r in pend], axis=0)  # [B,C,H,W]
            logits, value = self.net(torch.from_numpy(xs).to(self.device))
            logits = logits.float()  # [B,A]
            value = value.float().squeeze(1)  # [B]

            # 每个样本掩码+softmax
            P_list: List[np.ndarray] = []
            v_list: List[float] = []
            A = logits.shape[-1]
            for i, r in enumerate(pend):
                li = logits[i]
                legal_mask = torch.zeros(A, dtype=torch.bool, device=li.device)
                if r.legal_actions:
                    legal_mask[r.legal_actions] = True
                li = li.masked_fill(~legal_mask, -1e9)
                Pi = torch.softmax(li, dim=-1).detach().cpu().numpy().astype(np.float32)
                P_list.append(Pi)
                v_list.append(float(value[i].item()))

            # 回调发回
            for (r, P, v) in zip(pend, P_list, v_list):
                try:
                    r.cb(P, v)
                except Exception as e:
                    logger.exception(f"AsyncBatcher callback error: {e}")

            self._batch_counter += 1
            if time.time() - last_log > 5.0:
                logger.info(f"AsyncBatcher: served {self._batch_counter} batches, last_batch={B}")
                last_log = time.time()
            pend.clear()
