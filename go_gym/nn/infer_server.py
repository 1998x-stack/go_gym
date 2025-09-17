from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable
import threading, queue, time
import numpy as np
import torch
from loguru import logger

@dataclass
class _Req:
    planes: np.ndarray               # [C,H,W] float32
    legal_actions: List[int]         # indices incl PASS
    cb: Callable[[np.ndarray, float], None]  # (P, v)->None

class AsyncBatcher:
    """Threaded async batcher that merges eval requests into GPU batches."""
    def __init__(self, net: torch.nn.Module, device: torch.device,
                 planes: int, board_size: int,
                 max_batch: int = 64, timeout_ms: int = 5):
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
        self._served = 0
        logger.info(f"AsyncBatcher started: max_batch={self.max_batch}, timeout_ms={self.timeout_ms}")

    def update_net(self, net: torch.nn.Module):
        self.net = net.eval()
        logger.info("AsyncBatcher: net hot-swapped (EMA)")

    def stop(self):
        self._stop.set()
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass

    def submit(self, planes: np.ndarray, legal_actions: List[int], cb: Callable[[np.ndarray, float], None]):
        self.q.put(_Req(planes.astype(np.float32), list(legal_actions), cb))

    @torch.no_grad()
    def _loop(self):
        pend: List[_Req] = []
        last_log = time.time()
        while not self._stop.is_set():
            try:
                req = self.q.get(timeout=0.01)
            except queue.Empty:
                continue
            pend.append(req)
            t0 = time.time()
            while len(pend) < self.max_batch:
                dt = (time.time() - t0) * 1000.0
                if dt >= self.timeout_ms:
                    break
                try:
                    pend.append(self.q.get(timeout=(self.timeout_ms - dt) / 1000.0))
                except queue.Empty:
                    break

            B = len(pend)
            xs = np.stack([r.planes for r in pend], axis=0)  # [B,C,H,W]
            logits, value = self.net(torch.from_numpy(xs).to(self.device))
            logits = logits.float()                     # [B,A]
            value = value.float().squeeze(1)           # [B]

            A = logits.shape[-1]
            for i, r in enumerate(pend):
                li = logits[i]
                legal_mask = torch.zeros(A, dtype=torch.bool, device=li.device)
                if r.legal_actions:
                    legal_mask[r.legal_actions] = True
                li = li.masked_fill(~legal_mask, -1e9)
                Pi = torch.softmax(li, dim=-1).detach().cpu().numpy().astype(np.float32)
                vi = float(value[i].item())
                try:
                    r.cb(Pi, vi)
                except Exception as e:
                    logger.exception(f"AsyncBatcher callback error: {e}")

            self._served += 1
            if time.time() - last_log > 5.0:
                logger.info(f"AsyncBatcher: served {self._served} batches, last_batch={B}")
                last_log = time.time()
            pend.clear()
