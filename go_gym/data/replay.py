from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from loguru import logger


@dataclass
class Sample:
    planes: np.ndarray  # [C,H,W] float32
    pi:     np.ndarray  # [A]     float32
    z:      float       # scalar  float (黑视角)


def _symmetries(planes: np.ndarray, pi: np.ndarray, size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """8-重对称增强：旋转0/90/180/270 × (不翻/水平翻)。同步变换 policy 分布（含 PASS）。"""
    A = size * size + 1
    assert pi.shape[0] == A
    pass_idx = A - 1

    def map_pi(pi_in: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
        pi_grid = pi_in[:-1].reshape(size, size)  # 不含PASS
        g = np.rot90(pi_grid, k=k_rot)
        if flip:
            g = np.fliplr(g)
        out = np.concatenate([g.reshape(size*size), np.array([pi_in[pass_idx]], dtype=pi_in.dtype)], axis=0)
        return out

    outs = []
    for k in range(4):
        p = np.rot90(planes, k=k, axes=(1, 2))
        pi_k = map_pi(pi, k, False)
        outs.append((p, pi_k))
        p_f = np.flip(p, axis=2)
        pi_kf = map_pi(pi, k, True)
        outs.append((p_f, pi_kf))
    return outs


class ReplayBuffer:
    """简单环形回放池（可选对称增强在采样时进行）。"""
    def __init__(self, capacity: int, enable_symmetry: bool = True):
        self.capacity = int(capacity)
        self.enable_symmetry = bool(enable_symmetry)
        self._buf: List[Sample] = []
        self._ptr = 0

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, planes: np.ndarray, pi: np.ndarray, z: float):
        s = Sample(planes.astype(np.float32), pi.astype(np.float32), float(z))
        if len(self._buf) < self.capacity:
            self._buf.append(s)
        else:
            self._buf[self._ptr] = s
            self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int, board_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert len(self._buf) > 0, "ReplayBuffer is empty"
        idxs = np.random.randint(0, len(self._buf), size=batch_size)
        planes_list, pi_list, z_list = [], [], []
        for i in idxs:
            s = self._buf[i]
            if self.enable_symmetry and np.random.rand() < 0.5:
                sym = _symmetries(s.planes, s.pi, board_size)
                planes, pi = sym[np.random.randint(0, len(sym))]
            else:
                planes, pi = s.planes, s.pi
            planes_list.append(planes)
            pi_list.append(pi)
            z_list.append(s.z)
        return (np.stack(planes_list, axis=0).astype(np.float32),
                np.stack(pi_list,    axis=0).astype(np.float32),
                np.array(z_list,     dtype=np.float32))
