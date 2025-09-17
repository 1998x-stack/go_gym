from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import os, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from loguru import logger

from ..nn.policy_value_net import PolicyValueNet
from ..data.replay import ReplayBuffer
from ..utils.tb import create_writer, StepTimer
from ..train.evaluator import arena, EvalConfig

@dataclass
class OptimConfig:
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    total_steps: int = 200000
    clip_grad_norm: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=True)

def cosine_lr(step: int, cfg: OptimConfig) -> float:
    if step < cfg.warmup_steps:
        return float(cfg.lr) * (step / max(1, cfg.warmup_steps))
    t = (step - cfg.warmup_steps) / max(1, (cfg.total_steps - cfg.warmup_steps))
    t = min(max(t, 0.0), 1.0)
    return 0.5 * float(cfg.lr) * (1 + math.cos(math.pi * t))

def save_checkpoint(path: str, model: torch.nn.Module, ema: EMA, step: int):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step), "model": model.state_dict(), "ema": ema.shadow}, path)
    logger.info(f"Checkpoint saved: {path} (step={step})")

def train_step(batch, model, optim, scaler, device, cfg: OptimConfig, step: int) -> Dict[str, float]:
    planes, pi, z = batch
    planes = torch.from_numpy(planes).to(device)
    pi     = torch.from_numpy(pi).to(device)              # [B,A]
    z      = torch.from_numpy(z).to(device).unsqueeze(-1) # [B,1]
    with autocast(enabled=bool(cfg.amp)):
        logits, v = model(planes)
        logp = torch.log_softmax(logits, dim=-1)
        policy_loss = (- (pi * logp).sum(dim=-1)).mean()
        value_loss  = F.mse_loss(v, z)
        loss = policy_loss + value_loss
    scaler.scale(loss).backward()
    if float(cfg.clip_grad_norm) > 0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad_norm))
    scaler.step(optim); scaler.update()
    optim.zero_grad(set_to_none=True)
    with torch.no_grad():
        ent = (-torch.softmax(logits, dim=-1) * logp).sum(dim=-1).mean()
        z_m = v.mean()
    return {"loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(ent.item()),
            "z_mean": float(z_m.item())}

def learner_loop(
    replay: ReplayBuffer,
    board_size: int,
    planes: int,
    log_dir: str,
    ckpt_dir: str,
    optim_cfg: OptimConfig,
    channels: int = 96,
    blocks: int = 5,
    batch_size: int = 256,
    save_interval: int = 2000,
    max_steps: int = 200000,
    device: str = "cuda",
    ema_push_model: Optional[torch.nn.Module] = None,
    ema_push_interval: int = 2000,
    on_ema_pushed: Optional[Callable[[torch.nn.Module], None]] = None,  # NEW
    komi: float = 7.5,
    eval_cfg: Optional[EvalConfig] = None,
    best_model: Optional[PolicyValueNet] = None,
    eval_interval: Optional[int] = None,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(planes=int(planes), channels=int(channels), blocks=int(blocks), board_size=int(board_size)).to(device)
    ema = EMA(model, decay=float(optim_cfg.ema_decay))
    optim = AdamW(model.parameters(), lr=float(optim_cfg.lr),
                  weight_decay=float(optim_cfg.weight_decay),
                  betas=(float(optim_cfg.betas[0]), float(optim_cfg.betas[1])))
    scaler = GradScaler(enabled=bool(optim_cfg.amp))
    writer = create_writer(log_dir); timer = StepTimer(window=50)

    gate_every = int(eval_interval) if eval_interval is not None else int(save_interval)
    step = 0
    while step < int(max_steps):
        if len(replay) < int(batch_size):
            time.sleep(0.2); continue
        for g in optim.param_groups: g["lr"] = cosine_lr(step, optim_cfg)
        batch = replay.sample(int(batch_size), int(board_size))
        metrics = train_step(batch, model, optim, scaler, device, optim_cfg, step)
        ema.update(model)

        ips = timer.tick()
        if ips > 0:
            logger.info(f"step {step} | loss {metrics['loss']:.4f} (p {metrics['policy_loss']:.4f}, v {metrics['value_loss']:.4f}) "
                        f"| ent {metrics['entropy']:.3f} | z_mean {metrics['z_mean']:.3f} | {ips:.1f} it/s")
        writer.add_scalar("train/loss", metrics["loss"], step)
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], step)
        writer.add_scalar("train/value_loss", metrics["value_loss"], step)
        writer.add_scalar("train/entropy", metrics["entropy"], step)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], step)

        if (step + 1) % int(save_interval) == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"step_{step+1}.pt"), model, ema, step+1)
        if ema_push_model is not None and (step + 1) % int(ema_push_interval) == 0:
            with torch.no_grad():
                ema.apply_to(ema_push_model); ema_push_model.eval()
            logger.info(f"EMA weights pushed to selfplay model at step {step+1}")
            if on_ema_pushed is not None:
                on_ema_pushed(ema_push_model)  # notify batcher

        if eval_cfg is not None and best_model is not None and (step + 1) % gate_every == 0:
            cand = PolicyValueNet(planes=int(planes), channels=int(channels), blocks=int(blocks), board_size=int(board_size)).to(device).eval()
            cand.load_state_dict(ema.shadow, strict=True)
            winrate = arena(cand, best_model, device, board_size=int(board_size), komi=float(komi), cfg=eval_cfg)
            writer.add_scalar("eval/winrate_vs_best", winrate, step+1)
            thr = float(getattr(eval_cfg, "gate_winrate", 0.55))
            if winrate >= thr:
                best_model.load_state_dict(ema.shadow, strict=True); best_model.eval()
                if ema_push_model is not None:
                    with torch.no_grad():
                        ema.apply_to(ema_push_model); ema_push_model.eval()
                    if on_ema_pushed is not None:
                        on_ema_pushed(ema_push_model)
                save_checkpoint(os.path.join(ckpt_dir, f"best_step_{step+1}.pt"), model, ema, step+1)
                logger.success(f"[GATE PASS] winrate={winrate:.3f} >= {thr:.3f} → Promote BEST & broadcast")
            else:
                logger.warning(f"[GATE FAIL] winrate={winrate:.3f} < {thr:.3f}")

        step += 1

    save_checkpoint(os.path.join(ckpt_dir, f"final.pt"), model, ema, step)
