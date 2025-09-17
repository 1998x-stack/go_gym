from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import threading, time, yaml, torch, os
from loguru import logger

from ..nn.policy_value_net import PolicyValueNet
from ..data.replay import ReplayBuffer
from ..train.selfplay import SelfPlayConfig, play_one_game
from ..train.learner import learner_loop, OptimConfig
from ..train.evaluator import EvalConfig   # ← 引入

@dataclass
class ZeroConfig:
    SEED: int
    BOARD_SIZE: int
    KOMI: float
    NET: Dict[str, Any]
    MCTS: Dict[str, Any]
    SELFPLAY: Dict[str, Any]
    REPLAY: Dict[str, Any]
    OPTIM: Dict[str, Any]
    EVAL: Dict[str, Any]
    LOG: Dict[str, Any]

def load_config(path: str) -> ZeroConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ZeroConfig(**cfg)

def selfplay_producer_thread(replay: ReplayBuffer, net_ema: PolicyValueNet, device: torch.device, sp_cfg: SelfPlayConfig):
    torch.manual_seed(int(time.time()) % 10_000_000)
    while True:
        try:
            planes_list, pi_list, z = play_one_game(net_ema, device, sp_cfg)
            for p, pi in zip(planes_list, pi_list):
                replay.add(p, pi, z)
        except Exception as e:
            logger.exception(f"Selfplay worker crashed and auto-restarted: {e}")
            time.sleep(0.5)

def main(config_path: str):
    cfg = load_config(config_path)
    torch.manual_seed(int(cfg.SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 回放池
    replay = ReplayBuffer(capacity=int(cfg.REPLAY["capacity"]), enable_symmetry=bool(cfg.REPLAY.get("symmetries", True)))

    # 自博弈用 EMA 模型 & 初始 best 模型
    net_ema = PolicyValueNet(planes=int(cfg.NET["planes"]), channels=int(cfg.NET["channels"]),
                             blocks=int(cfg.NET["blocks"]), board_size=int(cfg.BOARD_SIZE)).to(device).eval()
    best_model = PolicyValueNet(planes=int(cfg.NET["planes"]), channels=int(cfg.NET["channels"]),
                                blocks=int(cfg.NET["blocks"]), board_size=int(cfg.BOARD_SIZE)).to(device).eval()
    best_model.load_state_dict(net_ema.state_dict(), strict=True)

    # 自博弈配置
    sp_cfg = SelfPlayConfig(
        board_size=int(cfg.BOARD_SIZE), komi=float(cfg.KOMI),
        sims=int(cfg.MCTS["n_simulations"]), c_puct=float(cfg.MCTS["c_puct"]),
        dirichlet_epsilon=float(cfg.MCTS["dirichlet_epsilon"]), dirichlet_alpha=float(cfg.MCTS["dirichlet_alpha"]),
        use_planes=int(cfg.NET["planes"]),
        temperature_moves=int(cfg.MCTS["temperature_moves"]), temperature_after=float(cfg.MCTS["temperature_after"]),
        resign_threshold=float(cfg.SELFPLAY["resign_threshold"]),
        resign_check_steps=int(cfg.SELFPLAY["resign_check_steps"]),
        max_moves=int(cfg.SELFPLAY["max_moves"])
    )

    # 自博弈线程
    workers = max(1, int(cfg.SELFPLAY["workers"]))
    for i in range(workers):
        t = threading.Thread(target=selfplay_producer_thread, args=(replay, net_ema, device, sp_cfg), daemon=True)
        t.start()
        logger.info(f"Selfplay worker #{i} started.")

    # 优化器配置
    optim_cfg = OptimConfig(
        lr=float(cfg.OPTIM["lr"]), weight_decay=float(cfg.OPTIM["weight_decay"]),
        betas=(float(cfg.OPTIM["betas"][0]), float(cfg.OPTIM["betas"][1])),
        warmup_steps=int(cfg.OPTIM["warmup_steps"]), total_steps=int(cfg.OPTIM["total_steps"]),
        clip_grad_norm=float(cfg.OPTIM["clip_grad_norm"]), amp=bool(cfg.OPTIM["amp"]),
        ema_decay=float(cfg.NET["ema_decay"])
    )

    # 评测配置（带默认值，允许 YAML 缺项）
    eval_dict = cfg.EVAL or {}
    eval_cfg = EvalConfig(
        games=int(eval_dict.get("games", 40)),
        sims=int(eval_dict.get("sims", 600)),
        c_puct=float(eval_dict.get("c_puct", 1.25)),
        temperature=0.0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=float(eval_dict.get("dirichlet_alpha", 0.03)),
        use_planes=int(cfg.NET["planes"]),
    )
    gate_winrate = float(eval_dict.get("gate_winrate", 0.55))
    setattr(eval_cfg, "gate_winrate", gate_winrate)  # 兼容在 arena 后取阈值
    eval_interval = int(eval_dict.get("eval_interval", cfg.LOG.get("save_interval", 2000)))

    # 训练（集成门控 + 推送 EMA）
    learner_loop(
        replay=replay, board_size=int(cfg.BOARD_SIZE), planes=int(cfg.NET["planes"]),
        log_dir=str(cfg.LOG["tb_dir"]), ckpt_dir=str(cfg.LOG["ckpt_dir"]),
        optim_cfg=optim_cfg, channels=int(cfg.NET["channels"]), blocks=int(cfg.NET["blocks"]),
        batch_size=int(cfg.REPLAY["batch_size"]), save_interval=int(cfg.LOG["save_interval"]),
        max_steps=int(cfg.OPTIM["total_steps"]), device=str(device),
        ema_push_model=net_ema, ema_push_interval=int(cfg.LOG["save_interval"]),
        komi=float(cfg.KOMI), eval_cfg=eval_cfg, best_model=best_model, eval_interval=eval_interval
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("AlphaZero-style training loop")
    parser.add_argument("--cfg", type=str, required=True, help="configs/zero_9x9.yaml")
    args = parser.parse_args()
    main(args.cfg)
