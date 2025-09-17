import numpy as np
from go_gym import GoChineseEnv

env = GoChineseEnv(board_size=19, komi=7.5, render_mode="ansi")
obs, info = env.reset(seed=42)
print("legal moves:", info["action_mask"].sum(), "of", info["action_mask"].size)

# 随机下几手
for t in range(10):
    action = int(np.random.choice(np.where(info["action_mask"]==1)[0]))
    obs, reward, done, trunc, info = env.step(action)
    print(env.render())
    if done:
        print("Game end. reward=", reward, "info:", info)
        break

# 在 Python REPL 里快速 sanity check
from go_gym.agents.zero_mcts import ZeroMCTS, GameState
from go_gym.nn.policy_value_net import PolicyValueNet
from go_gym.core.board import Board, BLACK
import torch, numpy as np

net = PolicyValueNet(planes=8, channels=64, blocks=3, board_size=9).eval()
dev = torch.device('cpu')
mcts = ZeroMCTS(net, dev, board_size=9, n_simulations=8)

state = GameState(board=Board(9), to_play=BLACK, pass_count=0, komi=7.5)
pi, a = mcts.run(state, temperature=1.0, add_root_noise=True)
assert np.isclose(pi.sum(), 1.0), "pi not normalized"
print("OK, pi shape:", pi.shape, "action:", a)
