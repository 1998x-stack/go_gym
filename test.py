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
