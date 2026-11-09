# go_gym

> 围棋（Go / Baduk / Weiqi）Gymnasium 环境与 MCTS / UCT 研究原型。

## 功能与边界

- `GoChineseEnv`：中国面积计分、贴目、禁自杀、普通劫禁（可选超级劫）、双停结束。
- 行动：`Discrete(size * size + 1)`；末位为 pass；`info["action_mask"]` 与当前局面的合法手一致。
- 观察：4 × H × W 的 `float32` 平面（己方、对方、当前执子方、最后落子点）。
- **奖励含义：** 双停正常终局时为**黑方视角**的 +1 / -1 / 0；非法手立即结束并返回 -1 处罚（当前行棋方失误）；纯步数预算耗尽时 `truncated=True, reward=0`，不是按中途棋面判断胜负。
- 正常结束、非法动作结束和时间截断后所有动作掩码元素均为 0，继续调用 `step` 将报错，需先 `reset`。
- MCTS 的各节点 `Q` 已是该节点行棋方视角，白方选择不能再额外取反。policy/value 缓存区分普通劫历史与最后落子点；超级劫时不使用不完整状态缓存。

**限制：** 这不是浏览器游戏；自动化测试与短局自对弈不等于九路/十九路实力评估或深度网络训练已复现。终局数子采用简化连通空域判断，死子协商/完整中国规则争议并未实现。

## 目录

```text
go_gym/
├── go_gym/
│   ├── core/                 # 棋盘合法性、提子、面积数子
│   ├── envs/                 # Gymnasium 环境
│   ├── agents/               # UCT / 神经网络 MCTS
│   ├── nn/ train/ data/      # 模型与训练组件
│   └── utils/
├── tests/test_go_contract.py # 固定棋形和 MCTS 回归
├── selfplay_uct.py
├── .github/workflows/go-contract.yml
└── install.sh
```

## CPU 环境与可复现运行

本仓库目前**没有 Python 包构建元数据**，请在仓库根目录运行脚本，不使用 `pip install -e .`：

```bash
python -m pip install numpy gymnasium loguru
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python selfplay_uct.py --size 3 --sims 2 --rollout_limit 6 --max_moves 12 --seed 2026
```

若要使用训练循环，请另按 `install.sh` 安装 `pyyaml`、`tensorboard` 等依赖并核查训练配置。上述验收只针对棋盘规则、Gymnasium episode 契约、简单劫、MCTS 选择与缓存、以及固定随机种子的短 UCT 对局。

## 证据

[GitHub Actions · Go contract](https://github.com/1998x-stack/go_gym/actions/workflows/go-contract.yml) 会运行代码语法检查、单元回归与无界面短局。只有工作流实际成功之后才能将状态标为“CI 通过”；真正的全规模训练与 19×19 对局仍待验收。

## License

MIT（保留原项目许可证声明；依赖与第三方模型资产请分别核查）。
