# go_gym

> 围棋 Gymnasium 强化学习环境。

## 📌 项目概述

`go_gym` 是一个**围棋（Go / Baduk / Weiqi）**的强化学习环境，遵循 Gymnasium 接口规范，包含自对弈 / UCT 搜索等玩法实现，可用于训练围棋 AI 与相关研究。

## 🏗️ 核心结构

```
go_gym/
├─ go_gym/          # 环境主实现
│  ├─ envs/         # 环境定义
│  ├─ agents/       # 智能体
│  ├─ core/、nn/、train/、utils/、data/
├─ selfplay_uct.py  # 自对弈 + UCT 搜索
├─ configs/         # 配置
├─ test.py          # 测试
└─ install.sh       # 安装脚本
```

## 🚀 快速开始

```bash
bash install.sh
# 或
pip install -e .
python selfplay_uct.py
```

## 📄 License

MIT