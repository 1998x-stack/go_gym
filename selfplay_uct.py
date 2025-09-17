# examples/selfplay_uct.py
from __future__ import annotations
import argparse
import random
from typing import Optional, Tuple, List

from go_gym.core.board import Board, BLACK, WHITE, Color, Point
from go_gym.core.scoring import chinese_area_score
from go_gym.agents.uct import GameState, uct_search, action_to_point, point_to_action
from go_gym.utils.sgf import write_sgf


def main():
    parser = argparse.ArgumentParser("UCT Selfplay (Chinese rules, simple-ko)")
    parser.add_argument("--size", type=int, default=19)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--sims", type=int, default=600, help="UCT simulations per move")
    parser.add_argument("--c_puct", type=float, default=1.4)
    parser.add_argument("--rollout_limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_moves", type=int, default=None, help="safety cap")
    parser.add_argument("--sgf_out", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    board = Board(args.size, allow_suicide=False, use_superko=False)
    to_play: Color = BLACK
    pass_count = 0
    last_move: Optional[Point] = None
    moves_sgf: List[Tuple[str, Optional[Point]]] = []

    move_cap = args.max_moves or (args.size * args.size * 2)

    for t in range(move_cap):
        state = GameState(board=board.copy(), to_play=to_play, pass_count=pass_count, komi=args.komi, last_move=last_move)

        # UCT 决策
        action = uct_search(state, n_simulations=args.sims, c_puct=args.c_puct, rollout_limit=args.rollout_limit)
        p = action_to_point(action, args.size)

        # 执行
        color_char = "B" if to_play == BLACK else "W"
        moves_sgf.append((color_char, p))
        if p is None:
            board.play(to_play, None, to_play=to_play)
            pass_count += 1
            last_move = None
        else:
            board.play(to_play, p, to_play=to_play)
            pass_count = 0
            last_move = p

        # 切换
        to_play = BLACK if to_play == WHITE else WHITE

        # 终局条件：两次 pass
        if pass_count >= 2:
            break

    b, w = chinese_area_score(board, args.komi)
    diff = abs(b - w)
    if b > w:
        result = f"B+{diff:.1f}"
    elif w > b:
        result = f"W+{diff:.1f}"
    else:
        result = "0"

    print(f"Final Score — Black: {b:.1f}, White: {w:.1f} (komi {args.komi}); Result: {result}")

    if args.sgf_out:
        sgf_text = write_sgf(moves_sgf, size=args.size, komi=args.komi, result=result, pb="UCT", pw="UCT")
        with open(args.sgf_out, "w", encoding="utf-8") as f:
            f.write(sgf_text)
        print(f"SGF saved to: {args.sgf_out}")


if __name__ == "__main__":
    main()