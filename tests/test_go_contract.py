"""Regression tests for board legality, Gymnasium episodes and MCTS decisions."""
import unittest

import numpy as np
import torch

from go_gym import GoChineseEnv
from go_gym.core.board import BLACK, WHITE, EMPTY, Board
from go_gym.agents.zero_mcts import GameState, Node, ZeroMCTS


def ko_board():
    """A 5x5 simple ko: black captures at (1,2); white cannot recapture (1,1)."""
    board = Board(5)
    for point in ((0, 1), (1, 0), (2, 1)):
        board.set(point, BLACK)
    for point in ((1, 1), (0, 2), (2, 2), (1, 3)):
        board.set(point, WHITE)
    return board


class BoardContractTests(unittest.TestCase):
    def test_wrong_side_cannot_play_or_pass(self):
        board = Board(5)
        self.assertFalse(board.is_legal(WHITE, None, to_play=BLACK))
        self.assertFalse(board.is_legal(WHITE, (0, 0), to_play=BLACK))
        self.assertFalse(board.is_legal(7, None, to_play=7))
        with self.assertRaises(ValueError):
            board.play(WHITE, (0, 0), to_play=BLACK)
        self.assertEqual(board.stones_count(BLACK) + board.stones_count(WHITE), 0)

    def test_capture_and_simple_ko_mask(self):
        board = ko_board()
        self.assertTrue(board.is_legal(BLACK, (1, 2), to_play=BLACK))
        self.assertEqual(board.play(BLACK, (1, 2), to_play=BLACK), 1)
        self.assertEqual(board.get((1, 1)), EMPTY)
        self.assertFalse(board.is_legal(WHITE, (1, 1), to_play=WHITE))
        self.assertTrue(board.is_legal(WHITE, None, to_play=WHITE))

    def test_suicide_rejected(self):
        board = Board(3)
        for p in ((0, 1), (1, 0), (1, 2), (2, 1)):
            board.set(p, WHITE)
        self.assertFalse(board.is_legal(BLACK, (1, 1), to_play=BLACK))


class EnvironmentContractTests(unittest.TestCase):
    def test_reset_observation_mask_and_seed(self):
        env = GoChineseEnv(board_size=3, komi=0.5)
        obs, info = env.reset(seed=2026)
        self.assertEqual(obs.shape, (4, 3, 3))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        np.testing.assert_array_equal(info["action_mask"], np.ones(10, dtype=np.int8))
        self.assertTrue((obs[2] == 1).all())
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["action_mask"][0], 0)
        self.assertEqual(info["action_mask"][-1], 1)
        self.assertTrue((obs[2] == 0).all())
        self.assertEqual(obs[3, 0, 0], 1)
        obs2, info2 = env.reset(seed=2026)
        self.assertEqual(env.steps, 0)
        self.assertTrue((obs2[2] == 1).all())
        self.assertEqual(int(info2["action_mask"].sum()), 10)

    def test_two_passes_score_and_post_terminal_guard(self):
        env = GoChineseEnv(board_size=3, komi=0.5)
        env.reset(seed=2026)
        env.step(9)
        obs, reward, terminated, truncated, info = env.step(9)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)  # empty board, white wins by komi
        self.assertEqual(info["terminated_reason"], "two_passes")
        self.assertEqual(info["winner"], "white")
        self.assertEqual(info["score_black"], 0.0)
        self.assertEqual(info["score_white"], 0.5)
        self.assertFalse(info["action_mask"].any())
        with self.assertRaises(RuntimeError):
            env.step(9)
        env.reset()
        self.assertTrue(env.legal_action_mask().all())

    def test_illegal_action_freezes_position(self):
        env = GoChineseEnv(board_size=3)
        env.reset()
        env.step(0)
        before = env.board.grid.copy()
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)
        self.assertTrue(info["illegal_move"])
        self.assertFalse(info["action_mask"].any())
        np.testing.assert_array_equal(env.board.grid, before)
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_move_limit_is_truncation_not_scored_victory(self):
        env = GoChineseEnv(board_size=3, komi=0.5, max_steps=1)
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["terminated_reason"], "max_steps")
        self.assertNotIn("winner", info)
        self.assertFalse(info["action_mask"].any())
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_out_of_range_action_rejected(self):
        env = GoChineseEnv(board_size=3)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(100)
        self.assertEqual(env.steps, 0)


class ConstantNet(torch.nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.action_count = action_count

    def forward(self, x):
        return (torch.zeros((x.shape[0], self.action_count), device=x.device),
                torch.zeros((x.shape[0], 1), device=x.device))


class MCTSContractTests(unittest.TestCase):
    def test_white_selects_highest_own_perspective_q(self):
        node = Node(WHITE)
        node.legal_actions = [0, 1]
        node.P = {0: 0.5, 1: 0.5}
        node.N = {0: 10, 1: 10}
        node.Q = {0: 0.7, 1: -0.6}
        mcts = ZeroMCTS(ConstantNet(10), torch.device("cpu"), board_size=3)
        self.assertEqual(mcts._select_puct(node), 0)

    def test_ko_history_prevents_stale_cached_legal_mask(self):
        board = ko_board()
        board.play(BLACK, (1, 2), to_play=BLACK)
        restricted = GameState(board, WHITE, 0, 0.5, last_move=(1, 2))
        unrestricted_board = board.copy()
        unrestricted_board.history.clear()
        unrestricted = GameState(unrestricted_board, WHITE, 0, 0.5, last_move=(1, 2))
        mcts = ZeroMCTS(ConstantNet(26), torch.device("cpu"), board_size=5)
        restricted_prior, _ = mcts._eval_policy_value(restricted)
        unrestricted_prior, _ = mcts._eval_policy_value(unrestricted)
        recapture = 1 * 5 + 1
        self.assertEqual(float(restricted_prior[recapture]), 0.0)
        self.assertGreater(float(unrestricted_prior[recapture]), 0.0)
        self.assertEqual(len(mcts.cache), 2)

    def test_superko_never_reuses_incomplete_cache_key(self):
        board = Board(3, use_superko=True)
        state = GameState(board, BLACK, 0, 0.5)
        mcts = ZeroMCTS(ConstantNet(10), torch.device("cpu"), board_size=3)
        mcts._eval_policy_value(state)
        self.assertEqual(len(mcts.cache), 0)

    def test_short_search_returns_legal_probability_distribution(self):
        board = Board(3)
        state = GameState(board, BLACK, 0, 0.5)
        mcts = ZeroMCTS(ConstantNet(10), torch.device("cpu"), board_size=3,
                        n_simulations=3, dirichlet_epsilon=0.0)
        policy, action = mcts.run(state, temperature=1.0, add_root_noise=False)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertIn(action, list(range(10)))


if __name__ == "__main__":
    unittest.main()
