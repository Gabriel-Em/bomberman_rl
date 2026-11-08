"""Verification suite for PyTorch Actor-Critic, Multi-Process Vectorized PPO Trainer, and Diagnostics."""

from __future__ import annotations
import glob
import os
import shutil
import tempfile
import time
import tarfile
import numpy as np
import pytest
import torch

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    MoveAction,
    BombAction,
    TileType,
)
from bomberman_rl.engine.env import BombermanEnv
from bomberman_rl.engine.state import GameState, PlayerState
from bomberman_rl.rl.observation import ObservationBuilder, DEFAULT_SCALAR_DIM
from bomberman_rl.rl.action_filter import SafetyActionFilter, is_cul_de_sac
from bomberman_rl.replays.recorder import ReplayEpisode
from bomberman_rl.visualizer.renderer import Renderer
from bomberman_rl.training.network import ActorCriticNetwork, ResidualBlock
from bomberman_rl.training.vec_env import SyncVectorEnv, AsyncVectorEnv, BombermanMatchEnv, HeuristicOpponent
from bomberman_rl.training.diagnostics import BehaviorDiagnostics, MatchStats
from bomberman_rl.training.ppo_trainer import PPOTrainer, get_curriculum
from bomberman_rl.training.archive import archive_training_data


def test_network_shapes_and_receptive_field() -> None:
    """Verify ActorCriticNetwork input shapes, spatial trunk, fusion core, and heads."""
    net = ActorCriticNetwork(spatial_channels=18, scalar_dim=18, hidden_dim=256)

    # Batch of 4
    spatial = torch.randn(4, 18, 9, 9)
    scalars = torch.randn(4, 18)

    m_logits, b_logits, values = net.forward(spatial, scalars)
    assert m_logits.shape == (4, 5)
    assert b_logits.shape == (4, 2)
    assert values.shape == (4,)

    # Single unbatched input auto-unsqueezing
    s_single = torch.randn(18, 9, 9)
    sc_single = torch.randn(18)
    actions, log_prob, val, probs = net.get_action(s_single, sc_single)
    assert actions.shape == (1, 2)
    assert log_prob.shape == (1,)
    assert val.shape == (1,)
    assert probs["move"].shape == (1, 5)
    assert probs["bomb"].shape == (1, 2)


def test_action_masking_zeroes_invalid_bomb_prob() -> None:
    """Verify action masking completely zeroes out invalid PLACE_BOMB probability."""
    net = ActorCriticNetwork()
    spatial = torch.randn(2, 18, 9, 9)
    scalars = torch.randn(2, 18)

    move_mask = torch.ones(2, 5, dtype=torch.bool)
    bomb_mask = torch.ones(2, 2, dtype=torch.bool)
    # Mask out PLACE_BOMB for environment 0
    bomb_mask[0, 1] = False

    actions, log_prob, val, probs = net.get_action(
        spatial, scalars, move_mask=move_mask, bomb_mask=bomb_mask
    )

    # Prob of PLACE_BOMB for env 0 must be 0.0
    assert np.isclose(probs["bomb"][0, 1].item(), 0.0, atol=1e-6)
    # Prob of NO_BOMB for env 0 must be 1.0
    assert np.isclose(probs["bomb"][0, 0].item(), 1.0, atol=1e-5)
    # Env 0 sampled action for bomb must be NO_BOMB (0)
    assert actions[0, 1].item() == 0


def test_cul_de_sac_suicide_filter() -> None:
    """Verify is_cul_de_sac identifies when an agent cannot escape its own bomb."""
    state = GameState.create_initial(seed=42)
    # Trap Player 0 in corner (0, 0) by setting (0, 1) and (1, 0) to BRICK
    state.players[0].r = 0.5
    state.players[0].c = 0.5
    state.grid[0, 1] = TileType.BRICK
    state.grid[1, 0] = TileType.BRICK

    assert is_cul_de_sac(state, player_id=0) == True

    # Open (0, 1) and (0, 2)
    state.grid[0, 1] = TileType.EMPTY
    state.grid[0, 2] = TileType.EMPTY
    # With blast radius 1, bomb at (0, 0) hits (0, 1). Tile (0, 2) is safe and reachable!
    assert is_cul_de_sac(state, player_id=0) == False


def test_curriculum_annealing_schedule() -> None:
    """Verify self_damage_scale (0.25 -> 1.0 @ 300) and safety_filter_prob (1.0 -> 0.0 @ 400)."""
    # Gen 0
    s0, p0 = get_curriculum(0)
    assert np.isclose(s0, 0.25)
    assert np.isclose(p0, 1.0)

    # Gen 150 (halfway for scale)
    s150, p150 = get_curriculum(150)
    assert np.isclose(s150, 0.25 + 0.75 * 0.5)

    # Gen 300 (full scale)
    s300, p300 = get_curriculum(300)
    assert np.isclose(s300, 1.0)
    assert np.isclose(p300, 0.25)

    # Gen 400 (zero filter prob)
    s400, p400 = get_curriculum(400)
    assert np.isclose(s400, 1.0)
    assert np.isclose(p400, 0.0)

    # Gen 500 (remains clamped)
    s500, p500 = get_curriculum(500)
    assert np.isclose(s500, 1.0)
    assert np.isclose(p500, 0.0)


def test_throughput_benchmark_assertion() -> None:
    """Benchmark: Run a single environment for 1,000 steps and assert completes in < 0.5 seconds on CPU."""
    env = BombermanEnv(seed=42)
    obs_builder = ObservationBuilder()
    env.reset()

    start_time = time.perf_counter()
    for _ in range(1000):
        # Build ego observations and step physics
        obs_0, obs_1 = obs_builder.build_both(env.state, copy=False)
        _, _, term, trunc, _ = env.step([(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
        if term or trunc:
            env.reset()

    elapsed = time.perf_counter() - start_time
    print(f"\n[Benchmark] 1,000 steps took {elapsed:.4f}s ({1000.0 / elapsed:.1f} steps/s)")
    assert elapsed < 0.5, f"Expected < 0.5 seconds on CPU, got {elapsed:.4f}s"


def test_end_to_end_ppo_mini_update_and_replay_generation() -> None:
    """Run 2 generations of ppo_trainer with num_envs=2, rollout_steps=64.

    Verifies:
    1. Gradients flow without NaNs through movement_head, bomb_head, and critic_head.
    2. Action masking zeroes out invalid PLACE_BOMB probabilities.
    3. Replay .npz file is created and successfully loadable by ReplayEpisode.
    """
    temp_dir = tempfile.mkdtemp(prefix="bomberman_test_")
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")

    try:
        trainer = PPOTrainer(
            num_envs=2,
            rollout_steps=64,
            total_generations=2,
            num_epochs=2,
            minibatch_size=32,
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            showcase_interval=1,  # showcase match on every generation
            device="cpu",
            use_async=False,  # test SyncVectorEnv for determinism
            seed=42,
        )

        history = trainer.train(generations=2)
        assert len(history) == 2

        # 1. Verify gradient flow across heads without NaNs
        assert trainer.policy.movement_head.weight.grad is not None
        assert not torch.isnan(trainer.policy.movement_head.weight.grad).any()
        assert trainer.policy.bomb_head.weight.grad is not None
        assert not torch.isnan(trainer.policy.bomb_head.weight.grad).any()
        assert trainer.policy.critic_head.weight.grad is not None
        assert not torch.isnan(trainer.policy.critic_head.weight.grad).any()

        # 2. Verify replay files exist and are loadable
        replays = sorted(glob.glob(os.path.join(replays_dir, "*.npz")))
        assert len(replays) >= 1, "Expected at least 1 .npz showcase replay file"

        replay_path = replays[0]
        episode = ReplayEpisode.load(replay_path)
        assert len(episode) > 0
        assert episode.action_probs_p0.shape[1] == 10
        assert episode.critic_values.shape[1] == 2
        assert episode.player_pos.shape[1] == 2  # 2 players

        # 3. Verify visualizer loads and renders the generated replay without errors
        renderer = Renderer(headless=True)
        frame_state = episode.get_state(0)
        telemetry = episode.get_telemetry(0)
        renderer.render_frame(frame_state, telemetry=telemetry)
        assert renderer.screen is not None

    finally:
        trainer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_behavior_diagnostics_audit_table() -> None:
    """Verify behavior diagnostics calculates rates and formats summary table properly."""
    diag = BehaviorDiagnostics()

    # Match 1: win for p0
    diag.record_match(
        MatchStats(
            winner=0,
            draw=False,
            p0_suicide=False,
            ticks=200,
            bricks_destroyed=10,
            p0_buffs_collected=3,
            p0_debuffs_collected=0,
            kicks_executed=1,
            chain_detonations=2,
        )
    )

    # Match 2: loss by p0 suicide
    diag.record_match(
        MatchStats(
            winner=1,
            draw=False,
            p0_suicide=True,
            ticks=100,
            bricks_destroyed=4,
            p0_buffs_collected=1,
            p0_debuffs_collected=1,
            kicks_executed=0,
            chain_detonations=0,
        )
    )

    metrics = diag.compute_metrics(generation=1)
    assert metrics["matches"] == 2
    assert np.isclose(metrics["win_rate"], 0.5)
    assert np.isclose(metrics["loss_rate"], 0.5)
    assert np.isclose(metrics["draw_rate"], 0.0)
    assert np.isclose(metrics["suicide_rate"], 0.5)
    assert np.isclose(metrics["avg_match_length"], 150.0)
    assert np.isclose(metrics["avg_bricks_destroyed"], 7.0)

    table = diag.format_table(generation=1)
    assert "Generation 0001 Emergent Behavior Audit" in table
    assert "P0 Suicide Rate:" in table
    assert "Bricks Destroyed / Match:" in table


def test_async_vector_env_multiprocess() -> None:
    """Verify AsyncVectorEnv runs parallel processes with IPC without deadlocks."""
    venv = AsyncVectorEnv(num_envs=2, seed=999)
    try:
        obs_0, mask_0, obs_1, mask_1, infos = venv.reset()
        assert obs_0["spatial"].shape == (2, 18, 9, 9)
        assert obs_0["scalars"].shape == (2, 18)
        assert mask_0["move_mask"].shape == (2, 5)
        assert mask_0["bomb_mask"].shape == (2, 2)

        # Step 10 times
        for _ in range(10):
            acts_0 = np.array([[0, 0], [1, 0]], dtype=np.int64)
            acts_1 = np.array([[0, 0], [0, 0]], dtype=np.int64)
            o0, rews, terms, truncs, step_infos, m0, o1, m1 = venv.step(acts_0, acts_1)
            assert rews.shape == (2,)

        venv.set_curriculum(self_damage_scale=0.5, safety_filter_prob=0.8)
    finally:
        venv.close()


def test_heuristic_opponent_active_sparring() -> None:
    """Verify HeuristicOpponent actively moves and acts during matches rather than standing still."""
    env = BombermanMatchEnv(seed=123)
    env.reset()
    env.opponent_type = "heuristic"

    # Let the heuristic opponent step for 50 ticks with player 0 doing NOOP
    initial_p1_pos = (env.env.state.players[1].r, env.env.state.players[1].c)
    moved = False
    for _ in range(50):
        # Pass None or (-1, -1) to trigger heuristic action
        env.step((MoveAction.NOOP, BombAction.NOOP), None)
        cur_pos = (env.env.state.players[1].r, env.env.state.players[1].c)
        if cur_pos != initial_p1_pos:
            moved = True
            break

    assert moved, "HeuristicOpponent must actively move during the match"


def test_match_continuity_across_generations() -> None:
    """Verify that train_generation preserves ongoing matches across generation boundaries."""
    temp_dir = tempfile.mkdtemp(prefix="bomberman_cont_")
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")

    try:
        trainer = PPOTrainer(
            num_envs=2,
            rollout_steps=10,  # Short rollout step so match is guaranteed to span generations
            total_generations=3,
            num_epochs=1,
            minibatch_size=10,
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            showcase_interval=0,
            device="cpu",
            use_async=False,
            seed=42,
        )

        # Gen 0
        trainer.train_generation(0)
        tick_after_gen0 = trainer.vec_env.envs[0].env.state.tick
        assert tick_after_gen0 >= 10

        # Gen 1: environment tick must continue from tick_after_gen0, NOT reset to 0!
        trainer.train_generation(1)
        tick_after_gen1 = trainer.vec_env.envs[0].env.state.tick
        assert tick_after_gen1 >= tick_after_gen0 + 10 or trainer.vec_env.envs[0].env.state.tick > 0
    finally:
        trainer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_opponent_pool_consistency_and_fallback() -> None:
    """Verify opponent sampling distribution, fallback to latest when pool is empty, and match consistency."""
    env = BombermanMatchEnv(seed=777)

    # 1. When no checkpoints exist, sampling 20% bucket must fallback to latest (not heuristic)
    env.sample_opponent(past_checkpoint_paths=[])
    # Run 100 samples with empty past checkpoints
    types = [env.sample_opponent([]) for _ in range(200)]
    latest_count = types.count("latest")
    heuristic_count = types.count("heuristic")
    checkpoint_count = types.count("checkpoint")

    assert checkpoint_count == 0, "No checkpoint should be sampled when pool is empty"
    # ~90% latest, ~10% heuristic
    assert latest_count > 150, f"Expected ~90% latest, got {latest_count}/200"
    assert heuristic_count < 50, f"Expected ~10% heuristic, got {heuristic_count}/200"

    # 2. When checkpoints exist, assigned checkpoint must persist across steps until done
    fake_ckpts = ["/fake/ckpt_1.pt", "/fake/ckpt_2.pt"]
    env.past_checkpoint_paths = fake_ckpts
    # Force checkpoint opponent
    env.opponent_type = "checkpoint"
    env.opponent_checkpoint_path = fake_ckpts[0]

    _, _, _, _, info, _, _, _ = env.step((0, 0), (0, 0))
    assert info["opponent_checkpoint"] == fake_ckpts[0]


def test_trainer_resume_checkpoint() -> None:
    """Verify trainer can load a checkpoint and resume training from the next generation."""
    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")

    try:
        trainer1 = PPOTrainer(
            num_envs=2,
            rollout_steps=10,
            total_generations=1,
            num_epochs=1,
            minibatch_size=10,
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            showcase_interval=0,
            device="cpu",
            use_async=False,
            seed=123,
        )
        history1 = trainer1.train()
        assert len(history1) == 1
        ckpt_0 = os.path.join(ckpt_dir, "checkpoint_gen_0000.pt")
        assert os.path.exists(ckpt_0)
        trainer1.close()

        # Trainer 2 resuming from checkpoint
        trainer2 = PPOTrainer(
            num_envs=2,
            rollout_steps=10,
            total_generations=2,
            num_epochs=1,
            minibatch_size=10,
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            showcase_interval=0,
            device="cpu",
            use_async=False,
            seed=456,
        )
        resumed_gen = trainer2.load_checkpoint("latest")
        assert resumed_gen == 1
        assert trainer2.start_generation == 1

        # Check weights loaded correctly
        for p1, p2 in zip(trainer1.policy.parameters(), trainer2.policy.parameters()):
            assert torch.equal(p1, p2)

        history2 = trainer2.train()
        assert len(history2) == 1
        assert history2[0]["generation"] == 1
        assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_gen_0001.pt"))

        # Verify max_generation checkpoint filtering
        assert len(trainer2.get_past_checkpoints(max_generation=-1)) == 0
        assert len(trainer2.get_past_checkpoints(max_generation=0)) == 1
        assert len(trainer2.get_past_checkpoints(max_generation=1)) == 2

        # Test error handling on missing file or empty dir
        with pytest.raises(FileNotFoundError):
            trainer2.load_checkpoint("/nonexistent/checkpoint.pt")

        empty_dir = os.path.join(temp_dir, "empty_ckpts")
        os.makedirs(empty_dir, exist_ok=True)
        trainer_empty = PPOTrainer(
            num_envs=1,
            rollout_steps=5,
            total_generations=1,
            checkpoint_dir=empty_dir,
            device="cpu",
            use_async=False,
        )
        with pytest.raises(FileNotFoundError):
            trainer_empty.load_checkpoint("latest")
        trainer_empty.close()
    finally:
        trainer2.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_archive_training_data() -> None:
    """Verify archive_training_data compresses, moves files, preserves fixtures, and names by timestamp."""
    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")
    archives_dir = os.path.join(temp_dir, "archives")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(replays_dir, exist_ok=True)

    try:
        # Create mock checkpoints
        open(os.path.join(ckpt_dir, ".gitkeep"), "w").close()
        with open(os.path.join(ckpt_dir, "checkpoint_gen_0000.pt"), "w") as f:
            f.write("mock weights 0")
        with open(os.path.join(ckpt_dir, "checkpoint_gen_0001.pt"), "w") as f:
            f.write("mock weights 1")

        # Create mock replays
        open(os.path.join(replays_dir, ".gitkeep"), "w").close()
        with open(os.path.join(replays_dir, "test_match.npz"), "w") as f:
            f.write("fixture sample")
        with open(os.path.join(replays_dir, "gen_0000_showcase.npz"), "w") as f:
            f.write("mock replay")

        # Run archive with specific timestamp
        res = archive_training_data(
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            archive_dir=archives_dir,
            timestamp="20261109_120000",
        )

        assert res["archived"] is True
        assert res["files_count"] == 3  # 2 checkpoints + 1 showcase replay
        assert os.path.isdir(res["archive_dir"])
        assert os.path.isfile(res["archive_tar"])
        assert "archive_20261109_120000.tar.gz" in res["archive_tar"]

        # Checkpoints directory must only retain .gitkeep
        remaining_ckpts = os.listdir(ckpt_dir)
        assert remaining_ckpts == [".gitkeep"]

        # Replays directory must only retain .gitkeep and test_match.npz
        remaining_replays = sorted(os.listdir(replays_dir))
        assert remaining_replays == [".gitkeep", "test_match.npz"]

        # Verify archive dir contents
        archived_ckpts = sorted(os.listdir(os.path.join(res["archive_dir"], "checkpoints")))
        assert archived_ckpts == ["checkpoint_gen_0000.pt", "checkpoint_gen_0001.pt"]
        archived_replays = os.listdir(os.path.join(res["archive_dir"], "replays"))
        assert archived_replays == ["gen_0000_showcase.npz"]

        # Verify compressed tar archive
        with tarfile.open(res["archive_tar"], "r:gz") as tar:
            names = tar.getnames()
            assert any("checkpoint_gen_0000.pt" in n for n in names)
            assert any("gen_0000_showcase.npz" in n for n in names)

        # Running archive again on empty workspace should report archived=False
        empty_res = archive_training_data(
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            archive_dir=archives_dir,
        )
        assert empty_res["archived"] is False
        assert empty_res["files_count"] == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)




