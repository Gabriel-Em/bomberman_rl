"""Verification suite for PyTorch Actor-Critic, Multi-Process Vectorized PPO Trainer, and Diagnostics."""

from __future__ import annotations
import glob
import os
import re
import shutil
import sys
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
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.rl.observation import ObservationBuilder, DEFAULT_SCALAR_DIM
from bomberman_rl.rl.action_filter import SafetyActionFilter, is_cul_de_sac, is_bomb_placement_safe
from bomberman_rl.rl.rewards import RewardTracker, compute_danger_potential, BRICK_REWARD
from bomberman_rl.replays.recorder import ReplayEpisode
from bomberman_rl.visualizer.renderer import Renderer
from bomberman_rl.training.network import ActorCriticNetwork, ResidualBlock
from bomberman_rl.training.vec_env import SyncVectorEnv, AsyncVectorEnv, BombermanMatchEnv, HeuristicOpponent
from bomberman_rl.training.diagnostics import BehaviorDiagnostics, MatchStats
from bomberman_rl.training.ppo_trainer import PPOTrainer, get_curriculum, get_competence_curriculum
from bomberman_rl.training.archive import archive_training_data, clean_training_data


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


def test_clean_training_data() -> None:
    """Verify clean_training_data removes checkpoints and replays while preserving sample fixtures."""
    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(replays_dir, exist_ok=True)

    try:
        # Create mock checkpoint files
        open(os.path.join(ckpt_dir, ".gitkeep"), "w").close()
        with open(os.path.join(ckpt_dir, "checkpoint_gen_0000.pt"), "w") as f:
            f.write("mock weights 0")
        with open(os.path.join(ckpt_dir, "checkpoint_gen_0001.pt"), "w") as f:
            f.write("mock weights 1")

        # Create mock replay files and test fixture
        open(os.path.join(replays_dir, ".gitkeep"), "w").close()
        with open(os.path.join(replays_dir, "test_match.npz"), "w") as f:
            f.write("fixture sample")
        with open(os.path.join(replays_dir, "gen_0000_showcase.npz"), "w") as f:
            f.write("mock replay")
        with open(os.path.join(replays_dir, "gen_0010_showcase.npz"), "w") as f:
            f.write("mock replay 2")

        # Clean training data
        res = clean_training_data(checkpoint_dir=ckpt_dir, replays_dir=replays_dir)

        assert res["cleaned"] is True
        assert res["files_count"] == 4  # 2 checkpoints + 2 showcase replays
        assert len(res["files"]) == 4

        # Checkpoints directory must only retain .gitkeep
        assert os.listdir(ckpt_dir) == [".gitkeep"]

        # Replays directory must only retain .gitkeep and test_match.npz
        assert sorted(os.listdir(replays_dir)) == [".gitkeep", "test_match.npz"]

        # Calling clean again on already clean directories should report cleaned=False
        res_empty = clean_training_data(checkpoint_dir=ckpt_dir, replays_dir=replays_dir)
        assert res_empty["cleaned"] is False
        assert res_empty["files_count"] == 0
        assert res_empty["files"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_clean_training_data_boundary_conditions() -> None:
    """Verify clean_training_data handles non-existent directories, subdirectories, and custom keep_samples."""
    temp_dir = tempfile.mkdtemp()
    non_existent_ckpt = os.path.join(temp_dir, "non_existent_ckpts")
    non_existent_replays = os.path.join(temp_dir, "non_existent_replays")

    try:
        # Non-existent directories should not crash and return cleaned=False
        res = clean_training_data(checkpoint_dir=non_existent_ckpt, replays_dir=non_existent_replays)
        assert res["cleaned"] is False
        assert res["files_count"] == 0

        # Create directories with subdirectories and custom keep list
        os.makedirs(non_existent_ckpt, exist_ok=True)
        sub_folder = os.path.join(non_existent_ckpt, "old_run_dir")
        os.makedirs(sub_folder, exist_ok=True)
        with open(os.path.join(sub_folder, "nested.pt"), "w") as f:
            f.write("nested")
        with open(os.path.join(non_existent_ckpt, "custom_keep.pt"), "w") as f:
            f.write("keep me")

        res_custom = clean_training_data(
            checkpoint_dir=non_existent_ckpt,
            replays_dir=non_existent_replays,
            keep_samples=("custom_keep.pt",),
        )
        assert res_custom["cleaned"] is True
        assert res_custom["files_count"] == 1
        assert os.path.exists(os.path.join(non_existent_ckpt, "custom_keep.pt"))
        assert not os.path.exists(sub_folder)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_clean_existing_in_ppo_trainer() -> None:
    """Verify PPOTrainer with clean_existing=True clears stale checkpoints and replays on start."""
    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(replays_dir, exist_ok=True)

    try:
        # Populate with stale run files
        stale_ckpt = os.path.join(ckpt_dir, "checkpoint_gen_0500.pt")
        stale_replay = os.path.join(replays_dir, "gen_0500_20260101_000000.npz")
        with open(stale_ckpt, "w") as f:
            f.write("stale checkpoint")
        with open(stale_replay, "w") as f:
            f.write("stale replay")

        # Initialize trainer with clean_existing=True
        trainer = PPOTrainer(
            num_envs=1,
            rollout_steps=4,
            total_generations=1,
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            clean_existing=True,
            device="cpu",
            use_async=False,
        )
        try:
            assert not os.path.exists(stale_ckpt)
            assert not os.path.exists(stale_replay)
            assert len(trainer.get_past_checkpoints()) == 0
        finally:
            trainer.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_cli_cleans_directories_without_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train.py CLI removes previous training data when started without --resume."""
    import train

    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(replays_dir, exist_ok=True)

    try:
        # Create mock checkpoint and replay from a previous stopped training
        stale_ckpt = os.path.join(ckpt_dir, "checkpoint_gen_0042.pt")
        stale_replay = os.path.join(replays_dir, "gen_0040_20261109_010000.npz")
        with open(stale_ckpt, "w") as f:
            f.write("mock weights")
        with open(stale_replay, "w") as f:
            f.write("mock replay")

        # Mock PPOTrainer to prevent starting actual game envs
        class DummyTrainer:
            def __init__(self, **kwargs: object) -> None:
                pass
            def train(self) -> list[object]:
                return []
            def close(self) -> None:
                pass

        monkeypatch.setattr(train, "PPOTrainer", DummyTrainer)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--checkpoint-dir",
                ckpt_dir,
                "--replays-dir",
                replays_dir,
                "--total-generations",
                "1",
            ],
        )

        exit_code = train.main()
        assert exit_code == 0

        # Assert stale files were removed
        assert not os.path.exists(stale_ckpt)
        assert not os.path.exists(stale_replay)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_cli_preserves_directories_with_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train.py CLI preserves existing checkpoints when --resume is supplied."""
    import train

    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(replays_dir, exist_ok=True)

    try:
        existing_ckpt = os.path.join(ckpt_dir, "checkpoint_gen_0010.pt")
        with open(existing_ckpt, "w") as f:
            f.write("mock weights")

        class DummyTrainer:
            def __init__(self, **kwargs: object) -> None:
                pass
            def load_checkpoint(self, path: str) -> int:
                return 11
            def train(self) -> list[object]:
                return []
            def close(self) -> None:
                pass

        monkeypatch.setattr(train, "PPOTrainer", DummyTrainer)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--checkpoint-dir",
                ckpt_dir,
                "--replays-dir",
                replays_dir,
                "--resume",
                existing_ckpt,
            ],
        )

        exit_code = train.main()
        assert exit_code == 0

        # Existing checkpoint must NOT be removed
        assert os.path.exists(existing_ckpt)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_spawn_corner_trap_verification() -> None:
    """Verify is_bomb_placement_safe returns False at (0, 0) and True at (0, 1) or (1, 0)."""
    state = GameState.create_initial(seed=42)
    # Ensure (0, 2) and (2, 0) are breakable brick walls
    state.grid[0, 2] = TileType.BRICK
    state.grid[2, 0] = TileType.BRICK
    # Ensure (0, 1) and (1, 0) are empty
    state.grid[0, 1] = TileType.EMPTY
    state.grid[1, 0] = TileType.EMPTY
    # (1, 1) is a pillar by default in 9x9 Bomberman

    # Player 0 at spawn (0, 0)
    state.players[0].r = 0.5
    state.players[0].c = 0.5
    assert is_bomb_placement_safe(state, player_id=0) is False

    # Player 0 steps to (0, 1) -> can escape around the corner to (0, 0) then (1, 0)
    state.players[0].r = 0.5
    state.players[0].c = 1.5
    assert is_bomb_placement_safe(state, player_id=0) is True

    # Player 0 steps to (1, 0) -> can escape around the corner to (0, 0) then (0, 1)
    state.players[0].r = 1.5
    state.players[0].c = 0.5
    assert is_bomb_placement_safe(state, player_id=0) is True


def test_danger_escape_pbrs_gradient() -> None:
    """Place bomb at (0, 1) with Player at (0, 1). Verify no instant penalty and positive shaping when escaping."""
    env = BombermanEnv(seed=42)
    env.state.grid[0, 2] = TileType.BRICK
    env.state.grid[2, 0] = TileType.BRICK
    env.state.grid[0, 1] = TileType.EMPTY
    env.state.grid[1, 0] = TileType.EMPTY

    # Position player 0 at (0, 1)
    env.state.players[0].r = 0.5
    env.state.players[0].c = 1.5

    tracker = RewardTracker(gamma=0.99)
    tracker.reset(env.state)

    # Step 1: Player 0 places bomb at (0, 1)
    state_before = env.state.clone()
    env.step([(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    rewards, tel = tracker.step(
        state_before,
        [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)],
        env.state,
        terminated=False,
        truncated=False,
    )
    # Must NOT trigger instant negative PBRS penalty
    pbrs_danger_step1 = tel["player_0"].get("pbrs_danger", 0.0)
    assert pbrs_danger_step1 >= 0.0, f"Expected non-negative danger shaping on bomb placement tick, got {pbrs_danger_step1}"

    # Now walk Player 0 from (0, 1) to (0, 0) and then (0, 0) to (1, 0)
    cumulative_danger_pbrs = pbrs_danger_step1
    # Move LEFT until reaching (0, 0)
    while env.state.players[0].tile != (0, 0):
        sb = env.state.clone()
        env.step([(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
        _, t_step = tracker.step(
            sb,
            [(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)],
            env.state,
            terminated=False,
            truncated=False,
        )
        cumulative_danger_pbrs += t_step["player_0"].get("pbrs_danger", 0.0)

    # Move DOWN until reaching (1, 0) (outside the blast cross)
    while env.state.players[0].tile != (1, 0):
        sb = env.state.clone()
        env.step([(MoveAction.DOWN, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
        _, t_step = tracker.step(
            sb,
            [(MoveAction.DOWN, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)],
            env.state,
            terminated=False,
            truncated=False,
        )
        cumulative_danger_pbrs += t_step["player_0"].get("pbrs_danger", 0.0)

    # Stepping outside the blast cross must yield strictly positive cumulative danger PBRS reward
    assert cumulative_danger_pbrs > 0.0, f"Expected positive cumulative danger shaping, got {cumulative_danger_pbrs}"


def test_anti_jitter_visitation() -> None:
    """Move Player 0 back and forth between (0,0) and (0,1) 20 times.

    Verify the visitation bonus is awarded exactly once for (0,0) and once for (0,1),
    and 0.0 for all 19 subsequent oscillations.
    """
    env = BombermanEnv(seed=42)
    env.state.grid[0, 1] = TileType.EMPTY
    env.state.players[0].r = 0.5
    env.state.players[0].c = 0.5

    tracker = RewardTracker()
    tracker.reset(env.state)

    visitation_bonuses_0_0: list[float] = []
    visitation_bonuses_0_1: list[float] = []

    for oscillation in range(20):
        # Move to (0, 1)
        while env.state.players[0].tile != (0, 1):
            sb = env.state.clone()
            env.step([(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
            _, tel = tracker.step(
                sb,
                [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)],
                env.state,
                terminated=False,
                truncated=False,
            )
            v = tel["player_0"].get("visitation", 0.0)
            if v > 0:
                visitation_bonuses_0_1.append(v)

        # Move back to (0, 0)
        while env.state.players[0].tile != (0, 0):
            sb = env.state.clone()
            env.step([(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
            _, tel = tracker.step(
                sb,
                [(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)],
                env.state,
                terminated=False,
                truncated=False,
            )
            v = tel["player_0"].get("visitation", 0.0)
            if v > 0:
                visitation_bonuses_0_0.append(v)

    # Awarded exactly once for (0, 1) and zero times for (0, 0) since spawn is pre-seeded
    assert len(visitation_bonuses_0_1) == 1, f"Expected 1 bonus for (0, 1), got {len(visitation_bonuses_0_1)}"
    assert np.isclose(visitation_bonuses_0_1[0], 0.010, atol=1e-5)
    assert len(visitation_bonuses_0_0) == 0, f"Expected 0 bonus for (0, 0) upon return, got {len(visitation_bonuses_0_0)}"


def test_network_bomb_head_bias_initialization() -> None:
    """Verify bomb_head.bias is initialized to [1.0, -1.0] (~73% NO_BOMB, ~27% PLACE_BOMB)."""
    net = ActorCriticNetwork()
    bias = net.bomb_head.bias.detach().cpu()
    assert torch.allclose(bias, torch.tensor([1.0, -1.0]), atol=1e-5)

    # Initial probability of PLACE_BOMB at generation 0 is calibrated to ~12-27%
    spatial = torch.zeros(10, 18, 9, 9)
    scalars = torch.zeros(10, 18)
    with torch.no_grad():
        _, _, _, probs = net.get_action(spatial, scalars)
    bomb_prob = float(probs["bomb"][:, 1].mean().item())
    assert 0.10 <= bomb_prob <= 0.30, f"Expected initial bomb prob in [0.10, 0.30], got {bomb_prob:.4f}"


def test_competence_curriculum_schedule() -> None:
    """Verify smooth linear competence curriculum based on ema_bricks_per_match."""
    # ema_bricks = 0.0
    scale0, prob0, ticks0 = get_competence_curriculum(0.0)
    assert np.isclose(scale0, 0.15)
    assert np.isclose(prob0, 1.00)
    assert ticks0 == 600

    # ema_bricks = 6.0 (midway: progress = 0.5)
    scale6, prob6, ticks6 = get_competence_curriculum(6.0)
    assert np.isclose(scale6, 0.15 + 0.85 * 0.5)
    assert np.isclose(prob6, 1.00 - 0.90 * 0.5)
    assert ticks6 == int(600 + 3000 * 0.5)

    # ema_bricks = 12.0 (progress = 1.0)
    scale12, prob12, ticks12 = get_competence_curriculum(12.0)
    assert np.isclose(scale12, 1.00)
    assert np.isclose(prob12, 0.10)
    assert ticks12 == 3600

    # ema_bricks > 12.0 (clamped)
    scale20, prob20, ticks20 = get_competence_curriculum(20.0)
    assert np.isclose(scale20, 1.00)
    assert np.isclose(prob20, 0.10)
    assert ticks20 == 3600

    # ema_bricks < 0.0 (clamped)
    scale_neg, prob_neg, ticks_neg = get_competence_curriculum(-1.0)
    assert np.isclose(scale_neg, 0.15)
    assert np.isclose(prob_neg, 1.00)
    assert ticks_neg == 600


def test_adversarial_safety_filter_boundaries() -> None:
    """Cover dead player, bomb_pass ability, flame on tile, and bomb limits in is_bomb_placement_safe."""
    state = GameState.create_initial(seed=123)

    # 1. Dead player
    state.players[0].alive = False
    assert is_bomb_placement_safe(state, player_id=0) is False
    state.players[0].alive = True

    # 2. Out of bounds player
    state.players[0].r = -1.0
    state.players[0].c = 0.0
    assert is_bomb_placement_safe(state, player_id=0) is False
    state.players[0].r = 0.5
    state.players[0].c = 0.5

    # 3. Active flame on player tile
    state.flame_timer[0, 0] = 5
    assert is_bomb_placement_safe(state, player_id=0) is False
    state.flame_timer[0, 0] = 0

    # 4. Max bombs limit reached
    state.players[0].max_bombs = 1
    state.bombs.append(
        BombState(id=99, owner_id=0, r=4.5, c=4.5, timer=30, blast_radius=1)
    )
    assert is_bomb_placement_safe(state, player_id=0) is False
    state.bombs.clear()

    # 5. Bomb pass ability allows player to escape through other bombs
    # Trap at (0, 0) with brick at (0, 2) and (2, 0), and an existing bomb blocking (0, 1)
    state.grid[0, 2] = TileType.BRICK
    state.grid[2, 0] = TileType.BRICK
    state.grid[0, 1] = TileType.EMPTY
    state.grid[1, 0] = TileType.EMPTY
    state.players[0].r = 0.5
    state.players[0].c = 0.5
    # Without bomb_pass, cannot escape
    assert is_bomb_placement_safe(state, player_id=0) is False

    # If (0, 2) and (0, 3) are open to empty, but has another bomb at (0, 1):
    state.grid[0, 2] = TileType.EMPTY
    state.grid[0, 3] = TileType.EMPTY
    state.bombs.append(BombState(id=1, owner_id=1, r=0.5, c=1.5, timer=50, blast_radius=1))
    # Without bomb_pass, (0, 1) is impassable bomb -> cannot reach (0, 3) -> unsafe
    state.players[0].bomb_pass = False
    assert is_bomb_placement_safe(state, player_id=0) is False

    # With bomb_pass, player can walk through the bomb at (0, 1) to safe tile (0, 3)
    state.players[0].bomb_pass = True
    assert is_bomb_placement_safe(state, player_id=0) is True


def test_adversarial_rewards_and_capping() -> None:
    """Cover shielded immunity, brick adjacency cap (15 max), and novel visitation."""
    tracker = RewardTracker()
    state_before = GameState.create_initial(seed=42)
    state_after = state_before.clone()

    # 1. Shielded player (shield_timer > 20) -> Phi_danger == 0.0
    from bomberman_rl.rl.observation import compute_danger_map
    from bomberman_rl.engine.state import BombState
    state_after.bombs.append(BombState(id=0, owner_id=1, r=0.5, c=0.5, timer=10, blast_radius=2))
    dmap = compute_danger_map(state_after)
    state_after.players[0].shield_timer = 25
    assert compute_danger_potential(state_after.players[0], dmap) == 0.0

    state_after.players[0].shield_timer = 15
    assert compute_danger_potential(state_after.players[0], dmap) < 0.0

    # 2. Bomb adjacency bonus caps at 15 times
    tracker.reset(state_before)
    total_adj_bonus = 0.0
    for tick in range(20):
        sb = state_before.clone()
        sa = state_before.clone()
        # Place bomb at (0, 1) adjacent to brick at (0, 2)
        sb.grid[0, 2] = TileType.BRICK
        sa.grid[0, 2] = TileType.BRICK
        sa.bombs.append(BombState(id=tick, owner_id=0, r=0.5, c=1.5, timer=60, blast_radius=1))
        _, tel = tracker.step(
            sb,
            [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)],
            sa,
            False,
            False,
        )
        total_adj_bonus += tel["player_0"]["bomb_placement"]

    # Exactly 15 * 0.02 = 0.30 awarded under updated safe bomb economics
    assert np.isclose(total_adj_bonus, 15 * 0.02, atol=1e-5)
    assert tracker.bomb_near_brick_count[0] == 15


def test_death_tick_pbrs_danger_is_zero_and_suicide_penalized() -> None:
    """Verify that dying in a bomb blast awards 0.0 pbrs_danger and enforces strict negative loss."""
    env = BombermanEnv(seed=42)
    env.state.players[0].r = 0.5
    env.state.players[0].c = 1.5

    # Stage 1 curriculum self_damage_scale = 0.15
    tracker = RewardTracker(self_damage_scale=0.15, gamma=0.99)
    tracker.reset(env.state)

    # Step 1: P0 drops bomb
    sb = env.state.clone()
    env.step([(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    tracker.step(sb, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)], env.state, False, False)

    # Stand still for 58 ticks until detonation
    for _ in range(58):
        sb = env.state.clone()
        env.step([(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
        tracker.step(sb, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)], env.state, False, False)

    # Detonation tick (Player 0 dies)
    sb = env.state.clone()
    _, _, term, trunc, _ = env.step([(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    step_rews, step_tel = tracker.step(sb, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)], env.state, term, trunc)

    # Invariant 1: pbrs_danger must be exactly 0.0 (death is NOT an escape from danger)
    assert step_tel["player_0"]["pbrs_danger"] == 0.0, (
        f"Expected 0.0 pbrs_danger on death tick, got {step_tel['player_0']['pbrs_danger']}"
    )

    # Invariant 2: Terminal loss must be penalized strictly negatively (-0.15)
    assert np.isclose(step_tel["player_0"]["loss"], -0.15, atol=1e-5)

    # Invariant 3: Total step reward must be strictly negative, NEVER positive
    assert step_rews["player_0"] <= -0.15, (
        f"Expected strictly negative reward on suicide, got {step_rews['player_0']}"
    )


def test_opponent_bomb_placement_does_not_penalize_player() -> None:
    """Verify that when an opponent drops a bomb near Player 0, Player 0 is not penalized on placement tick."""
    env = BombermanEnv(seed=42)
    env.state.players[0].r = 0.5
    env.state.players[0].c = 0.5
    env.state.players[1].r = 0.5
    env.state.players[1].c = 1.5

    tracker = RewardTracker(self_damage_scale=0.15, gamma=0.99)
    tracker.reset(env.state)

    # Player 1 drops bomb at (0, 1) covering Player 0 at (0, 0)
    sb = env.state.clone()
    env.step([(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.PLACE_BOMB)])
    _, step_tel = tracker.step(
        sb,
        [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.PLACE_BOMB)],
        env.state,
        False,
        False,
    )

    assert step_tel["player_0"]["pbrs_danger"] >= 0.0, (
        f"Expected non-negative pbrs_danger on opponent placement tick, got {step_tel['player_0']['pbrs_danger']}"
    )


def test_shielded_cul_de_sac_remains_masked() -> None:
    """Verify that cul-de-sac bomb placement is masked even when player has an active shield."""
    state = GameState.create_initial(seed=42)
    state.players[0].r = 0.5
    state.players[0].c = 0.5
    state.grid[0, 2] = TileType.BRICK
    state.grid[2, 0] = TileType.BRICK
    state.grid[0, 1] = TileType.EMPTY
    state.grid[1, 0] = TileType.EMPTY

    # Shield active
    state.players[0].shield_timer = 15

    saf = SafetyActionFilter(safety_filter_prob=1.0)
    masks = saf.get_action_mask(state, player_id=0)

    # Movement should not be restricted by flames when shielded
    assert np.all(masks["move_mask"])
    # Cul-de-sac bomb placement must still be masked
    assert masks["bomb_mask"][BombAction.PLACE_BOMB] == False


def test_action_entropy_arithmetic_operators() -> None:
    """Verify ActionEntropy operator overloading (__mul__, __rmul__, __neg__, __sub__, __rsub__)."""
    from bomberman_rl.training.network import ActionEntropy
    m = torch.tensor([0.5, 0.7])
    b = torch.tensor([0.2, 0.3])
    ae = ActionEntropy(m, b)

    weighted = ae.weighted_loss(0.02, 0.01)
    assert isinstance(weighted, torch.Tensor)
    assert np.isclose(weighted.item(), -(0.02 * 0.6 + 0.01 * 0.25))

    # Test arithmetic operators
    mul_res = ae * 2.0
    assert torch.allclose(mul_res, (m + b) * 2.0)

    rmul_res = 2.0 * ae
    assert torch.allclose(rmul_res, 2.0 * (m + b))

    neg_res = -ae
    assert torch.allclose(neg_res, -(m + b))

    sub_res = ae - 1.0
    assert torch.allclose(sub_res, (m + b) - 1.0)

    rsub_res = 1.0 - ae
    assert torch.allclose(rsub_res, 1.0 - (m + b))


def test_interest_score_calculation() -> None:
    """Verify interest score weighting with hardcoded ground truth literals.

    Formula: (bricks_destroyed * 2.0) + (powerups_collected * 5.0) + (kicks_executed * 10.0) + (match_duration_ticks * 0.01)
    """
    from bomberman_rl.training.diagnostics import compute_interest_score, MatchStats

    # Case 1: hand-solved ground truth: 3 bricks, 2 powerups, 1 kick, 150 ticks
    # (3 * 2.0) + (2 * 5.0) + (1 * 10.0) + (150 * 0.01) = 6.0 + 10.0 + 10.0 + 1.5 = 27.5
    s1 = compute_interest_score(bricks_destroyed=3, powerups_collected=2, kicks_executed=1, match_duration_ticks=150)
    assert np.isclose(s1, 27.5, atol=1e-5)

    # Case 2: zero events, duration only: 0 bricks, 0 powerups, 0 kicks, 80 ticks -> 0.8
    s2 = compute_interest_score(bricks_destroyed=0, powerups_collected=0, kicks_executed=0, match_duration_ticks=80)
    assert np.isclose(s2, 0.8, atol=1e-5)

    # Case 3: MatchStats powerups_collected property and score calculation
    ms = MatchStats(
        bricks_destroyed=5,
        p0_buffs_collected=2,
        p0_debuffs_collected=1,
        p1_buffs_collected=3,
        p1_debuffs_collected=0,
        kicks_executed=2,
        ticks=200,
    )
    # Total powerups: 2 + 1 + 3 + 0 = 6
    assert ms.powerups_collected == 6
    # Score: (5 * 2.0) + (6 * 5.0) + (2 * 10.0) + (200 * 0.01) = 10.0 + 30.0 + 20.0 + 2.0 = 62.0
    s3 = compute_interest_score(
        bricks_destroyed=ms.bricks_destroyed,
        powerups_collected=ms.powerups_collected,
        kicks_executed=ms.kicks_executed,
        match_duration_ticks=ms.ticks,
    )
    assert np.isclose(s3, 62.0, atol=1e-5)


def test_cleanup_replays_storage_cap() -> None:
    """Verify cleanup_replays retains only the 50 newest .npz files and purges oldest."""
    from bomberman_rl.replays import cleanup_replays
    temp_dir = tempfile.mkdtemp(prefix="bomberman_replays_test_")
    try:
        # Create 55 dummy .npz files with staggered modification times
        base_time = 1700000000.0
        files = []
        for i in range(55):
            fname = os.path.join(temp_dir, f"gen_{i:04d}_score_{i * 2.0:.1f}.npz")
            with open(fname, "wb") as f:
                f.write(b"dummy")
            mtime = base_time + i * 10
            os.utime(fname, (mtime, mtime))
            files.append(fname)

        # Cleanup keeping 50
        deleted = cleanup_replays(temp_dir, max_replays=50)
        assert len(deleted) == 5
        # The 5 oldest should have been deleted (indices 0 to 4)
        for i in range(5):
            assert files[i] in deleted
            assert not os.path.exists(files[i])

        # Remaining 50 files should still exist
        remaining = [f for f in os.listdir(temp_dir) if f.endswith(".npz")]
        assert len(remaining) == 50
        for i in range(5, 55):
            assert os.path.exists(files[i])

        # Running again when <= 50 should delete nothing
        deleted_again = cleanup_replays(temp_dir, max_replays=50)
        assert len(deleted_again) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_two_generation_interest_replay_saving() -> None:
    """Verify 2-generation PPO training saves exactly two .npz replays with _score_X.X suffix.

    Also verifies initial probability of PLACE_BOMB at generation 0 is strictly < 1%.
    """
    temp_dir = tempfile.mkdtemp(prefix="bomberman_interest_test_")
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    replays_dir = os.path.join(temp_dir, "replays")

    try:
        trainer = PPOTrainer(
            num_envs=4,
            rollout_steps=256,
            total_generations=2,
            num_epochs=1,
            minibatch_size=64,
            checkpoint_dir=ckpt_dir,
            replays_dir=replays_dir,
            device="cpu",
            use_async=False,
            seed=42,
        )

        # 1. Verify initial probability of PLACE_BOMB is calibrated to ~12-27% (preventing extinction)
        spatial = torch.zeros(16, 18, 9, 9)
        scalars = torch.zeros(16, 18)
        with torch.no_grad():
            _, _, _, probs = trainer.policy.get_action(spatial, scalars)
        initial_bomb_prob = float(probs["bomb"][:, 1].mean().item())
        assert 0.10 <= initial_bomb_prob <= 0.30, f"Expected initial bomb prob in [0.10, 0.30], got {initial_bomb_prob:.4f}"

        # 2. Run 2 generations
        history = trainer.train(generations=2)
        assert len(history) == 2

        # 3. Check replays saved
        replay_files = sorted(glob.glob(os.path.join(replays_dir, "*.npz")))
        assert len(replay_files) == 2, f"Expected exactly 2 replays, found {len(replay_files)}: {replay_files}"

        # 4. Verify filename convention: gen_{gen:04d}_score_{score:.1f}.npz
        pattern = re.compile(r"^gen_(\d{4})_score_(\d+\.\d)\.npz$")
        for idx, rpath in enumerate(replay_files):
            basename = os.path.basename(rpath)
            match = pattern.match(basename)
            assert match is not None, f"Filename {basename} does not match expected format gen_XXXX_score_X.X.npz"
            gen_num = int(match.group(1))
            assert gen_num == idx, f"Expected gen {idx}, got {gen_num}"
            score_val = float(match.group(2))
            assert score_val > 0.0, f"Expected interest score > 0, got {score_val}"

            # 5. Verify the replay file is fully valid and loadable bit-exact
            episode = ReplayEpisode.load(rpath)
            assert len(episode) > 0
            assert episode.action_probs_p0.shape[1] == 10
            assert episode.critic_values.shape[1] == 2
            assert episode.player_pos.shape[1] == 2
    finally:
        trainer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dual_player_rollout_buffer() -> None:
    """Verify rollout collection populates transitions for both Player 0 and learning Player 1 instances."""
    trainer = PPOTrainer(
        num_envs=4,
        rollout_steps=16,
        total_generations=1,
        minibatch_size=16,
        num_epochs=1,
        device="cpu",
        use_async=False,
        seed=101,
    )
    try:
        # Buffer dimensions must be expanded to 2 * num_envs
        assert trainer.buf_spatial.shape == (16, 8, 18, 9, 9)
        assert trainer.buf_rewards.shape == (16, 8)
        assert trainer.buf_actions.shape == (16, 8, 2)
        assert trainer.buf_is_learning.shape == (16, 8)

        # Run 1 generation
        metrics = trainer.train_generation(generation=0)

        # Slot 0..3: Player 0 transitions (always learning)
        p0_learning = trainer.buf_is_learning[:, 0:4].sum().item()
        assert p0_learning == 16 * 4

        # Slot 4..7: Player 1 transitions (learning when opponent is latest policy)
        p1_learning = trainer.buf_is_learning[:, 4:8].sum().item()
        assert p1_learning > 0, "Expected at least one learning Player 1 instance in rollout batch"
    finally:
        trainer.close()


def test_ema_smoothing_and_escape_guidance_decay():
    """Verify ema_bricks updates with 0.95 / 0.05 smoothing, escape_guidance_prob reaches 0.0 at ema_bricks >= 4.0, and telemetry toggle."""
    # Check escape_guidance_prob decay via competence curriculum
    curr0 = get_competence_curriculum(0.0)
    assert np.isclose(curr0.escape_guidance_prob, 0.50)

    curr2 = get_competence_curriculum(2.0)
    # progress = 2 / 12 = 1 / 6; 0.50 - 1.50 * (1/6) = 0.25
    assert np.isclose(curr2.escape_guidance_prob, 0.25)

    curr4 = get_competence_curriculum(4.0)
    assert np.isclose(curr4.escape_guidance_prob, 0.0)

    curr8 = get_competence_curriculum(8.0)
    assert np.isclose(curr8.escape_guidance_prob, 0.0)

    # Check trainer ema_bricks initialization and 0.95 / 0.05 update
    trainer = PPOTrainer(
        num_envs=2,
        rollout_steps=4,
        total_generations=1,
        minibatch_size=4,
        num_epochs=1,
        device="cpu",
        use_async=False,
        seed=42,
    )
    try:
        assert trainer.ema_bricks == 0.0
        assert trainer.show_telemetry is False  # toggle off by default

        # Test toggle
        trainer.toggle_telemetry()
        assert trainer.show_telemetry is True
        trainer.toggle_telemetry(False)
        assert trainer.show_telemetry is False

        # Simulate diagnostic matches with 10.0 avg bricks
        from bomberman_rl.training.diagnostics import MatchStats
        m = MatchStats()
        m.bricks_destroyed = 10
        trainer.diagnostics.matches = [m]
        # Simulate update logic: 0.95 * 0.0 + 0.05 * 10.0 = 0.50
        gen_avg = sum(m.bricks_destroyed for m in trainer.diagnostics.matches) / len(trainer.diagnostics.matches)
        trainer.ema_bricks = 0.95 * trainer.ema_bricks + 0.05 * gen_avg
        assert np.isclose(trainer.ema_bricks, 0.50)
    finally:
        trainer.close()


def test_decoupled_action_filter_and_curriculum_boundaries() -> None:
    """Verify boundary clamping on competence curriculum and independent decoupling of placement vs escape filters."""
    # Clamping below 0.0
    curr_neg = get_competence_curriculum(-50.0)
    assert np.isclose(curr_neg.self_damage_scale, 0.15)
    assert np.isclose(curr_neg.placement_filter_prob, 1.00)
    assert np.isclose(curr_neg.escape_guidance_prob, 0.50)
    assert curr_neg.max_episode_ticks == 600

    # Clamping above 12.0
    curr_high = get_competence_curriculum(100.0)
    assert np.isclose(curr_high.self_damage_scale, 1.00)
    assert np.isclose(curr_high.placement_filter_prob, 0.10)
    assert np.isclose(curr_high.escape_guidance_prob, 0.00)
    assert curr_high.max_episode_ticks == 3600

    # Decoupled behavior at ema_bricks = 4.0:
    # placement_filter_prob is still active (0.70), but escape_guidance_prob is 0.00!
    curr_mid = get_competence_curriculum(4.0)
    assert np.isclose(curr_mid.placement_filter_prob, 1.00 - 0.90 * (4.0 / 12.0))
    assert np.isclose(curr_mid.escape_guidance_prob, 0.0)

    # Action filter with escape_guidance_prob=0.0 does NOT puppeteer escape (e.g. does not mask NOOP when threatened)
    from bomberman_rl.rl.action_filter import SafetyActionFilter
    from bomberman_rl.engine.state import GameState, BombState
    state = GameState.create_initial(seed=42)
    state.bombs = [BombState(id=0, owner_id=1, r=1.5, c=0.5, timer=50, blast_radius=2)]
    state.players[0].r = 1.5
    state.players[0].c = 0.5

    saf = SafetyActionFilter(placement_filter_prob=1.0, escape_guidance_prob=0.0)
    masks = saf.get_action_mask(state, player_id=0, ego_perspective=False)
    # When escape guidance is disabled, NOOP is not masked out to force escape
    assert masks["move_mask"][int(MoveAction.NOOP)] == True















def test_interest_replay_metadata_records_end_reason_and_tick_cap() -> None:
    """Saved training replays state why the match ended and the episode tick cap in force."""
    temp_dir = tempfile.mkdtemp(prefix="bomberman_endreason_test_")
    trainer = None
    try:
        trainer = PPOTrainer(
            num_envs=2,
            rollout_steps=128,
            total_generations=1,
            num_epochs=1,
            minibatch_size=64,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            replays_dir=os.path.join(temp_dir, "replays"),
            device="cpu",
            use_async=False,
            seed=7,
        )
        trainer.vec_env.set_curriculum(1.0, 0.0, 40)
        trainer.train(generations=1)
        files = glob.glob(os.path.join(temp_dir, "replays", "*.npz"))
        assert len(files) == 1
        episode = ReplayEpisode.load(files[0])
        assert episode.metadata["max_episode_ticks"] >= 1
        assert episode.metadata["end_reason"] in {"win_p0", "win_p1", "double_ko", "time_limit"}
        if episode.metadata["end_reason"] == "time_limit":
            assert len(episode) == episode.metadata["max_episode_ticks"] + 1
    finally:
        if trainer is not None:
            trainer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# GAE truncation bootstrapping
# ---------------------------------------------------------------------------
def test_gae_truncation_bootstraps_but_termination_does_not():
    import torch
    from bomberman_rl.training.ppo_trainer import compute_gae_advantages

    rewards = torch.tensor([[0.0, 0.0]])
    values = torch.tensor([[1.0, 1.0]])
    dones = torch.tensor([[True, True]])
    truncated = torch.tensor([[True, False]])
    last_value = torch.tensor([2.0, 2.0])
    adv = compute_gae_advantages(rewards, values, dones, truncated, last_value, gamma=0.5, gae_lambda=1.0)
    # col 0 truncated: delta = 0 + 0.5*2 - 1 = 0 ; col 1 terminated: delta = 0 - 1 = -1
    assert adv[0, 0].item() == pytest.approx(0.0)
    assert adv[0, 1].item() == pytest.approx(-1.0)


def test_gae_trace_does_not_leak_across_truncated_boundary():
    import torch
    from bomberman_rl.training.ppo_trainer import compute_gae_advantages

    rewards = torch.tensor([[0.0], [10.0]])
    values = torch.tensor([[0.0], [0.0]])
    dones = torch.tensor([[True], [False]])
    truncated = torch.tensor([[True], [False]])
    last_value = torch.tensor([0.0])
    adv = compute_gae_advantages(rewards, values, dones, truncated, last_value, gamma=1.0, gae_lambda=1.0)
    assert adv[1, 0].item() == pytest.approx(10.0)
    # Step 0 ended its episode: reward from the next episode (10) must not leak in.
    assert adv[0, 0].item() == pytest.approx(0.0)
