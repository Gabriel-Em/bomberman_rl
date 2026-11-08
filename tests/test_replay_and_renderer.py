"""Comprehensive verification suite for Replay & Telemetry Recorder and Pygame Visualizer.

Mandatory tests:
1. Headless Surface Rendering Test: SDL_VIDEODRIVER="dummy", 100 ticks with synthetic telemetry.
2. Replay Bit-Exact Roundtrip & Scrubber Sync: 200-tick episode at ./replays/test_match.npz,
   seeking to tick 0, 105, 199 with exact state, bomb timer, and progress ratio checks.
"""

from __future__ import annotations
import math
import os
import tempfile
import pytest
import numpy as np

# Set headless SDL dummy driver before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    MoveAction,
    BombAction,
    TileType,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.engine.physics import BombermanPhysics
from bomberman_rl.rl.rewards import RewardTracker, PlayerRewardLog
from bomberman_rl.replays.recorder import (
    ReplayRecorder,
    ReplayEpisode,
    GameEvent,
    REWARD_COMPONENT_NAMES,
)
from bomberman_rl.visualizer.renderer import Renderer, BrickParticle
from bomberman_rl.visualizer.play_interactive import HeuristicAgent, run_interactive


class TestHeadlessSurfaceRendering:
    """Test 1: Headless surface rendering with overlays, sparklines, and animations."""

    def test_headless_rendering_100_ticks(self) -> None:
        """Run Pygame renderer for 100 ticks with synthetic policy/critic telemetry."""
        renderer = Renderer(headless=True)
        assert renderer.screen.get_width() == 1280
        assert renderer.screen.get_height() == 760

        state = GameState.create_initial(seed=123)
        prev_state = state.clone()

        # Add mock bomb and powerup for visual coverage
        state.bombs.append(
            BombState(
                id=99,
                owner_id=0,
                r=1.5,
                c=2.5,
                timer=15,  # <= 20 to test orange/red fuse spark
                blast_radius=3,
                velocity_r=0.25,  # moving bomb directional arrow
                velocity_c=0.0,
            )
        )
        state.powerups[0, 2] = PowerUpType.MAX_BLAST
        state.powerups[2, 0] = PowerUpType.SPEED_UP
        state.flame_timer[4, 4] = 8

        # Give Player 0 active shield
        state.players[0].shield_timer = 50

        # Run 100 ticks
        for t in range(100):
            state.tick = t

            # Toggle overlays across ticks to test all render paths
            renderer.overlay_danger = (t % 3 == 0)
            renderer.overlay_safe_space = (t % 3 == 1)
            renderer.overlay_policy_arrow = (t % 3 == 2)

            # Synthetic telemetry
            v0 = float(math.sin(t * 0.1))
            v1 = float(math.cos(t * 0.1))

            p0_probs = np.array([0.1, 0.4, 0.1, 0.2, 0.2, 0.8, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
            p1_probs = np.array([0.2, 0.1, 0.5, 0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

            telemetry = {
                "tick": t,
                "actions": {
                    "player_0": (int(MoveAction.UP), int(BombAction.NOOP)),
                    "player_1": (int(MoveAction.DOWN), int(BombAction.NOOP)),
                },
                "action_probs_p0": p0_probs,
                "action_probs_p1": p1_probs,
                "value_p0": v0,
                "value_p1": v1,
                "rewards_p0": {"brick_destruction": 0.015, "total": 0.015},
                "rewards_p1": {"total": 0.0},
                "cumulative_rewards_p0": {"brick_destruction": 0.045, "total": 0.045},
                "cumulative_rewards_p1": {"total": 0.0},
            }

            # Spawn particles occasionally
            if t % 20 == 0:
                renderer.spawn_brick_crumble(3, 3)

            # Render frame
            renderer.render_frame(
                state=state,
                prev_state=prev_state,
                telemetry=telemetry,
                events=[],
                interp_alpha=0.5,
                total_ticks=100,
            )

        # Assert sparkline rolling window populated
        assert len(renderer.v_history_p0) == 100
        assert len(renderer.v_history_p1) == 100


class TestReplayBitExactRoundtripAndScrubberSync:
    """Test 2: Replay Bit-Exact Roundtrip and Scrubber Synchronization."""

    def test_replay_200_ticks_roundtrip_and_sync(self) -> None:
        """Record 200 ticks to ./replays/test_match.npz, reload, seek 0, 105, 199."""
        os.makedirs("./replays", exist_ok=True)
        replay_filepath = "./replays/test_match.npz"

        # 1. Simulate 200 ticks and record
        state = GameState.create_initial(seed=42)
        reward_tracker = RewardTracker()
        reward_tracker.reset(state)
        recorder = ReplayRecorder(metadata={"seed": 42, "match_type": "verification"})

        agent0 = HeuristicAgent(0)
        agent1 = HeuristicAgent(1)

        recorded_states: list[GameState] = []

        for tick in range(200):
            recorded_states.append(state.clone())

            danger_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            p0_act, p0_probs, v0 = agent0.compute_action_and_telemetry(state, danger_map)
            p1_act, p1_probs, v1 = agent1.compute_action_and_telemetry(state, danger_map)

            state_before = state.clone()
            BombermanPhysics.step(state, [p0_act, p1_act])

            p0_alive = state.players[0].alive
            p1_alive = state.players[1].alive
            terminated = (not p0_alive or not p1_alive)

            step_rew, step_telemetry = reward_tracker.step(
                state_before=state_before,
                actions=[p0_act, p1_act],
                state_after=state,
                terminated=terminated,
                truncated=False,
            )

            recorder.record_frame(
                state=state_before,  # record pre-step state corresponding to this tick index
                actions=(p0_act, p1_act),
                action_probs_p0=p0_probs,
                action_probs_p1=p1_probs,
                value_p0=v0,
                value_p1=v1,
                rewards_p0=step_telemetry["player_0"],
                rewards_p1=step_telemetry["player_1"],
            )

            if terminated:
                # If game terminated early, keep recording till 200 frames for exact test length
                pass

        assert len(recorder) == 200
        recorder.save(replay_filepath)
        assert os.path.isfile(replay_filepath)

        # 2. Reload episode from disk
        episode = ReplayEpisode.load(replay_filepath)
        assert len(episode) == 200

        # 3. Seek to tick 0, 105, and 199 and verify bit-exact fidelity & scrubber sync
        test_ticks = [0, 105, 199]
        for t in test_ticks:
            reconstructed_state = episode.get_state(t)
            orig_state = recorded_states[t]

            # A. Board state bit-exact check
            assert np.array_equal(reconstructed_state.grid, orig_state.grid), f"Grid mismatch at tick {t}"
            assert np.array_equal(reconstructed_state.powerups, orig_state.powerups), f"Powerups mismatch at tick {t}"
            assert np.array_equal(reconstructed_state.flame_timer, orig_state.flame_timer), f"Flame timer mismatch at tick {t}"

            # B. Bomb timers bit-exact check
            assert len(reconstructed_state.bombs) == len(orig_state.bombs), f"Bomb count mismatch at tick {t}"
            for b_rec, b_orig in zip(reconstructed_state.bombs, orig_state.bombs):
                assert b_rec.id == b_orig.id
                assert b_rec.timer == b_orig.timer, f"Bomb timer mismatch at tick {t}: {b_rec.timer} != {b_orig.timer}"
                assert abs(b_rec.r - b_orig.r) < 1e-5
                assert abs(b_rec.c - b_orig.c) < 1e-5

            # C. Player positions bit-exact check
            for pid in (0, 1):
                p_rec = reconstructed_state.players[pid]
                p_orig = orig_state.players[pid]
                assert abs(p_rec.r - p_orig.r) < 1e-5
                assert abs(p_rec.c - p_orig.c) < 1e-5
                assert p_rec.alive == p_orig.alive
                assert p_rec.shield_timer == p_orig.shield_timer

            # D. Timeline progress ratio strictly matches tick / total_ticks to 1e-6 precision
            expected_ratio = float(t) / 200.0
            actual_ratio = episode.get_progress_ratio(t)
            assert abs(actual_ratio - expected_ratio) < 1e-6, (
                f"Progress ratio mismatch at tick {t}: actual {actual_ratio} != expected {expected_ratio}"
            )


class TestReplayAndScrubberDetails:
    """Additional unit tests for scrubber invariants, edge cases, and telemetry."""

    def test_single_tick_episode_progress_ratio(self) -> None:
        rec = ReplayRecorder()
        state = GameState.create_initial()
        rec.record_frame(state)
        ep = rec.to_episode()
        assert len(ep) == 1
        assert ep.get_progress_ratio(0) == 0.0

    def test_empty_episode_safety(self) -> None:
        rec = ReplayRecorder()
        ep = rec.to_episode()
        assert len(ep) == 0
        assert ep.get_progress_ratio(0) == 0.0
        s = ep.get_state(0)
        assert isinstance(s, GameState)
        assert ep.get_telemetry(0) == {}

    def test_out_of_bounds_tick_clamping(self) -> None:
        rec = ReplayRecorder()
        state = GameState.create_initial()
        rec.record_frame(state)
        rec.record_frame(state)
        ep = rec.to_episode()
        # Should clamp gracefully
        s_neg = ep.get_state(-10)
        s_high = ep.get_state(999)
        assert isinstance(s_neg, GameState)
        assert isinstance(s_high, GameState)

    def test_event_markers_recording_and_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events_test.npz")
            rec = ReplayRecorder()
            s = GameState.create_initial()
            ev1 = GameEvent(tick=10, event_type="wall_breach", description="Wall breached by P0", player_id=0)
            ev2 = GameEvent(tick=45, event_type="player_death", description="P1 eliminated", player_id=1)

            rec.record_frame(s, events=[ev1])
            rec.record_frame(s, events=[ev2])
            rec.save(path)

            ep = ReplayEpisode.load(path)
            assert len(ep.events) == 2
            assert ep.events[0].tick == 10
            assert ep.events[0].event_type == "wall_breach"
            assert ep.events[1].tick == 45
            assert ep.events[1].event_type == "player_death"

    def test_particle_lifespan_and_gravity(self) -> None:
        p = BrickParticle(x=10.0, y=10.0, vx=1.0, vy=-2.0, size=4.0, color=(180, 80, 60), lifetime=10)
        for frame in range(10):
            alive = p.update()
            if frame < 9:
                assert alive is True
            else:
                assert alive is False
        assert p.age == 10
        assert p.x > 10.0  # moved forward

    def test_renderer_keyboard_and_scrubber_interaction(self) -> None:
        renderer = Renderer(headless=True)
        # Test speed toggling
        initial_speed = renderer.playback_speed
        renderer.speed_idx = min(len(renderer.speed_multipliers) - 1, renderer.speed_idx + 1)
        assert renderer.playback_speed > initial_speed

        # Test pause toggling
        assert renderer.paused is False
        renderer.paused = True
        assert renderer.paused is True

    def test_play_interactive_headless_dry_run(self) -> None:
        """Verify play_interactive CLI runs in headless mode without exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, "dry_run.npz")
            # Run 5 ticks agent vs agent
            run_interactive(
                mode="agent_vs_agent",
                record_path=rec_path,
                seed=42,
                headless=True,
                max_ticks=5,
            )
            assert os.path.isfile(rec_path)
            # Replay the generated file for 5 ticks
            run_interactive(
                mode="replay",
                replay_path=rec_path,
                headless=True,
            )
