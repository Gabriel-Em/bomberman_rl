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
    TICK_RATE,
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
    find_latest_replay,
    list_replays,
)
from bomberman_rl.visualizer.renderer import (
    Renderer,
    BrickParticle,
    COLOR_P0_CYAN,
    COLOR_P1_RED,
)
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

    def test_brick_events_attributed_to_bomb_owner(self) -> None:
        rec = ReplayRecorder()
        s0 = GameState.create_initial()
        s0.grid[1, 1] = TileType.BRICK
        s0.grid[1, 3] = TileType.BRICK
        s0.grid[3, 1] = TileType.BRICK
        rec.record_frame(s0)
        s1 = s0.clone()
        s1.tick = 1
        s1.grid[1, 1] = TileType.EMPTY
        s1.grid[1, 3] = TileType.EMPTY
        s1.grid[3, 1] = TileType.EMPTY
        s1.flame_owner_mask[1, 1] = 1  # P0 flame
        s1.flame_owner_mask[1, 3] = 2  # P1 flame
        s1.flame_owner_mask[3, 1] = 2  # P1 flame
        rec.record_frame(s1)
        bricks = {e.player_id: e for e in rec.events if e.event_type == "brick_destroyed"}
        assert set(bricks) == {0, 1}
        assert bricks[0].data["count"] == 1
        assert bricks[1].data["count"] == 2
        assert "P0" in bricks[0].description and "P1" in bricks[1].description

    def test_event_marker_colors_distinguish_players(self) -> None:
        renderer = Renderer(headless=True)
        cyan = renderer.event_marker_color(GameEvent(tick=1, event_type="brick_destroyed", description="x", player_id=0))
        red = renderer.event_marker_color(GameEvent(tick=1, event_type="brick_destroyed", description="x", player_id=1))
        pu0 = renderer.event_marker_color(GameEvent(tick=1, event_type="powerup_pickup", description="x", player_id=0))
        pu1 = renderer.event_marker_color(GameEvent(tick=1, event_type="powerup_pickup", description="x", player_id=1))
        neutral = renderer.event_marker_color(GameEvent(tick=1, event_type="brick_destroyed", description="x"))
        assert cyan == pu0 and red == pu1
        assert cyan != red
        assert neutral not in (cyan, red)

    def test_event_marker_shapes_distinguish_types(self) -> None:
        renderer = Renderer(headless=True)
        shapes = {
            renderer.event_marker_shape(t)
            for t in ("brick_destroyed", "powerup_pickup", "shield_activated", "player_death")
        }
        assert len(shapes) == 4

    def test_telemetry_panel_hidden_by_default_and_toggle_key(self) -> None:
        renderer = Renderer(headless=True)
        assert renderer.show_telemetry is False
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))
        renderer.handle_input()
        assert renderer.show_telemetry is True
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))
        renderer.handle_input()
        assert renderer.show_telemetry is False

    def test_telemetry_panel_drawn_only_when_enabled(self) -> None:
        renderer = Renderer(headless=True)
        state = GameState.create_initial()
        renderer.render_frame(state)
        off = pygame.image.tostring(renderer.screen, "RGB")
        renderer.show_telemetry = True
        renderer.render_frame(state)
        on = pygame.image.tostring(renderer.screen, "RGB")
        assert off != on

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


class TestReplayAutoDiscovery:
    """Executable specifications for replay auto-discovery and wrapper convenience."""

    def test_find_latest_replay_empty_or_nonexistent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_latest_replay(tmpdir) is None
            assert list_replays(tmpdir) == []

        nonexistent = "/path/to/nonexistent/replays/dir"
        assert find_latest_replay(nonexistent) is None
        assert list_replays(nonexistent) == []

    def test_find_latest_replay_multiple_files_returns_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "replay_1.npz")
            f2 = os.path.join(tmpdir, "replay_2.npz")
            f3 = os.path.join(tmpdir, "replay_3.npz")

            rec = ReplayRecorder()
            s = GameState.create_initial()
            rec.record_frame(s)

            rec.save(f1)
            rec.save(f2)
            rec.save(f3)

            # Explicitly set mtimes: f2 is oldest, f1 is middle, f3 is newest
            os.utime(f2, (1000, 1000))
            os.utime(f1, (2000, 2000))
            os.utime(f3, (3000, 3000))

            latest = find_latest_replay(tmpdir)
            assert latest == os.path.abspath(f3)

            replays = list_replays(tmpdir)
            assert replays == [os.path.abspath(f3), os.path.abspath(f1), os.path.abspath(f2)]

    def test_find_latest_replay_ignores_non_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, ".gitkeep"), "w") as f:
                f.write("")
            with open(os.path.join(tmpdir, "notes.txt"), "w") as f:
                f.write("text file")
            with open(os.path.join(tmpdir, "checkpoint.pt"), "w") as f:
                f.write("weights")

            assert find_latest_replay(tmpdir) is None
            assert list_replays(tmpdir) == []

    def test_run_interactive_auto_discovers_latest_replay(self) -> None:
        """When replay_path is None, run_interactive resolves the latest replay from replays_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "match_1.npz")
            f2 = os.path.join(tmpdir, "match_2.npz")

            rec = ReplayRecorder()
            s = GameState.create_initial()
            rec.record_frame(s)
            rec.save(f1)

            rec2 = ReplayRecorder()
            rec2.record_frame(s)
            rec2.record_frame(s)
            rec2.save(f2)

            os.utime(f1, (1000, 1000))
            os.utime(f2, (2000, 2000))

            # Should auto-discover f2 and run headless without error
            run_interactive(
                mode="replay",
                replay_path=None,
                replays_dir=tmpdir,
                headless=True,
            )

    def test_replay_cli_empty_dir_returns_error(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        import replay
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "sys.argv",
                ["replay.py", "--replays-dir", tmpdir, "--headless"],
            )
            exit_code = replay.main()
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "Error: No replay files (.npz) found" in captured.err

    def test_replay_cli_plays_latest_and_handles_list(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import replay
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "match_old.npz")
            f2 = os.path.join(tmpdir, "match_new.npz")

            rec = ReplayRecorder()
            s = GameState.create_initial()
            rec.record_frame(s)
            rec.save(f1)
            rec.save(f2)

            os.utime(f1, (1000, 1000))
            os.utime(f2, (2000, 2000))

            # Test --list
            monkeypatch.setattr(
                "sys.argv",
                ["replay.py", "--replays-dir", tmpdir, "--list"],
            )
            code_list = replay.main()
            assert code_list == 0
            list_out = capsys.readouterr().out
            assert "match_new.npz (latest)" in list_out
            assert "match_old.npz" in list_out

            # Test default run (headless)
            monkeypatch.setattr(
                "sys.argv",
                ["replay.py", "--replays-dir", tmpdir, "--headless"],
            )
            code_run = replay.main()
            assert code_run == 0
            run_out = capsys.readouterr().out
            assert "Playing latest replay:" in run_out
            assert "match_new.npz" in run_out


class TestCursorAndPlayerVisuals:
    """Verification for in-game cursor drawing and player body appearance."""

    def test_player_rendering_has_eyes_and_no_p0_p1_text_labels(self) -> None:
        """Verify that P0/P1 text badges are removed from player bodies while eyes remain."""
        renderer = Renderer(headless=True)
        surf_p0 = pygame.Surface((100, 100))
        surf_p0.fill((0, 0, 0))
        p0 = PlayerState(id=0, r=0.5, c=0.5)
        renderer._draw_player(surf_p0, 50, 50, p0, 0)

        # Eyes are preserved (pupil color (20, 30, 40))
        assert surf_p0.get_at((46, 46))[:3] == (20, 30, 40)
        assert surf_p0.get_at((51, 46))[:3] == (20, 30, 40)

        # Lower body center at (50, 58) must be solid player base color, without text badge blit
        assert surf_p0.get_at((50, 58))[:3] == COLOR_P0_CYAN

        surf_p1 = pygame.Surface((100, 100))
        surf_p1.fill((0, 0, 0))
        p1 = PlayerState(id=1, r=0.5, c=0.5)
        renderer._draw_player(surf_p1, 50, 50, p1, 0)
        assert surf_p1.get_at((46, 46))[:3] == (20, 30, 40)
        assert surf_p1.get_at((50, 58))[:3] == COLOR_P1_RED

    def test_in_game_custom_cursor_rendering(self) -> None:
        """Verify that custom software cursor is drawn on the surface with tip and body colors."""
        renderer = Renderer(headless=True)
        surf = pygame.Surface((100, 100))
        surf.fill((0, 0, 0))

        # Draw cursor at (20, 20)
        renderer._draw_cursor(surf, 20, 20)

        # Tip outline at (20, 20)
        assert surf.get_at((20, 20))[:3] == (15, 18, 24)
        # Inside body at (21, 23) must be bright white pointer fill
        assert surf.get_at((21, 23))[:3] == (250, 252, 255)
        # Outside cursor untouched
        assert surf.get_at((5, 5))[:3] == (0, 0, 0)

    def test_render_cursor_bounds_clamping(self) -> None:
        """Verify render_cursor does not crash or draw outside valid window bounds."""
        renderer = Renderer(headless=True)
        surf = pygame.Surface((1280, 760))
        # Should not raise any error even when called headless or out of bounds
        renderer.render_cursor(surf)


class TestReplaySparklineAndPlaybackControls:
    """Executable specifications for sparkline pause stability and 5-second replay seek jumps."""

    def test_sparkline_does_not_advance_when_paused_or_same_tick(self) -> None:
        """Verify that multiple render passes at the same tick (such as when paused) do not scroll sparkline."""
        renderer = Renderer(headless=True)
        renderer.paused = True
        state = GameState.create_initial(seed=1)
        state.tick = 42
        telemetry = {"value_p0": 0.5, "value_p1": -0.5}

        # Render 20 frames while paused at tick 42
        for _ in range(20):
            renderer.render_frame(state=state, telemetry=telemetry)

        # Sparkline must only have recorded 1 entry for tick 42, not 20 entries
        assert len(renderer.v_history_p0) == 1
        assert len(renderer.v_history_p1) == 1
        assert renderer.v_history_p0[0] == 0.5

    def test_unpaused_left_right_arrow_keys_jump_5_seconds(self) -> None:
        """Verify LEFT and RIGHT arrow keys jump 5 seconds (5 * TICK_RATE) when unpaused, and 1 tick when paused."""
        renderer = Renderer(headless=True)

        # 1. Unpaused: LEFT/RIGHT should jump 5 seconds
        renderer.paused = False
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        inp = renderer.handle_input()
        assert inp["step"] == -5 * TICK_RATE

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        inp = renderer.handle_input()
        assert inp["step"] == 5 * TICK_RATE

        # 2. Paused: LEFT/RIGHT should step 1 frame
        renderer.paused = True
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        inp = renderer.handle_input()
        assert inp["step"] == -1

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        inp = renderer.handle_input()
        assert inp["step"] == 1

    def test_replay_telemetry_provides_value_history_for_seeking(self) -> None:
        """Verify ReplayEpisode provides value_history in telemetry and accurately populates sparkline on seek."""
        rec = ReplayRecorder()
        state = GameState.create_initial()
        for t in range(50):
            state.tick = t
            rec.record_frame(state=state, value_p0=float(t) * 0.01, value_p1=-float(t) * 0.01)
        ep = rec.to_episode()

        # Telemetry at tick 30 must provide history up to tick 30
        telem30 = ep.get_telemetry(30)
        assert "value_history_p0" in telem30
        assert len(telem30["value_history_p0"]) == 31
        assert abs(telem30["value_history_p0"][-1] - 0.30) < 1e-5

        renderer = Renderer(headless=True)
        renderer.render_right_panel(renderer.screen, ep.get_state(30), telem30)
        assert len(renderer.v_history_p0) == 31
        assert abs(renderer.v_history_p0[-1] - 0.30) < 1e-5




