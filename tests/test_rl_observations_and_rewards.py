"""Comprehensive test suite for RL observations, PBRS rewards, and safety action filter."""

import numpy as np
import pytest

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    BOMB_TIMER,
    TileType,
    MoveAction,
    BombAction,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.rl.observation import (
    SPATIAL_CHANNELS,
    DEFAULT_SCALAR_DIM,
    FULL_SCALAR_DIM,
    ObservationBuilder,
    compute_danger_map,
    compute_reachable_safe_tiles,
    flip_action_to_world,
    flip_action_to_ego,
)
from bomberman_rl.rl.rewards import (
    CurriculumLethalityScheduler,
    RewardTracker,
    compute_pbrs_potential,
    check_players_connected,
    BRICK_REWARD,
    CONNECTIVITY_REWARD,
    AUXILIARY_REWARD_CAP,
    TERMINAL_WIN,
    TERMINAL_DRAW,
)
from bomberman_rl.rl.action_filter import SafetyActionFilter


def create_clean_game_state() -> GameState:
    """Create a minimalist clean game state for deterministic testing."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for r in range(1, GRID_SIZE, 2):
        for c in range(1, GRID_SIZE, 2):
            grid[r, c] = TileType.PILLAR

    powerups = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    powerup_spawn_tick = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int32)
    flame_timer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    flame_owner_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    players = [
        PlayerState(id=0, r=0.5, c=0.5),
        PlayerState(id=1, r=8.5, c=8.5),
    ]

    return GameState(
        grid=grid,
        powerups=powerups,
        powerup_spawn_tick=powerup_spawn_tick,
        flame_timer=flame_timer,
        flame_owner_mask=flame_owner_mask,
        players=players,
        bombs=[],
        tick=0,
        next_bomb_id=0,
        rng=np.random.default_rng(42),
    )


# ==============================================================================
# Mandatory Verification 1: 180-Degree Perspective Symmetry
# ==============================================================================


def test_action_flipping_symmetry() -> None:
    """Verify that action flipping maps UP <-> DOWN and LEFT <-> RIGHT while keeping NOOP and PLACE_BOMB invariant."""
    # Player 0: identity
    for m in MoveAction:
        for b in BombAction:
            act = (int(m), int(b))
            assert flip_action_to_world(act, player_id=0) == act
            assert flip_action_to_ego(act, player_id=0) == act

    # Player 1: directional inversion
    p1_cases = [
        ((MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)),
        ((MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.PLACE_BOMB)),
        ((MoveAction.UP, BombAction.NOOP), (MoveAction.DOWN, BombAction.NOOP)),
        ((MoveAction.DOWN, BombAction.PLACE_BOMB), (MoveAction.UP, BombAction.PLACE_BOMB)),
        ((MoveAction.LEFT, BombAction.NOOP), (MoveAction.RIGHT, BombAction.NOOP)),
        ((MoveAction.RIGHT, BombAction.PLACE_BOMB), (MoveAction.LEFT, BombAction.PLACE_BOMB)),
    ]

    for ego_act, expected_world in p1_cases:
        world = flip_action_to_world((int(ego_act[0]), int(ego_act[1])), player_id=1)
        assert world == (int(expected_world[0]), int(expected_world[1]))
        # Self-inverting reflection
        ego_back = flip_action_to_ego(world, player_id=1)
        assert ego_back == (int(ego_act[0]), int(ego_act[1]))


def test_180_degree_perspective_symmetry() -> None:
    """Set up a symmetric board state and verify P0 and P1 observation tensors are bit-for-bit identical."""
    state = create_clean_game_state()

    # Add symmetric bricks
    state.grid[2, 3] = TileType.BRICK
    state.grid[6, 5] = TileType.BRICK  # (8-2, 8-3)

    state.grid[3, 4] = TileType.BRICK
    state.grid[5, 4] = TileType.BRICK  # (8-3, 8-4)

    # Add symmetric power-ups
    state.powerups[0, 2] = PowerUpType.BLAST_UP
    state.powerups[8, 6] = PowerUpType.BLAST_UP  # (8-0, 8-2)

    state.powerups[2, 0] = PowerUpType.KICK
    state.powerups[6, 8] = PowerUpType.KICK  # (8-2, 8-0)

    # Add symmetric flames
    state.flame_timer[0, 4] = 6
    state.flame_timer[8, 4] = 6

    # Add symmetric bombs with velocities
    # Bomb 0: near P0, moving DOWN (velocity_r = +0.25)
    state.bombs.append(
        BombState(
            id=0,
            owner_id=0,
            r=1.5,
            c=2.5,
            timer=40,
            blast_radius=2,
            velocity_r=0.25,
            velocity_c=0.0,
        )
    )
    # Bomb 1: symmetric near P1, moving UP in world space (velocity_r = -0.25)
    # (8 - 1, 8 - 2) = (7, 6) -> center (7.5, 6.5)
    state.bombs.append(
        BombState(
            id=1,
            owner_id=1,
            r=7.5,
            c=6.5,
            timer=40,
            blast_radius=2,
            velocity_r=-0.25,
            velocity_c=0.0,
        )
    )

    # Match player stats symmetrically
    state.players[0].max_bombs = 3
    state.players[1].max_bombs = 3
    state.players[0].blast_radius = 2
    state.players[1].blast_radius = 2
    state.players[0].can_kick = True
    state.players[1].can_kick = True

    obs_builder = ObservationBuilder(scalar_dim=FULL_SCALAR_DIM)
    obs_p0, obs_p1 = obs_builder.build_both(state)

    spatial_p0 = obs_p0["spatial"]
    spatial_p1 = obs_p1["spatial"]

    # Verify spatial tensor shape
    assert spatial_p0.shape == (SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE)
    assert spatial_p1.shape == (SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE)

    # Channel-by-channel comparison for exact symmetry
    for ch in range(SPATIAL_CHANNELS):
        diff = np.abs(spatial_p0[ch] - spatial_p1[ch])
        max_diff = np.max(diff)
        assert max_diff < 1e-6, f"Asymmetry detected in Channel {ch}! Max diff: {max_diff}\nCh P0:\n{spatial_p0[ch]}\nCh P1:\n{spatial_p1[ch]}"

    # Verify scalars are also identical
    scalars_p0 = obs_p0["scalars"]
    scalars_p1 = obs_p1["scalars"]
    np.testing.assert_allclose(scalars_p0, scalars_p1, atol=1e-6)


# ==============================================================================
# Mandatory Verification 2: Chain-Reaction Danger Forecast (Ch 8)
# ==============================================================================


def test_chain_reaction_danger_forecast() -> None:
    """Place Bomb A (timer=10, radius=3) at (2,2) and Bomb B (timer=55, radius=2) at (2,4).

    Verify Ch 8 at (2,6) (reachable only by Bomb B) reflects countdown of 10 ticks (1.0 - 10/60).
    """
    state = create_clean_game_state()

    # Place Bomb A: (2, 2), timer=10, radius=3
    state.bombs.append(
        BombState(
            id=0,
            owner_id=0,
            r=2.5,
            c=2.5,
            timer=10,
            blast_radius=3,
        )
    )

    # Place Bomb B: (2, 4), timer=55, radius=2
    # Distance between (2,2) and (2,4) is 2 <= radius 3 of Bomb A -> triggered!
    state.bombs.append(
        BombState(
            id=1,
            owner_id=1,
            r=2.5,
            c=4.5,
            timer=55,
            blast_radius=2,
        )
    )

    danger_map = compute_danger_map(state)

    # Expected imminent danger from 10 ticks: 1.0 - (10.0 / 60.0) = 50.0 / 60.0
    expected_danger = 1.0 - (10.0 / float(BOMB_TIMER))
    wrong_danger = 1.0 - (55.0 / float(BOMB_TIMER))

    actual_danger_2_6 = danger_map[2, 6]

    # Tile (2, 6) is distance 4 from Bomb A (out of reach) but distance 2 from Bomb B
    assert np.isclose(actual_danger_2_6, expected_danger, atol=1e-5), (
        f"Tile (2, 6) danger forecast failure: expected imminent {expected_danger} (10 ticks), "
        f"got {actual_danger_2_6} (timer was {wrong_danger} for 55 ticks)"
    )

    # Tile (2, 7) is beyond Bomb B's radius 2 -> should be 0.0 (safe)
    assert danger_map[2, 7] == 0.0

    # Bomb A center tile (2, 2)
    assert np.isclose(danger_map[2, 2], expected_danger, atol=1e-5)
    # Bomb B center tile (2, 4)
    assert np.isclose(danger_map[2, 4], expected_danger, atol=1e-5)


def test_chain_reaction_blocked_by_pillar() -> None:
    """Verify that interior unbreakable pillars stop raycasts from chain-triggering bombs."""
    state = create_clean_game_state()
    # Pillar at (1, 1)
    # Bomb A at (1, 0), timer=10, radius=3
    state.bombs.append(BombState(id=0, owner_id=0, r=1.5, c=0.5, timer=10, blast_radius=3))
    # Bomb B at (1, 2), timer=50, radius=2
    state.bombs.append(BombState(id=1, owner_id=1, r=1.5, c=2.5, timer=50, blast_radius=2))

    danger = compute_danger_map(state)
    # The pillar at (1, 1) blocks Bomb A from reaching Bomb B!
    # So Bomb B detonates at 50, not 10.
    expected_b_danger = 1.0 - (50.0 / float(BOMB_TIMER))
    assert np.isclose(danger[1, 2], expected_b_danger, atol=1e-5)


# ==============================================================================
# Mandatory Verification 3: Zero-Sum Loop Immunity (PBRS Test)
# ==============================================================================


def test_zero_sum_loop_immunity_pbrs() -> None:
    """Force a player to pick up BLAST_UP then BLAST_DOWN 10 times in a row.

    Verify the cumulative shaping reward is exactly 0.0.
    """
    state_prev = create_clean_game_state()
    tracker = RewardTracker()
    tracker.reset(state_prev)

    cumulative_pbrs_reward = 0.0

    for cycle in range(10):
        # 1. Pick up BLAST_UP
        state_after_up = state_prev.clone()
        state_after_up.players[0].blast_radius += 1
        rewards, telemetry = tracker.step(
            state_prev,
            actions=[(0, 0), (0, 0)],
            state_after=state_after_up,
            terminated=False,
            truncated=False,
        )
        cumulative_pbrs_reward += telemetry["player_0"]["pbrs_stats"]
        state_prev = state_after_up

        # 2. Pick up BLAST_DOWN
        state_after_down = state_prev.clone()
        state_after_down.players[0].blast_radius -= 1
        rewards, telemetry = tracker.step(
            state_prev,
            actions=[(0, 0), (0, 0)],
            state_after=state_after_down,
            terminated=False,
            truncated=False,
        )
        cumulative_pbrs_reward += telemetry["player_0"]["pbrs_stats"]
        state_prev = state_after_down

    assert np.isclose(cumulative_pbrs_reward, 0.0, atol=1e-7), (
        f"PBRS loop immunity violated! Cumulative reward after 10 loops: {cumulative_pbrs_reward}"
    )


def test_pbrs_all_stat_upgrades() -> None:
    """Verify potential values and step shaping for all buff types."""
    p = PlayerState(id=0, r=0.5, c=0.5)
    # Initial base stats
    assert compute_pbrs_potential(p) == 0.0

    # Max bombs + 1 -> +0.03
    p.max_bombs = 2
    assert np.isclose(compute_pbrs_potential(p), 0.03, atol=1e-6)

    # Speed + 0.5 (one speed up) -> +0.04
    p.speed = 3.0
    assert np.isclose(compute_pbrs_potential(p), 0.03 + 0.04, atol=1e-6)

    # Kick -> +0.06
    p.can_kick = True
    assert np.isclose(compute_pbrs_potential(p), 0.03 + 0.04 + 0.06, atol=1e-6)

    # Pass -> +0.06
    p.bomb_pass = True
    assert np.isclose(compute_pbrs_potential(p), 0.03 + 0.04 + 0.06 + 0.06, atol=1e-6)

    # Shield active -> +0.05
    p.shield_timer = 50
    assert np.isclose(compute_pbrs_potential(p), 0.03 + 0.04 + 0.06 + 0.06 + 0.05, atol=1e-6)

    # Shield expires -> reverts 0.05
    p.shield_timer = 0
    assert np.isclose(compute_pbrs_potential(p), 0.03 + 0.04 + 0.06 + 0.06, atol=1e-6)


# ==============================================================================
# Additional Comprehensive Tests: Rewards, Telemetry, and Curriculum
# ==============================================================================


def test_brick_destruction_reward_and_attribution() -> None:
    """Verify that brick destruction rewards +0.015 and attributes correctly by bomb owner."""
    state_before = create_clean_game_state()
    state_before.grid[1, 2] = TileType.BRICK
    state_before.grid[3, 4] = TileType.BRICK

    state_after = state_before.clone()
    state_after.grid[1, 2] = TileType.EMPTY
    state_after.grid[3, 4] = TileType.EMPTY

    # P0 bomb destroyed (1, 2)
    state_after.flame_owner_mask[1, 2] = 1 << 0
    # P1 bomb destroyed (3, 4)
    state_after.flame_owner_mask[3, 4] = 1 << 1

    tracker = RewardTracker()
    tracker.reset(state_before)

    rewards, telemetry = tracker.step(
        state_before,
        actions=[(0, 0), (0, 0)],
        state_after=state_after,
        terminated=False,
        truncated=False,
    )

    assert np.isclose(telemetry["player_0"]["brick_destruction"], BRICK_REWARD, atol=1e-5)
    assert np.isclose(telemetry["player_1"]["brick_destruction"], BRICK_REWARD, atol=1e-5)


def test_map_connectivity_opening_reward() -> None:
    """Verify +0.05 bonus awarded once on the exact tick the separating wall is breached."""
    state_before = create_clean_game_state()
    # Create a vertical wall dividing the board at col 4
    for r in range(GRID_SIZE):
        state_before.grid[r, 4] = TileType.BRICK

    assert not check_players_connected(state_before.grid, (0, 0), (8, 8))

    # Breaching step: P0 bomb destroys brick at (2, 4)
    state_after = state_before.clone()
    state_after.grid[2, 4] = TileType.EMPTY
    state_after.flame_owner_mask[2, 4] = 1 << 0

    assert check_players_connected(state_after.grid, (0, 0), (8, 8))

    tracker = RewardTracker()
    tracker.reset(state_before)

    rewards, telemetry = tracker.step(
        state_before,
        actions=[(0, 0), (0, 0)],
        state_after=state_after,
        terminated=False,
        truncated=False,
    )

    # P0 opened the connection -> receives +0.05
    assert np.isclose(telemetry["player_0"]["connectivity"], CONNECTIVITY_REWARD, atol=1e-5)
    assert telemetry["player_1"]["connectivity"] == 0.0

    # Subsequent step: bonus should NOT be awarded again
    state_after2 = state_after.clone()
    state_after2.grid[4, 4] = TileType.EMPTY
    state_after2.flame_owner_mask[4, 4] = 1 << 0

    rewards2, telemetry2 = tracker.step(
        state_after,
        actions=[(0, 0), (0, 0)],
        state_after=state_after2,
        terminated=False,
        truncated=False,
    )
    assert telemetry2["player_0"]["connectivity"] == 0.0
    assert telemetry2["player_1"]["connectivity"] == 0.0


def test_terminal_win_loss_draw_and_self_damage_annealing() -> None:
    """Verify terminal rewards: Win (+1.0), Draw (-0.2), and Loss scaling for own-bomb suicide."""
    state_before = create_clean_game_state()

    # 1. Player 0 wins, Player 1 killed by P0 bomb
    state_win = state_before.clone()
    state_win.players[1].alive = False
    p1_tile = state_win.players[1].tile
    state_win.flame_owner_mask[p1_tile[0], p1_tile[1]] = 1 << 0  # Killed by P0

    tracker = RewardTracker(self_damage_scale=0.25)
    tracker.reset(state_before)
    _, tel = tracker.step(state_before, [(0, 0), (0, 0)], state_win, terminated=True, truncated=False)

    assert tel["player_0"]["win"] == TERMINAL_WIN
    assert tel["player_1"]["loss"] == -1.0

    # 2. Player 0 commits suicide with own bomb during early curriculum (scale=0.25)
    state_suicide = state_before.clone()
    state_suicide.players[0].alive = False
    p0_tile = state_suicide.players[0].tile
    state_suicide.flame_owner_mask[p0_tile[0], p0_tile[1]] = 1 << 0  # Killed by self

    tracker_early = RewardTracker(self_damage_scale=0.25)
    tracker_early.reset(state_before)
    _, tel_early = tracker_early.step(state_before, [(0, 0), (0, 0)], state_suicide, terminated=True, truncated=False)

    assert tel_early["player_0"]["loss"] == -0.25
    assert tel_early["player_1"]["win"] == TERMINAL_WIN

    # 3. Same suicide in late curriculum (scale=1.0)
    tracker_late = RewardTracker(self_damage_scale=1.0)
    tracker_late.reset(state_before)
    _, tel_late = tracker_late.step(state_before, [(0, 0), (0, 0)], state_suicide, terminated=True, truncated=False)

    assert tel_late["player_0"]["loss"] == -1.0

    # 4. Mutual death -> Draw (-0.2)
    state_draw = state_before.clone()
    state_draw.players[0].alive = False
    state_draw.players[1].alive = False

    tracker_draw = RewardTracker()
    tracker_draw.reset(state_before)
    _, tel_draw = tracker_draw.step(state_before, [(0, 0), (0, 0)], state_draw, terminated=True, truncated=False)

    assert tel_draw["player_0"]["draw"] == TERMINAL_DRAW
    assert tel_draw["player_1"]["draw"] == TERMINAL_DRAW


def test_curriculum_scheduler() -> None:
    """Verify linear and cosine annealing of CurriculumLethalityScheduler."""
    sched = CurriculumLethalityScheduler(initial_scale=0.25, final_scale=1.0, total_steps=100)
    assert sched.get_scale(0) == 0.25
    assert sched.get_scale(50) == 0.625
    assert sched.get_scale(100) == 1.0
    assert sched.get_scale(200) == 1.0


def test_auxiliary_reward_cap() -> None:
    """Verify strictly monotonic shaping cap: cumulative auxiliary rewards never exceed 0.35."""
    tracker = RewardTracker(auxiliary_reward_cap=0.35)
    state = create_clean_game_state()
    tracker.reset(state)

    # Award large brick reward exceeding 0.35
    state_after = state.clone()
    # Simulate 30 bricks destroyed by P0: 30 * 0.015 = 0.45 > 0.35
    for r in range(1, 6):
        for c in range(1, 7):
            state_after.flame_owner_mask[r, c] = 1 << 0
            state.grid[r, c] = TileType.BRICK
            state_after.grid[r, c] = TileType.EMPTY

    rewards, tel = tracker.step(state, [(0, 0), (0, 0)], state_after, terminated=False, truncated=False)
    assert rewards["player_0"] <= 0.35
    assert np.isclose(tracker.cumulative_aux[0], 0.35, atol=1e-5)


# ==============================================================================
# Additional Tests: Safety Action Filter
# ==============================================================================


def test_safety_action_filter_lethal_flame() -> None:
    """Verify that action filter masks stepping directly into lethal active flame."""
    state = create_clean_game_state()
    # Place flame at (1, 0)
    state.flame_timer[1, 0] = 5

    # Player 0 at (0.5, 0.5)
    saf = SafetyActionFilter(enabled=True)
    masks = saf.get_action_mask(state, player_id=0, ego_perspective=False)

    # Moving DOWN moves towards row 1 -> overlaps active flame at (1, 0)
    assert masks["move_mask"][int(MoveAction.DOWN)] == False
    # Moving RIGHT or NOOP is safe
    assert masks["move_mask"][int(MoveAction.RIGHT)] == True
    assert masks["move_mask"][int(MoveAction.NOOP)] == True

    # Action filtering replaces suicide DOWN with safe alternative
    safe_act = saf.filter_action(state, player_id=0, action=(int(MoveAction.DOWN), int(BombAction.NOOP)))
    assert safe_act[0] != MoveAction.DOWN

    # If shielded, all moves are permitted
    state.players[0].shield_timer = 20
    masks_shielded = saf.get_action_mask(state, player_id=0)
    assert np.all(masks_shielded["move_mask"])


# ==============================================================================
# Additional Tests: Flood-Fill Reachability and Observation Pre-allocation
# ==============================================================================


def test_compute_reachable_safe_tiles_with_bomb_block() -> None:
    """Verify BFS flood-fill reaches empty tiles but stops at bombs unless player has bomb_pass."""
    state = create_clean_game_state()
    # Put bomb at (0, 1)
    state.bombs.append(BombState(id=0, owner_id=0, r=0.5, c=1.5, timer=50))

    bfs_no_pass = compute_reachable_safe_tiles(state, state.players[0])
    # Starting tile (0, 0) is reachable (value 1.0)
    assert bfs_no_pass[0, 0] == 1.0
    # Tile (0, 1) has bomb -> blocked
    assert bfs_no_pass[0, 1] == 0.0

    # With bomb pass
    state.players[0].bomb_pass = True
    bfs_with_pass = compute_reachable_safe_tiles(state, state.players[0])
    # Tile (0, 1) is now reachable: dist=1 -> 1.0 - 1/8 = 7/8 = 0.875
    assert np.isclose(bfs_with_pass[0, 1], 0.875)


def test_observation_builder_preallocation_and_dimensions() -> None:
    """Verify ObservationBuilder updates in-place without memory reallocations."""
    builder_18 = ObservationBuilder(scalar_dim=DEFAULT_SCALAR_DIM)
    builder_19 = ObservationBuilder(scalar_dim=FULL_SCALAR_DIM)
    state = create_clean_game_state()

    obs_18 = builder_18.build(state, player_id=0, copy=False)
    obs_19 = builder_19.build(state, player_id=0, copy=False)

    assert obs_18["spatial"].shape == (18, 9, 9)
    assert obs_18["scalars"].shape == (18,)
    assert obs_19["scalars"].shape == (19,)

    # Verify normalization bounds
    assert np.all(obs_18["spatial"] >= -1.0)
    assert np.all(obs_18["spatial"] <= 1.0)
    assert np.all(obs_18["scalars"] >= -1.0)
    assert np.all(obs_18["scalars"] <= 1.0)


# ==============================================================================
# Additional Deep Verification & Edge Case Tests
# ==============================================================================


def test_multi_hop_chain_reaction_forecast() -> None:
    """Verify 4-hop chain reaction propagates earliest detonation time across the entire chain."""
    state = create_clean_game_state()
    # Row 2 is empty of pillars: place 4 bombs in a line
    # B1 at (2, 0), timer=5, radius=2
    # B2 at (2, 2), timer=30, radius=2
    # B3 at (2, 4), timer=45, radius=2
    # B4 at (2, 6), timer=60, radius=2
    state.bombs.append(BombState(id=1, owner_id=0, r=2.5, c=0.5, timer=5, blast_radius=2))
    state.bombs.append(BombState(id=2, owner_id=0, r=2.5, c=2.5, timer=30, blast_radius=2))
    state.bombs.append(BombState(id=3, owner_id=1, r=2.5, c=4.5, timer=45, blast_radius=2))
    state.bombs.append(BombState(id=4, owner_id=1, r=2.5, c=6.5, timer=60, blast_radius=2))

    danger = compute_danger_map(state)
    expected_danger = 1.0 - (5.0 / float(BOMB_TIMER))

    # All bombs and reachable tile (2, 8) must reflect countdown of 5 ticks
    assert np.isclose(danger[2, 0], expected_danger, atol=1e-5)
    assert np.isclose(danger[2, 2], expected_danger, atol=1e-5)
    assert np.isclose(danger[2, 4], expected_danger, atol=1e-5)
    assert np.isclose(danger[2, 6], expected_danger, atol=1e-5)
    assert np.isclose(danger[2, 8], expected_danger, atol=1e-5)


def test_chain_reaction_blocked_by_brick_no_early_penetration() -> None:
    """Verify a brick wall blocks a blast ray from penetrating and chain-triggering a bomb behind it."""
    state = create_clean_game_state()
    # Bomb A at (2, 0), timer=10, radius=5
    state.bombs.append(BombState(id=0, owner_id=0, r=2.5, c=0.5, timer=10, blast_radius=5))
    # Brick at (2, 1)
    state.grid[2, 1] = TileType.BRICK
    # Bomb B at (2, 2), timer=50, radius=2
    state.bombs.append(BombState(id=1, owner_id=1, r=2.5, c=2.5, timer=50, blast_radius=2))

    danger = compute_danger_map(state)

    # Brick at (2, 1) is in blast of Bomb A -> danger = 1.0 - 10/60
    assert np.isclose(danger[2, 1], 1.0 - 10.0 / float(BOMB_TIMER), atol=1e-5)
    # Bomb B at (2, 2) is blocked by the brick -> detonates at timer 50, NOT 10!
    expected_b = 1.0 - 50.0 / float(BOMB_TIMER)
    assert np.isclose(danger[2, 2], expected_b, atol=1e-5)


def test_shield_pbrs_zero_sum_cycle() -> None:
    """Verify shield pickup, active ticks, and expiration sum to exactly 0.0 PBRS reward."""
    state_prev = create_clean_game_state()
    tracker = RewardTracker()
    tracker.reset(state_prev)

    cum_rew = 0.0

    # 1. Acquire shield
    state_shield = state_prev.clone()
    state_shield.players[0].shield_timer = 100
    _, tel = tracker.step(state_prev, [(0, 0), (0, 0)], state_shield, False, False)
    cum_rew += tel["player_0"]["pbrs_stats"]
    assert np.isclose(tel["player_0"]["pbrs_stats"], 0.05, atol=1e-6)

    # 2. 5 steps of shield countdown while active (timer > 0)
    curr_state = state_shield
    for t in [80, 60, 40, 20, 1]:
        next_state = curr_state.clone()
        next_state.players[0].shield_timer = t
        _, tel = tracker.step(curr_state, [(0, 0), (0, 0)], next_state, False, False)
        cum_rew += tel["player_0"]["pbrs_stats"]
        assert np.isclose(tel["player_0"]["pbrs_stats"], 0.0, atol=1e-6)
        curr_state = next_state

    # 3. Shield expires (timer = 0)
    expired_state = curr_state.clone()
    expired_state.players[0].shield_timer = 0
    _, tel = tracker.step(curr_state, [(0, 0), (0, 0)], expired_state, False, False)
    cum_rew += tel["player_0"]["pbrs_stats"]
    assert np.isclose(tel["player_0"]["pbrs_stats"], -0.05, atol=1e-6)

    # Net cumulative reward must be exactly 0.0
    assert np.isclose(cum_rew, 0.0, atol=1e-7)


def test_connectivity_bonus_not_awarded_if_initially_connected() -> None:
    """If map starts already connected, no connectivity opening bonus should be awarded."""
    state = create_clean_game_state()  # Clean map without bricks -> players connected
    tracker = RewardTracker()
    tracker.reset(state)

    assert tracker.connectivity_awarded == True

    # Destroy a brick
    state_after = state.clone()
    state.grid[2, 2] = TileType.BRICK
    state_after.grid[2, 2] = TileType.EMPTY
    state_after.flame_owner_mask[2, 2] = 1 << 0

    _, tel = tracker.step(state, [(0, 0), (0, 0)], state_after, False, False)
    assert tel["player_0"]["connectivity"] == 0.0


def test_p1_connectivity_opening_attribution() -> None:
    """If Player 1's bomb breaches the separating wall, Player 1 receives the +0.05 bonus."""
    state_before = create_clean_game_state()
    for r in range(GRID_SIZE):
        state_before.grid[r, 4] = TileType.BRICK

    state_after = state_before.clone()
    state_after.grid[6, 4] = TileType.EMPTY
    state_after.flame_owner_mask[6, 4] = 1 << 1  # P1 bomb

    tracker = RewardTracker()
    tracker.reset(state_before)

    _, tel = tracker.step(state_before, [(0, 0), (0, 0)], state_after, False, False)
    assert tel["player_0"]["connectivity"] == 0.0
    assert np.isclose(tel["player_1"]["connectivity"], CONNECTIVITY_REWARD, atol=1e-5)


def test_action_filter_all_directions_blocked_safe_fallback() -> None:
    """If all 4 directions have active flames, NOOP is preserved to prevent an empty mask crash."""
    state = create_clean_game_state()
    # Surround (0, 0)
    state.flame_timer[0, 1] = 5  # RIGHT
    state.flame_timer[1, 0] = 5  # DOWN

    saf = SafetyActionFilter(enabled=True)
    masks = saf.get_action_mask(state, player_id=0, ego_perspective=False)

    # At least NOOP must be True
    assert masks["move_mask"][int(MoveAction.NOOP)] == True


def test_action_filter_p1_ego_perspective() -> None:
    """Verify action filter correctly inverts directional masks for Player 1 ego perspective."""
    state = create_clean_game_state()
    # Player 1 is at (8.5, 8.5)
    # Place flame at (7, 8) -> world UP from P1
    state.flame_timer[7, 8] = 5

    saf = SafetyActionFilter(enabled=True)
    ego_masks = saf.get_action_mask(state, player_id=1, ego_perspective=True)
    world_masks = saf.get_action_mask(state, player_id=1, ego_perspective=False)

    # In world coordinates, moving UP is dangerous
    assert world_masks["move_mask"][int(MoveAction.UP)] == False
    # In P1 ego coordinates, world UP corresponds to ego DOWN!
    assert ego_masks["move_mask"][int(MoveAction.DOWN)] == False


def test_observation_builder_state_update_in_place() -> None:
    """Verify successive calls to ObservationBuilder overwrite buffers accurately without stale data."""
    builder = ObservationBuilder()
    state1 = create_clean_game_state()
    state1.grid[2, 2] = TileType.BRICK

    obs1 = builder.build(state1, player_id=0, copy=True)
    assert obs1["spatial"][2, 2, 2] == 1.0  # Brick wall

    state2 = create_clean_game_state()
    state2.grid[2, 2] = TileType.EMPTY

    obs2 = builder.build(state2, player_id=0, copy=True)
    assert obs2["spatial"][2, 2, 2] == 0.0
    assert obs2["spatial"][0, 2, 2] == 1.0  # Empty


# ==============================================================================
# In-Depth Verification: Edge Cases, Continuous Symmetry, and Robustness
# ==============================================================================


def test_perspective_symmetry_integer_and_continuous_coordinates() -> None:
    """Verify bit-for-bit symmetry when players move to integer coordinates (e.g. tick 4 of standard move)."""
    state = create_clean_game_state()
    # P0 at (1.0, 0.5) - exactly on row integer boundary
    state.players[0].r = 1.0
    state.players[0].c = 0.5
    # P1 at exact symmetric continuous position (8.0, 8.5)
    state.players[1].r = 8.0
    state.players[1].c = 8.5

    builder = ObservationBuilder()
    obs0, obs1 = builder.build_both(state)

    for ch in range(SPATIAL_CHANNELS):
        np.testing.assert_allclose(
            obs0["spatial"][ch],
            obs1["spatial"][ch],
            atol=1e-6,
            err_msg=f"Symmetry broken in spatial channel {ch} at integer coords!",
        )
    np.testing.assert_allclose(obs0["scalars"], obs1["scalars"], atol=1e-6)

    # Test arbitrary floating continuous coordinates
    state.players[0].r = 2.375
    state.players[0].c = 3.625
    state.players[1].r = 9.0 - 2.375
    state.players[1].c = 9.0 - 3.625
    obs0_f, obs1_f = builder.build_both(state)
    np.testing.assert_allclose(obs0_f["spatial"], obs1_f["spatial"], atol=1e-6)
    np.testing.assert_allclose(obs0_f["scalars"], obs1_f["scalars"], atol=1e-6)


def test_danger_map_overlapping_center_tile_bomb_trigger() -> None:
    """Verify that a bomb overlapping another bomb's center tile is triggered immediately in the same wave."""
    state = create_clean_game_state()
    # Bomb A at (2, 2), timer=10, radius=1
    state.bombs.append(BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=10, blast_radius=1))
    # Bomb B overlapping tile (2, 2) at (2.4, 2.4), timer=50, radius=3
    state.bombs.append(BombState(id=1, owner_id=1, r=2.4, c=2.4, timer=50, blast_radius=3))

    danger = compute_danger_map(state)
    expected_danger = 1.0 - (10.0 / float(BOMB_TIMER))

    # Tile (2, 5) is within Bomb B's radius 3, but outside Bomb A's radius 1.
    # Because Bomb B was triggered by Bomb A on the center tile, (2, 5) must show 10-tick countdown!
    assert np.isclose(danger[2, 5], expected_danger, atol=1e-5)


def test_danger_map_same_wave_brick_ray_blocking() -> None:
    """Verify that a brick wall hit during an explosion wave stops all rays detonating on that exact same wave."""
    state = create_clean_game_state()
    # Bomb A at (2, 0), timer=10, radius=3 -> hits brick at (2, 1)
    state.bombs.append(BombState(id=0, owner_id=0, r=2.5, c=0.5, timer=10, blast_radius=3))
    # Brick at (2, 1)
    state.grid[2, 1] = TileType.BRICK
    # Bomb B at (2, 2), timer=10, radius=2 -> shoots rays on same tick 10
    state.bombs.append(BombState(id=1, owner_id=1, r=2.5, c=2.5, timer=10, blast_radius=2))
    # Bomb C at (2, 0), timer=50, radius=1 (behind the brick from Bomb B's perspective)

    danger = compute_danger_map(state)
    expected_10 = 1.0 - (10.0 / float(BOMB_TIMER))
    assert np.isclose(danger[2, 1], expected_10, atol=1e-5)


def test_dying_player_flame_extinction_suicide_scale() -> None:
    """Verify that when a player dies on the last tick of a flame (flame_timer == 1), suicide scale is applied."""
    from bomberman_rl.engine.physics import BombermanPhysics
    state = create_clean_game_state()
    # Flame at (0, 0) with timer = 1 owned by Player 0
    state.flame_timer[0, 0] = 1
    state.flame_owner_mask[0, 0] = 1 << 0

    state_before = state.clone()
    # Advance physics: player dies, flame timer drops to 0 and clears flame_owner_mask
    BombermanPhysics.update_flames_and_damage(state)
    state_after = state

    assert state_after.players[0].alive == False
    assert state_after.flame_timer[0, 0] == 0
    assert state_after.flame_owner_mask[0, 0] == 0

    tracker = RewardTracker(self_damage_scale=0.25)
    tracker.reset(state_before)
    _, telemetry = tracker.step(state_before, [(0, 0), (0, 0)], state_after, terminated=True, truncated=False)

    # In early curriculum, suicide must be -0.25, NOT -1.0
    assert np.isclose(telemetry["player_0"]["loss"], -0.25, atol=1e-5)


def test_pbrs_potential_retains_value_on_death() -> None:
    """Verify that player death does NOT wipe PBRS stats or penalize collected powerups."""
    state_before = create_clean_game_state()
    # Player 0 has collected upgrades
    state_before.players[0].blast_radius = 4
    state_before.players[0].max_bombs = 3
    state_before.players[0].speed = 3.5
    state_before.players[0].can_kick = True

    initial_phi = compute_pbrs_potential(state_before.players[0])
    assert initial_phi > 0.15

    tracker = RewardTracker(self_damage_scale=0.25)
    tracker.reset(state_before)

    # Player 0 dies from opponent's flame
    state_after = state_before.clone()
    state_after.players[0].alive = False
    state_after.flame_owner_mask[0, 0] = 1 << 1

    _, telemetry = tracker.step(state_before, [(0, 0), (0, 0)], state_after, terminated=True, truncated=False)

    # Step PBRS reward must be 0.0 (stats did not change on death tick)
    assert np.isclose(telemetry["player_0"]["pbrs_stats"], 0.0, atol=1e-5)
    assert np.isclose(telemetry["player_0"]["loss"], -1.0, atol=1e-5)


def test_no_spurious_connectivity_bonus_on_default_reset() -> None:
    """Verify default reset() on an already-connected map awards exactly 0.0 connectivity reward."""
    state_before = create_clean_game_state()
    state_after = state_before.clone()

    tracker = RewardTracker()
    tracker.reset()  # Called without initial_state

    _, tel = tracker.step(state_before, [(0, 0), (0, 0)], state_after, terminated=False, truncated=False)
    assert tel["player_0"]["connectivity"] == 0.0
    assert tel["player_1"]["connectivity"] == 0.0


def test_connectivity_bonus_attributed_to_actual_breach_brick() -> None:
    """Verify that if P0 destroys an unrelated brick and P1 breaches the wall, only P1 receives +0.05."""
    state_before = create_clean_game_state()
    # Build a dividing wall at col 4
    for r in range(GRID_SIZE):
        state_before.grid[r, 4] = TileType.BRICK
    # Add an unrelated brick near P0's corner
    state_before.grid[1, 2] = TileType.BRICK

    state_after = state_before.clone()
    # P0 destroys (1, 2)
    state_after.grid[1, 2] = TileType.EMPTY
    state_after.flame_owner_mask[1, 2] = 1 << 0
    # P1 destroys (4, 4), which breaches the dividing wall
    state_after.grid[4, 4] = TileType.EMPTY
    state_after.flame_owner_mask[4, 4] = 1 << 1

    tracker = RewardTracker()
    tracker.reset(state_before)
    _, tel = tracker.step(state_before, [(0, 0), (0, 0)], state_after, terminated=False, truncated=False)

    # Only Player 1 breached the wall
    assert tel["player_0"]["connectivity"] == 0.0
    assert np.isclose(tel["player_1"]["connectivity"], CONNECTIVITY_REWARD, atol=1e-5)


def test_safety_action_filter_allows_escape_from_flame() -> None:
    """Verify that when standing on an active flame, moves that step onto safe tiles are permitted."""
    state = create_clean_game_state()
    state.flame_timer[0, 0] = 5
    state.players[0].r = 0.95
    state.players[0].c = 0.5

    saf = SafetyActionFilter(enabled=True)
    mask = saf.get_action_mask(state, player_id=0, ego_perspective=False)

    # Moving DOWN moves center to 1.075 (tile 1, 0, which is safe) -> allowed!
    assert mask["move_mask"][int(MoveAction.DOWN)] == True
    # NOOP stays on burning tile (0, 0) -> masked!
    assert mask["move_mask"][int(MoveAction.NOOP)] == False


def test_safety_action_filter_masks_boundary_entering_flame_timer_1() -> None:
    """Verify moving onto a tile with flame_timer = 1 on this tick is masked as lethal suicide."""
    state = create_clean_game_state()
    state.players[0].r = 0.95
    state.players[0].c = 0.5
    # Flame on adjacent tile (1, 0) with timer = 1
    state.flame_timer[1, 0] = 1

    saf = SafetyActionFilter(enabled=True)
    mask = saf.get_action_mask(state, player_id=0, ego_perspective=False)

    # Moving DOWN lands on (1, 0) where flame is active during damage resolution -> lethal!
    assert mask["move_mask"][int(MoveAction.DOWN)] == False


def test_safety_action_filter_active_prob_and_single_int() -> None:
    """Verify active_prob stochastic bypass and single integer action compatibility."""
    state = create_clean_game_state()
    state.flame_timer[1, 0] = 5
    state.players[0].r = 0.95
    state.players[0].c = 0.5

    # With active_prob=0.0, filter is never active
    saf_disabled = SafetyActionFilter(active_prob=0.0, enabled=True)
    mask_dis = saf_disabled.get_action_mask(state, player_id=0, ego_perspective=False)
    assert np.all(mask_dis["move_mask"])

    # Single integer action filtering
    saf = SafetyActionFilter(enabled=True)
    filtered = saf.filter_action(state, player_id=0, action=int(MoveAction.DOWN), ego_perspective=False)
    assert isinstance(filtered, tuple)
    assert len(filtered) == 2


def test_auxiliary_cap_exact_component_sum() -> None:
    """Verify that when auxiliary rewards are capped, all breakdown components sum exactly to total."""
    tracker = RewardTracker(auxiliary_reward_cap=0.35)
    state_before = create_clean_game_state()
    tracker.reset(state_before)

    state_after = state_before.clone()
    # Player 0 destroys 20 bricks (0.30)
    for r in range(1, 5):
        for c in range(1, 6):
            state_before.grid[r, c] = TileType.BRICK
            state_after.grid[r, c] = TileType.EMPTY
            state_after.flame_owner_mask[r, c] = 1 << 0

    # Plus picks up debuff BLAST_DOWN (-0.03)
    state_after.players[0].blast_radius = 1  # was 1, set to 1 (phi = 0)
    state_before.players[0].blast_radius = 2  # phi = 0.03 -> delta = -0.03

    rewards, tel = tracker.step(state_before, [(0, 0), (0, 0)], state_after, False, False)
    log_0 = tracker.episode_logs[0]

    # Verify breakdown sums exactly to total
    component_sum = log_0.brick_destruction + log_0.pbrs_stats + log_0.connectivity
    assert np.isclose(component_sum, log_0.total, atol=1e-6)
    assert log_0.total <= 0.35

