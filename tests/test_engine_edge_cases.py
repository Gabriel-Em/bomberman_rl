"""Edge case and invariant tests for Bomberman RL physics and simulation engine."""

import pytest
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    SAFE_ZONE_P0,
    SAFE_ZONE_P1,
    NUM_BRICK_WALLS,
    BASE_PLAYER_SPEED,
    MAX_PLAYER_SPEED,
    SPEED_UP_INCREMENT,
    DEFAULT_MAX_BOMBS,
    MAX_BOMBS_LIMIT,
    MIN_BOMBS_LIMIT,
    DEFAULT_BLAST_RADIUS,
    MAX_BLAST_RADIUS,
    MIN_BLAST_RADIUS,
    SHIELD_DURATION,
    TileType,
    MoveAction,
    BombAction,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.engine.physics import BombermanPhysics
from bomberman_rl.engine.env import BombermanEnv


# ---------------------------------------------------------------------------
# Test 1: Spawn Safety & Map Invariants
# ---------------------------------------------------------------------------
def test_spawn_safety_and_map_invariants():
    """Across 50 seeds, verify:

    - Spawn safe zones (0,0), (0,1), (1,0) and (8,8), (8,7), (7,8) never contain walls.
    - Exactly 16 interior unbreakable pillars at odd coordinates (r%2==1 and c%2==1).
    - Exactly 41 breakable brick walls (70% of 59 empty tiles).
    """
    for seed in range(50):
        state = GameState.create_initial(seed=seed)

        # 1. Verify safe zones never contain walls
        for r, c in SAFE_ZONE_P0:
            assert state.grid[r, c] == TileType.EMPTY, f"Seed {seed}: P0 safe tile ({r},{c}) has wall"
        for r, c in SAFE_ZONE_P1:
            assert state.grid[r, c] == TileType.EMPTY, f"Seed {seed}: P1 safe tile ({r},{c}) has wall"

        # 2. Verify all 16 interior pillars
        pillar_count = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if r % 2 == 1 and c % 2 == 1:
                    assert state.grid[r, c] == TileType.PILLAR, f"Seed {seed}: Expected pillar at ({r},{c})"
                    pillar_count += 1
                else:
                    assert state.grid[r, c] != TileType.PILLAR, f"Seed {seed}: Unexpected pillar at ({r},{c})"
        assert pillar_count == 16, f"Seed {seed}: Expected 16 pillars, found {pillar_count}"

        # 3. Verify exact brick wall count
        brick_count = int(np.sum(state.grid == TileType.BRICK))
        assert brick_count == NUM_BRICK_WALLS, (
            f"Seed {seed}: Expected {NUM_BRICK_WALLS} bricks, found {brick_count}"
        )


# ---------------------------------------------------------------------------
# Test 2: Bomb Hysteresis (Leaving & Re-entering)
# ---------------------------------------------------------------------------
def test_bomb_hysteresis_leaving_and_reentering():
    """Verify a player placing a bomb can walk off the tile, cannot walk back

    onto the tile without BOMB_PASS, and *can* walk back onto it with BOMB_PASS.
    """
    state = GameState.create_initial(seed=42)
    # Clear grid to isolate bomb hysteresis
    state.grid.fill(TileType.EMPTY)

    # Player 0 places bomb at tile (0, 0)
    p0 = state.players[0]
    p0.r, p0.c = 0.5, 0.5
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])

    assert len(state.bombs) == 1
    bomb = state.bombs[0]
    assert bomb.tile == (0, 0)
    assert 0 in bomb.overlapping_players, "Player 0 must be in bomb overlapping_players upon placement"

    # Player 0 walks RIGHT away from tile (0, 0) into (0, 1)
    # Speed is 0.125 tiles/tick. It takes 7 ticks to reach c = 0.5 + 7 * 0.125 = 1.375
    # When c = 1.375, player's left edge is 1.375 - 0.35 = 1.025 > 1.0 (exited tile 0)
    for _ in range(8):
        BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert 0 not in bomb.overlapping_players, "Player 0 must have exited bomb.overlapping_players"
    pos_after_exit_c = p0.c

    # Now attempt to walk LEFT back onto the bomb tile without BOMB_PASS
    assert not p0.bomb_pass
    for _ in range(5):
        BombermanPhysics.step(state, [(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    # Left edge cannot penetrate tile (0, 0) (c - 0.35 must remain >= 1.0 -> c >= 1.35)
    assert p0.c >= 1.35 - 1e-5, f"Player entered bomb tile without BOMB_PASS: c={p0.c}"
    assert p0.tile == (0, 1)

    # Now grant BOMB_PASS power-up to Player 0
    p0.bomb_pass = True
    for _ in range(8):
        BombermanPhysics.step(state, [(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    # Player 0 should now easily enter tile (0, 0)
    assert p0.c < 1.0, f"Player with BOMB_PASS should walk back onto bomb tile: c={p0.c}"


# ---------------------------------------------------------------------------
# Test 3: Recursive Chain Detonation & Wall Stopping
# ---------------------------------------------------------------------------
def test_recursive_chain_detonation_and_wall_stopping():
    """Place Bomb A (radius 2) at (2,2) and Bomb B (timer 50) at (2,4) with a

    breakable wall at (2,5). Detonate Bomb A -> verify Bomb B detonates on the
    *exact same tick*, destroys the wall at (2,5), does NOT penetrate to (2,6),
    and any power-up spawned at (2,5) survives that tick's blast.
    """
    state = GameState.create_initial(seed=123)
    state.grid.fill(TileType.EMPTY)

    # Set up breakable brick wall at (2, 5)
    state.grid[2, 5] = TileType.BRICK

    # Bomb A at (2, 2) with radius 2, timer = 1 (will detonate on next tick)
    bomb_a = BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=1, blast_radius=2)
    # Bomb B at (2, 4) with radius 1, timer = 50
    bomb_b = BombState(id=1, owner_id=1, r=2.5, c=4.5, timer=50, blast_radius=1)
    state.bombs = [bomb_a, bomb_b]

    # Pre-place existing powerup at (2, 3) to test that existing exposed powerup is destroyed
    state.powerups[2, 3] = PowerUpType.SPEED_UP
    state.powerup_spawn_tick[2, 3] = -1

    # Force RNG so that a power-up WILL spawn at (2, 5) when brick is destroyed
    class MockRNG:
        def random(self):
            return 0.10

        def choice(self, seq, size=None, replace=True, p=None):
            return PowerUpType.SPEED_UP

    state.rng = MockRNG()

    # Advance 1 tick -> Bomb A detonates, triggers Bomb B in the SAME tick
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    # Both bombs must have detonated in this exact tick
    assert len(state.bombs) == 0, "Both bombs should have detonated"

    # Wall at (2, 5) must be destroyed
    assert state.grid[2, 5] == TileType.EMPTY, "Brick wall at (2, 5) should be destroyed"

    # Flame must be active at (2, 2), (2, 3), (2, 4), (2, 5)
    assert state.flame_timer[2, 2] > 0
    assert state.flame_timer[2, 3] > 0
    assert state.flame_timer[2, 4] > 0
    assert state.flame_timer[2, 5] > 0

    # Ray must NOT penetrate to (2, 6)
    assert state.flame_timer[2, 6] == 0, "Explosion must not penetrate beyond destroyed wall to (2, 6)"

    # Existing exposed powerup at (2, 3) must be destroyed
    assert state.powerups[2, 3] == PowerUpType.NONE, "Exposed powerup at (2, 3) should have been destroyed"

    # Newly spawned power-up at (2, 5) must survive that tick's blast!
    assert state.powerups[2, 5] != PowerUpType.NONE, "Powerup spawned at (2, 5) must survive current tick blast"

    # Telemetry: verify flame owner mask at (2, 4) contains both P0 (from Bomb A ray) and P1 (Bomb B center)
    assert state.flame_owner_mask[2, 4] == (1 | 2), "Flame at (2,4) should track both P0 and P1 owners"


# ---------------------------------------------------------------------------
# Test 4: Kicked Bomb Collisions
# ---------------------------------------------------------------------------
def test_kicked_bomb_collisions():
    """Verify a kicked bomb rolls at 0.25 tiles/tick, stops cleanly centered on

    the tile before a brick wall, another bomb, a player, or the map border,
    and can still detonate while rolling if its timer expires mid-flight.
    """
    # 4A. Stops cleanly centered before a brick wall
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.grid[2, 5] = TileType.BRICK

    bomb = BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=60, velocity_r=0.0, velocity_c=0.25)
    state.bombs = [bomb]
    # Move players out of the way
    state.players[0].r, state.players[0].c = 0.5, 0.5
    state.players[1].r, state.players[1].c = 8.5, 8.5

    # Rolling right at 0.25 tiles/tick:
    # Starts at c = 2.5. Tile (2, 5) is brick.
    # Tile 3 center: 3.5 (tick 4)
    # Tile 4 center: 4.5 (tick 8)
    # Next tile (2, 5) is blocked, so bomb stops at tile 4 center (2.5, 4.5)
    for tick in range(1, 9):
        BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
        expected_c = 2.5 + tick * 0.25
        assert abs(bomb.c - expected_c) < 1e-5, f"Tick {tick}: Bomb expected at c={expected_c}, got {bomb.c}"

    assert bomb.velocity_r == 0.0 and bomb.velocity_c == 0.0, "Bomb must stop before brick wall"
    assert abs(bomb.c - 4.5) < 1e-5 and abs(bomb.r - 2.5) < 1e-5, "Bomb must stop cleanly centered on tile (2, 4)"

    # 4B. Stops cleanly centered before map border
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.players[0].r, state.players[0].c = 0.5, 0.5
    state.players[1].r, state.players[1].c = 0.5, 1.5

    bomb = BombState(id=0, owner_id=0, r=2.5, c=7.5, timer=60, velocity_r=0.0, velocity_c=0.25)
    state.bombs = [bomb]
    # Center of tile 8 is 8.5 (tick 4). Next tile 9 is border!
    for _ in range(4):
        BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert abs(bomb.c - 8.5) < 1e-5
    assert bomb.velocity_c == 0.0, "Bomb must stop cleanly centered at map border"

    # 4C. Stops cleanly before another bomb
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.players[0].r, state.players[0].c = 0.5, 0.5
    state.players[1].r, state.players[1].c = 0.5, 1.5

    bomb1 = BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=60, velocity_r=0.0, velocity_c=0.25)
    bomb2 = BombState(id=1, owner_id=1, r=2.5, c=5.5, timer=60, velocity_r=0.0, velocity_c=0.0)
    state.bombs = [bomb1, bomb2]

    # bomb1 rolls towards tile 5 (where bomb2 is) -> should stop at center of tile 4 (4.5)
    for _ in range(8):
        BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert abs(bomb1.c - 4.5) < 1e-5
    assert bomb1.velocity_c == 0.0, "Bomb must stop cleanly before another bomb"

    # 4D. Stops cleanly before a player
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.players[0].r, state.players[0].c = 0.5, 0.5
    state.players[1].r, state.players[1].c = 2.5, 5.5  # Player 1 is on tile (2, 5)

    bomb = BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=60, velocity_r=0.0, velocity_c=0.25)
    state.bombs = [bomb]
    for _ in range(8):
        BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert abs(bomb.c - 4.5) < 1e-5
    assert bomb.velocity_c == 0.0, "Bomb must stop cleanly before a player"

    # 4E. Can still detonate while rolling if timer expires mid-flight
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.players[0].r, state.players[0].c = 0.5, 0.5
    state.players[1].r, state.players[1].c = 8.5, 8.5

    bomb = BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=3, velocity_r=0.0, velocity_c=0.25)
    state.bombs = [bomb]

    # After 3 ticks, bomb is mid-flight at c = 2.5 + 3 * 0.25 = 3.25 (tile 3) and detonates!
    for _ in range(3):
        BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert len(state.bombs) == 0, "Rolling bomb must detonate when timer expires mid-flight"
    # Detonated at tile (2, 3)
    assert state.flame_timer[2, 3] > 0, "Flame must spawn where rolling bomb was when timer expired"


# ---------------------------------------------------------------------------
# Test 5: Debuff Floors & Buff Caps
# ---------------------------------------------------------------------------
def test_debuff_floors_and_buff_caps():
    """Verify picking up 5 BLAST_DOWN and BOMB_DOWN items never drops

    blast_radius or max_bombs below 1, and MAX_BLAST caps at 8.
    """
    player = PlayerState(id=0, r=0.5, c=0.5, blast_radius=3, max_bombs=3)

    # Pick up 5 BLAST_DOWN
    for _ in range(5):
        BombermanPhysics.apply_powerup(player, PowerUpType.BLAST_DOWN)
    assert player.blast_radius == MIN_BLAST_RADIUS, f"Blast radius floor is {MIN_BLAST_RADIUS}, got {player.blast_radius}"

    # Pick up 5 BOMB_DOWN
    for _ in range(5):
        BombermanPhysics.apply_powerup(player, PowerUpType.BOMB_DOWN)
    assert player.max_bombs == MIN_BOMBS_LIMIT, f"Max bombs floor is {MIN_BOMBS_LIMIT}, got {player.max_bombs}"

    # MAX_BLAST sets blast_radius to 8
    BombermanPhysics.apply_powerup(player, PowerUpType.MAX_BLAST)
    assert player.blast_radius == MAX_BLAST_RADIUS

    # Adding BLAST_UP does not exceed 8
    BombermanPhysics.apply_powerup(player, PowerUpType.BLAST_UP)
    assert player.blast_radius == MAX_BLAST_RADIUS

    # BOMB_UP caps at 6
    for _ in range(10):
        BombermanPhysics.apply_powerup(player, PowerUpType.BOMB_UP)
    assert player.max_bombs == MAX_BOMBS_LIMIT

    # SPEED_UP caps at 4.5
    for _ in range(10):
        BombermanPhysics.apply_powerup(player, PowerUpType.SPEED_UP)
    assert player.speed == MAX_PLAYER_SPEED


# ---------------------------------------------------------------------------
# Test 6: Shield Expiration Boundary
# ---------------------------------------------------------------------------
def test_shield_expiration_boundary():
    """Player with shield_timer = 2 stands in a 10-tick flame.

    Verify they survive tick 1 and tick 2, and die on tick 3 when shield_timer
    hits 0.
    """
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.r, p0.c = 2.5, 2.5
    p0.shield_timer = 2
    p0.alive = True

    # Active 10-tick flame at (2, 2)
    state.flame_timer[2, 2] = 10
    state.flame_owner_mask[2, 2] = 2

    # Tick 1: Shield protects, timer goes 2 -> 1, flame goes 10 -> 9
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert p0.alive, "Player must survive tick 1 with active shield"
    assert p0.shield_timer == 1

    # Tick 2: Shield protects, timer goes 1 -> 0, flame goes 9 -> 8
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert p0.alive, "Player must survive tick 2 as shield expires"
    assert p0.shield_timer == 0

    # Tick 3: Shield is 0, flame is 8 > 0 -> player dies!
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert not p0.alive, "Player must die on tick 3 when shield_timer reaches 0 in active flame"


# ---------------------------------------------------------------------------
# Test 7: Simultaneous Death Tie
# ---------------------------------------------------------------------------
def test_simultaneous_death_tie():
    """Both players caught in the same blast on tick T results in done = True,

    winner = None (Draw).
    """
    env = BombermanEnv(seed=42)
    env.state.grid.fill(TileType.EMPTY)

    # Place both players adjacent to a bomb at (4, 4)
    env.state.players[0].r, env.state.players[0].c = 4.5, 3.5  # tile (4, 3)
    env.state.players[1].r, env.state.players[1].c = 4.5, 5.5  # tile (4, 5)

    bomb = BombState(id=0, owner_id=0, r=4.5, c=4.5, timer=1, blast_radius=2)
    env.state.bombs = [bomb]

    obs, rewards, terminated, truncated, info = env.step({
        "player_0": (MoveAction.NOOP, BombAction.NOOP),
        "player_1": (MoveAction.NOOP, BombAction.NOOP),
    })

    assert terminated, "Match must terminate when both players die"
    assert not env.state.players[0].alive, "Player 0 should be dead"
    assert not env.state.players[1].alive, "Player 1 should be dead"
    assert info["winner"] is None, "Winner must be None in simultaneous death"
    assert info["draw"] is True, "Draw must be True in simultaneous death"
    assert rewards["player_0"] == 0.0 and rewards["player_1"] == 0.0


# ---------------------------------------------------------------------------
# Test 8: Corner Sliding Mechanics
# ---------------------------------------------------------------------------
def test_corner_sliding_smooth_navigation():
    """Verify corner sliding nudges perpendicular coordinate toward aisle

    center when clipping an obstacle edge by < 0.30 tiles.
    """
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    # Pillar at (1, 1) occupies [1, 2] x [1, 2]
    state.grid[1, 1] = TileType.PILLAR

    # Open aisle is column 0 (center 0.5).
    # Player moves UP from row 2.0.
    # If c = 0.75, left edge is 0.40, right edge is 1.10.
    # Overlap into pillar (col 1) is 1.10 - 1.0 = 0.10 tiles (< 0.30 threshold).
    p0 = state.players[0]
    p0.r, p0.c = 2.0, 0.75

    # Move UP: should slide perpendicular toward c = 0.5 and advance in r
    BombermanPhysics.step(state, [(MoveAction.UP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    # Nudge moves c closer to 0.5 (from 0.75 by 0.125 -> 0.625)
    assert p0.c < 0.75, "Player c must nudge toward aisle center 0.5"
    assert abs(p0.c - 0.625) < 1e-5
    # Since c = 0.625 has right edge 0.975 < 1.0 (cleared pillar), player also advances UP
    assert p0.r < 2.0, "Player must smoothly advance UP around the corner"

    # Now verify that clipping by >= 0.30 tiles does NOT slide
    state.grid.fill(TileType.EMPTY)
    state.grid[1, 1] = TileType.PILLAR
    # If c = 1.0, right edge is 1.35. Overlap into col 1 is 0.35 >= 0.30 -> Snagged!
    p0.r, p0.c = 2.0, 1.0
    BombermanPhysics.step(state, [(MoveAction.UP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    # Player should not corner slide into aisle 0, forward movement blocked by pillar
    assert p0.r >= 2.0 - 1e-5, "Player clipping by >= 0.30 must snag on the wall"


# ---------------------------------------------------------------------------
# Test 9: Bomb Kicking Collision and Velocity Transfer
# ---------------------------------------------------------------------------
def test_bomb_kicking_transfer():
    """Verify that a player with can_kick=True walking into a stationary bomb

    imparts 0.25 tiles/tick velocity to the bomb in movement direction.
    """
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.r, p0.c = 2.5, 1.60  # Right against tile (2, 2)
    p0.can_kick = True

    bomb = BombState(id=0, owner_id=1, r=2.5, c=2.5, timer=60)
    state.bombs = [bomb]

    # Player 0 walks RIGHT into the stationary bomb
    BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    # Bomb must now have velocity_c = 0.25 (rolling RIGHT)
    assert bomb.velocity_c == 0.25, f"Bomb must be kicked with velocity_c=0.25, got {bomb.velocity_c}"
    assert bomb.velocity_r == 0.0


# ---------------------------------------------------------------------------
# Test 10: Environment Max Ticks Truncation & Draw
# ---------------------------------------------------------------------------
def test_env_max_ticks_truncation():
    """Verify episode truncates at MAX_TICKS with a draw if both players remain

    alive.
    """
    env = BombermanEnv(seed=42)
    env.state.grid.fill(TileType.EMPTY)

    # Set tick just before MAX_TICKS
    env.state.tick = MAX_TICKS - 1
    obs, rewards, terminated, truncated, info = env.step([
        (MoveAction.NOOP, BombAction.NOOP),
        (MoveAction.NOOP, BombAction.NOOP),
    ])

    assert truncated, "Episode must be truncated at tick 3600"
    assert not terminated
    assert info["draw"] is True
    assert info["winner"] is None


# ---------------------------------------------------------------------------
# Test 11: Corner Sliding In All Directions (DOWN, LEFT, RIGHT)
# ---------------------------------------------------------------------------
def test_corner_sliding_all_directions():
    """Verify corner sliding in DOWN, LEFT, and RIGHT directions."""
    # 11A. Moving DOWN clipping pillar at (3, 3)
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.grid[3, 3] = TileType.PILLAR  # pillar at [3, 4] x [3, 4]

    # Player at r = 2.65, c = 2.75 (open aisle is col 2, center 2.5)
    # Moving DOWN enters row 3. Right edge is 2.75 + 0.35 = 3.10. Overlap into col 3 is 0.10 < 0.30
    p0 = state.players[0]
    p0.r, p0.c = 2.65, 2.75
    BombermanPhysics.step(state, [(MoveAction.DOWN, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert p0.c < 2.75, "Moving DOWN must nudge c toward aisle center 2.5"
    assert p0.r > 2.65, "Moving DOWN must advance player around corner"

    # 11B. Moving LEFT clipping pillar at (3, 3)
    state.grid.fill(TileType.EMPTY)
    state.grid[3, 3] = TileType.PILLAR
    # Player at r = 2.75, c = 4.35 (open aisle is row 2, center 2.5)
    # Moving LEFT enters col 3. Bottom edge is 2.75 + 0.35 = 3.10. Overlap into row 3 is 0.10 < 0.30
    p0.r, p0.c = 2.75, 4.35
    BombermanPhysics.step(state, [(MoveAction.LEFT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert p0.r < 2.75, "Moving LEFT must nudge r toward aisle center 2.5"
    assert p0.c < 4.35, "Moving LEFT must advance player around corner"

    # 11C. Moving RIGHT clipping pillar at (3, 3)
    state.grid.fill(TileType.EMPTY)
    state.grid[3, 3] = TileType.PILLAR
    # Player at r = 2.75, c = 2.65 (open aisle is row 2, center 2.5)
    # Moving RIGHT enters col 3. Bottom edge is 2.75 + 0.35 = 3.10. Overlap into row 3 is 0.10 < 0.30
    p0.r, p0.c = 2.75, 2.65
    BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert p0.r < 2.75, "Moving RIGHT must nudge r toward aisle center 2.5"
    assert p0.c > 2.65, "Moving RIGHT must advance player around corner"


# ---------------------------------------------------------------------------
# Test 12: Bomb Placement Limits & Cooldown
# ---------------------------------------------------------------------------
def test_bomb_placement_limits_and_cooldown():
    """Verify player cannot place more bombs than max_bombs, and can place

    again after a bomb explodes.
    """
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.r, p0.c = 2.5, 2.5
    p0.max_bombs = 1

    # Place first bomb
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    assert len(state.bombs) == 1

    # Move to (2, 4) (safe distance away from bomb at (2, 2) with radius 1)
    for _ in range(16):
        BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    assert len(state.bombs) == 1, "Cannot exceed max_bombs = 1"

    # Wait until bomb explodes
    while len(state.bombs) > 0:
        BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert p0.alive, "Player 0 should be alive safely outside blast radius"
    # Now player can place a bomb again
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    assert len(state.bombs) == 1, "Player should be able to place bomb after previous detonated"


# ---------------------------------------------------------------------------
# Test 13: Cannot Place Bomb On Occupied Tile
# ---------------------------------------------------------------------------
def test_cannot_place_bomb_on_occupied_tile():
    """Even if max_bombs allows it, cannot place two bombs on the exact same tile."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.r, p0.c = 2.5, 2.5
    p0.max_bombs = 3

    # Place first bomb at (2, 2)
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    assert len(state.bombs) == 1

    # Try to place second bomb without moving away from tile (2, 2)
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.PLACE_BOMB), (MoveAction.NOOP, BombAction.NOOP)])
    assert len(state.bombs) == 1, "Cannot place a second bomb on the same tile"


# ---------------------------------------------------------------------------
# Test 14: Multi-Bomb Chain Detonation (A -> B -> C)
# ---------------------------------------------------------------------------
def test_three_bomb_recursive_chain_detonation():
    """Verify recursive chain detonation across 3 bombs detonating simultaneously."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    bomb_a = BombState(id=0, owner_id=0, r=4.5, c=1.5, timer=1, blast_radius=2)
    bomb_b = BombState(id=1, owner_id=1, r=4.5, c=3.5, timer=40, blast_radius=2)
    bomb_c = BombState(id=2, owner_id=0, r=4.5, c=5.5, timer=60, blast_radius=2)
    state.bombs = [bomb_a, bomb_b, bomb_c]

    # Advance 1 tick -> A triggers B, which triggers C in the EXACT SAME tick
    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert len(state.bombs) == 0, "All 3 bombs must detonate recursively on the exact same tick"
    assert state.flame_timer[4, 1] > 0
    assert state.flame_timer[4, 3] > 0
    assert state.flame_timer[4, 5] > 0
    assert state.flame_timer[4, 7] > 0


# ---------------------------------------------------------------------------
# Test 15: Pillars and Map Border Block Explosion Rays
# ---------------------------------------------------------------------------
def test_pillars_and_border_stop_explosion_rays():
    """Verify unbreakable pillars and map borders strictly stop explosion rays."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.grid[1, 1] = TileType.PILLAR

    # Bomb at (1, 2) with large radius 5
    bomb = BombState(id=0, owner_id=0, r=1.5, c=2.5, timer=1, blast_radius=5)
    state.bombs = [bomb]

    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    # Tile (1, 2) has flame
    assert state.flame_timer[1, 2] > 0
    # Pillar at (1, 1) must NOT have flame
    assert state.flame_timer[1, 1] == 0, "Pillars must never have flame"
    # Tile behind pillar at (1, 0) must NOT have flame
    assert state.flame_timer[1, 0] == 0, "Flame ray must not penetrate through unbreakable pillar"


# ---------------------------------------------------------------------------
# Test 16: Kicking Bombs in All Directions (UP, DOWN, LEFT)
# ---------------------------------------------------------------------------
def test_kicking_in_all_directions():
    """Verify kicking stationary bombs UP, DOWN, and LEFT."""
    for act, expected_v in [
        (MoveAction.UP, (-0.25, 0.0)),
        (MoveAction.DOWN, (0.25, 0.0)),
        (MoveAction.LEFT, (0.0, -0.25)),
    ]:
        state = GameState.create_initial(seed=42)
        state.grid.fill(TileType.EMPTY)
        p0 = state.players[0]
        p0.can_kick = True

        if act == MoveAction.UP:
            p0.r, p0.c = 3.40, 2.5
            bomb = BombState(id=0, owner_id=1, r=2.5, c=2.5, timer=60)
        elif act == MoveAction.DOWN:
            p0.r, p0.c = 1.60, 2.5
            bomb = BombState(id=0, owner_id=1, r=2.5, c=2.5, timer=60)
        elif act == MoveAction.LEFT:
            p0.r, p0.c = 2.5, 3.40
            bomb = BombState(id=0, owner_id=1, r=2.5, c=2.5, timer=60)

        state.bombs = [bomb]
        BombermanPhysics.step(state, [(act, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

        assert (bomb.velocity_r, bomb.velocity_c) == expected_v, (
            f"Action {act.name} expected velocity {expected_v}, got {(bomb.velocity_r, bomb.velocity_c)}"
        )


# ---------------------------------------------------------------------------
# Test 17: Cannot Kick Bomb Without can_kick
# ---------------------------------------------------------------------------
def test_cannot_kick_without_can_kick():
    """Player without can_kick colliding with a stationary bomb is blocked."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.r, p0.c = 2.5, 1.60
    p0.can_kick = False

    bomb = BombState(id=0, owner_id=1, r=2.5, c=2.5, timer=60)
    state.bombs = [bomb]

    BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert bomb.velocity_c == 0.0, "Bomb must not be kicked without can_kick"
    assert p0.c <= 1.65 + 1e-5, "Player must be stopped by the solid bomb"


# ---------------------------------------------------------------------------
# Test 18: Gymnasium Environment Reset, MultiDiscrete Actions, and Observation Format
# ---------------------------------------------------------------------------
def test_gym_env_integration():
    """Verify Gymnasium interface: reset, spaces, observation shapes and types."""
    env = BombermanEnv(seed=77)
    obs, info = env.reset(seed=77)

    assert "grid" in obs and obs["grid"].shape == (9, 9)
    assert "player_0" in obs and obs["player_0"].shape == (9,)
    assert "player_1" in obs and obs["player_1"].shape == (9,)
    assert obs["tick"] == 0
    assert info["winner"] is None
    assert info["draw"] is False

    # Test MultiDiscrete action step with dict
    actions = {
        "player_0": np.array([MoveAction.RIGHT, BombAction.PLACE_BOMB]),
        "player_1": np.array([MoveAction.LEFT, BombAction.NOOP]),
    }
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert obs["tick"] == 1
    assert not terminated
    assert not truncated
    assert rewards["player_0"] == 0.0 and rewards["player_1"] == 0.0


# ---------------------------------------------------------------------------
# Test 19: Performance Benchmark (Thousands of steps per second on CPU)
# ---------------------------------------------------------------------------
def test_simulation_speed_benchmark():
    """Verify simulation executes thousands of steps per second on CPU."""
    import time

    env = BombermanEnv(seed=123)
    env.reset(seed=123)

    num_steps = 2000
    start_time = time.perf_counter()
    for _ in range(num_steps):
        env.step([
            (MoveAction.RIGHT, BombAction.NOOP),
            (MoveAction.LEFT, BombAction.NOOP),
        ])
    elapsed = time.perf_counter() - start_time
    sps = num_steps / elapsed

    print(f"\nEngine speed: {sps:.0f} steps/second")
    # Must run at least 2,000 steps/sec
    assert sps > 2000, f"Engine should run >2000 steps/sec, achieved {sps:.0f}"


# ---------------------------------------------------------------------------
# Test 20: Game State Clone Determinism & Independence
# ---------------------------------------------------------------------------
def test_game_state_clone_determinism():
    """Verify state cloning preserves full determinism and bit generator state."""
    state1 = GameState.create_initial(seed=42)
    state2 = state1.clone()

    actions = [(MoveAction.RIGHT, BombAction.PLACE_BOMB), (MoveAction.LEFT, BombAction.NOOP)]
    for _ in range(10):
        BombermanPhysics.step(state1, actions)
        BombermanPhysics.step(state2, actions)

    assert state1.tick == state2.tick
    assert np.array_equal(state1.grid, state2.grid)
    assert np.array_equal(state1.flame_timer, state2.flame_timer)
    assert np.array_equal(state1.flame_owner_mask, state2.flame_owner_mask)
    assert state1.players[0].r == state2.players[0].r
    assert state1.players[0].c == state2.players[0].c
    assert state1.rng.random() == state2.rng.random()


# ---------------------------------------------------------------------------
# Test 21: Corner Sliding Monotonic Progress (No Backward Teleport)
# ---------------------------------------------------------------------------
def test_corner_sliding_no_backward_teleport():
    """Verify player moving in cardinal direction never moves in opposite direction."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.grid[1, 1] = TileType.PILLAR

    p0 = state.players[0]
    p0.r, p0.c = 2.30, 0.80

    BombermanPhysics.step(state, [(MoveAction.UP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert p0.r <= 2.30, f"Moving UP must not increase r (was 2.30, now {p0.r})"
    assert p0.c < 0.80, "Player must slide horizontally toward aisle center"


# ---------------------------------------------------------------------------
# Test 22: Outer Border Motion Precision (No Phantom Perpendicular Nudge)
# ---------------------------------------------------------------------------
def test_outer_border_sliding_no_phantom_snag():
    """Verify moving parallel to outer border wall does not trigger phantom snag or nudge."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.r, p0.c = 0.35, 0.50

    BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])
    assert abs(p0.r - 0.35) < 1e-5, f"Moving RIGHT along top border should not alter r, got {p0.r}"
    assert p0.c > 0.50, "Player should advance smoothly along border"


# ---------------------------------------------------------------------------
# Test 23: Brick Wall Blocks Multiple Detonations Within Same Tick
# ---------------------------------------------------------------------------
def test_brick_wall_blocks_second_explosion_same_tick():
    """Verify that a brick wall destroyed by one bomb stops subsequent flame rays in the exact same tick."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.grid[2, 5] = TileType.BRICK

    # Bomb 1 at (2, 4) with radius 1 destroys brick at (2, 5)
    b1 = BombState(id=0, owner_id=0, r=2.5, c=4.5, timer=1, blast_radius=1)
    # Bomb 2 at (2, 7) with radius 6 raycasts left towards (2, 5)
    b2 = BombState(id=1, owner_id=1, r=2.5, c=7.5, timer=1, blast_radius=6)
    state.bombs = [b1, b2]

    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert state.grid[2, 5] == TileType.EMPTY
    assert state.flame_timer[2, 5] > 0
    # Both bombs hit the wall at (2, 5)
    assert state.flame_owner_mask[2, 5] == (1 | 2)
    # Bomb 2's ray does NOT penetrate through the wall to (2, 3) or beyond
    assert state.flame_owner_mask[2, 3] == 1, "Tile (2, 3) must only have Bomb 1 owner mask"
    assert state.flame_timer[2, 1] == 0, "Flame ray must not penetrate through brick wall destroyed on same tick"


# ---------------------------------------------------------------------------
# Test 24: Rolling Bomb Collides With Entity Mid-Tile
# ---------------------------------------------------------------------------
def test_rolling_bomb_stops_at_player_mid_tile():
    """Verify that a rolling bomb does not roll through a player standing in its path."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)
    state.players[0].r, state.players[0].c = 0.5, 0.5
    state.players[1].r, state.players[1].c = 2.5, 3.5

    bomb = BombState(id=0, owner_id=0, r=2.5, c=3.0, timer=60, velocity_r=0.0, velocity_c=0.25)
    state.bombs = [bomb]

    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert bomb.velocity_c == 0.0, "Rolling bomb must stop when colliding with player"
    assert bomb.c <= 3.5, f"Bomb must not roll through player: c={bomb.c}"


# ---------------------------------------------------------------------------
# Test 25: Simultaneous Opposing Kicks Cancel Out
# ---------------------------------------------------------------------------
def test_simultaneous_opposing_kicks():
    """Verify opposing kicks from two players on the same tick cancel out neutrally."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p1 = state.players[1]
    p0.can_kick = True
    p1.can_kick = True

    bomb = BombState(id=0, owner_id=0, r=2.5, c=2.5, timer=60)
    state.bombs = [bomb]

    p0.r, p0.c = 2.5, 1.60
    p1.r, p1.c = 2.5, 3.40

    BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.LEFT, BombAction.NOOP)])

    assert bomb.velocity_c == 0.0, f"Opposing kicks must cancel out, got velocity_c={bomb.velocity_c}"
    assert bomb.velocity_r == 0.0


# ---------------------------------------------------------------------------
# Test 26: Kicking With Bomb Pass When Colliding From Outside
# ---------------------------------------------------------------------------
def test_can_kick_with_bomb_pass_from_outside():
    """Verify player with both can_kick and bomb_pass kicks stationary bomb when walking into it from outside."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    p0 = state.players[0]
    p0.can_kick = True
    p0.bomb_pass = True
    p0.r, p0.c = 2.5, 1.60

    bomb = BombState(id=0, owner_id=1, r=2.5, c=2.5, timer=60)
    state.bombs = [bomb]

    BombermanPhysics.step(state, [(MoveAction.RIGHT, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert bomb.velocity_c == 0.25, "Player with can_kick and bomb_pass should kick stationary bomb from outside"


# ---------------------------------------------------------------------------
# Test 27: Chain Reaction Triggers Rolling Bomb Across Tile Boundary
# ---------------------------------------------------------------------------
def test_chain_reaction_rolling_bomb_crossing_tile():
    """Verify rolling bomb spanning tile boundary is triggered by flame on either overlapped tile."""
    state = GameState.create_initial(seed=42)
    state.grid.fill(TileType.EMPTY)

    bomb_roll = BombState(id=0, owner_id=0, r=2.5, c=3.8, timer=50, velocity_c=0.25)
    bomb_det = BombState(id=1, owner_id=1, r=2.5, c=5.5, timer=1, blast_radius=1)
    state.bombs = [bomb_roll, bomb_det]

    BombermanPhysics.step(state, [(MoveAction.NOOP, BombAction.NOOP), (MoveAction.NOOP, BombAction.NOOP)])

    assert len(state.bombs) == 0, "Rolling bomb spanning tile boundary should chain detonate when hit at tile 4"

