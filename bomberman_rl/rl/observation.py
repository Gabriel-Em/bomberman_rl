"""Vectorized ego-centric observation builder for 1v1 Bomberman.

Constructs 18x9x9 spatial feature tensors and normalized scalar vectors
with canonical 180-degree perspective invariance for self-play agents.
"""

from __future__ import annotations
from collections import deque
from typing import Any
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    BOMB_TIMER,
    FLAME_DURATION,
    MAX_BLAST_RADIUS,
    MAX_BOMBS_LIMIT,
    MAX_PLAYER_SPEED,
    BASE_PLAYER_SPEED,
    SPEED_UP_INCREMENT,
    SHIELD_DURATION,
    TileType,
    MoveAction,
    BombAction,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.engine.physics import BombermanPhysics

SPATIAL_CHANNELS = 18
DEFAULT_SCALAR_DIM = 18
FULL_SCALAR_DIM = 19


def flip_action_to_world(
    action: tuple[int, int] | list[int] | np.ndarray | int,
    player_id: int,
) -> tuple[int, int]:
    """Convert an agent's ego-centric action to world-coordinate space.

    For Player 0 (Top-Left spawn): identity.
    For Player 1 (Bottom-Right spawn): rotates directional moves 180 degrees
    (UP <-> DOWN, LEFT <-> RIGHT), leaving NOOP and bomb placement invariant.
    """
    if isinstance(action, (tuple, list, np.ndarray)):
        m_act = int(action[0])
        b_act = int(action[1]) if len(action) > 1 else int(BombAction.NOOP)
    else:
        val = int(action)
        if val == 5:
            m_act = int(MoveAction.NOOP)
            b_act = int(BombAction.PLACE_BOMB)
        else:
            m_act = val
            b_act = int(BombAction.NOOP)

    if player_id == 0:
        return (m_act, b_act)

    # Player 1 action flip
    if m_act == MoveAction.UP:
        flipped_m = int(MoveAction.DOWN)
    elif m_act == MoveAction.DOWN:
        flipped_m = int(MoveAction.UP)
    elif m_act == MoveAction.LEFT:
        flipped_m = int(MoveAction.RIGHT)
    elif m_act == MoveAction.RIGHT:
        flipped_m = int(MoveAction.LEFT)
    else:
        flipped_m = m_act

    return (flipped_m, b_act)


def flip_action_to_ego(
    action: tuple[int, int] | list[int] | np.ndarray | int,
    player_id: int,
) -> tuple[int, int]:
    """Convert a world-coordinate action to the agent's ego-centric space.

    Symmetric to flip_action_to_world since 180-degree reflection is self-inverting.
    """
    return flip_action_to_world(action, player_id)


def compute_danger_map(state: GameState, out: np.ndarray | None = None) -> np.ndarray:
    """Pre-compute the Chain-Aware Danger / Time-to-Impact Map in world coordinates.

    Accounts for immediate and recursive chain detonations across all active bombs.
    For every tile in a future blast path, stores 1.0 - (effective_ticks_to_explosion / 60.0).
    If an active flame is already present on the tile, sets the value to 1.0.
    """
    if out is None:
        danger_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    else:
        danger_map = out
        danger_map.fill(0.0)

    # Active flame tiles are immediately lethal (0 ticks to impact)
    danger_map[state.flame_timer > 0] = 1.0

    if not state.bombs:
        return danger_map

    # Multi-wave chain-reaction event simulation
    # Track the earliest explosion tick each tile experiences
    curr_grid = state.grid.copy()
    bomb_timers = {b.id: int(b.timer) for b in state.bombs}
    detonated_ids: set[int] = set()
    tile_earliest_tick: dict[tuple[int, int], int] = {}

    while len(detonated_ids) < len(state.bombs):
        # Find undetonated bomb with minimum remaining timer
        min_timer = None
        for b in state.bombs:
            if b.id not in detonated_ids:
                t = bomb_timers[b.id]
                if min_timer is None or t < min_timer:
                    min_timer = t

        if min_timer is None:
            break

        event_tick = min_timer
        wave = [b for b in state.bombs if b.id not in detonated_ids and bomb_timers[b.id] <= event_tick]
        for b in wave:
            detonated_ids.add(b.id)

        destroyed_in_this_wave: set[tuple[int, int]] = set()

        while wave:
            b = wave.pop(0)
            br = int(np.floor(b.r))
            bc = int(np.floor(b.c))

            # Bomb center tile is in blast path
            if 0 <= br < GRID_SIZE and 0 <= bc < GRID_SIZE:
                prev = tile_earliest_tick.get((br, bc), 9999)
                if event_tick < prev:
                    tile_earliest_tick[(br, bc)] = event_tick

                # Center tile overlap: trigger any other undetonated bomb overlapping center tile
                for other_b in state.bombs:
                    if other_b.id not in detonated_ids:
                        if BombermanPhysics.bomb_overlaps_tile(other_b, br, bc):
                            detonated_ids.add(other_b.id)
                            bomb_timers[other_b.id] = event_tick
                            wave.append(other_b)

                # Raycast along 4 cardinal directions
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for step in range(1, b.blast_radius + 1):
                        rr = br + dr * step
                        cc = bc + dc * step

                        if rr < 0 or rr >= GRID_SIZE or cc < 0 or cc >= GRID_SIZE:
                            break

                        if curr_grid[rr, cc] == TileType.PILLAR:
                            # Pillars stop ray, no flame on pillar
                            break

                        prev_t = tile_earliest_tick.get((rr, cc), 9999)
                        if event_tick < prev_t:
                            tile_earliest_tick[(rr, cc)] = event_tick

                        if (rr, cc) in destroyed_in_this_wave:
                            # Brick hit earlier on this wave stops ray penetration
                            break

                        if curr_grid[rr, cc] == TileType.BRICK:
                            # Brick stops ray and is scheduled for destruction after wave
                            destroyed_in_this_wave.add((rr, cc))
                            break

                        # Tile is passable: check if another undetonated bomb is hit
                        for other_b in state.bombs:
                            if other_b.id not in detonated_ids:
                                if BombermanPhysics.bomb_overlaps_tile(other_b, rr, cc):
                                    detonated_ids.add(other_b.id)
                                    bomb_timers[other_b.id] = event_tick
                                    wave.append(other_b)

        # After the wave finishes, destroyed bricks become empty for future waves
        for rr, cc in destroyed_in_this_wave:
            curr_grid[rr, cc] = TileType.EMPTY

    # Populate danger map from tile_earliest_tick
    for (r, c), ticks in tile_earliest_tick.items():
        danger = 1.0 - (float(ticks) / float(BOMB_TIMER))
        danger = float(np.clip(danger, 0.0, 1.0))
        if danger > danger_map[r, c]:
            danger_map[r, c] = danger

    # Preserve 1.0 for active flames
    danger_map[state.flame_timer > 0] = 1.0
    return danger_map


def compute_reachable_safe_tiles(
    state: GameState,
    player: PlayerState,
    out: np.ndarray | None = None,
    start_tile: tuple[int, int] | None = None,
    grid: np.ndarray | None = None,
    flame_timer: np.ndarray | None = None,
    bomb_tiles: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Flood-fill (BFS) from player tile up to depth 8 over passable tiles.

    Tile values are 1.0 - dist / 8.0.
    A tile is passable if:
    1. It is within the 9x9 grid.
    2. grid == TileType.EMPTY.
    3. flame_timer == 0 (not currently burning).
    4. Does not contain a bomb, unless the player has bomb_pass.
    """
    if out is None:
        flood_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    else:
        flood_map = out
        flood_map.fill(0.0)

    if not player.alive:
        return flood_map

    if start_tile is None:
        pr, pc = player.tile
    else:
        pr, pc = start_tile

    if not (0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE):
        return flood_map

    eff_grid = state.grid if grid is None else grid
    eff_flames = state.flame_timer if flame_timer is None else flame_timer
    eff_bomb_tiles = {b.tile for b in state.bombs} if bomb_tiles is None else bomb_tiles

    queue: deque[tuple[int, int, int]] = deque([(pr, pc, 0)])
    visited: set[tuple[int, int]] = {(pr, pc)}
    flood_map[pr, pc] = 1.0

    while queue:
        r, c, dist = queue.popleft()
        if dist >= 8:
            continue

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                if eff_grid[nr, nc] == TileType.EMPTY and eff_flames[nr, nc] == 0:
                    if player.bomb_pass or ((nr, nc) not in eff_bomb_tiles):
                        visited.add((nr, nc))
                        next_dist = dist + 1
                        flood_map[nr, nc] = 1.0 - (float(next_dist) / 8.0)
                        queue.append((nr, nc, next_dist))

    return flood_map


class ObservationBuilder:
    """Vectorized ego-centric observation generator with pre-allocated buffers.

    Pre-computes world-space channels once per tick, then maps to the perspective
    of Player 0 or Player 1 via 180-degree tensor rotation and velocity inversion.
    """

    def __init__(self, scalar_dim: int = DEFAULT_SCALAR_DIM) -> None:
        self.scalar_dim = scalar_dim

        # Pre-allocated world-space buffer for common channels 0..14
        self._world_spatial = np.zeros((15, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._world_danger = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._world_bfs_p0 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._world_bfs_p1 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Pre-allocated player ego observation buffers
        self._spatial_p0 = np.zeros((SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._spatial_p1 = np.zeros((SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._scalars_p0 = np.zeros(self.scalar_dim, dtype=np.float32)
        self._scalars_p1 = np.zeros(self.scalar_dim, dtype=np.float32)

    def _compute_world_channels(self, state: GameState) -> None:
        """Compute the 15 world-space channels once per tick."""
        self._world_spatial.fill(0.0)

        # Ch 0: Passable Empty Tile
        self._world_spatial[0] = (state.grid == TileType.EMPTY).astype(np.float32)
        # Ch 1: Interior Unbreakable Pillar
        self._world_spatial[1] = (state.grid == TileType.PILLAR).astype(np.float32)
        # Ch 2: Breakable Brick Wall
        self._world_spatial[2] = (state.grid == TileType.BRICK).astype(np.float32)

        # Ch 3..7: Active Bombs
        for b in state.bombs:
            tr, tc = b.tile
            if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                self._world_spatial[3, tr, tc] = 1.0
                self._world_spatial[4, tr, tc] = float(b.timer) / float(BOMB_TIMER)
                self._world_spatial[5, tr, tc] = float(b.blast_radius) / float(MAX_BLAST_RADIUS)
                # Bomb velocity: 0.25 tiles/tick = 5.0 tiles/sec -> normalized to [-1, 1]
                vy_norm = float(np.clip(b.velocity_r / 0.25, -1.0, 1.0))
                vx_norm = float(np.clip(b.velocity_c / 0.25, -1.0, 1.0))
                self._world_spatial[6, tr, tc] = vy_norm
                self._world_spatial[7, tr, tc] = vx_norm

        # Ch 8: Chain-Aware Danger Map
        compute_danger_map(state, out=self._world_danger)
        self._world_spatial[8] = self._world_danger

        # Ch 9: Active Flame Remaining Duration
        self._world_spatial[9] = state.flame_timer.astype(np.float32) / float(FLAME_DURATION)

        # Ch 10..14: Power-Up Channels
        powerup_coords = np.argwhere(state.powerups != PowerUpType.NONE)
        for r, c in powerup_coords:
            ptype = state.powerups[r, c]
            if ptype == PowerUpType.BLAST_UP:
                self._world_spatial[10, r, c] = 1.0 / 8.0
            elif ptype == PowerUpType.BLAST_DOWN:
                self._world_spatial[10, r, c] = -1.0 / 8.0
            elif ptype == PowerUpType.MAX_BLAST:
                self._world_spatial[10, r, c] = 1.0
            elif ptype == PowerUpType.BOMB_UP:
                self._world_spatial[11, r, c] = 1.0 / 6.0
            elif ptype == PowerUpType.BOMB_DOWN:
                self._world_spatial[11, r, c] = -1.0 / 6.0
            elif ptype == PowerUpType.SPEED_UP:
                self._world_spatial[12, r, c] = float(SPEED_UP_INCREMENT / (MAX_PLAYER_SPEED - BASE_PLAYER_SPEED))
            elif ptype == PowerUpType.KICK:
                self._world_spatial[13, r, c] = 1.0
            elif ptype == PowerUpType.BOMB_PASS:
                self._world_spatial[13, r, c] = -1.0
            elif ptype == PowerUpType.SHIELD:
                self._world_spatial[14, r, c] = 1.0

    def _build_scalars(
        self,
        state: GameState,
        player_id: int,
        out: np.ndarray,
    ) -> np.ndarray:
        """Construct the normalized scalar telemetry vector."""
        self_p = state.players[player_id]
        opp_p = state.players[1 - player_id]

        if player_id == 0:
            self_r, self_c = self_p.r, self_p.c
            opp_r, opp_c = opp_p.r, opp_p.c
        else:
            self_r, self_c = 9.0 - self_p.r, 9.0 - self_p.c
            opp_r, opp_c = 9.0 - opp_p.r, 9.0 - opp_p.c

        # Sub-tile offsets and relative displacement in ego frame
        self_off_r = self_r - np.floor(self_r) - 0.5
        self_off_c = self_c - np.floor(self_c) - 0.5
        rel_dy = (opp_r - self_r) / 9.0
        rel_dx = (opp_c - self_c) / 9.0

        # Self stats
        self_bombs = sum(1 for b in state.bombs if b.owner_id == self_p.id)
        self_active_bombs_ratio = float(self_bombs) / float(self_p.max_bombs)
        self_max_bombs_norm = float(self_p.max_bombs) / float(MAX_BOMBS_LIMIT)
        self_blast_radius_norm = float(self_p.blast_radius) / float(MAX_BLAST_RADIUS)
        self_speed_norm = (self_p.speed - BASE_PLAYER_SPEED) / (MAX_PLAYER_SPEED - BASE_PLAYER_SPEED)
        self_kick = 1.0 if self_p.can_kick else 0.0
        self_pass = 1.0 if self_p.bomb_pass else 0.0
        self_shield = float(self_p.shield_timer) / float(SHIELD_DURATION)

        # Opponent stats
        opp_bombs = sum(1 for b in state.bombs if b.owner_id == opp_p.id)
        opp_active_bombs_ratio = float(opp_bombs) / float(opp_p.max_bombs)
        opp_max_bombs_norm = float(opp_p.max_bombs) / float(MAX_BOMBS_LIMIT)
        opp_blast_radius_norm = float(opp_p.blast_radius) / float(MAX_BLAST_RADIUS)
        opp_speed_norm = (opp_p.speed - BASE_PLAYER_SPEED) / (MAX_PLAYER_SPEED - BASE_PLAYER_SPEED)
        opp_kick = 1.0 if opp_p.can_kick else 0.0
        opp_pass = 1.0 if opp_p.bomb_pass else 0.0
        opp_shield = float(opp_p.shield_timer) / float(SHIELD_DURATION)

        # Match time remaining
        match_time = float(max(0, MAX_TICKS - state.tick)) / float(MAX_TICKS)

        if len(out) == 18:
            # 18 floats: self offset (2), opp rel (2), self stats (7), opp upgrades (6), match time (1)
            out[0] = self_off_r
            out[1] = self_off_c
            out[2] = rel_dy
            out[3] = rel_dx
            out[4] = self_active_bombs_ratio
            out[5] = self_max_bombs_norm
            out[6] = self_blast_radius_norm
            out[7] = self_speed_norm
            out[8] = self_kick
            out[9] = self_pass
            out[10] = self_shield
            out[11] = opp_max_bombs_norm
            out[12] = opp_blast_radius_norm
            out[13] = opp_speed_norm
            out[14] = opp_kick
            out[15] = opp_pass
            out[16] = opp_shield
            out[17] = match_time
        else:
            # 19 floats (full): includes opp_active_bombs_ratio
            out[0] = self_off_r
            out[1] = self_off_c
            out[2] = rel_dy
            out[3] = rel_dx
            out[4] = self_active_bombs_ratio
            out[5] = self_max_bombs_norm
            out[6] = self_blast_radius_norm
            out[7] = self_speed_norm
            out[8] = self_kick
            out[9] = self_pass
            out[10] = self_shield
            out[11] = opp_active_bombs_ratio
            out[12] = opp_max_bombs_norm
            out[13] = opp_blast_radius_norm
            out[14] = opp_speed_norm
            out[15] = opp_kick
            out[16] = opp_pass
            out[17] = opp_shield
            if len(out) > 18:
                out[18] = match_time

        return out

    def build_both(
        self,
        state: GameState,
        copy: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute ego observations for both players in a single vectorized pass."""
        self._compute_world_channels(state)

        p0 = state.players[0]
        p1 = state.players[1]

        # Player 0 ego coordinates (identical to world)
        tr0 = int(np.clip(np.floor(p0.r), 0, GRID_SIZE - 1))
        tc0 = int(np.clip(np.floor(p0.c), 0, GRID_SIZE - 1))
        opp_tr0 = int(np.clip(np.floor(p1.r), 0, GRID_SIZE - 1))
        opp_tc0 = int(np.clip(np.floor(p1.c), 0, GRID_SIZE - 1))

        # Player 1 ego coordinates (180-degree reflection)
        p1_ego_r = 9.0 - p1.r
        p1_ego_c = 9.0 - p1.c
        tr1 = int(np.clip(np.floor(p1_ego_r), 0, GRID_SIZE - 1))
        tc1 = int(np.clip(np.floor(p1_ego_c), 0, GRID_SIZE - 1))

        p0_ego_r = 9.0 - p0.r
        p0_ego_c = 9.0 - p0.c
        opp_tr1 = int(np.clip(np.floor(p0_ego_r), 0, GRID_SIZE - 1))
        opp_tc1 = int(np.clip(np.floor(p0_ego_c), 0, GRID_SIZE - 1))

        # Compute reachable safe tiles flood-fill for both
        compute_reachable_safe_tiles(
            state,
            p0,
            out=self._world_bfs_p0,
            start_tile=(tr0, tc0),
        )

        ego_grid_p1 = np.rot90(state.grid, 2)
        ego_flames_p1 = np.rot90(state.flame_timer, 2)
        ego_bomb_tiles_p1 = {
            (
                int(np.clip(np.floor(9.0 - b.r), 0, GRID_SIZE - 1)),
                int(np.clip(np.floor(9.0 - b.c), 0, GRID_SIZE - 1)),
            )
            for b in state.bombs
        }
        compute_reachable_safe_tiles(
            state,
            p1,
            out=self._world_bfs_p1,
            start_tile=(tr1, tc1),
            grid=ego_grid_p1,
            flame_timer=ego_flames_p1,
            bomb_tiles=ego_bomb_tiles_p1,
        )

        # Assemble Player 0 Spatial Tensor
        self._spatial_p0[:15] = self._world_spatial
        self._spatial_p0[15].fill(0.0)
        if p0.alive:
            self._spatial_p0[15, tr0, tc0] = 1.0

        self._spatial_p0[16].fill(0.0)
        if p1.alive:
            self._spatial_p0[16, opp_tr0, opp_tc0] = 1.0

        self._spatial_p0[17] = self._world_bfs_p0
        self._build_scalars(state, 0, self._scalars_p0)

        # Assemble Player 1 Spatial Tensor via 180-degree rotation of world channels
        for k in range(15):
            self._spatial_p1[k] = np.rot90(self._world_spatial[k], 2)

        # Invert directional velocities for Player 1 ego view
        self._spatial_p1[6] = -self._spatial_p1[6]
        self._spatial_p1[7] = -self._spatial_p1[7]

        # For bomb channels 3..7, populate directly in ego frame to ensure exact continuous symmetry
        self._spatial_p1[3:8].fill(0.0)
        for b in state.bombs:
            b_ego_r = 9.0 - b.r
            b_ego_c = 9.0 - b.c
            b_tr = int(np.clip(np.floor(b_ego_r), 0, GRID_SIZE - 1))
            b_tc = int(np.clip(np.floor(b_ego_c), 0, GRID_SIZE - 1))
            self._spatial_p1[3, b_tr, b_tc] = 1.0
            self._spatial_p1[4, b_tr, b_tc] = float(b.timer) / float(BOMB_TIMER)
            self._spatial_p1[5, b_tr, b_tc] = float(b.blast_radius) / float(MAX_BLAST_RADIUS)
            vy_norm = float(np.clip(-b.velocity_r / 0.25, -1.0, 1.0))
            vx_norm = float(np.clip(-b.velocity_c / 0.25, -1.0, 1.0))
            self._spatial_p1[6, b_tr, b_tc] = vy_norm
            self._spatial_p1[7, b_tr, b_tc] = vx_norm

        # P1 Self and Opponent maps in ego perspective
        self._spatial_p1[15].fill(0.0)
        if p1.alive:
            self._spatial_p1[15, tr1, tc1] = 1.0

        self._spatial_p1[16].fill(0.0)
        if p0.alive:
            self._spatial_p1[16, opp_tr1, opp_tc1] = 1.0

        self._spatial_p1[17] = self._world_bfs_p1
        self._build_scalars(state, 1, self._scalars_p1)

        if copy:
            obs_0 = {
                "spatial": self._spatial_p0.copy(),
                "scalars": self._scalars_p0.copy(),
            }
            obs_1 = {
                "spatial": self._spatial_p1.copy(),
                "scalars": self._scalars_p1.copy(),
            }
        else:
            obs_0 = {
                "spatial": self._spatial_p0,
                "scalars": self._scalars_p0,
            }
            obs_1 = {
                "spatial": self._spatial_p1,
                "scalars": self._scalars_p1,
            }

        return obs_0, obs_1

    def build(
        self,
        state: GameState,
        player_id: int,
        copy: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate ego observation for a single player."""
        obs_0, obs_1 = self.build_both(state, copy=copy)
        return obs_0 if player_id == 0 else obs_1
