"""Physics simulation: sub-tile movement, corner sliding, bomb kicking, and raycast explosions."""

from __future__ import annotations
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    DT,
    BASE_PLAYER_SPEED,
    SPEED_UP_INCREMENT,
    MAX_PLAYER_SPEED,
    PLAYER_HALF_SIZE,
    CORNER_SLIDE_THRESHOLD,
    BOMB_TIMER,
    FLAME_DURATION,
    KICK_SPEED,
    MAX_BOMBS_LIMIT,
    MIN_BOMBS_LIMIT,
    MAX_BLAST_RADIUS,
    MIN_BLAST_RADIUS,
    SHIELD_DURATION,
    POWERUP_DROP_PROB,
    POWERUP_TYPES_LIST,
    POWERUP_WEIGHTS_LIST,
    TileType,
    MoveAction,
    BombAction,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState


class BombermanPhysics:
    """Headless, deterministic physics engine for 1v1 Bomberman."""

    @staticmethod
    def step(
        state: GameState,
        actions: list[tuple[int, int]] | dict[int, tuple[int, int]] | np.ndarray,
    ) -> None:
        """Advance the simulation by one tick."""
        # 1. Parse actions: list of (move_action, bomb_action) for each player
        parsed_actions: list[tuple[int, int]] = []
        for pid in range(len(state.players)):
            if isinstance(actions, dict):
                act = actions.get(pid, (MoveAction.NOOP, BombAction.NOOP))
            elif isinstance(actions, (list, tuple, np.ndarray)):
                act = actions[pid]
            else:
                act = (MoveAction.NOOP, BombAction.NOOP)

            if isinstance(act, (list, tuple, np.ndarray)):
                m_act = int(act[0])
                b_act = int(act[1]) if len(act) > 1 else BombAction.NOOP
            else:
                # Fallback for discrete single int (0..5)
                val = int(act)
                if val == 5:
                    m_act = MoveAction.NOOP
                    b_act = BombAction.PLACE_BOMB
                else:
                    m_act = val
                    b_act = BombAction.NOOP
            parsed_actions.append((m_act, b_act))

        # 2. Process bomb placement
        for pid, (m_act, b_act) in enumerate(parsed_actions):
            player = state.players[pid]
            if not player.alive or b_act != BombAction.PLACE_BOMB:
                continue

            # Check max active bombs
            active_count = sum(1 for b in state.bombs if b.owner_id == player.id)
            if active_count >= player.max_bombs:
                continue

            tr = int(np.floor(player.r))
            tc = int(np.floor(player.c))

            # Tile must be within bounds and not already contain a bomb
            if not (0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE):
                continue
            tile_occupied = any(b.tile == (tr, tc) for b in state.bombs)
            if tile_occupied:
                continue

            # Identify overlapping players
            overlapping: set[int] = set()
            for p in state.players:
                if p.alive and BombermanPhysics.player_overlaps_tile(p, tr, tc):
                    overlapping.add(p.id)

            new_bomb = BombState(
                id=state.next_bomb_id,
                owner_id=player.id,
                r=tr + 0.5,
                c=tc + 0.5,
                timer=BOMB_TIMER,
                blast_radius=player.blast_radius,
                overlapping_players=overlapping,
            )
            state.next_bomb_id += 1
            state.bombs.append(new_bomb)

        # 3. Move rolling bombs
        for bomb in state.bombs:
            BombermanPhysics.update_rolling_bomb(bomb, state)

        # 4. Move players (including kicking stationary bombs & corner sliding)
        kick_impulses: list[tuple[BombState, float, float]] = []
        for pid, (m_act, _) in enumerate(parsed_actions):
            player = state.players[pid]
            if player.alive and m_act != MoveAction.NOOP:
                BombermanPhysics.move_player(
                    player, MoveAction(m_act), state, kick_impulses=kick_impulses
                )

        # Resolve accumulated kick impulses (opposing simultaneous kicks cancel out)
        bomb_net_impulse: dict[int, list[tuple[float, float]]] = {}
        for b, ir, ic in kick_impulses:
            bomb_net_impulse.setdefault(b.id, []).append((ir, ic))

        for b in state.bombs:
            if b.id in bomb_net_impulse:
                impulses = bomb_net_impulse[b.id]
                total_r = sum(ir for ir, ic in impulses)
                total_c = sum(ic for ir, ic in impulses)
                if abs(total_r) < 1e-5:
                    b.velocity_r = 0.0
                else:
                    b.velocity_r = 0.25 if total_r > 0 else -0.25

                if abs(total_c) < 1e-5:
                    b.velocity_c = 0.0
                else:
                    b.velocity_c = 0.25 if total_c > 0 else -0.25

        # 5. Update bomb pass-through hysteresis
        for bomb in state.bombs:
            b_tr, b_tc = bomb.tile
            to_remove: list[int] = []
            for pid in bomb.overlapping_players:
                p = state.players[pid]
                if not (p.alive and BombermanPhysics.player_overlaps_tile(p, b_tr, b_tc)):
                    to_remove.append(pid)
            for pid in to_remove:
                bomb.overlapping_players.remove(pid)

        # 6. Power-up collection
        for player in state.players:
            if not player.alive:
                continue
            pr = int(np.floor(player.r))
            pc = int(np.floor(player.c))
            if 0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE:
                ptype = state.powerups[pr, pc]
                if ptype != PowerUpType.NONE:
                    BombermanPhysics.apply_powerup(player, PowerUpType(ptype))
                    state.powerups[pr, pc] = PowerUpType.NONE

        # 7. Bomb detonations & recursive chain reactions
        BombermanPhysics.resolve_explosions(state)

        # 8. Active flame decay & player damage
        BombermanPhysics.update_flames_and_damage(state)

        # 9. Increment tick counter
        state.tick += 1

    @staticmethod
    def player_overlaps_tile(player: PlayerState, tr: int, tc: int) -> bool:
        """Check if player AABB intersects tile [tr, tr+1] x [tc, tc+1]."""
        p_min_r = player.r - PLAYER_HALF_SIZE
        p_max_r = player.r + PLAYER_HALF_SIZE
        p_min_c = player.c - PLAYER_HALF_SIZE
        p_max_c = player.c + PLAYER_HALF_SIZE
        return (
            p_max_r > tr and p_min_r < tr + 1 and
            p_max_c > tc and p_min_c < tc + 1
        )

    @staticmethod
    def bomb_overlaps_tile(bomb: BombState, tr: int, tc: int) -> bool:
        """Check if bomb bounding box (half-size 0.35) intersects tile [tr, tr+1] x [tc, tc+1]."""
        b_min_r = bomb.r - 0.35
        b_max_r = bomb.r + 0.35
        b_min_c = bomb.c - 0.35
        b_max_c = bomb.c + 0.35
        return (
            b_max_r > tr and b_min_r < tr + 1 and
            b_max_c > tc and b_min_c < tc + 1
        )

    @staticmethod
    def is_tile_blocked_for_bomb(r: int, c: int, bomb_id: int, state: GameState) -> bool:
        """Determine if a tile blocks a rolling bomb."""
        if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
            return True
        if state.grid[r, c] != TileType.EMPTY:
            return True
        for other in state.bombs:
            if other.id != bomb_id:
                if BombermanPhysics.bomb_overlaps_tile(other, r, c):
                    return True
        for p in state.players:
            if p.alive:
                if BombermanPhysics.player_overlaps_tile(p, r, c):
                    return True
        return False

    @staticmethod
    def update_rolling_bomb(bomb: BombState, state: GameState) -> None:
        """Simulate rolling bomb continuous motion and collisions."""
        if bomb.velocity_r == 0.0 and bomb.velocity_c == 0.0:
            return

        step_dist = 0.25  # KICK_SPEED (5.0 tiles/sec * 0.05 DT)

        if bomb.velocity_c > 0:  # Moving RIGHT
            curr_tr = int(round(bomb.r - 0.5))
            is_at_center = abs(bomb.c - (round(bomb.c - 0.5) + 0.5)) < 1e-5

            if is_at_center:
                tc = int(round(bomb.c - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(curr_tr, tc + 1, bomb.id, state):
                    bomb.c = tc + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.c += step_dist
            else:
                tc = int(np.floor(bomb.c - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(curr_tr, tc + 1, bomb.id, state):
                    bomb.c = tc + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.c += step_dist

            new_tc = int(round(bomb.c - 0.5))
            if abs(bomb.c - (new_tc + 0.5)) < 1e-5:
                bomb.c = new_tc + 0.5
                if BombermanPhysics.is_tile_blocked_for_bomb(curr_tr, new_tc + 1, bomb.id, state):
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0

        elif bomb.velocity_c < 0:  # Moving LEFT
            curr_tr = int(round(bomb.r - 0.5))
            is_at_center = abs(bomb.c - (round(bomb.c - 0.5) + 0.5)) < 1e-5

            if is_at_center:
                tc = int(round(bomb.c - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(curr_tr, tc - 1, bomb.id, state):
                    bomb.c = tc + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.c -= step_dist
            else:
                tc = int(np.ceil(bomb.c - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(curr_tr, tc - 1, bomb.id, state):
                    bomb.c = tc + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.c -= step_dist

            new_tc = int(round(bomb.c - 0.5))
            if abs(bomb.c - (new_tc + 0.5)) < 1e-5:
                bomb.c = new_tc + 0.5
                if BombermanPhysics.is_tile_blocked_for_bomb(curr_tr, new_tc - 1, bomb.id, state):
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0

        elif bomb.velocity_r > 0:  # Moving DOWN
            curr_tc = int(round(bomb.c - 0.5))
            is_at_center = abs(bomb.r - (round(bomb.r - 0.5) + 0.5)) < 1e-5

            if is_at_center:
                tr = int(round(bomb.r - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(tr + 1, curr_tc, bomb.id, state):
                    bomb.r = tr + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.r += step_dist
            else:
                tr = int(np.floor(bomb.r - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(tr + 1, curr_tc, bomb.id, state):
                    bomb.r = tr + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.r += step_dist

            new_tr = int(round(bomb.r - 0.5))
            if abs(bomb.r - (new_tr + 0.5)) < 1e-5:
                bomb.r = new_tr + 0.5
                if BombermanPhysics.is_tile_blocked_for_bomb(new_tr + 1, curr_tc, bomb.id, state):
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0

        elif bomb.velocity_r < 0:  # Moving UP
            curr_tc = int(round(bomb.c - 0.5))
            is_at_center = abs(bomb.r - (round(bomb.r - 0.5) + 0.5)) < 1e-5

            if is_at_center:
                tr = int(round(bomb.r - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(tr - 1, curr_tc, bomb.id, state):
                    bomb.r = tr + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.r -= step_dist
            else:
                tr = int(np.ceil(bomb.r - 0.5))
                if BombermanPhysics.is_tile_blocked_for_bomb(tr - 1, curr_tc, bomb.id, state):
                    bomb.r = tr + 0.5
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0
                    return
                bomb.r -= step_dist

            new_tr = int(round(bomb.r - 0.5))
            if abs(bomb.r - (new_tr + 0.5)) < 1e-5:
                bomb.r = new_tr + 0.5
                if BombermanPhysics.is_tile_blocked_for_bomb(new_tr - 1, curr_tc, bomb.id, state):
                    bomb.velocity_r = 0.0
                    bomb.velocity_c = 0.0

    @staticmethod
    def is_tile_obstacle_for_player(
        player: PlayerState, r: int, c: int, state: GameState
    ) -> tuple[bool, BombState | None]:
        """Check if tile (r, c) blocks player movement."""
        if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
            return True, None
        if state.grid[r, c] != TileType.EMPTY:
            return True, None
        for b in state.bombs:
            b_tr, b_tc = b.tile
            if b_tr == r and b_tc == c:
                if player.id in b.overlapping_players:
                    return False, b
                if player.bomb_pass:
                    # can_kick overrides bomb_pass when walking into a stationary bomb from outside
                    if player.can_kick and (b.velocity_r == 0.0 and b.velocity_c == 0.0):
                        return True, b
                    return False, b
                return True, b
        return False, None

    @staticmethod
    def move_player(
        player: PlayerState,
        move_act: MoveAction,
        state: GameState,
        kick_impulses: list[tuple[BombState, float, float]] | None = None,
    ) -> None:
        """Handle cardinal movement, corner sliding, and bomb kicking."""
        step_dist = player.speed * DT

        def record_kick(bomb: BombState, ir: float, ic: float) -> None:
            if kick_impulses is not None:
                kick_impulses.append((bomb, ir, ic))
            else:
                bomb.velocity_r = ir
                bomb.velocity_c = ic

        if move_act == MoveAction.UP:
            dr = -step_dist
            target_r = player.r + dr
            if target_r - PLAYER_HALF_SIZE < 0:
                player.r = PLAYER_HALF_SIZE
                return

            front_r = max(0.0, target_r - PLAYER_HALF_SIZE)
            tile_r = max(0, min(GRID_SIZE - 1, int(np.floor(front_r))))
            col_left = max(0.0, player.c - PLAYER_HALF_SIZE)
            col_right = min(float(GRID_SIZE) - 1e-7, player.c + PLAYER_HALF_SIZE)
            col1 = max(0, min(GRID_SIZE - 1, int(np.floor(col_left))))
            col2 = max(0, min(GRID_SIZE - 1, int(np.floor(col_right))))

            if col1 == col2:
                obs, b = BombermanPhysics.is_tile_obstacle_for_player(player, tile_r, col1, state)
                if obs:
                    if b is not None and player.can_kick and (b.velocity_r == 0 and b.velocity_c == 0):
                        record_kick(b, -0.25, 0.0)
                    max_r = tile_r + 1.0 + PLAYER_HALF_SIZE
                    player.r = min(player.r, max(target_r, max_r))
                else:
                    player.r = target_r
            else:
                obs1, b1 = BombermanPhysics.is_tile_obstacle_for_player(player, tile_r, col1, state)
                obs2, b2 = BombermanPhysics.is_tile_obstacle_for_player(player, tile_r, col2, state)

                if obs1 and obs2:
                    if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                        record_kick(b1, -0.25, 0.0)
                    elif b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                        record_kick(b2, -0.25, 0.0)
                    max_r = tile_r + 1.0 + PLAYER_HALF_SIZE
                    player.r = min(player.r, max(target_r, max_r))
                elif obs1 and not obs2:
                    clip = (col1 + 1.0) - (player.c - PLAYER_HALF_SIZE)
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = col2 + 0.5
                        nudge = min(step_dist, aisle_center - player.c)
                        player.c += nudge
                        if player.c - PLAYER_HALF_SIZE >= col1 + 1.0:
                            player.r = target_r
                        else:
                            max_r = tile_r + 1.0 + PLAYER_HALF_SIZE
                            player.r = min(player.r, max(target_r, max_r))
                    else:
                        if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                            record_kick(b1, -0.25, 0.0)
                        max_r = tile_r + 1.0 + PLAYER_HALF_SIZE
                        player.r = min(player.r, max(target_r, max_r))
                elif not obs1 and obs2:
                    clip = (player.c + PLAYER_HALF_SIZE) - col2
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = col1 + 0.5
                        nudge = min(step_dist, player.c - aisle_center)
                        player.c -= nudge
                        if player.c + PLAYER_HALF_SIZE <= col2:
                            player.r = target_r
                        else:
                            max_r = tile_r + 1.0 + PLAYER_HALF_SIZE
                            player.r = min(player.r, max(target_r, max_r))
                    else:
                        if b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                            record_kick(b2, -0.25, 0.0)
                        max_r = tile_r + 1.0 + PLAYER_HALF_SIZE
                        player.r = min(player.r, max(target_r, max_r))
                else:
                    player.r = target_r

        elif move_act == MoveAction.DOWN:
            dr = step_dist
            target_r = player.r + dr
            if target_r + PLAYER_HALF_SIZE > GRID_SIZE:
                player.r = float(GRID_SIZE) - PLAYER_HALF_SIZE
                return

            front_r = min(float(GRID_SIZE) - 1e-7, target_r + PLAYER_HALF_SIZE)
            tile_r = max(0, min(GRID_SIZE - 1, int(np.floor(front_r))))
            col_left = max(0.0, player.c - PLAYER_HALF_SIZE)
            col_right = min(float(GRID_SIZE) - 1e-7, player.c + PLAYER_HALF_SIZE)
            col1 = max(0, min(GRID_SIZE - 1, int(np.floor(col_left))))
            col2 = max(0, min(GRID_SIZE - 1, int(np.floor(col_right))))

            if col1 == col2:
                obs, b = BombermanPhysics.is_tile_obstacle_for_player(player, tile_r, col1, state)
                if obs:
                    if b is not None and player.can_kick and (b.velocity_r == 0 and b.velocity_c == 0):
                        record_kick(b, 0.25, 0.0)
                    min_r = tile_r - PLAYER_HALF_SIZE
                    player.r = max(player.r, min(target_r, min_r))
                else:
                    player.r = target_r
            else:
                obs1, b1 = BombermanPhysics.is_tile_obstacle_for_player(player, tile_r, col1, state)
                obs2, b2 = BombermanPhysics.is_tile_obstacle_for_player(player, tile_r, col2, state)

                if obs1 and obs2:
                    if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                        record_kick(b1, 0.25, 0.0)
                    elif b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                        record_kick(b2, 0.25, 0.0)
                    min_r = tile_r - PLAYER_HALF_SIZE
                    player.r = max(player.r, min(target_r, min_r))
                elif obs1 and not obs2:
                    clip = (col1 + 1.0) - (player.c - PLAYER_HALF_SIZE)
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = col2 + 0.5
                        nudge = min(step_dist, aisle_center - player.c)
                        player.c += nudge
                        if player.c - PLAYER_HALF_SIZE >= col1 + 1.0:
                            player.r = target_r
                        else:
                            min_r = tile_r - PLAYER_HALF_SIZE
                            player.r = max(player.r, min(target_r, min_r))
                    else:
                        if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                            record_kick(b1, 0.25, 0.0)
                        min_r = tile_r - PLAYER_HALF_SIZE
                        player.r = max(player.r, min(target_r, min_r))
                elif not obs1 and obs2:
                    clip = (player.c + PLAYER_HALF_SIZE) - col2
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = col1 + 0.5
                        nudge = min(step_dist, player.c - aisle_center)
                        player.c -= nudge
                        if player.c + PLAYER_HALF_SIZE <= col2:
                            player.r = target_r
                        else:
                            min_r = tile_r - PLAYER_HALF_SIZE
                            player.r = max(player.r, min(target_r, min_r))
                    else:
                        if b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                            record_kick(b2, 0.25, 0.0)
                        min_r = tile_r - PLAYER_HALF_SIZE
                        player.r = max(player.r, min(target_r, min_r))
                else:
                    player.r = target_r

        elif move_act == MoveAction.LEFT:
            dc = -step_dist
            target_c = player.c + dc
            if target_c - PLAYER_HALF_SIZE < 0:
                player.c = PLAYER_HALF_SIZE
                return

            front_c = max(0.0, target_c - PLAYER_HALF_SIZE)
            tile_c = max(0, min(GRID_SIZE - 1, int(np.floor(front_c))))
            row_top = max(0.0, player.r - PLAYER_HALF_SIZE)
            row_bottom = min(float(GRID_SIZE) - 1e-7, player.r + PLAYER_HALF_SIZE)
            row1 = max(0, min(GRID_SIZE - 1, int(np.floor(row_top))))
            row2 = max(0, min(GRID_SIZE - 1, int(np.floor(row_bottom))))

            if row1 == row2:
                obs, b = BombermanPhysics.is_tile_obstacle_for_player(player, row1, tile_c, state)
                if obs:
                    if b is not None and player.can_kick and (b.velocity_r == 0 and b.velocity_c == 0):
                        record_kick(b, 0.0, -0.25)
                    max_c = tile_c + 1.0 + PLAYER_HALF_SIZE
                    player.c = min(player.c, max(target_c, max_c))
                else:
                    player.c = target_c
            else:
                obs1, b1 = BombermanPhysics.is_tile_obstacle_for_player(player, row1, tile_c, state)
                obs2, b2 = BombermanPhysics.is_tile_obstacle_for_player(player, row2, tile_c, state)

                if obs1 and obs2:
                    if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                        record_kick(b1, 0.0, -0.25)
                    elif b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                        record_kick(b2, 0.0, -0.25)
                    max_c = tile_c + 1.0 + PLAYER_HALF_SIZE
                    player.c = min(player.c, max(target_c, max_c))
                elif obs1 and not obs2:
                    clip = (row1 + 1.0) - (player.r - PLAYER_HALF_SIZE)
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = row2 + 0.5
                        nudge = min(step_dist, aisle_center - player.r)
                        player.r += nudge
                        if player.r - PLAYER_HALF_SIZE >= row1 + 1.0:
                            player.c = target_c
                        else:
                            max_c = tile_c + 1.0 + PLAYER_HALF_SIZE
                            player.c = min(player.c, max(target_c, max_c))
                    else:
                        if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                            record_kick(b1, 0.0, -0.25)
                        max_c = tile_c + 1.0 + PLAYER_HALF_SIZE
                        player.c = min(player.c, max(target_c, max_c))
                elif not obs1 and obs2:
                    clip = (player.r + PLAYER_HALF_SIZE) - row2
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = row1 + 0.5
                        nudge = min(step_dist, player.r - aisle_center)
                        player.r -= nudge
                        if player.r + PLAYER_HALF_SIZE <= row2:
                            player.c = target_c
                        else:
                            max_c = tile_c + 1.0 + PLAYER_HALF_SIZE
                            player.c = min(player.c, max(target_c, max_c))
                    else:
                        if b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                            record_kick(b2, 0.0, -0.25)
                        max_c = tile_c + 1.0 + PLAYER_HALF_SIZE
                        player.c = min(player.c, max(target_c, max_c))
                else:
                    player.c = target_c

        elif move_act == MoveAction.RIGHT:
            dc = step_dist
            target_c = player.c + dc
            if target_c + PLAYER_HALF_SIZE > GRID_SIZE:
                player.c = float(GRID_SIZE) - PLAYER_HALF_SIZE
                return

            front_c = min(float(GRID_SIZE) - 1e-7, target_c + PLAYER_HALF_SIZE)
            tile_c = max(0, min(GRID_SIZE - 1, int(np.floor(front_c))))
            row_top = max(0.0, player.r - PLAYER_HALF_SIZE)
            row_bottom = min(float(GRID_SIZE) - 1e-7, player.r + PLAYER_HALF_SIZE)
            row1 = max(0, min(GRID_SIZE - 1, int(np.floor(row_top))))
            row2 = max(0, min(GRID_SIZE - 1, int(np.floor(row_bottom))))

            if row1 == row2:
                obs, b = BombermanPhysics.is_tile_obstacle_for_player(player, row1, tile_c, state)
                if obs:
                    if b is not None and player.can_kick and (b.velocity_r == 0 and b.velocity_c == 0):
                        record_kick(b, 0.0, 0.25)
                    min_c = tile_c - PLAYER_HALF_SIZE
                    player.c = max(player.c, min(target_c, min_c))
                else:
                    player.c = target_c
            else:
                obs1, b1 = BombermanPhysics.is_tile_obstacle_for_player(player, row1, tile_c, state)
                obs2, b2 = BombermanPhysics.is_tile_obstacle_for_player(player, row2, tile_c, state)

                if obs1 and obs2:
                    if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                        record_kick(b1, 0.0, 0.25)
                    elif b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                        record_kick(b2, 0.0, 0.25)
                    min_c = tile_c - PLAYER_HALF_SIZE
                    player.c = max(player.c, min(target_c, min_c))
                elif obs1 and not obs2:
                    clip = (row1 + 1.0) - (player.r - PLAYER_HALF_SIZE)
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = row2 + 0.5
                        nudge = min(step_dist, aisle_center - player.r)
                        player.r += nudge
                        if player.r - PLAYER_HALF_SIZE >= row1 + 1.0:
                            player.c = target_c
                        else:
                            min_c = tile_c - PLAYER_HALF_SIZE
                            player.c = max(player.c, min(target_c, min_c))
                    else:
                        if b1 is not None and player.can_kick and (b1.velocity_r == 0 and b1.velocity_c == 0):
                            record_kick(b1, 0.0, 0.25)
                        min_c = tile_c - PLAYER_HALF_SIZE
                        player.c = max(player.c, min(target_c, min_c))
                elif not obs1 and obs2:
                    clip = (player.r + PLAYER_HALF_SIZE) - row2
                    if clip < CORNER_SLIDE_THRESHOLD:
                        aisle_center = row1 + 0.5
                        nudge = min(step_dist, player.r - aisle_center)
                        player.r -= nudge
                        if player.r + PLAYER_HALF_SIZE <= row2:
                            player.c = target_c
                        else:
                            min_c = tile_c - PLAYER_HALF_SIZE
                            player.c = max(player.c, min(target_c, min_c))
                    else:
                        if b2 is not None and player.can_kick and (b2.velocity_r == 0 and b2.velocity_c == 0):
                            record_kick(b2, 0.0, 0.25)
                        min_c = tile_c - PLAYER_HALF_SIZE
                        player.c = max(player.c, min(target_c, min_c))
                else:
                    player.c = target_c

    @staticmethod
    def apply_powerup(player: PlayerState, powerup: PowerUpType) -> None:
        """Apply powerup buff or debuff with boundary clamps."""
        if powerup == PowerUpType.BLAST_UP:
            player.blast_radius = min(MAX_BLAST_RADIUS, player.blast_radius + 1)
        elif powerup == PowerUpType.BOMB_UP:
            player.max_bombs = min(MAX_BOMBS_LIMIT, player.max_bombs + 1)
        elif powerup == PowerUpType.SPEED_UP:
            player.speed = min(MAX_PLAYER_SPEED, player.speed + SPEED_UP_INCREMENT)
        elif powerup == PowerUpType.BLAST_DOWN:
            player.blast_radius = max(MIN_BLAST_RADIUS, player.blast_radius - 1)
        elif powerup == PowerUpType.BOMB_DOWN:
            player.max_bombs = max(MIN_BOMBS_LIMIT, player.max_bombs - 1)
        elif powerup == PowerUpType.KICK:
            player.can_kick = True
        elif powerup == PowerUpType.BOMB_PASS:
            player.bomb_pass = True
        elif powerup == PowerUpType.SHIELD:
            player.shield_timer = SHIELD_DURATION
        elif powerup == PowerUpType.MAX_BLAST:
            player.blast_radius = MAX_BLAST_RADIUS

    @staticmethod
    def resolve_explosions(state: GameState) -> None:
        """Detonate expiring bombs and recursively resolve chain reactions within the same tick."""
        # 1. Decrement timers of all active bombs
        for bomb in state.bombs:
            bomb.timer -= 1

        # 2. Queue all bombs whose timer reached <= 0
        detonating_bombs = [b for b in state.bombs if b.timer <= 0]
        detonated_ids: set[int] = set()
        destroyed_bricks_this_tick: set[tuple[int, int]] = set()

        while detonating_bombs:
            b = detonating_bombs.pop(0)
            if b.id in detonated_ids:
                continue
            detonated_ids.add(b.id)

            br = int(np.floor(b.r))
            bc = int(np.floor(b.c))

            # Flame on bomb center tile
            if 0 <= br < GRID_SIZE and 0 <= bc < GRID_SIZE:
                state.flame_timer[br, bc] = FLAME_DURATION
                state.flame_owner_mask[br, bc] |= (1 << b.owner_id)

                # Destroy existing exposed power-up on bomb tile if not newly spawned
                if state.powerups[br, bc] != PowerUpType.NONE:
                    if state.powerup_spawn_tick[br, bc] != state.tick:
                        state.powerups[br, bc] = PowerUpType.NONE

                # Check other bombs overlapping this center tile
                for other_b in state.bombs:
                    if other_b.id not in detonated_ids and other_b not in detonating_bombs:
                        if BombermanPhysics.bomb_overlaps_tile(other_b, br, bc):
                            other_b.timer = 0
                            detonating_bombs.append(other_b)

            # Raycast along 4 cardinal directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for step in range(1, b.blast_radius + 1):
                    rr = br + dr * step
                    cc = bc + dc * step

                    # Map border stops ray immediately (no flame outside)
                    if rr < 0 or rr >= GRID_SIZE or cc < 0 or cc >= GRID_SIZE:
                        break

                    # Pillar stops ray immediately (no flame on pillar)
                    if state.grid[rr, cc] == TileType.PILLAR:
                        break

                    # Brick destroyed earlier on this exact tick stops ray (no penetration)
                    if (rr, cc) in destroyed_bricks_this_tick:
                        state.flame_timer[rr, cc] = FLAME_DURATION
                        state.flame_owner_mask[rr, cc] |= (1 << b.owner_id)
                        break

                    # Brick wall stops ray, breaks, rolls powerup
                    if state.grid[rr, cc] == TileType.BRICK:
                        state.grid[rr, cc] = TileType.EMPTY
                        destroyed_bricks_this_tick.add((rr, cc))
                        state.flame_timer[rr, cc] = FLAME_DURATION
                        state.flame_owner_mask[rr, cc] |= (1 << b.owner_id)

                        # Roll for powerup spawn
                        if state.rng.random() < POWERUP_DROP_PROB:
                            chosen_powerup = state.rng.choice(
                                POWERUP_TYPES_LIST, p=POWERUP_WEIGHTS_LIST
                            )
                            state.powerups[rr, cc] = chosen_powerup
                            state.powerup_spawn_tick[rr, cc] = state.tick

                        # Ray does NOT penetrate further
                        break

                    # Tile is EMPTY: flame spawns
                    state.flame_timer[rr, cc] = FLAME_DURATION
                    state.flame_owner_mask[rr, cc] |= (1 << b.owner_id)

                    # Existing exposed power-up hit by flame ray is destroyed
                    if state.powerups[rr, cc] != PowerUpType.NONE:
                        if state.powerup_spawn_tick[rr, cc] != state.tick:
                            state.powerups[rr, cc] = PowerUpType.NONE

                    # Chain detonation: hit active bomb detonates immediately in same tick
                    for other_b in state.bombs:
                        if other_b.id not in detonated_ids and other_b not in detonating_bombs:
                            if BombermanPhysics.bomb_overlaps_tile(other_b, rr, cc):
                                other_b.timer = 0
                                detonating_bombs.append(other_b)

        # Remove all detonated bombs
        state.bombs = [b for b in state.bombs if b.id not in detonated_ids]

    @staticmethod
    def update_flames_and_damage(state: GameState) -> None:
        """Apply flame damage to players and decrement flame timers."""
        # 1. Damage check for each player
        for player in state.players:
            if not player.alive:
                continue

            pr = int(np.floor(player.r))
            pc = int(np.floor(player.c))

            had_shield = player.shield_timer > 0
            if had_shield:
                player.shield_timer -= 1

            if 0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE:
                if state.flame_timer[pr, pc] > 0:
                    if not had_shield:
                        player.alive = False

        # 2. Decrement flame timers
        active_flames = state.flame_timer > 0
        state.flame_timer[active_flames] -= 1
        extinguished = (state.flame_timer == 0) & active_flames
        state.flame_owner_mask[extinguished] = 0
