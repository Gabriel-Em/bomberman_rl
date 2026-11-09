"""Safety Action Filter for trivial 1-tick suicides during early curriculum exploration.

Prevents agents from mindlessly walking into active lethal flames or committing
immediate suicide, avoiding the Pommerman cowardice trap without enforcing rigid
scripted paths or overengineered positional heuristics.
"""

from __future__ import annotations
from collections import deque
from typing import Any
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    PLAYER_HALF_SIZE,
    MoveAction,
    BombAction,
    TileType,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.engine.physics import BombermanPhysics
from bomberman_rl.rl.observation import flip_action_to_world, flip_action_to_ego, compute_danger_map


def is_tile_threatened(
    state: GameState,
    r: int,
    c: int,
    bombs: list[BombState] | None = None,
) -> bool:
    """Check if discrete tile (r, c) is covered by any active bomb's blast cross."""
    target_bombs = state.bombs if bombs is None else bombs
    for b in target_bombs:
        br, bc = int(np.floor(b.r)), int(np.floor(b.c))
        if (r, c) == (br, bc):
            return True
        if r == br and abs(c - bc) <= b.blast_radius:
            step = 1 if c > bc else -1
            blocked = False
            for cc in range(bc + step, c, step):
                if state.grid[r, cc] != TileType.EMPTY:
                    blocked = True
                    break
            if not blocked:
                return True
        elif c == bc and abs(r - br) <= b.blast_radius:
            step = 1 if r > br else -1
            blocked = False
            for rr in range(br + step, r, step):
                if state.grid[rr, c] != TileType.EMPTY:
                    blocked = True
                    break
            if not blocked:
                return True
    return False


def find_escape_waypoint(
    state: GameState,
    player: PlayerState,
    danger_map: np.ndarray | None = None,
    max_steps: int = 6,
) -> tuple[tuple[int, int], float, int] | None:
    """Find the immediate next waypoint tile along shortest BFS escape path to nearest safe tile.

    Returns:
        tuple of (immediate_next_tile, d_path, rem_steps) if a safe tile is reachable within max_steps,
        else None.
    """
    if danger_map is None:
        danger_map = compute_danger_map(state)

    pr, pc = int(np.floor(player.r)), int(np.floor(player.c))
    if not (0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE):
        return None

    bomb_tiles = {b.tile for b in state.bombs}

    def is_safe_tile(r: int, c: int) -> bool:
        if danger_map[r, c] > 0.0:
            return False
        if state.flame_timer[r, c] > 0:
            return False
        if (r, c) in bomb_tiles:
            return False
        if is_tile_threatened(state, r, c):
            return False
        return True

    if is_safe_tile(pr, pc):
        return None

    queue: deque[tuple[int, int, int, tuple[int, int] | None]] = deque([(pr, pc, 0, None)])
    visited: set[tuple[int, int]] = {(pr, pc)}
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    min_safe_dist: int | None = None

    while queue:
        r, c, dist, first_step = queue.popleft()

        if min_safe_dist is not None and dist > min_safe_dist:
            break

        if dist > 0 and is_safe_tile(r, c):
            candidates.append((dist, first_step, (r, c)))  # type: ignore
            min_safe_dist = dist
            continue

        if dist >= max_steps:
            continue

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                if state.grid[nr, nc] != TileType.EMPTY:
                    continue
                if state.flame_timer[nr, nc] > 0:
                    continue
                if (nr, nc) in bomb_tiles and not player.bomb_pass:
                    continue

                visited.add((nr, nc))
                f_step = (nr, nc) if first_step is None else first_step
                queue.append((nr, nc, dist + 1, f_step))

    if not candidates:
        return None

    min_d_path = float("inf")
    best_rem_steps = 0
    best_next_tile: tuple[int, int] | None = None

    for dist, f_step, (sr, sc) in candidates:
        rem_steps = dist - 1
        fnr, fnc = f_step
        d_cand = float(rem_steps + abs(player.r - (fnr + 0.5)) + abs(player.c - (fnc + 0.5)))
        if d_cand < min_d_path:
            min_d_path = d_cand
            best_rem_steps = rem_steps
            best_next_tile = f_step

    if best_next_tile is None:
        return None

    return best_next_tile, min_d_path, best_rem_steps


def is_bomb_placement_safe(
    state: GameState,
    player_id: int,
) -> bool:
    """Check if placing a bomb at player's current tile permits reaching at least one safe tile outside the blast cross.

    1. Simulates placing a bomb at the player's current tile (r_b, c_b) with player.blast_radius.
    2. Computes the union of all blast tiles resulting from existing bombs + this hypothetical new bomb
       (using raycast logic identical to observation.py, stopped by pillars and breakable brick walls).
    3. Runs an exact BFS starting from (r_b, c_b) up to max_tile_steps = int(player.speed * 2.5).
    4. Safe condition: Returns True ONLY IF the BFS reaches at least ONE passable tile strictly
       outside the hypothetical blast zone (danger == 0 and not on an active flame).
    """
    player = state.players[player_id]
    if not player.alive:
        return False

    pr, pc = player.tile
    if not (0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE):
        return False

    # Hard syntax checks: cannot place if at max bombs or bomb already on tile
    active_bombs = sum(1 for b in state.bombs if b.owner_id == player.id)
    if active_bombs >= player.max_bombs:
        return False
    if any(b.tile == (pr, pc) for b in state.bombs):
        return False
    if state.flame_timer[pr, pc] > 0:
        return False

    # Union of blast tiles from existing bombs + hypothetical bomb at (pr, pc)
    blast_tiles: set[tuple[int, int]] = {(pr, pc)}
    all_bombs: list[tuple[int, int, int]] = [
        (b.tile[0], b.tile[1], b.blast_radius) for b in state.bombs
    ]
    all_bombs.append((pr, pc, player.blast_radius))

    for br, bc, rad in all_bombs:
        blast_tiles.add((br, bc))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for step in range(1, rad + 1):
                rr = br + dr * step
                cc = bc + dc * step
                if not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE):
                    break
                if state.grid[rr, cc] == TileType.PILLAR:
                    break
                blast_tiles.add((rr, cc))
                if state.grid[rr, cc] == TileType.BRICK:
                    break

    # Active flame tiles are also lethal / dangerous
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if state.flame_timer[r, c] > 0:
                blast_tiles.add((r, c))

    # All bombs that block movement
    blocking_bomb_tiles = {b.tile for b in state.bombs} | {(pr, pc)}

    # Max step distance in BFS (conservative 2.5s of travel: int(player.speed * 2.5))
    max_tile_steps = max(1, int(player.speed * 2.5))

    queue: deque[tuple[int, int, int]] = deque([(pr, pc, 0)])
    visited: set[tuple[int, int]] = {(pr, pc)}

    while queue:
        r, c, dist = queue.popleft()

        # If this tile is strictly outside blast tiles and not an active bomb, safe escape found!
        if (r, c) not in blast_tiles and (r, c) not in {b.tile for b in state.bombs}:
            return True

        if dist >= max_tile_steps:
            continue

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                if state.grid[nr, nc] == TileType.EMPTY and state.flame_timer[nr, nc] == 0:
                    # Once stepped off (pr, pc), cannot step back onto (pr, pc) nor other bombs
                    if player.bomb_pass or (nr, nc) not in blocking_bomb_tiles:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))

    return False


def is_cul_de_sac(
    state: GameState,
    player_id: int,
    max_ticks: int = 60,
) -> bool:
    """Check if placing a bomb at player's current tile is an unsafe cul-de-sac self-trap."""
    return not is_bomb_placement_safe(state, player_id)


class SafetyActionFilter:
    """Computes valid action masks to eliminate trivial 1-tick suicide actions and syntax violations."""

    def __init__(
        self,
        active_prob: float = 1.0,
        enabled: bool = True,
        safety_filter_prob: float | None = None,
        placement_filter_prob: float | None = None,
        escape_guidance_prob: float | None = None,
        threat_escape_filter_prob: float | None = None,
        bomb_placement_filter_prob: float | None = None,
    ) -> None:
        self.active_prob = float(np.clip(active_prob, 0.0, 1.0))
        self.enabled = enabled

        # Placement filter probability (defaults to 1.0)
        p_prob = 1.0
        if bomb_placement_filter_prob is not None:
            p_prob = bomb_placement_filter_prob
        elif placement_filter_prob is not None:
            p_prob = placement_filter_prob
        elif safety_filter_prob is not None:
            p_prob = safety_filter_prob
        self.placement_filter_prob = float(np.clip(p_prob, 0.0, 1.0))

        # Threat escape guidance probability (defaults to 0.5)
        e_prob = 0.5
        if threat_escape_filter_prob is not None:
            e_prob = threat_escape_filter_prob
        elif escape_guidance_prob is not None:
            e_prob = escape_guidance_prob
        elif safety_filter_prob is not None:
            e_prob = safety_filter_prob
        self.escape_guidance_prob = float(np.clip(e_prob, 0.0, 1.0))

    @property
    def safety_filter_prob(self) -> float:
        return self.placement_filter_prob

    @safety_filter_prob.setter
    def safety_filter_prob(self, val: float) -> None:
        v = float(np.clip(val, 0.0, 1.0))
        self.placement_filter_prob = v
        self.escape_guidance_prob = v

    @property
    def bomb_placement_filter_prob(self) -> float:
        return self.placement_filter_prob

    @bomb_placement_filter_prob.setter
    def bomb_placement_filter_prob(self, val: float) -> None:
        self.placement_filter_prob = float(np.clip(val, 0.0, 1.0))

    @property
    def threat_escape_filter_prob(self) -> float:
        return self.escape_guidance_prob

    @threat_escape_filter_prob.setter
    def threat_escape_filter_prob(self, val: float) -> None:
        self.escape_guidance_prob = float(np.clip(val, 0.0, 1.0))

    def get_action_mask(
        self,
        state: GameState,
        player_id: int,
        ego_perspective: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute boolean action masks for move and bomb sub-actions.

        Returns:
            dict with 'move_mask' (5, bool) and 'bomb_mask' (2, bool).
            If ego_perspective is True, masks correspond to the player's ego view.
        """
        # All actions valid by default
        move_mask = np.ones(5, dtype=bool)
        bomb_mask = np.ones(2, dtype=bool)

        if not self.enabled:
            return {"move_mask": move_mask, "bomb_mask": bomb_mask}

        player = state.players[player_id]
        if not player.alive:
            return {"move_mask": move_mask, "bomb_mask": bomb_mask}

        # Hard syntax checks on bomb placement: always enforced when filter enabled
        cur_tr, cur_tc = player.tile
        active_bombs = sum(1 for b in state.bombs if b.owner_id == player.id)
        if active_bombs >= player.max_bombs:
            bomb_mask[BombAction.PLACE_BOMB] = False

        if any(b.tile == (cur_tr, cur_tc) for b in state.bombs):
            bomb_mask[BombAction.PLACE_BOMB] = False

        if 0 <= cur_tr < GRID_SIZE and 0 <= cur_tc < GRID_SIZE:
            if state.flame_timer[cur_tr, cur_tc] > 0:
                bomb_mask[BombAction.PLACE_BOMB] = False

        # Check movement filter bypass: shielded immunity or stochastic active_prob
        skip_move_filter = (
            player.shield_timer > 0
            or (self.active_prob < 1.0 and np.random.random() > self.active_prob)
        )

        if not skip_move_filter:
            # Check lethal flame risk for each move action
            step_dist = player.speed * 0.05
            pr, pc = player.r, player.c
            curr_is_burning = (
                0 <= cur_tr < GRID_SIZE
                and 0 <= cur_tc < GRID_SIZE
                and state.flame_timer[cur_tr, cur_tc] > 0
            )

            min_bound = PLAYER_HALF_SIZE
            max_bound = GRID_SIZE - PLAYER_HALF_SIZE

            move_deltas = (
                (MoveAction.NOOP, 0.0, 0.0, 0, 0),
                (MoveAction.UP, -step_dist, 0.0, -1, 0),
                (MoveAction.DOWN, step_dist, 0.0, 1, 0),
                (MoveAction.LEFT, 0.0, -step_dist, 0, -1),
                (MoveAction.RIGHT, 0.0, step_dist, 0, 1),
            )

            world_move_safe = {}
            for m_act, dr, dc, tile_dr, tile_dc in move_deltas:
                raw_r = pr + dr
                raw_c = pc + dc
                target_r = min_bound if raw_r < min_bound else (max_bound if raw_r > max_bound else raw_r)
                target_c = min_bound if raw_c < min_bound else (max_bound if raw_c > max_bound else raw_c)
                target_tr = int(target_r)
                target_tc = int(target_c)

                # 1. Direct lethal destination on this tick
                if state.flame_timer[target_tr, target_tc] > 0:
                    world_move_safe[m_act] = False
                    continue

                # 2. Moving towards an active flame on an adjacent tile from a safe tile
                if not curr_is_burning and (tile_dr != 0 or tile_dc != 0):
                    adj_r = cur_tr + tile_dr
                    adj_c = cur_tc + tile_dc
                    if 0 <= adj_r < GRID_SIZE and 0 <= adj_c < GRID_SIZE:
                        if state.flame_timer[adj_r, adj_c] > 1:
                            world_move_safe[m_act] = False
                            continue

                # 3. Wall/Obstacle Flush Bump Check
                if m_act != MoveAction.NOOP:
                    p_clone = player.clone()
                    kick_impulses: list[Any] = []
                    BombermanPhysics.move_player(p_clone, m_act, state, kick_impulses=kick_impulses)
                    disp = abs(p_clone.r - pr) + abs(p_clone.c - pc)
                    if disp < 1e-4 and not kick_impulses:
                        world_move_safe[m_act] = False
                        continue

                world_move_safe[m_act] = True

            # Threat Escape Action Masking
            # When danger_map[pr, pc] > 0.0 or tile threatened by active bomb
            # AND curriculum escape guidance triggers (escape_guidance_prob):
            if 0 <= cur_tr < GRID_SIZE and 0 <= cur_tc < GRID_SIZE and state.bombs:
                danger_map = compute_danger_map(state)
                in_threat = (danger_map[cur_tr, cur_tc] > 0.0 or is_tile_threatened(state, cur_tr, cur_tc))
                trigger_curriculum = (
                    self.escape_guidance_prob >= 1.0
                    or (self.escape_guidance_prob > 0.0 and np.random.random() < self.escape_guidance_prob)
                )
                if in_threat:
                    if trigger_curriculum:
                        escape_res = find_escape_waypoint(state, player, danger_map)
                        if escape_res is not None:
                            (nr, nc), _, _ = escape_res
                            target_r = nr + 0.5
                            target_c = nc + 0.5
                            curr_dist = abs(pr - target_r) + abs(pc - target_c)

                            # 1. Mask out NOOP so agent cannot freeze in place
                            world_move_safe[MoveAction.NOOP] = False

                            # 2. Only keep cardinal movement actions that strictly reduce distance
                            for m_act in (MoveAction.UP, MoveAction.DOWN, MoveAction.LEFT, MoveAction.RIGHT):
                                if not world_move_safe.get(m_act, False):
                                    continue
                                p_clone = player.clone()
                                dummy_kicks: list[Any] = []
                                BombermanPhysics.move_player(p_clone, m_act, state, kick_impulses=dummy_kicks)
                                new_dist = abs(p_clone.r - target_r) + abs(p_clone.c - target_c)
                                if new_dist >= curr_dist - 1e-4:
                                    world_move_safe[m_act] = False

                            # 3. Fallback: if no action reduces distance, keep at least primary cardinal direction
                            if not any(world_move_safe.get(m, False) for m in (MoveAction.UP, MoveAction.DOWN, MoveAction.LEFT, MoveAction.RIGHT)):
                                if nr < cur_tr:
                                    prim = MoveAction.UP
                                elif nr > cur_tr:
                                    prim = MoveAction.DOWN
                                elif nc < cur_tc:
                                    prim = MoveAction.LEFT
                                elif nc > cur_tc:
                                    prim = MoveAction.RIGHT
                                else:
                                    prim = None

                                if prim is not None:
                                    world_move_safe[prim] = True
                                else:
                                    world_move_safe[MoveAction.NOOP] = True
                else:
                    # Player is on a safe tile: prevent walking into active bomb blast cross
                    if trigger_curriculum:
                        for m_act in (MoveAction.UP, MoveAction.DOWN, MoveAction.LEFT, MoveAction.RIGHT):
                            if not world_move_safe.get(m_act, False):
                                continue
                            p_clone = player.clone()
                            dummy_kicks: list[Any] = []
                            BombermanPhysics.move_player(p_clone, m_act, state, kick_impulses=dummy_kicks)
                            dest_tr, dest_tc = int(np.floor(p_clone.r)), int(np.floor(p_clone.c))
                            if 0 <= dest_tr < GRID_SIZE and 0 <= dest_tc < GRID_SIZE:
                                if is_tile_threatened(state, dest_tr, dest_tc) or danger_map[dest_tr, dest_tc] > 0.0:
                                    world_move_safe[m_act] = False

                        if not any(world_move_safe.values()):
                            world_move_safe[MoveAction.NOOP] = True

            # Ensure at least one action remains unmasked (fallback to NOOP)
            if not any(world_move_safe.values()):
                world_move_safe[MoveAction.NOOP] = True

            # Apply to move_mask
            for m_act in MoveAction:
                is_safe = world_move_safe[m_act]
                if ego_perspective and player_id == 1:
                    # Map world action to ego action
                    ego_m, _ = flip_action_to_ego((int(m_act), 0), 1)
                    move_mask[ego_m] = is_safe
                else:
                    move_mask[int(m_act)] = is_safe

        # Cul-de-sac suicide filter (annealed via placement_filter_prob)
        if bomb_mask[BombAction.PLACE_BOMB] and self.placement_filter_prob > 0.0:
            if self.placement_filter_prob >= 1.0 or np.random.random() < self.placement_filter_prob:
                if is_cul_de_sac(state, player_id):
                    bomb_mask[BombAction.PLACE_BOMB] = False

        return {"move_mask": move_mask, "bomb_mask": bomb_mask}

    def get_action_masks(
        self,
        state: GameState,
        player_id: int,
        ego_perspective: bool = True,
    ) -> dict[str, np.ndarray]:
        """Alias for get_action_mask."""
        return self.get_action_mask(state, player_id, ego_perspective=ego_perspective)

    def filter_action(
        self,
        state: GameState,
        player_id: int,
        action: tuple[int, int] | list[int] | np.ndarray | int,
        ego_perspective: bool = True,
    ) -> tuple[int, int]:
        """Filter an action, replacing masked choices with safe alternatives."""
        if not self.enabled:
            if isinstance(action, (tuple, list, np.ndarray)):
                return (int(action[0]), int(action[1]) if len(action) > 1 else int(BombAction.NOOP))
            val = int(action)
            return (int(MoveAction.NOOP), int(BombAction.PLACE_BOMB)) if val == 5 else (val, int(BombAction.NOOP))

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

        masks = self.get_action_mask(state, player_id, ego_perspective=ego_perspective)

        # Filter move action
        if not masks["move_mask"][m_act]:
            # Fallback to NOOP if safe, else first safe move
            if masks["move_mask"][int(MoveAction.NOOP)]:
                m_act = int(MoveAction.NOOP)
            else:
                safe_moves = np.where(masks["move_mask"])[0]
                m_act = int(safe_moves[0]) if len(safe_moves) > 0 else int(MoveAction.NOOP)

        # Filter bomb action
        if not masks["bomb_mask"][b_act]:
            b_act = int(BombAction.NOOP)

        return (m_act, b_act)


# Alias for convenience
ActionFilter = SafetyActionFilter

