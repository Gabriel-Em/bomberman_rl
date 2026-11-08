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
from bomberman_rl.engine.state import GameState, PlayerState
from bomberman_rl.rl.observation import flip_action_to_world, flip_action_to_ego


def is_cul_de_sac(
    state: GameState,
    player_id: int,
    max_ticks: int = 60,
) -> bool:
    """Check if placing a bomb at player's current tile leaves 0 reachable safe tiles.

    Fast check:
    1. Computes the blast ray coverage of a new bomb placed at the player's tile.
    2. Explores reachable tiles from the player's tile within `max_ticks` (BFS).
    3. If 0 reachable tiles exist outside the blast rays (and existing lethal flames),
       the placement is a guaranteed cul-de-sac self-trap / suicide.
    """
    player = state.players[player_id]
    if not player.alive:
        return False

    pr, pc = player.tile
    if not (0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE):
        return False

    # Blast rays from the new bomb at (pr, pc)
    blast_radius = player.blast_radius
    blast_tiles: set[tuple[int, int]] = {(pr, pc)}

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for step in range(1, blast_radius + 1):
            nr = pr + dr * step
            nc = pc + dc * step
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                break
            if state.grid[nr, nc] == TileType.PILLAR:
                break
            blast_tiles.add((nr, nc))
            if state.grid[nr, nc] == TileType.BRICK:
                break

    # Existing bombs tiles that block movement (unless player has bomb_pass)
    bomb_tiles = {b.tile for b in state.bombs}

    # Dynamic maximum travel steps in max_ticks (DT = 0.05s)
    max_steps = max(1, int(np.ceil(player.speed * (max_ticks * 0.05))))

    # BFS from (pr, pc) to see if any safe tile outside blast_tiles is reachable
    queue: deque[tuple[int, int, int]] = deque([(pr, pc, 0)])
    visited: set[tuple[int, int]] = {(pr, pc)}

    while queue:
        r, c, dist = queue.popleft()
        if dist > max_steps:
            continue

        # If this tile is safe (outside blast_tiles and not on active flame), we found an escape!
        if (r, c) not in blast_tiles and state.flame_timer[r, c] == 0:
            return False

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                if state.grid[nr, nc] == TileType.EMPTY and state.flame_timer[nr, nc] == 0:
                    # Can step onto (nr, nc) if no bomb or player has bomb_pass
                    if player.bomb_pass or ((nr, nc) not in bomb_tiles and (nr, nc) != (pr, pc)):
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))

    # No safe tile reached outside the blast rays
    return True


class SafetyActionFilter:
    """Computes valid action masks to eliminate trivial 1-tick suicide actions and syntax violations."""

    def __init__(
        self,
        active_prob: float = 1.0,
        enabled: bool = True,
        safety_filter_prob: float = 1.0,
    ) -> None:
        self.active_prob = float(np.clip(active_prob, 0.0, 1.0))
        self.enabled = enabled
        self.safety_filter_prob = float(np.clip(safety_filter_prob, 0.0, 1.0))

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

        if self.active_prob < 1.0 and np.random.random() > self.active_prob:
            return {"move_mask": move_mask, "bomb_mask": bomb_mask}

        player = state.players[player_id]
        if not player.alive:
            return {"move_mask": move_mask, "bomb_mask": bomb_mask}

        # Syntax checks on bomb placement
        cur_tr, cur_tc = player.tile
        active_bombs = sum(1 for b in state.bombs if b.owner_id == player.id)
        if active_bombs >= player.max_bombs:
            bomb_mask[BombAction.PLACE_BOMB] = False

        if any(b.tile == (cur_tr, cur_tc) for b in state.bombs):
            bomb_mask[BombAction.PLACE_BOMB] = False

        if 0 <= cur_tr < GRID_SIZE and 0 <= cur_tc < GRID_SIZE:
            if state.flame_timer[cur_tr, cur_tc] > 0:
                bomb_mask[BombAction.PLACE_BOMB] = False

        # If shielded, player is immune to flame damage
        if player.shield_timer > 0:
            return {"move_mask": move_mask, "bomb_mask": bomb_mask}

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

            world_move_safe[m_act] = True

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

        # Cul-de-sac suicide filter (annealed)
        if bomb_mask[BombAction.PLACE_BOMB] and self.safety_filter_prob > 0.0:
            if self.safety_filter_prob >= 1.0 or np.random.random() < self.safety_filter_prob:
                if is_cul_de_sac(state, player_id):
                    bomb_mask[BombAction.PLACE_BOMB] = False

        return {"move_mask": move_mask, "bomb_mask": bomb_mask}

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

