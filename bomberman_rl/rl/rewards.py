"""Anti-Exploit Reward Architecture and Potential-Based Reward Shaping (PBRS).

Provides mathematically grounded, loop-immune reward shaping and curriculum
self-damage annealing to solve the Pommerman cowardice trap while keeping terminal
win incentives dominant.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    TileType,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState
from bomberman_rl.rl.observation import compute_danger_map
from bomberman_rl.rl.action_filter import is_bomb_placement_safe, find_escape_waypoint, is_tile_threatened

# Potential-based shaping weights
WEIGHT_BLAST = 0.03
WEIGHT_BOMBS = 0.03
WEIGHT_SPEED = 0.04
WEIGHT_KICK = 0.06
WEIGHT_PASS = 0.06
WEIGHT_SHIELD = 0.05

DANGER_POTENTIAL_WEIGHT = 0.08
BRICK_REWARD = 0.06
CONNECTIVITY_REWARD = 0.05
VISITATION_REWARD = 0.010
FRONTIER_REWARD_WEIGHT = 0.015
BOMB_WALL_ADJACENT_REWARD = 0.02
AUXILIARY_REWARD_CAP = 3.50

TERMINAL_WIN = 1.0
SURVIVAL_BONUS = 0.05
TERMINAL_DRAW = -0.2

ANTI_THRASH_PENALTY = -0.001
CALM_SHELTER_BONUS = 0.0005
OPPOSITE_DIRECTIONS = {
    1: 2,  # UP: DOWN
    2: 1,  # DOWN: UP
    3: 4,  # LEFT: RIGHT
    4: 3,  # RIGHT: LEFT
}


class CurriculumLethalityScheduler:
    """Anneals self-damage penalty scale from early exploration to authentic competitive play."""

    def __init__(
        self,
        initial_scale: float = 0.25,
        final_scale: float = 1.0,
        total_steps: int = 100_000,
        anneal_type: str = "linear",
    ) -> None:
        self.initial_scale = float(np.clip(initial_scale, 0.15, 1.0))
        self.final_scale = float(np.clip(final_scale, 0.15, 1.0))
        self.total_steps = max(1, total_steps)
        self.anneal_type = anneal_type
        self.current_step = 0

    @property
    def scale(self) -> float:
        return self.get_scale(self.current_step)

    def get_scale(self, step: int) -> float:
        progress = float(np.clip(step / self.total_steps, 0.0, 1.0))
        if self.anneal_type == "linear":
            return self.initial_scale + progress * (self.final_scale - self.initial_scale)
        elif self.anneal_type == "cosine":
            cos_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            return self.initial_scale + float(cos_progress) * (self.final_scale - self.initial_scale)
        return self.initial_scale

    def step(self) -> float:
        val = self.scale
        self.current_step += 1
        return val

    def set_step(self, step: int) -> None:
        self.current_step = max(0, step)


def compute_pbrs_potential(player: PlayerState) -> float:
    """Calculate the stat potential function Phi_stats(s).

    Phi_stats(s) = 0.03 * (blast_radius - 1)
                 + 0.03 * (max_bombs - 1)
                 + 0.04 * ((speed - 2.5) / 0.5)
                 + 0.06 * I(kick)
                 + 0.06 * I(pass)
                 + 0.05 * I(shield > 0)
    """
    phi_blast = WEIGHT_BLAST * max(0, player.blast_radius - 1)
    phi_bombs = WEIGHT_BOMBS * max(0, player.max_bombs - 1)
    phi_speed = WEIGHT_SPEED * max(0.0, (player.speed - 2.5) / 0.5)
    phi_kick = WEIGHT_KICK if player.can_kick else 0.0
    phi_pass = WEIGHT_PASS if player.bomb_pass else 0.0
    phi_shield = WEIGHT_SHIELD if player.shield_timer > 0 else 0.0

    return float(phi_blast + phi_bombs + phi_speed + phi_kick + phi_pass + phi_shield)


def check_players_connected(grid: np.ndarray, p0_tile: tuple[int, int], p1_tile: tuple[int, int]) -> bool:
    """Check if passable paths exist between Player 0 and Player 1 using BFS."""
    if p0_tile == p1_tile:
        return True

    tr0, tc0 = p0_tile
    tr1, tc1 = p1_tile
    if not (0 <= tr0 < GRID_SIZE and 0 <= tc0 < GRID_SIZE):
        return False
    if not (0 <= tr1 < GRID_SIZE and 0 <= tc1 < GRID_SIZE):
        return False
    if grid[tr0, tc0] != TileType.EMPTY or grid[tr1, tc1] != TileType.EMPTY:
        return False

    queue = deque([(tr0, tc0)])
    visited = {(tr0, tc0)}

    while queue:
        r, c = queue.popleft()
        if (r, c) == (tr1, tc1):
            return True

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                if grid[nr, nc] == TileType.EMPTY:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return False


def compute_danger_potential(
    arg1: GameState | PlayerState,
    arg2: PlayerState | np.ndarray,
    arg3: np.ndarray | None = None,
    state: GameState | None = None,
) -> float:
    """Calculate purely spatial, continuous sub-tile danger potential Phi_danger(s).

    Phi_danger(s) = 0.0 if not alive, shielded (>20), or current tile is safe (not in any blast cross).
    Otherwise:
    Phi_danger(s) = -0.08 * (1.0 + d_path)
    where d_path is the continuous Waypoint Geodesic Distance along the 6-step BFS shortest path
    to the nearest safe tile (or 4.0 if unreachable).
    """
    if isinstance(arg1, GameState):
        actual_state: GameState | None = arg1
        player: PlayerState = arg2  # type: ignore
        danger_map: np.ndarray = arg3  # type: ignore
    else:
        player = arg1  # type: ignore
        danger_map = arg2  # type: ignore
        actual_state = state

    if not player.alive or player.shield_timer > 20:
        return 0.0

    pr = int(np.floor(player.r))
    pc = int(np.floor(player.c))
    if not (0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE):
        return 0.0

    in_danger = (danger_map[pr, pc] > 0.0)
    if not in_danger and actual_state is not None and actual_state.bombs:
        in_danger = is_tile_threatened(actual_state, pr, pc)
    if not in_danger and actual_state is not None and actual_state.flame_timer[pr, pc] > 0:
        in_danger = True

    if not in_danger:
        return 0.0

    # Player is currently on a threatened tile
    if actual_state is not None:
        res = find_escape_waypoint(actual_state, player, danger_map)
        if res is not None:
            _, d_path, _ = res
        else:
            d_path = 8.0
    else:
        # Fallback when actual_state is not provided
        queue: deque[tuple[int, int, int, tuple[int, int] | None]] = deque([(pr, pc, 0, None)])
        visited: set[tuple[int, int]] = {(pr, pc)}
        candidates: list[tuple[int, tuple[int, int]]] = []
        min_safe_dist: int | None = None

        while queue:
            r, c, dist, first_step = queue.popleft()
            if min_safe_dist is not None and dist > min_safe_dist:
                break
            if dist > 0 and danger_map[r, c] == 0.0 and not (r % 2 == 1 and c % 2 == 1):
                candidates.append((dist, first_step))  # type: ignore
                min_safe_dist = dist
                continue
            if dist >= 6:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                    if (nr % 2 == 1 and nc % 2 == 1):
                        continue
                    if danger_map[nr, nc] < 1.0:
                        visited.add((nr, nc))
                        f_step = (nr, nc) if first_step is None else first_step
                        queue.append((nr, nc, dist + 1, f_step))

        if candidates:
            d_path = min(
                float((dist - 1) + abs(player.r - (fnr + 0.5)) + abs(player.c - (fnc + 0.5)))
                for dist, (fnr, fnc) in candidates
            )
        else:
            d_path = 8.0

    return float(-DANGER_POTENTIAL_WEIGHT * (1.0 + d_path))


@dataclass
class PlayerRewardLog:
    """Telemetry record of modular reward components for a single player."""
    win: float = 0.0
    loss: float = 0.0
    draw: float = 0.0
    brick_destruction: float = 0.0
    pbrs_stats: float = 0.0
    connectivity: float = 0.0
    pbrs_danger: float = 0.0
    visitation: float = 0.0
    frontier: float = 0.0
    bomb_placement: float = 0.0
    anti_thrash: float = 0.0
    calm_shelter: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "win": self.win,
            "loss": self.loss,
            "draw": self.draw,
            "brick_destruction": self.brick_destruction,
            "pbrs_stats": self.pbrs_stats,
            "connectivity": self.connectivity,
            "pbrs_danger": self.pbrs_danger,
            "visitation": self.visitation,
            "frontier": self.frontier,
            "bomb_placement": self.bomb_placement,
            "anti_thrash": self.anti_thrash,
            "calm_shelter": self.calm_shelter,
            "total": self.total,
        }


class RewardTracker:
    """Modular reward calculator with telemetry logging and anti-loop guarantees."""

    def __init__(
        self,
        self_damage_scale: float | CurriculumLethalityScheduler = 0.25,
        auxiliary_reward_cap: float = AUXILIARY_REWARD_CAP,
        gamma: float = 0.99,
    ) -> None:
        self.scheduler: CurriculumLethalityScheduler | None = None
        if isinstance(self_damage_scale, CurriculumLethalityScheduler):
            self.scheduler = self_damage_scale
            self._self_damage_scale = self.scheduler.scale
        else:
            self._self_damage_scale = float(np.clip(self_damage_scale, 0.15, 1.0))

        self.auxiliary_reward_cap = auxiliary_reward_cap
        self.gamma = gamma
        self.reset()

    @property
    def self_damage_scale(self) -> float:
        if self.scheduler is not None:
            return self.scheduler.scale
        return self._self_damage_scale

    @self_damage_scale.setter
    def self_damage_scale(self, val: float) -> None:
        self._self_damage_scale = float(np.clip(val, 0.15, 1.0))

    def reset(self, initial_state: GameState | None = None) -> None:
        """Reset per-episode state trackers."""
        self.prev_potentials: dict[int, float] = {0: 0.0, 1: 0.0}
        self.prev_danger_potentials: dict[int, float] = {0: 0.0, 1: 0.0}
        self.cumulative_aux: dict[int, float] = {0: 0.0, 1: 0.0}
        self.connectivity_awarded: bool = False
        self.visited_tiles = np.zeros((2, GRID_SIZE, GRID_SIZE), dtype=bool)
        # Pre-seed spawn tiles so returning to spawn never awards visitation bonus
        self.visited_tiles[0, 0, 0] = True
        self.visited_tiles[1, 8, 8] = True
        self.max_frontier_dist: dict[int, float] = {0: 0.0, 1: 0.0}
        self.spawn_coords: dict[int, tuple[float, float]] = {0: (0.5, 0.5), 1: (8.5, 8.5)}
        self.bomb_near_brick_count: dict[int, int] = {0: 0, 1: 0}
        self.prev_move_action: dict[int, int | None] = {0: None, 1: None}
        self.prev_bomb_placed: dict[int, bool] = {0: False, 1: False}
        self.prev_tile: dict[int, tuple[int, int] | None] = {0: None, 1: None}
        self.episode_logs: dict[int, PlayerRewardLog] = {0: PlayerRewardLog(), 1: PlayerRewardLog()}

        if initial_state is not None:
            p0 = initial_state.players[0]
            p1 = initial_state.players[1]
            self.prev_tile[0] = p0.tile
            self.prev_tile[1] = p1.tile
            self.spawn_coords[0] = (float(p0.r), float(p0.c))
            self.spawn_coords[1] = float(p1.r), float(p1.c)
            if 0 <= p0.tile[0] < GRID_SIZE and 0 <= p0.tile[1] < GRID_SIZE:
                self.visited_tiles[0, p0.tile[0], p0.tile[1]] = True
            if 0 <= p1.tile[0] < GRID_SIZE and 0 <= p1.tile[1] < GRID_SIZE:
                self.visited_tiles[1, p1.tile[0], p1.tile[1]] = True
            self.prev_potentials[0] = compute_pbrs_potential(p0)
            self.prev_potentials[1] = compute_pbrs_potential(p1)
            init_danger = compute_danger_map(initial_state)
            self.prev_danger_potentials[0] = compute_danger_potential(initial_state, p0, init_danger)
            self.prev_danger_potentials[1] = compute_danger_potential(initial_state, p1, init_danger)
            # If map starts already connected, do not award bonus later
            if check_players_connected(initial_state.grid, p0.tile, p1.tile):
                self.connectivity_awarded = True

    def step(
        self,
        state_before: GameState,
        actions: Any,
        state_after: GameState,
        terminated: bool,
        truncated: bool,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Compute the step reward for each player and return telemetry breakdown."""
        scale = self.self_damage_scale
        step_rewards: dict[str, float] = {"player_0": 0.0, "player_1": 0.0}
        step_telemetry: dict[str, dict[str, float]] = {
            "player_0": {},
            "player_1": {},
        }

        # 1. Detect Brick Destructions & identify breaker
        newly_destroyed_bricks: list[tuple[int, int]] = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if state_before.grid[r, c] == TileType.BRICK and state_after.grid[r, c] == TileType.EMPTY:
                    newly_destroyed_bricks.append((r, c))

        p0_brick_rew = 0.0
        p1_brick_rew = 0.0
        for r, c in newly_destroyed_bricks:
            mask = state_after.flame_owner_mask[r, c]
            if mask & (1 << 0):
                p0_brick_rew += BRICK_REWARD
            if mask & (1 << 1):
                p1_brick_rew += BRICK_REWARD

        # 2. PBRS Stat Potentials: Phi_stats(s') - Phi_stats(s)
        curr_phi_0 = compute_pbrs_potential(state_after.players[0])
        curr_phi_1 = compute_pbrs_potential(state_after.players[1])
        p0_pbrs = curr_phi_0 - self.prev_potentials[0]
        p1_pbrs = curr_phi_1 - self.prev_potentials[1]
        self.prev_potentials[0] = curr_phi_0
        self.prev_potentials[1] = curr_phi_1

        # 3. Danger-Zone Potential PBRS: gamma_shaping = 1.0 (Undiscounted Spatial Difference)
        danger_map_after = compute_danger_map(state_after)
        before_bomb_ids = {b.id for b in state_before.bombs}
        new_bombs = [b for b in state_after.bombs if b.id not in before_bomb_ids]
        new_bombs_by_player = {
            0: [b for b in new_bombs if b.owner_id == 0],
            1: [b for b in new_bombs if b.owner_id == 1],
        }

        pbrs_danger_rewards = [0.0, 0.0]
        for pid in (0, 1):
            if not state_after.players[pid].alive:
                # Terminal transition into death: dying in an explosion is not an escape.
                # Do not award positive shaping on death tick.
                pbrs_danger_rewards[pid] = 0.0
                self.prev_danger_potentials[pid] = 0.0
                continue

            curr_danger_phi = compute_danger_potential(state_after, state_after.players[pid], danger_map_after)
            pr_aft, pc_aft = state_after.players[pid].tile
            was_safe = (self.prev_danger_potentials[pid] == 0.0)
            now_in_danger = (curr_danger_phi < 0.0)
            threatened_by_new = bool(new_bombs and is_tile_threatened(state_after, pr_aft, pc_aft, bombs=new_bombs))

            if was_safe and now_in_danger and threatened_by_new:
                # Boundary Reset on Bomb Placement: on the exact tick a new bomb is placed
                # causing player to transition into danger, set baseline to curr_danger_phi
                # so the placement tick gets 0.0 penalty.
                pbrs_danger_rewards[pid] = 0.0
            else:
                pbrs_danger_rewards[pid] = curr_danger_phi - self.prev_danger_potentials[pid]
            self.prev_danger_potentials[pid] = curr_danger_phi

        p0_pbrs_danger = pbrs_danger_rewards[0]
        p1_pbrs_danger = pbrs_danger_rewards[1]

        # 4. One-Time Tile Visitation Bonus (Anti-Jitter)
        visitation_rewards = [0.0, 0.0]
        for pid in (0, 1):
            t_bef = state_before.players[pid].tile
            t_aft = state_after.players[pid].tile
            if t_aft != t_bef:
                tr, tc = t_aft
                if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE and not self.visited_tiles[pid, tr, tc]:
                    self.visited_tiles[pid, tr, tc] = True
                    visitation_rewards[pid] = VISITATION_REWARD
        p0_visitation = visitation_rewards[0]
        p1_visitation = visitation_rewards[1]

        # Continuous Frontier Distance Potential (+0.015 * delta monotonic high-water mark)
        frontier_rewards = [0.0, 0.0]
        for pid in (0, 1):
            if state_after.players[pid].alive:
                sr, sc = self.spawn_coords[pid]
                dist_spawn = abs(state_after.players[pid].r - sr) + abs(state_after.players[pid].c - sc)
                if dist_spawn > self.max_frontier_dist[pid]:
                    delta = dist_spawn - self.max_frontier_dist[pid]
                    frontier_rewards[pid] = float(FRONTIER_REWARD_WEIGHT * delta)
                    self.max_frontier_dist[pid] = dist_spawn
        p0_frontier = frontier_rewards[0]
        p1_frontier = frontier_rewards[1]

        # 5. Bomb Placed Adjacent to Breakable Wall Bonus (+0.02, capped at 15 times/episode)
        bomb_near_brick_rewards = [0.0, 0.0]
        for pid in (0, 1):
            if new_bombs_by_player[pid] and self.bomb_near_brick_count[pid] < 15:
                for b in new_bombs_by_player[pid]:
                    b_tr, b_tc = b.tile
                    # Check if bomb placement is safe at the bomb's tile
                    if b.tile == state_before.players[pid].tile:
                        is_safe = is_bomb_placement_safe(state_before, pid)
                    else:
                        temp_state = state_before.clone()
                        temp_state.players[pid].r = b_tr + 0.5
                        temp_state.players[pid].c = b_tc + 0.5
                        is_safe = is_bomb_placement_safe(temp_state, pid)

                    if not is_safe:
                        continue

                    is_adj = False
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = b_tr + dr, b_tc + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                            if state_before.grid[nr, nc] == TileType.BRICK:
                                is_adj = True
                                break
                    if is_adj:
                        self.bomb_near_brick_count[pid] += 1
                        bomb_near_brick_rewards[pid] += BOMB_WALL_ADJACENT_REWARD
                        if self.bomb_near_brick_count[pid] >= 15:
                            break
        p0_bomb_wall = bomb_near_brick_rewards[0]
        p1_bomb_wall = bomb_near_brick_rewards[1]

        # 6. Map Control / Connectivity Opening (+0.05 one-time bonus)
        p0_conn_rew = 0.0
        p1_conn_rew = 0.0
        if not self.connectivity_awarded:
            connected_before = check_players_connected(
                state_before.grid, state_before.players[0].tile, state_before.players[1].tile
            )
            if connected_before:
                # Players were already connected before this step
                self.connectivity_awarded = True
            elif newly_destroyed_bricks:
                connected_after = check_players_connected(
                    state_after.grid, state_after.players[0].tile, state_after.players[1].tile
                )
                if connected_after:
                    self.connectivity_awarded = True
                    # Identify which destroyed brick(s) actually breached the separating wall
                    breach_bricks: list[tuple[int, int]] = []
                    for br, bc in newly_destroyed_bricks:
                        test_grid = state_after.grid.copy()
                        test_grid[br, bc] = TileType.BRICK
                        if not check_players_connected(
                            test_grid, state_after.players[0].tile, state_after.players[1].tile
                        ):
                            breach_bricks.append((br, bc))

                    if not breach_bricks:
                        breach_bricks = newly_destroyed_bricks

                    p0_opened = any(bool(state_after.flame_owner_mask[br, bc] & 1) for br, bc in breach_bricks)
                    p1_opened = any(bool(state_after.flame_owner_mask[br, bc] & 2) for br, bc in breach_bricks)

                    if p0_opened and not p1_opened:
                        p0_conn_rew = CONNECTIVITY_REWARD
                    elif p1_opened and not p0_opened:
                        p1_conn_rew = CONNECTIVITY_REWARD
                    elif p0_opened and p1_opened:
                        p0_conn_rew = CONNECTIVITY_REWARD / 2.0
                        p1_conn_rew = CONNECTIVITY_REWARD / 2.0

        # Parse actions for anti-thrash and calm shelter evaluation
        parsed_actions: list[tuple[int, int]] = []
        for pid in (0, 1):
            if isinstance(actions, dict):
                act = actions.get(pid, actions.get(f"player_{pid}", (0, 0)))
            elif isinstance(actions, (list, tuple, np.ndarray)) and pid < len(actions):
                act = actions[pid]
            else:
                act = (0, 0)

            if isinstance(act, (list, tuple, np.ndarray)):
                m_act = int(act[0])
                b_act = int(act[1]) if len(act) > 1 else 0
            else:
                val = int(act)
                if val == 5:
                    m_act = 0
                    b_act = 1
                else:
                    m_act = val
                    b_act = 0
            parsed_actions.append((m_act, b_act))

        # 7. Anti-Thrash Penalty (-0.001) & Calm Shelter Waiting Bonus (+0.0005)
        anti_thrash_rewards = [0.0, 0.0]
        calm_shelter_rewards = [0.0, 0.0]
        has_ticking_bomb = (len(state_before.bombs) > 0 or len(state_after.bombs) > 0)

        for pid in (0, 1):
            m_act, b_act = parsed_actions[pid]
            t_bef = state_before.players[pid].tile
            t_aft = state_after.players[pid].tile
            p_alive = state_after.players[pid].alive

            # 1-Tick Opposite-Direction Thrash Penalty (-0.001):
            # If move_action != NOOP and move_action is opposite of prev_move_action on the same tile without bomb on prev tick
            is_opposite = (
                self.prev_move_action[pid] is not None
                and m_act != 0
                and OPPOSITE_DIRECTIONS.get(m_act) == self.prev_move_action[pid]
            )
            same_tile = (t_bef == t_aft and (self.prev_tile[pid] is None or self.prev_tile[pid] == t_bef))
            no_prev_bomb = not self.prev_bomb_placed[pid]

            if p_alive and is_opposite and same_tile and no_prev_bomb:
                anti_thrash_rewards[pid] = ANTI_THRASH_PENALTY

            # Calm Shelter Waiting Bonus (+0.0005):
            # When active ticking bomb on board, player on safe tile (danger_map[tr, tc] == 0.0), and selecting NOOP
            tr, tc = t_aft
            is_safe_tile = (
                0 <= tr < GRID_SIZE
                and 0 <= tc < GRID_SIZE
                and danger_map_after[tr, tc] == 0.0
            )
            if p_alive and has_ticking_bomb and is_safe_tile and m_act == 0:
                calm_shelter_rewards[pid] = CALM_SHELTER_BONUS

        # Update per-player action and placement trackers for the next tick
        for pid in (0, 1):
            m_act, b_act = parsed_actions[pid]
            self.prev_move_action[pid] = m_act
            self.prev_bomb_placed[pid] = bool(new_bombs_by_player[pid])
            self.prev_tile[pid] = state_after.players[pid].tile

        # Apply auxiliary reward cap per match
        # Positive components are proportionally capped; negative PBRS is applied in full
        def apply_cap_and_scale(
            pid: int,
            brick_rew: float,
            pbrs: float,
            conn_rew: float,
            visitation: float,
            frontier: float,
            bomb_wall: float,
        ) -> tuple[float, float, float, float, float, float, float]:
            pos_pbrs = max(0.0, pbrs)
            neg_pbrs = min(0.0, pbrs)
            raw_pos = brick_rew + pos_pbrs + conn_rew + visitation + frontier + bomb_wall
            available = max(0.0, self.auxiliary_reward_cap - self.cumulative_aux[pid])
            capped_pos = min(raw_pos, available)
            self.cumulative_aux[pid] += capped_pos + neg_pbrs

            factor = (capped_pos / raw_pos) if raw_pos > 1e-7 else 1.0
            actual_brick = brick_rew * factor
            actual_pbrs = (pos_pbrs * factor) + neg_pbrs
            actual_conn = conn_rew * factor
            actual_visitation = visitation * factor
            actual_frontier = frontier * factor
            actual_bomb_wall = bomb_wall * factor
            actual_total = (
                actual_brick + actual_pbrs + actual_conn +
                actual_visitation + actual_frontier + actual_bomb_wall
            )
            return (
                actual_brick, actual_pbrs, actual_conn,
                actual_visitation, actual_frontier, actual_bomb_wall, actual_total
            )

        p0_b_act, p0_p_act, p0_c_act, p0_v_act, p0_f_act, p0_bw_act, actual_aux_0 = apply_cap_and_scale(
            0, p0_brick_rew, p0_pbrs, p0_conn_rew, p0_visitation, p0_frontier, p0_bomb_wall
        )
        p1_b_act, p1_p_act, p1_c_act, p1_v_act, p1_f_act, p1_bw_act, actual_aux_1 = apply_cap_and_scale(
            1, p1_brick_rew, p1_pbrs, p1_conn_rew, p1_visitation, p1_frontier, p1_bomb_wall
        )

        log_0 = PlayerRewardLog(
            brick_destruction=p0_b_act,
            pbrs_stats=p0_p_act,
            connectivity=p0_c_act,
            pbrs_danger=p0_pbrs_danger,
            visitation=p0_v_act,
            frontier=p0_f_act,
            bomb_placement=p0_bw_act,
            anti_thrash=anti_thrash_rewards[0],
            calm_shelter=calm_shelter_rewards[0],
        )
        log_1 = PlayerRewardLog(
            brick_destruction=p1_b_act,
            pbrs_stats=p1_p_act,
            connectivity=p1_c_act,
            pbrs_danger=p1_pbrs_danger,
            visitation=p1_v_act,
            frontier=p1_f_act,
            bomb_placement=p1_bw_act,
            anti_thrash=anti_thrash_rewards[1],
            calm_shelter=calm_shelter_rewards[1],
        )

        # 4. Terminal Outcomes
        p0_after = state_after.players[0]
        p1_after = state_after.players[1]

        if terminated or truncated:
            if not p0_after.alive and not p1_after.alive:
                # Mutual death -> Draw
                log_0.draw = TERMINAL_DRAW
                log_1.draw = TERMINAL_DRAW
            elif p0_after.alive and not p1_after.alive:
                # Determine who killed Player 1
                p1_tr, p1_tc = p1_after.tile
                mask = 0
                if 0 <= p1_tr < GRID_SIZE and 0 <= p1_tc < GRID_SIZE:
                    mask = int(state_after.flame_owner_mask[p1_tr, p1_tc])
                    if mask == 0:
                        mask = int(state_before.flame_owner_mask[p1_tr, p1_tc])
                killed_by_p0 = bool(mask & (1 << 0))
                killed_by_self = bool(mask & (1 << 1))
                if killed_by_self and not killed_by_p0:
                    log_1.loss = -1.0 * scale
                    log_0.win = SURVIVAL_BONUS
                else:
                    log_0.win = TERMINAL_WIN
                    log_1.loss = -1.0
            elif p1_after.alive and not p0_after.alive:
                # Determine who killed Player 0
                p0_tr, p0_tc = p0_after.tile
                mask = 0
                if 0 <= p0_tr < GRID_SIZE and 0 <= p0_tc < GRID_SIZE:
                    mask = int(state_after.flame_owner_mask[p0_tr, p0_tc])
                    if mask == 0:
                        mask = int(state_before.flame_owner_mask[p0_tr, p0_tc])
                killed_by_p1 = bool(mask & (1 << 1))
                killed_by_self = bool(mask & (1 << 0))
                if killed_by_self and not killed_by_p1:
                    log_0.loss = -1.0 * scale
                    log_1.win = SURVIVAL_BONUS
                else:
                    log_1.win = TERMINAL_WIN
                    log_0.loss = -1.0
            elif truncated:
                # Timeout draw
                log_0.draw = TERMINAL_DRAW
                log_1.draw = TERMINAL_DRAW

        # Compute totals
        log_0.total = (
            log_0.win + log_0.loss + log_0.draw + actual_aux_0 +
            log_0.pbrs_danger + log_0.anti_thrash + log_0.calm_shelter
        )
        log_1.total = (
            log_1.win + log_1.loss + log_1.draw + actual_aux_1 +
            log_1.pbrs_danger + log_1.anti_thrash + log_1.calm_shelter
        )

        # Accumulate into episode totals
        for attr in [
            "win", "loss", "draw", "brick_destruction", "pbrs_stats",
            "connectivity", "pbrs_danger", "visitation", "frontier", "bomb_placement",
            "anti_thrash", "calm_shelter", "total"
        ]:
            setattr(self.episode_logs[0], attr, getattr(self.episode_logs[0], attr) + getattr(log_0, attr))
            setattr(self.episode_logs[1], attr, getattr(self.episode_logs[1], attr) + getattr(log_1, attr))

        step_rewards["player_0"] = float(log_0.total)
        step_rewards["player_1"] = float(log_1.total)
        step_telemetry["player_0"] = log_0.as_dict()
        step_telemetry["player_1"] = log_1.as_dict()

        return step_rewards, step_telemetry
