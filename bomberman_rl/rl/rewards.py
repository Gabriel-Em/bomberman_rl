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

# Potential-based shaping weights
WEIGHT_BLAST = 0.03
WEIGHT_BOMBS = 0.03
WEIGHT_SPEED = 0.04
WEIGHT_KICK = 0.06
WEIGHT_PASS = 0.06
WEIGHT_SHIELD = 0.05

BRICK_REWARD = 0.015
CONNECTIVITY_REWARD = 0.05
AUXILIARY_REWARD_CAP = 0.35

TERMINAL_WIN = 1.0
TERMINAL_DRAW = -0.2


class CurriculumLethalityScheduler:
    """Anneals self-damage penalty scale from early exploration to authentic competitive play."""

    def __init__(
        self,
        initial_scale: float = 0.25,
        final_scale: float = 1.0,
        total_steps: int = 100_000,
        anneal_type: str = "linear",
    ) -> None:
        self.initial_scale = float(np.clip(initial_scale, 0.2, 1.0))
        self.final_scale = float(np.clip(final_scale, 0.2, 1.0))
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


@dataclass
class PlayerRewardLog:
    """Telemetry record of modular reward components for a single player."""
    win: float = 0.0
    loss: float = 0.0
    draw: float = 0.0
    brick_destruction: float = 0.0
    pbrs_stats: float = 0.0
    connectivity: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "win": self.win,
            "loss": self.loss,
            "draw": self.draw,
            "brick_destruction": self.brick_destruction,
            "pbrs_stats": self.pbrs_stats,
            "connectivity": self.connectivity,
            "total": self.total,
        }


class RewardTracker:
    """Modular reward calculator with telemetry logging and anti-loop guarantees."""

    def __init__(
        self,
        self_damage_scale: float | CurriculumLethalityScheduler = 0.25,
        auxiliary_reward_cap: float = AUXILIARY_REWARD_CAP,
    ) -> None:
        self.scheduler: CurriculumLethalityScheduler | None = None
        if isinstance(self_damage_scale, CurriculumLethalityScheduler):
            self.scheduler = self_damage_scale
            self._self_damage_scale = self.scheduler.scale
        else:
            self._self_damage_scale = float(np.clip(self_damage_scale, 0.2, 1.0))

        self.auxiliary_reward_cap = auxiliary_reward_cap
        self.reset()

    @property
    def self_damage_scale(self) -> float:
        if self.scheduler is not None:
            return self.scheduler.scale
        return self._self_damage_scale

    @self_damage_scale.setter
    def self_damage_scale(self, val: float) -> None:
        self._self_damage_scale = float(np.clip(val, 0.2, 1.0))

    def reset(self, initial_state: GameState | None = None) -> None:
        """Reset per-episode state trackers."""
        self.prev_potentials: dict[int, float] = {0: 0.0, 1: 0.0}
        self.cumulative_aux: dict[int, float] = {0: 0.0, 1: 0.0}
        self.connectivity_awarded: bool = False
        self.episode_logs: dict[int, PlayerRewardLog] = {0: PlayerRewardLog(), 1: PlayerRewardLog()}

        if initial_state is not None:
            self.prev_potentials[0] = compute_pbrs_potential(initial_state.players[0])
            self.prev_potentials[1] = compute_pbrs_potential(initial_state.players[1])
            # If map starts already connected, do not award bonus later
            p0 = initial_state.players[0]
            p1 = initial_state.players[1]
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

        # 2. PBRS Stat Potentials: Phi(s') - Phi(s)
        curr_phi_0 = compute_pbrs_potential(state_after.players[0])
        curr_phi_1 = compute_pbrs_potential(state_after.players[1])
        p0_pbrs = curr_phi_0 - self.prev_potentials[0]
        p1_pbrs = curr_phi_1 - self.prev_potentials[1]
        self.prev_potentials[0] = curr_phi_0
        self.prev_potentials[1] = curr_phi_1

        # 3. Map Control / Connectivity Opening (+0.05 one-time bonus)
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

        # Apply auxiliary reward cap per match (< 0.35)
        # Positive components are proportionally capped; negative PBRS is applied in full
        def apply_cap_and_scale(
            pid: int,
            brick_rew: float,
            pbrs: float,
            conn_rew: float,
        ) -> tuple[float, float, float, float]:
            pos_pbrs = max(0.0, pbrs)
            neg_pbrs = min(0.0, pbrs)
            raw_pos = brick_rew + pos_pbrs + conn_rew
            available = max(0.0, self.auxiliary_reward_cap - self.cumulative_aux[pid])
            capped_pos = min(raw_pos, available)
            self.cumulative_aux[pid] += capped_pos + neg_pbrs

            factor = (capped_pos / raw_pos) if raw_pos > 1e-7 else 1.0
            actual_brick = brick_rew * factor
            actual_pbrs = (pos_pbrs * factor) + neg_pbrs
            actual_conn = conn_rew * factor
            actual_total = actual_brick + actual_pbrs + actual_conn
            return actual_brick, actual_pbrs, actual_conn, actual_total

        p0_b_act, p0_p_act, p0_c_act, actual_aux_0 = apply_cap_and_scale(0, p0_brick_rew, p0_pbrs, p0_conn_rew)
        p1_b_act, p1_p_act, p1_c_act, actual_aux_1 = apply_cap_and_scale(1, p1_brick_rew, p1_pbrs, p1_conn_rew)

        log_0 = PlayerRewardLog(
            brick_destruction=p0_b_act,
            pbrs_stats=p0_p_act,
            connectivity=p0_c_act,
        )
        log_1 = PlayerRewardLog(
            brick_destruction=p1_b_act,
            pbrs_stats=p1_p_act,
            connectivity=p1_c_act,
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
                # Player 0 won, Player 1 lost
                log_0.win = TERMINAL_WIN
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
                else:
                    log_1.loss = -1.0
            elif p1_after.alive and not p0_after.alive:
                # Player 1 won, Player 0 lost
                log_1.win = TERMINAL_WIN
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
                else:
                    log_0.loss = -1.0
            elif truncated:
                # Timeout draw
                log_0.draw = TERMINAL_DRAW
                log_1.draw = TERMINAL_DRAW

        # Compute totals
        log_0.total = log_0.win + log_0.loss + log_0.draw + actual_aux_0
        log_1.total = log_1.win + log_1.loss + log_1.draw + actual_aux_1

        # Accumulate into episode totals
        for attr in ["win", "loss", "draw", "brick_destruction", "pbrs_stats", "connectivity", "total"]:
            setattr(self.episode_logs[0], attr, getattr(self.episode_logs[0], attr) + getattr(log_0, attr))
            setattr(self.episode_logs[1], attr, getattr(self.episode_logs[1], attr) + getattr(log_1, attr))

        step_rewards["player_0"] = float(log_0.total)
        step_rewards["player_1"] = float(log_1.total)
        step_telemetry["player_0"] = log_0.as_dict()
        step_telemetry["player_1"] = log_1.as_dict()

        return step_rewards, step_telemetry
