"""Gymnasium-compatible environment interface for 1v1 Bomberman."""

from __future__ import annotations
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    BOMB_TIMER,
    MAX_BLAST_RADIUS,
    MoveAction,
    BombAction,
    TileType,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState
from bomberman_rl.engine.physics import BombermanPhysics


class BombermanEnv(gym.Env):
    """Headless 1v1 Bomberman Gymnasium Environment."""

    metadata = {"render_modes": ["headless"], "render_fps": 20}

    def __init__(self, seed: int | None = None, max_ticks: int = MAX_TICKS) -> None:
        super().__init__()
        self.max_ticks = max_ticks
        self.state: GameState = GameState.create_initial(seed)

        # MultiDiscrete([5, 2]): 5 move actions (0: NOOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT),
        # 2 bomb actions (0: NOOP, 1: PLACE_BOMB) per player
        self.action_space = spaces.Dict({
            "player_0": spaces.MultiDiscrete([5, 2]),
            "player_1": spaces.MultiDiscrete([5, 2]),
        })

        # Observation space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=2, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            "bomb_timer": spaces.Box(low=0, high=BOMB_TIMER, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int16),
            "bomb_blast_radius": spaces.Box(low=0, high=MAX_BLAST_RADIUS, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            "flame_timer": spaces.Box(low=0, high=10, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int16),
            "flame_owner_mask": spaces.Box(low=0, high=3, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            "powerups": spaces.Box(low=0, high=9, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            "player_0": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            "player_1": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            "tick": spaces.Discrete(MAX_TICKS + 1),
        })

    def _get_obs(self) -> dict[str, Any]:
        p0 = self.state.players[0]
        p1 = self.state.players[1]

        p0_vec = np.array([
            p0.r,
            p0.c,
            p0.speed,
            float(p0.max_bombs),
            float(p0.blast_radius),
            1.0 if p0.can_kick else 0.0,
            1.0 if p0.bomb_pass else 0.0,
            float(p0.shield_timer),
            1.0 if p0.alive else 0.0,
        ], dtype=np.float32)

        p1_vec = np.array([
            p1.r,
            p1.c,
            p1.speed,
            float(p1.max_bombs),
            float(p1.blast_radius),
            1.0 if p1.can_kick else 0.0,
            1.0 if p1.bomb_pass else 0.0,
            float(p1.shield_timer),
            1.0 if p1.alive else 0.0,
        ], dtype=np.float32)

        bomb_timer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
        bomb_blast_radius = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        for b in self.state.bombs:
            tr = int(np.floor(b.r))
            tc = int(np.floor(b.c))
            if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                bomb_timer[tr, tc] = b.timer
                bomb_blast_radius[tr, tc] = b.blast_radius

        return {
            "grid": self.state.grid.copy(),
            "bomb_timer": bomb_timer,
            "bomb_blast_radius": bomb_blast_radius,
            "flame_timer": self.state.flame_timer.copy(),
            "flame_owner_mask": self.state.flame_owner_mask.copy(),
            "powerups": self.state.powerups.copy(),
            "player_0": p0_vec,
            "player_1": p1_vec,
            "tick": self.state.tick,
        }

    def _get_info(self, winner: int | None, draw: bool, truncated: bool = False) -> dict[str, Any]:
        p0 = self.state.players[0]
        p1 = self.state.players[1]
        return {
            "winner": winner,
            "draw": draw,
            "truncated": truncated,
            "tick": self.state.tick,
            "p0_alive": p0.alive,
            "p1_alive": p1.alive,
            "p0_bombs": sum(1 for b in self.state.bombs if b.owner_id == 0),
            "p1_bombs": sum(1 for b in self.state.bombs if b.owner_id == 1),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.state = GameState.create_initial(seed)
        obs = self._get_obs()
        info = self._get_info(winner=None, draw=False)
        return obs, info

    def step(
        self,
        actions: dict[str | int, Any] | list[Any] | tuple[Any, ...],
    ) -> tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        """Advance the environment by one tick given actions for both players."""
        # Normalize actions
        p0_act = (MoveAction.NOOP, BombAction.NOOP)
        p1_act = (MoveAction.NOOP, BombAction.NOOP)

        if isinstance(actions, dict):
            if "player_0" in actions:
                p0_act = actions["player_0"]
            elif 0 in actions:
                p0_act = actions[0]

            if "player_1" in actions:
                p1_act = actions["player_1"]
            elif 1 in actions:
                p1_act = actions[1]
        elif isinstance(actions, (list, tuple, np.ndarray)) and len(actions) >= 2:
            p0_act = actions[0]
            p1_act = actions[1]

        # Advance physics
        BombermanPhysics.step(self.state, [p0_act, p1_act])

        p0 = self.state.players[0]
        p1 = self.state.players[1]

        # Check termination & truncation
        terminated = False
        truncated = False
        winner: int | None = None
        draw = False
        rewards = {"player_0": 0.0, "player_1": 0.0}

        if not p0.alive and not p1.alive:
            # Both died on the same tick -> Draw
            terminated = True
            winner = None
            draw = True
            rewards = {"player_0": 0.0, "player_1": 0.0}
        elif p0.alive and not p1.alive:
            # Player 0 won
            terminated = True
            winner = 0
            rewards = {"player_0": 1.0, "player_1": -1.0}
        elif p1.alive and not p0.alive:
            # Player 1 won
            terminated = True
            winner = 1
            rewards = {"player_0": -1.0, "player_1": 1.0}
        elif self.state.tick >= self.max_ticks:
            # Tick limit reached with both players alive -> Draw
            truncated = True
            winner = None
            draw = True
            rewards = {"player_0": 0.0, "player_1": 0.0}

        obs = self._get_obs()
        # Only a shortened (curriculum) horizon is an artificial truncation; the
        # full MAX_TICKS limit is a genuine terminal draw.
        artificial_truncation = truncated and self.max_ticks < MAX_TICKS
        info = self._get_info(winner=winner, draw=draw, truncated=artificial_truncation)
        return obs, rewards, terminated, truncated, info
