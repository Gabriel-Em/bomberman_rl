"""Deterministic game state container."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    SAFE_ZONE_P0,
    SAFE_ZONE_P1,
    SPAWN_POS_P0,
    SPAWN_POS_P1,
    NUM_BRICK_WALLS,
    BASE_PLAYER_SPEED,
    DEFAULT_MAX_BOMBS,
    DEFAULT_BLAST_RADIUS,
    TileType,
    PowerUpType,
)


@dataclass
class PlayerState:
    id: int
    r: float
    c: float
    speed: float = BASE_PLAYER_SPEED
    max_bombs: int = DEFAULT_MAX_BOMBS
    blast_radius: int = DEFAULT_BLAST_RADIUS
    can_kick: bool = False
    bomb_pass: bool = False
    shield_timer: int = 0
    alive: bool = True

    @property
    def tile(self) -> tuple[int, int]:
        return int(np.floor(self.r)), int(np.floor(self.c))

    def clone(self) -> PlayerState:
        return PlayerState(
            id=self.id,
            r=self.r,
            c=self.c,
            speed=self.speed,
            max_bombs=self.max_bombs,
            blast_radius=self.blast_radius,
            can_kick=self.can_kick,
            bomb_pass=self.bomb_pass,
            shield_timer=self.shield_timer,
            alive=self.alive,
        )


@dataclass
class BombState:
    id: int
    owner_id: int
    r: float
    c: float
    timer: int = 60
    blast_radius: int = 1
    velocity_r: float = 0.0  # in tiles/tick
    velocity_c: float = 0.0  # in tiles/tick
    overlapping_players: set[int] = field(default_factory=set)

    @property
    def tile(self) -> tuple[int, int]:
        return int(np.floor(self.r)), int(np.floor(self.c))

    def clone(self) -> BombState:
        return BombState(
            id=self.id,
            owner_id=self.owner_id,
            r=self.r,
            c=self.c,
            timer=self.timer,
            blast_radius=self.blast_radius,
            velocity_r=self.velocity_r,
            velocity_c=self.velocity_c,
            overlapping_players=set(self.overlapping_players),
        )


@dataclass
class GameState:
    grid: np.ndarray  # (9, 9) int8: TileType (EMPTY, PILLAR, BRICK)
    powerups: np.ndarray  # (9, 9) int8: PowerUpType
    powerup_spawn_tick: np.ndarray  # (9, 9) int32: tick when powerup spawned
    flame_timer: np.ndarray  # (9, 9) int16: remaining flame duration in ticks
    flame_owner_mask: np.ndarray  # (9, 9) int8: bitmask (1 = P0, 2 = P1, 3 = Both)
    players: list[PlayerState]
    bombs: list[BombState]
    tick: int = 0
    next_bomb_id: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @classmethod
    def create_initial(cls, seed: int | None = None) -> GameState:
        rng = np.random.default_rng(seed)
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        # 1. Place 16 pillars at odd coordinates (r % 2 == 1 and c % 2 == 1)
        for r in range(1, GRID_SIZE, 2):
            for c in range(1, GRID_SIZE, 2):
                grid[r, c] = TileType.PILLAR

        # 2. Collect 59 valid empty tiles (excluding pillars and spawn safe zones)
        safe_set = set(SAFE_ZONE_P0).union(SAFE_ZONE_P1)
        valid_tiles: list[tuple[int, int]] = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == TileType.EMPTY and (r, c) not in safe_set:
                    valid_tiles.append((r, c))

        # 3. Randomly place exactly 41 breakable brick walls (70% of 59)
        chosen_indices = rng.choice(len(valid_tiles), size=NUM_BRICK_WALLS, replace=False)
        for idx in chosen_indices:
            r, c = valid_tiles[idx]
            grid[r, c] = TileType.BRICK

        powerups = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        powerup_spawn_tick = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int32)
        flame_timer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
        flame_owner_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        players = [
            PlayerState(id=0, r=SPAWN_POS_P0[0], c=SPAWN_POS_P0[1]),
            PlayerState(id=1, r=SPAWN_POS_P1[0], c=SPAWN_POS_P1[1]),
        ]

        return cls(
            grid=grid,
            powerups=powerups,
            powerup_spawn_tick=powerup_spawn_tick,
            flame_timer=flame_timer,
            flame_owner_mask=flame_owner_mask,
            players=players,
            bombs=[],
            tick=0,
            next_bomb_id=0,
            rng=rng,
        )

    def clone(self) -> GameState:
        bg = type(self.rng.bit_generator)()
        bg.state = self.rng.bit_generator.state
        rng = np.random.Generator(bg)
        return GameState(
            grid=self.grid.copy(),
            powerups=self.powerups.copy(),
            powerup_spawn_tick=self.powerup_spawn_tick.copy(),
            flame_timer=self.flame_timer.copy(),
            flame_owner_mask=self.flame_owner_mask.copy(),
            players=[p.clone() for p in self.players],
            bombs=[b.clone() for b in self.bombs],
            tick=self.tick,
            next_bomb_id=self.next_bomb_id,
            rng=rng,
        )
