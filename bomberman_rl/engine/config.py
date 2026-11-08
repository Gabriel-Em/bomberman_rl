"""Configuration, constants, enums, speeds, ticks, and probabilities."""

from enum import IntEnum

# Time & Tick Rate
TICK_RATE: int = 20
DT: float = 1.0 / TICK_RATE  # 0.05 seconds
MAX_TICKS: int = 3600  # 180 seconds / 3 minutes

# Grid & Map Generation
GRID_SIZE: int = 9  # 9x9 logical playable grid
SAFE_ZONE_P0: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0))
SAFE_ZONE_P1: tuple[tuple[int, int], ...] = ((8, 8), (8, 7), (7, 8))
SPAWN_POS_P0: tuple[float, float] = (0.5, 0.5)
SPAWN_POS_P1: tuple[float, float] = (8.5, 8.5)

# Total tiles: 81
# Pillars: 16 (where r % 2 == 1 and c % 2 == 1)
# Safe tiles: 6
# Remaining valid empty tiles: 59
# 70% of 59 = 41.3 -> 41 breakable brick walls
NUM_BRICK_WALLS: int = 41

# Player Physics & Capabilities
BASE_PLAYER_SPEED: float = 2.5  # tiles/second -> 0.125 tiles/tick
SPEED_UP_INCREMENT: float = 0.5  # tiles/second -> 0.025 tiles/tick
MAX_PLAYER_SPEED: float = 4.5  # tiles/second -> 0.225 tiles/tick
PLAYER_BBOX_SIZE: float = 0.70  # tile units
PLAYER_HALF_SIZE: float = PLAYER_BBOX_SIZE / 2.0  # 0.35 tile units
CORNER_SLIDE_THRESHOLD: float = 0.30  # tile units

# Bombs & Explosions
BOMB_TIMER: int = 60  # ticks (3.0 seconds)
FLAME_DURATION: int = 10  # ticks (0.5 seconds)
KICK_SPEED: float = 5.0  # tiles/second -> 0.25 tiles/tick
DEFAULT_MAX_BOMBS: int = 1
MAX_BOMBS_LIMIT: int = 6
MIN_BOMBS_LIMIT: int = 1
DEFAULT_BLAST_RADIUS: int = 1
MAX_BLAST_RADIUS: int = 8
MIN_BLAST_RADIUS: int = 1
SHIELD_DURATION: int = 100  # ticks (5.0 seconds)

# Power-Up Drops
POWERUP_DROP_PROB: float = 0.35


class TileType(IntEnum):
    EMPTY = 0
    PILLAR = 1  # Unbreakable wall
    BRICK = 2   # Breakable wall


class MoveAction(IntEnum):
    NOOP = 0
    UP = 1     # dr = -1
    DOWN = 2   # dr = +1
    LEFT = 3   # dc = -1
    RIGHT = 4  # dc = +1


class BombAction(IntEnum):
    NOOP = 0
    PLACE_BOMB = 1


class PowerUpType(IntEnum):
    NONE = 0
    BLAST_UP = 1
    BOMB_UP = 2
    SPEED_UP = 3
    BLAST_DOWN = 4
    BOMB_DOWN = 5
    KICK = 6
    BOMB_PASS = 7
    SHIELD = 8
    MAX_BLAST = 9


# Relative weights when a power-up drop occurs (sum to 1.0)
POWERUP_PROBABILITIES: dict[PowerUpType, float] = {
    PowerUpType.BLAST_UP: 0.24,
    PowerUpType.BOMB_UP: 0.24,
    PowerUpType.SPEED_UP: 0.16,
    PowerUpType.BLAST_DOWN: 0.08,
    PowerUpType.BOMB_DOWN: 0.08,
    PowerUpType.KICK: 0.09,
    PowerUpType.BOMB_PASS: 0.05,
    PowerUpType.SHIELD: 0.05,
    PowerUpType.MAX_BLAST: 0.01,
}

POWERUP_TYPES_LIST: list[PowerUpType] = list(POWERUP_PROBABILITIES.keys())
POWERUP_WEIGHTS_LIST: list[float] = [POWERUP_PROBABILITIES[p] for p in POWERUP_TYPES_LIST]
