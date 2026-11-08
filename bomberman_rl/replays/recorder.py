"""Zero-overhead Replay & Telemetry Recorder for 1v1 Bomberman.

Records deterministic game states, policy action probability distributions,
critic value estimates, and modular reward breakdowns at every tick.
Saves and loads compressed .npz episode trajectories with bit-exact roundtrips.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    TileType,
    PowerUpType,
    MoveAction,
    BombAction,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.rl.rewards import PlayerRewardLog

MAX_BOMBS_TRACKED: int = 32
REWARD_COMPONENT_NAMES: list[str] = [
    "win",
    "loss",
    "draw",
    "brick_destruction",
    "pbrs_stats",
    "connectivity",
    "total",
]


@dataclass
class GameEvent:
    """Discrete match event with frame timestamp for timeline scrubbing."""
    tick: int
    event_type: str
    description: str
    player_id: int | None = None
    tile: tuple[int, int] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "event_type": self.event_type,
            "description": self.description,
            "player_id": self.player_id,
            "tile": list(self.tile) if self.tile is not None else None,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameEvent:
        tile = tuple(d["tile"]) if d.get("tile") is not None else None
        return cls(
            tick=int(d["tick"]),
            event_type=str(d["event_type"]),
            description=str(d["description"]),
            player_id=int(d["player_id"]) if d.get("player_id") is not None else None,
            tile=tile,
            data=d.get("data", {}),
        )


class ReplayEpisode:
    """In-memory contiguous replay trajectory loaded from disk or recorder."""

    def __init__(
        self,
        total_ticks: int,
        grid: np.ndarray,
        powerups: np.ndarray,
        powerup_spawn_tick: np.ndarray,
        flame_timer: np.ndarray,
        flame_owner_mask: np.ndarray,
        player_pos: np.ndarray,
        player_speed: np.ndarray,
        player_max_bombs: np.ndarray,
        player_blast_radius: np.ndarray,
        player_can_kick: np.ndarray,
        player_bomb_pass: np.ndarray,
        player_shield_timer: np.ndarray,
        player_alive: np.ndarray,
        bomb_count: np.ndarray,
        bomb_id: np.ndarray,
        bomb_owner_id: np.ndarray,
        bomb_pos: np.ndarray,
        bomb_timer: np.ndarray,
        bomb_blast_radius: np.ndarray,
        bomb_velocity: np.ndarray,
        bomb_overlapping_mask: np.ndarray,
        actions: np.ndarray,
        action_probs_p0: np.ndarray,
        action_probs_p1: np.ndarray,
        critic_values: np.ndarray,
        rewards_p0: np.ndarray,
        rewards_p1: np.ndarray,
        cumulative_rewards_p0: np.ndarray,
        cumulative_rewards_p1: np.ndarray,
        ticks: np.ndarray,
        next_bomb_id: np.ndarray,
        events: list[GameEvent] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.total_ticks = int(total_ticks)
        self.grid = grid
        self.powerups = powerups
        self.powerup_spawn_tick = powerup_spawn_tick
        self.flame_timer = flame_timer
        self.flame_owner_mask = flame_owner_mask
        self.player_pos = player_pos
        self.player_speed = player_speed
        self.player_max_bombs = player_max_bombs
        self.player_blast_radius = player_blast_radius
        self.player_can_kick = player_can_kick
        self.player_bomb_pass = player_bomb_pass
        self.player_shield_timer = player_shield_timer
        self.player_alive = player_alive
        self.bomb_count = bomb_count
        self.bomb_id = bomb_id
        self.bomb_owner_id = bomb_owner_id
        self.bomb_pos = bomb_pos
        self.bomb_timer = bomb_timer
        self.bomb_blast_radius = bomb_blast_radius
        self.bomb_velocity = bomb_velocity
        self.bomb_overlapping_mask = bomb_overlapping_mask
        self.actions = actions
        self.action_probs_p0 = action_probs_p0
        self.action_probs_p1 = action_probs_p1
        self.critic_values = critic_values
        self.rewards_p0 = rewards_p0
        self.rewards_p1 = rewards_p1
        self.cumulative_rewards_p0 = cumulative_rewards_p0
        self.cumulative_rewards_p1 = cumulative_rewards_p1
        self.ticks = ticks
        self.next_bomb_id = next_bomb_id
        self.events: list[GameEvent] = events if events is not None else []
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    def __len__(self) -> int:
        return self.total_ticks

    def get_progress_ratio(self, tick: int) -> float:
        """Compute the strictly synchronized timeline progress ratio."""
        if self.total_ticks <= 0:
            return 0.0
        return float(tick) / float(self.total_ticks)

    def get_state(self, tick: int) -> GameState:
        """Reconstruct the bit-exact GameState at the given tick."""
        if self.total_ticks == 0:
            return GameState.create_initial()

        t = int(np.clip(tick, 0, self.total_ticks - 1))

        p0 = PlayerState(
            id=0,
            r=float(self.player_pos[t, 0, 0]),
            c=float(self.player_pos[t, 0, 1]),
            speed=float(self.player_speed[t, 0]),
            max_bombs=int(self.player_max_bombs[t, 0]),
            blast_radius=int(self.player_blast_radius[t, 0]),
            can_kick=bool(self.player_can_kick[t, 0]),
            bomb_pass=bool(self.player_bomb_pass[t, 0]),
            shield_timer=int(self.player_shield_timer[t, 0]),
            alive=bool(self.player_alive[t, 0]),
        )
        p1 = PlayerState(
            id=1,
            r=float(self.player_pos[t, 1, 0]),
            c=float(self.player_pos[t, 1, 1]),
            speed=float(self.player_speed[t, 1]),
            max_bombs=int(self.player_max_bombs[t, 1]),
            blast_radius=int(self.player_blast_radius[t, 1]),
            can_kick=bool(self.player_can_kick[t, 1]),
            bomb_pass=bool(self.player_bomb_pass[t, 1]),
            shield_timer=int(self.player_shield_timer[t, 1]),
            alive=bool(self.player_alive[t, 1]),
        )

        bombs: list[BombState] = []
        b_count = int(self.bomb_count[t])
        for i in range(b_count):
            mask = int(self.bomb_overlapping_mask[t, i])
            overlapping: set[int] = set()
            if mask & 1:
                overlapping.add(0)
            if mask & 2:
                overlapping.add(1)

            bombs.append(
                BombState(
                    id=int(self.bomb_id[t, i]),
                    owner_id=int(self.bomb_owner_id[t, i]),
                    r=float(self.bomb_pos[t, i, 0]),
                    c=float(self.bomb_pos[t, i, 1]),
                    timer=int(self.bomb_timer[t, i]),
                    blast_radius=int(self.bomb_blast_radius[t, i]),
                    velocity_r=float(self.bomb_velocity[t, i, 0]),
                    velocity_c=float(self.bomb_velocity[t, i, 1]),
                    overlapping_players=overlapping,
                )
            )

        return GameState(
            grid=self.grid[t].copy(),
            powerups=self.powerups[t].copy(),
            powerup_spawn_tick=self.powerup_spawn_tick[t].copy(),
            flame_timer=self.flame_timer[t].copy(),
            flame_owner_mask=self.flame_owner_mask[t].copy(),
            players=[p0, p1],
            bombs=bombs,
            tick=int(self.ticks[t]),
            next_bomb_id=int(self.next_bomb_id[t]),
        )

    def get_telemetry(self, tick: int) -> dict[str, Any]:
        """Retrieve neural and reward diagnostics at the given tick."""
        if self.total_ticks == 0:
            return {}

        t = int(np.clip(tick, 0, self.total_ticks - 1))

        def make_rew_dict(arr: np.ndarray) -> dict[str, float]:
            return {name: float(arr[i]) for i, name in enumerate(REWARD_COMPONENT_NAMES)}

        return {
            "tick": int(self.ticks[t]),
            "progress_ratio": self.get_progress_ratio(t),
            "actions": {
                "player_0": (int(self.actions[t, 0, 0]), int(self.actions[t, 0, 1])),
                "player_1": (int(self.actions[t, 1, 0]), int(self.actions[t, 1, 1])),
            },
            "action_probs_p0": self.action_probs_p0[t].copy(),
            "action_probs_p1": self.action_probs_p1[t].copy(),
            "value_p0": float(self.critic_values[t, 0]),
            "value_p1": float(self.critic_values[t, 1]),
            "rewards_p0": make_rew_dict(self.rewards_p0[t]),
            "rewards_p1": make_rew_dict(self.rewards_p1[t]),
            "cumulative_rewards_p0": make_rew_dict(self.cumulative_rewards_p0[t]),
            "cumulative_rewards_p1": make_rew_dict(self.cumulative_rewards_p1[t]),
        }

    def save(self, path: str) -> None:
        """Save episode trajectory to compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        events_json = json.dumps([e.to_dict() for e in self.events])
        metadata_json = json.dumps(self.metadata)

        np.savez_compressed(
            path,
            total_ticks=self.total_ticks,
            grid=self.grid,
            powerups=self.powerups,
            powerup_spawn_tick=self.powerup_spawn_tick,
            flame_timer=self.flame_timer,
            flame_owner_mask=self.flame_owner_mask,
            player_pos=self.player_pos,
            player_speed=self.player_speed,
            player_max_bombs=self.player_max_bombs,
            player_blast_radius=self.player_blast_radius,
            player_can_kick=self.player_can_kick,
            player_bomb_pass=self.player_bomb_pass,
            player_shield_timer=self.player_shield_timer,
            player_alive=self.player_alive,
            bomb_count=self.bomb_count,
            bomb_id=self.bomb_id,
            bomb_owner_id=self.bomb_owner_id,
            bomb_pos=self.bomb_pos,
            bomb_timer=self.bomb_timer,
            bomb_blast_radius=self.bomb_blast_radius,
            bomb_velocity=self.bomb_velocity,
            bomb_overlapping_mask=self.bomb_overlapping_mask,
            actions=self.actions,
            action_probs_p0=self.action_probs_p0,
            action_probs_p1=self.action_probs_p1,
            critic_values=self.critic_values,
            rewards_p0=self.rewards_p0,
            rewards_p1=self.rewards_p1,
            cumulative_rewards_p0=self.cumulative_rewards_p0,
            cumulative_rewards_p1=self.cumulative_rewards_p1,
            ticks=self.ticks,
            next_bomb_id=self.next_bomb_id,
            events_json=np.array(events_json),
            metadata_json=np.array(metadata_json),
        )

    @classmethod
    def load(cls, path: str) -> ReplayEpisode:
        """Load episode trajectory from compressed .npz archive."""
        data = np.load(path, allow_pickle=False)

        events: list[GameEvent] = []
        if "events_json" in data:
            raw_events = json.loads(str(data["events_json"]))
            events = [GameEvent.from_dict(d) for d in raw_events]

        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"]))

        total_ticks = int(data["total_ticks"])

        return cls(
            total_ticks=total_ticks,
            grid=data["grid"],
            powerups=data["powerups"],
            powerup_spawn_tick=data["powerup_spawn_tick"],
            flame_timer=data["flame_timer"],
            flame_owner_mask=data["flame_owner_mask"],
            player_pos=data["player_pos"],
            player_speed=data["player_speed"],
            player_max_bombs=data["player_max_bombs"],
            player_blast_radius=data["player_blast_radius"],
            player_can_kick=data["player_can_kick"],
            player_bomb_pass=data["player_bomb_pass"],
            player_shield_timer=data["player_shield_timer"],
            player_alive=data["player_alive"],
            bomb_count=data["bomb_count"],
            bomb_id=data["bomb_id"],
            bomb_owner_id=data["bomb_owner_id"],
            bomb_pos=data["bomb_pos"],
            bomb_timer=data["bomb_timer"],
            bomb_blast_radius=data["bomb_blast_radius"],
            bomb_velocity=data["bomb_velocity"],
            bomb_overlapping_mask=data["bomb_overlapping_mask"],
            actions=data["actions"],
            action_probs_p0=data["action_probs_p0"],
            action_probs_p1=data["action_probs_p1"],
            critic_values=data["critic_values"],
            rewards_p0=data["rewards_p0"],
            rewards_p1=data["rewards_p1"],
            cumulative_rewards_p0=data["cumulative_rewards_p0"],
            cumulative_rewards_p1=data["cumulative_rewards_p1"],
            ticks=data["ticks"],
            next_bomb_id=data["next_bomb_id"],
            events=events,
            metadata=metadata,
        )


class ReplayRecorder:
    """Lightweight, zero-overhead telemetry and episode trajectory recorder."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata if metadata is not None else {}
        self.frames_count = 0

        # Frame lists for fast append during simulation
        self._grid: list[np.ndarray] = []
        self._powerups: list[np.ndarray] = []
        self._powerup_spawn_tick: list[np.ndarray] = []
        self._flame_timer: list[np.ndarray] = []
        self._flame_owner_mask: list[np.ndarray] = []

        self._player_pos: list[np.ndarray] = []
        self._player_speed: list[np.ndarray] = []
        self._player_max_bombs: list[np.ndarray] = []
        self._player_blast_radius: list[np.ndarray] = []
        self._player_can_kick: list[np.ndarray] = []
        self._player_bomb_pass: list[np.ndarray] = []
        self._player_shield_timer: list[np.ndarray] = []
        self._player_alive: list[np.ndarray] = []

        self._bomb_count: list[int] = []
        self._bomb_id: list[np.ndarray] = []
        self._bomb_owner_id: list[np.ndarray] = []
        self._bomb_pos: list[np.ndarray] = []
        self._bomb_timer: list[np.ndarray] = []
        self._bomb_blast_radius: list[np.ndarray] = []
        self._bomb_velocity: list[np.ndarray] = []
        self._bomb_overlapping_mask: list[np.ndarray] = []

        self._actions: list[np.ndarray] = []
        self._action_probs_p0: list[np.ndarray] = []
        self._action_probs_p1: list[np.ndarray] = []
        self._critic_values: list[np.ndarray] = []

        self._rewards_p0: list[np.ndarray] = []
        self._rewards_p1: list[np.ndarray] = []
        self._cum_rewards_p0: np.ndarray = np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float32)
        self._cum_rewards_p1: np.ndarray = np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float32)
        self._cumulative_p0: list[np.ndarray] = []
        self._cumulative_p1: list[np.ndarray] = []

        self._ticks: list[int] = []
        self._next_bomb_id: list[int] = []
        self.events: list[GameEvent] = []

        self._prev_state: GameState | None = None

    def __len__(self) -> int:
        return self.frames_count

    def _normalize_probs(self, probs: Any, target_size: int = 10) -> np.ndarray:
        arr = np.zeros(target_size, dtype=np.float32)
        if probs is None:
            arr[0] = 1.0  # NOOP move
            arr[5] = 1.0  # NOOP bomb
            return arr
        if isinstance(probs, dict):
            m = np.asarray(probs.get("move", [1, 0, 0, 0, 0]), dtype=np.float32).ravel()
            b = np.asarray(probs.get("bomb", [1, 0]), dtype=np.float32).ravel()
            arr[:min(len(m), 5)] = m[:5]
            arr[5:5 + min(len(b), 2)] = b[:2]
            return arr
        raw = np.asarray(probs, dtype=np.float32).ravel()
        n = min(len(raw), target_size)
        arr[:n] = raw[:n]
        return arr

    def _normalize_rewards(self, rew: Any) -> np.ndarray:
        arr = np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float32)
        if rew is None:
            return arr
        if isinstance(rew, PlayerRewardLog):
            d = rew.as_dict()
        elif isinstance(rew, dict):
            d = rew
        elif isinstance(rew, (list, tuple, np.ndarray)):
            for i, val in enumerate(rew):
                if i < len(arr):
                    arr[i] = float(val)
            return arr
        else:
            arr[-1] = float(rew)
            return arr

        for i, name in enumerate(REWARD_COMPONENT_NAMES):
            if name in d:
                arr[i] = float(d[name])
            elif name == "brick_destruction" and "bricks" in d:
                arr[i] = float(d["bricks"])
            elif name == "pbrs_stats" and "pbrs" in d:
                arr[i] = float(d["pbrs"])
            elif name == "connectivity" and "conn" in d:
                arr[i] = float(d["conn"])
            elif name == "win" and "terminal" in d and float(d["terminal"]) > 0:
                arr[i] = float(d["terminal"])
            elif name == "loss" and "terminal" in d and float(d["terminal"]) < 0:
                arr[i] = float(d["terminal"])

        if "total" not in d:
            arr[-1] = float(np.sum(arr[:-1]))
        return arr

    def record_frame(
        self,
        state: GameState,
        actions: tuple[Any, Any] | list[Any] | dict[str | int, Any] | None = None,
        action_probs_p0: Any = None,
        action_probs_p1: Any = None,
        value_p0: float = 0.0,
        value_p1: float = 0.0,
        rewards_p0: Any = None,
        rewards_p1: Any = None,
        events: list[GameEvent] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Record a single simulation tick with telemetry."""
        # 1. State arrays
        self._grid.append(state.grid.copy())
        self._powerups.append(state.powerups.copy())
        self._powerup_spawn_tick.append(state.powerup_spawn_tick.copy())
        self._flame_timer.append(state.flame_timer.copy())
        self._flame_owner_mask.append(state.flame_owner_mask.copy())

        # 2. Player states
        p0 = state.players[0]
        p1 = state.players[1]
        self._player_pos.append(np.array([[p0.r, p0.c], [p1.r, p1.c]], dtype=np.float32))
        self._player_speed.append(np.array([p0.speed, p1.speed], dtype=np.float32))
        self._player_max_bombs.append(np.array([p0.max_bombs, p1.max_bombs], dtype=np.int16))
        self._player_blast_radius.append(np.array([p0.blast_radius, p1.blast_radius], dtype=np.int16))
        self._player_can_kick.append(np.array([p0.can_kick, p1.can_kick], dtype=bool))
        self._player_bomb_pass.append(np.array([p0.bomb_pass, p1.bomb_pass], dtype=bool))
        self._player_shield_timer.append(np.array([p0.shield_timer, p1.shield_timer], dtype=np.int16))
        self._player_alive.append(np.array([p0.alive, p1.alive], dtype=bool))

        # 3. Bombs state (dense packed with capacity MAX_BOMBS_TRACKED)
        n_bombs = min(len(state.bombs), MAX_BOMBS_TRACKED)
        self._bomb_count.append(n_bombs)

        b_id = np.zeros(MAX_BOMBS_TRACKED, dtype=np.int32)
        b_owner = np.zeros(MAX_BOMBS_TRACKED, dtype=np.int8)
        b_pos = np.zeros((MAX_BOMBS_TRACKED, 2), dtype=np.float32)
        b_timer = np.zeros(MAX_BOMBS_TRACKED, dtype=np.int16)
        b_radius = np.zeros(MAX_BOMBS_TRACKED, dtype=np.int16)
        b_vel = np.zeros((MAX_BOMBS_TRACKED, 2), dtype=np.float32)
        b_overlap = np.zeros(MAX_BOMBS_TRACKED, dtype=np.int8)

        for i in range(n_bombs):
            b = state.bombs[i]
            b_id[i] = b.id
            b_owner[i] = b.owner_id
            b_pos[i, 0] = b.r
            b_pos[i, 1] = b.c
            b_timer[i] = b.timer
            b_radius[i] = b.blast_radius
            b_vel[i, 0] = b.velocity_r
            b_vel[i, 1] = b.velocity_c
            mask = 0
            if 0 in b.overlapping_players:
                mask |= 1
            if 1 in b.overlapping_players:
                mask |= 2
            b_overlap[i] = mask

        self._bomb_id.append(b_id)
        self._bomb_owner_id.append(b_owner)
        self._bomb_pos.append(b_pos)
        self._bomb_timer.append(b_timer)
        self._bomb_blast_radius.append(b_radius)
        self._bomb_velocity.append(b_vel)
        self._bomb_overlapping_mask.append(b_overlap)

        # 4. Actions
        act_arr = np.zeros((2, 2), dtype=np.int16)
        if actions is not None:
            if isinstance(actions, dict):
                p0_a = actions.get("player_0", actions.get(0, (0, 0)))
                p1_a = actions.get("player_1", actions.get(1, (0, 0)))
            elif isinstance(actions, (list, tuple, np.ndarray)) and len(actions) >= 2:
                p0_a = actions[0]
                p1_a = actions[1]
            else:
                p0_a = (0, 0)
                p1_a = (0, 0)

            def parse_act(a: Any) -> tuple[int, int]:
                if isinstance(a, (list, tuple, np.ndarray)):
                    m = int(a[0])
                    b = int(a[1]) if len(a) > 1 else 0
                    return (m, b)
                val = int(a)
                return (0, 1) if val == 5 else (val, 0)

            act_arr[0] = parse_act(p0_a)
            act_arr[1] = parse_act(p1_a)

        self._actions.append(act_arr)

        # 5. Probabilities & Values
        norm_p0 = self._normalize_probs(action_probs_p0)
        norm_p1 = self._normalize_probs(action_probs_p1)
        self._action_probs_p0.append(norm_p0)
        self._action_probs_p1.append(norm_p1)
        self._critic_values.append(np.array([float(value_p0), float(value_p1)], dtype=np.float32))

        # 6. Rewards
        r0 = self._normalize_rewards(rewards_p0)
        r1 = self._normalize_rewards(rewards_p1)
        self._rewards_p0.append(r0)
        self._rewards_p1.append(r1)
        self._cum_rewards_p0 += r0
        self._cum_rewards_p1 += r1
        self._cumulative_p0.append(self._cum_rewards_p0.copy())
        self._cumulative_p1.append(self._cum_rewards_p1.copy())

        # 7. Ticks & Bomb counter
        self._ticks.append(int(state.tick))
        self._next_bomb_id.append(int(state.next_bomb_id))

        # 8. Event Detection
        if events is not None:
            for ev in events:
                if isinstance(ev, GameEvent):
                    self.events.append(ev)
                elif isinstance(ev, dict):
                    self.events.append(GameEvent.from_dict(ev))
        else:
            self._detect_events(state)

        self._prev_state = state.clone()
        self.frames_count += 1

    def _detect_events(self, curr_state: GameState) -> None:
        """Infer discrete events by comparing with previous frame."""
        if self._prev_state is None:
            return

        tick = curr_state.tick

        # Player deaths
        for pid in (0, 1):
            if self._prev_state.players[pid].alive and not curr_state.players[pid].alive:
                self.events.append(
                    GameEvent(
                        tick=tick,
                        event_type="player_death",
                        description=f"Player {pid} eliminated",
                        player_id=pid,
                        tile=curr_state.players[pid].tile,
                    )
                )

        # Shield activations
        for pid in (0, 1):
            if self._prev_state.players[pid].shield_timer == 0 and curr_state.players[pid].shield_timer > 0:
                self.events.append(
                    GameEvent(
                        tick=tick,
                        event_type="shield_activated",
                        description=f"Player {pid} activated Forcefield Shield",
                        player_id=pid,
                        tile=curr_state.players[pid].tile,
                    )
                )

        # Powerup pickups
        for pid in (0, 1):
            prev_p = self._prev_state.players[pid]
            curr_p = curr_state.players[pid]
            if curr_p.max_bombs > prev_p.max_bombs:
                self.events.append(GameEvent(tick=tick, event_type="powerup_pickup", description=f"P{pid} collected +Bomb", player_id=pid, tile=curr_p.tile))
            elif curr_p.blast_radius > prev_p.blast_radius:
                self.events.append(GameEvent(tick=tick, event_type="powerup_pickup", description=f"P{pid} collected +Blast", player_id=pid, tile=curr_p.tile))
            elif curr_p.speed > prev_p.speed:
                self.events.append(GameEvent(tick=tick, event_type="powerup_pickup", description=f"P{pid} collected +Speed", player_id=pid, tile=curr_p.tile))
            elif curr_p.can_kick and not prev_p.can_kick:
                self.events.append(GameEvent(tick=tick, event_type="powerup_pickup", description=f"P{pid} collected Kick", player_id=pid, tile=curr_p.tile))
            elif curr_p.bomb_pass and not prev_p.bomb_pass:
                self.events.append(GameEvent(tick=tick, event_type="powerup_pickup", description=f"P{pid} collected Bomb Pass", player_id=pid, tile=curr_p.tile))

        # Brick breaches / destructions
        bricks_destroyed = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self._prev_state.grid[r, c] == TileType.BRICK and curr_state.grid[r, c] == TileType.EMPTY:
                    bricks_destroyed += 1
        if bricks_destroyed > 0:
            self.events.append(
                GameEvent(
                    tick=tick,
                    event_type="brick_destroyed",
                    description=f"{bricks_destroyed} brick(s) destroyed",
                )
            )

    def to_episode(self) -> ReplayEpisode:
        """Pack recorded lists into a contiguous ReplayEpisode."""
        n = self.frames_count
        if n == 0:
            empty_9x9 = np.zeros((0, GRID_SIZE, GRID_SIZE), dtype=np.int8)
            return ReplayEpisode(
                total_ticks=0,
                grid=empty_9x9,
                powerups=empty_9x9,
                powerup_spawn_tick=np.zeros((0, GRID_SIZE, GRID_SIZE), dtype=np.int32),
                flame_timer=np.zeros((0, GRID_SIZE, GRID_SIZE), dtype=np.int16),
                flame_owner_mask=empty_9x9,
                player_pos=np.zeros((0, 2, 2), dtype=np.float32),
                player_speed=np.zeros((0, 2), dtype=np.float32),
                player_max_bombs=np.zeros((0, 2), dtype=np.int16),
                player_blast_radius=np.zeros((0, 2), dtype=np.int16),
                player_can_kick=np.zeros((0, 2), dtype=bool),
                player_bomb_pass=np.zeros((0, 2), dtype=bool),
                player_shield_timer=np.zeros((0, 2), dtype=np.int16),
                player_alive=np.zeros((0, 2), dtype=bool),
                bomb_count=np.zeros(0, dtype=np.int16),
                bomb_id=np.zeros((0, MAX_BOMBS_TRACKED), dtype=np.int32),
                bomb_owner_id=np.zeros((0, MAX_BOMBS_TRACKED), dtype=np.int8),
                bomb_pos=np.zeros((0, MAX_BOMBS_TRACKED, 2), dtype=np.float32),
                bomb_timer=np.zeros((0, MAX_BOMBS_TRACKED), dtype=np.int16),
                bomb_blast_radius=np.zeros((0, MAX_BOMBS_TRACKED), dtype=np.int16),
                bomb_velocity=np.zeros((0, MAX_BOMBS_TRACKED, 2), dtype=np.float32),
                bomb_overlapping_mask=np.zeros((0, MAX_BOMBS_TRACKED), dtype=np.int8),
                actions=np.zeros((0, 2, 2), dtype=np.int16),
                action_probs_p0=np.zeros((0, 10), dtype=np.float32),
                action_probs_p1=np.zeros((0, 10), dtype=np.float32),
                critic_values=np.zeros((0, 2), dtype=np.float32),
                rewards_p0=np.zeros((0, len(REWARD_COMPONENT_NAMES)), dtype=np.float32),
                rewards_p1=np.zeros((0, len(REWARD_COMPONENT_NAMES)), dtype=np.float32),
                cumulative_rewards_p0=np.zeros((0, len(REWARD_COMPONENT_NAMES)), dtype=np.float32),
                cumulative_rewards_p1=np.zeros((0, len(REWARD_COMPONENT_NAMES)), dtype=np.float32),
                ticks=np.zeros(0, dtype=np.int32),
                next_bomb_id=np.zeros(0, dtype=np.int32),
                events=self.events,
                metadata=self.metadata,
            )

        return ReplayEpisode(
            total_ticks=n,
            grid=np.stack(self._grid, axis=0),
            powerups=np.stack(self._powerups, axis=0),
            powerup_spawn_tick=np.stack(self._powerup_spawn_tick, axis=0),
            flame_timer=np.stack(self._flame_timer, axis=0),
            flame_owner_mask=np.stack(self._flame_owner_mask, axis=0),
            player_pos=np.stack(self._player_pos, axis=0),
            player_speed=np.stack(self._player_speed, axis=0),
            player_max_bombs=np.stack(self._player_max_bombs, axis=0),
            player_blast_radius=np.stack(self._player_blast_radius, axis=0),
            player_can_kick=np.stack(self._player_can_kick, axis=0),
            player_bomb_pass=np.stack(self._player_bomb_pass, axis=0),
            player_shield_timer=np.stack(self._player_shield_timer, axis=0),
            player_alive=np.stack(self._player_alive, axis=0),
            bomb_count=np.array(self._bomb_count, dtype=np.int16),
            bomb_id=np.stack(self._bomb_id, axis=0),
            bomb_owner_id=np.stack(self._bomb_owner_id, axis=0),
            bomb_pos=np.stack(self._bomb_pos, axis=0),
            bomb_timer=np.stack(self._bomb_timer, axis=0),
            bomb_blast_radius=np.stack(self._bomb_blast_radius, axis=0),
            bomb_velocity=np.stack(self._bomb_velocity, axis=0),
            bomb_overlapping_mask=np.stack(self._bomb_overlapping_mask, axis=0),
            actions=np.stack(self._actions, axis=0),
            action_probs_p0=np.stack(self._action_probs_p0, axis=0),
            action_probs_p1=np.stack(self._action_probs_p1, axis=0),
            critic_values=np.stack(self._critic_values, axis=0),
            rewards_p0=np.stack(self._rewards_p0, axis=0),
            rewards_p1=np.stack(self._rewards_p1, axis=0),
            cumulative_rewards_p0=np.stack(self._cumulative_p0, axis=0),
            cumulative_rewards_p1=np.stack(self._cumulative_p1, axis=0),
            ticks=np.array(self._ticks, dtype=np.int32),
            next_bomb_id=np.array(self._next_bomb_id, dtype=np.int32),
            events=list(self.events),
            metadata=dict(self.metadata),
        )

    def save(self, path: str) -> None:
        """Pack and save episode trajectory directly to compressed .npz archive."""
        episode = self.to_episode()
        episode.save(path)

    @classmethod
    def load(cls, path: str) -> ReplayEpisode:
        """Direct load convenience method."""
        return ReplayEpisode.load(path)
