"""Fast Vectorized Multi-Process and Synchronous Environments for 1v1 Bomberman.

Supports League Self-Play opponent distribution:
- 70% Latest Policy vs. Latest Policy
- 20% Latest Policy vs. Past Checkpoint
- 10% Latest Policy vs. Heuristic Baseline
"""

from __future__ import annotations
import multiprocessing as mp
from typing import Any, Callable
import numpy as np
import torch

from bomberman_rl.engine.config import (
    GRID_SIZE,
    TileType,
    PowerUpType,
    MoveAction,
    BombAction,
)
from bomberman_rl.engine.state import GameState
from bomberman_rl.engine.env import BombermanEnv
from bomberman_rl.engine.physics import BombermanPhysics
from bomberman_rl.rl.observation import ObservationBuilder, flip_action_to_world, SPATIAL_CHANNELS, DEFAULT_SCALAR_DIM
from bomberman_rl.rl.rewards import RewardTracker
from bomberman_rl.rl.action_filter import SafetyActionFilter
from bomberman_rl.training.diagnostics import MatchStats


BUFF_TYPES: set[PowerUpType] = {
    PowerUpType.BLAST_UP,
    PowerUpType.BOMB_UP,
    PowerUpType.SPEED_UP,
    PowerUpType.KICK,
    PowerUpType.BOMB_PASS,
    PowerUpType.SHIELD,
    PowerUpType.MAX_BLAST,
}

DEBUFF_TYPES: set[PowerUpType] = {
    PowerUpType.BLAST_DOWN,
    PowerUpType.BOMB_DOWN,
}


class HeuristicOpponent:
    """Lightweight heuristic explorer baseline for league sparring."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.RandomState(seed)
        self.action_filter = SafetyActionFilter(enabled=True, safety_filter_prob=1.0)

    def act(self, state: GameState, player_id: int = 1) -> tuple[int, int]:
        player = state.players[player_id]
        if not player.alive:
            return (int(MoveAction.NOOP), int(BombAction.NOOP))

        masks = self.action_filter.get_action_mask(state, player_id, ego_perspective=True)
        safe_moves = [m for m in range(5) if masks["move_mask"][m]]
        if not safe_moves:
            safe_moves = [int(MoveAction.NOOP)]

        pr, pc = player.tile
        can_bomb = bool(masks["bomb_mask"][int(BombAction.PLACE_BOMB)])
        adjacent_brick = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if state.grid[nr, nc] == TileType.BRICK:
                    adjacent_brick = True
                    break

        bomb_act = int(BombAction.PLACE_BOMB) if (can_bomb and adjacent_brick and self.rng.random() < 0.4) else int(BombAction.NOOP)
        move_act = int(self.rng.choice(safe_moves))
        return (move_act, bomb_act)


class BombermanMatchEnv:
    """Single match environment encapsulating physics, ego-obs, rewards, and telemetry."""

    def __init__(
        self,
        seed: int | None = None,
        self_damage_scale: float = 0.25,
        safety_filter_prob: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.env = BombermanEnv(seed=seed)
        self.obs_builder = ObservationBuilder(scalar_dim=DEFAULT_SCALAR_DIM)
        self.reward_tracker = RewardTracker(self_damage_scale=self_damage_scale)
        self.action_filter = SafetyActionFilter(enabled=True, safety_filter_prob=safety_filter_prob)
        self.heuristic = HeuristicOpponent(seed=seed)
        self.checkpoint_dir = checkpoint_dir

        self.self_damage_scale = self_damage_scale
        self.safety_filter_prob = safety_filter_prob
        self.past_checkpoint_paths: list[str] = []
        self.opponent_type = "latest"  # "latest", "checkpoint", or "heuristic"
        self.opponent_checkpoint_path: str | None = None
        self.checkpoint_model: Any = None

        # Telemetry tracking for current match
        self._match_stats = MatchStats()

    def set_curriculum(self, self_damage_scale: float, safety_filter_prob: float) -> None:
        self.self_damage_scale = float(self_damage_scale)
        self.safety_filter_prob = float(safety_filter_prob)
        self.reward_tracker.self_damage_scale = self.self_damage_scale
        self.action_filter.safety_filter_prob = self.safety_filter_prob

    def set_opponent_pool(self, past_checkpoint_paths: list[str] | None) -> None:
        """Update candidate checkpoint pool for subsequent opponent matchmaking."""
        self.past_checkpoint_paths = list(past_checkpoint_paths) if past_checkpoint_paths else []

    def sample_opponent(self, past_checkpoint_paths: list[str] | None = None) -> str:
        """Sample opponent according to 70% latest, 20% past checkpoint, 10% baseline."""
        if past_checkpoint_paths is not None:
            self.past_checkpoint_paths = list(past_checkpoint_paths)

        u = self.rng.random()
        if u < 0.70:
            self.opponent_type = "latest"
            self.opponent_checkpoint_path = None
        elif u < 0.90:
            if self.past_checkpoint_paths:
                self.opponent_type = "checkpoint"
                self.opponent_checkpoint_path = str(self.rng.choice(self.past_checkpoint_paths))
            else:
                self.opponent_type = "latest"
                self.opponent_checkpoint_path = None
        else:
            self.opponent_type = "heuristic"
            self.opponent_checkpoint_path = None
        return self.opponent_type

    def reset(
        self,
        seed: int | None = None,
        past_checkpoint_paths: list[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if past_checkpoint_paths is not None:
            self.past_checkpoint_paths = list(past_checkpoint_paths)

        self.env.reset(seed=seed)
        self.reward_tracker.reset(self.env.state)
        self.sample_opponent(self.past_checkpoint_paths)
        self._match_stats = MatchStats()

        obs_0, obs_1 = self.obs_builder.build_both(self.env.state, copy=False)
        mask_0 = self.action_filter.get_action_mask(self.env.state, 0, ego_perspective=True)
        mask_1 = self.action_filter.get_action_mask(self.env.state, 1, ego_perspective=True)

        info = {
            "opponent_type": self.opponent_type,
            "opponent_checkpoint": self.opponent_checkpoint_path,
            "tick": self.env.state.tick,
        }
        return obs_0, mask_0, obs_1, mask_1, info

    def step(
        self,
        action_p0: tuple[int, int] | list[int] | np.ndarray,
        action_p1: tuple[int, int] | list[int] | np.ndarray | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        state_before = self.env.state.clone()

        # Resolve Player 1 action if not provided externally or if heuristic
        if action_p1 is None or (isinstance(action_p1, (tuple, list, np.ndarray)) and len(action_p1) >= 2 and action_p1[0] == -1 and action_p1[1] == -1):
            if self.opponent_type == "heuristic":
                action_p1 = self.heuristic.act(self.env.state, 1)
            else:
                action_p1 = (int(MoveAction.NOOP), int(BombAction.NOOP))
        elif self.opponent_type == "heuristic" and isinstance(action_p1, (tuple, list, np.ndarray)) and action_p1[0] == 0 and action_p1[1] == 0:
            # Fallback for dummy (0, 0) when opponent is heuristic
            action_p1 = self.heuristic.act(self.env.state, 1)

        # Convert ego actions to world coordinate actions
        w_act_0 = flip_action_to_world(action_p0, 0)
        w_act_1 = flip_action_to_world(action_p1, 1)

        # Step physics
        raw_obs, env_rews, terminated, truncated, env_info = self.env.step([w_act_0, w_act_1])

        # Step modular rewards & telemetry
        step_rews, step_tel = self.reward_tracker.step(
            state_before, [w_act_0, w_act_1], self.env.state, terminated, truncated
        )

        p0_reward = float(step_rews["player_0"])
        p1_reward = float(step_rews["player_1"])

        # Track match events
        self._match_stats.ticks = self.env.state.tick

        # Track newly destroyed bricks (vectorized fast check)
        new_bricks = int(np.count_nonzero((state_before.grid == TileType.BRICK) & (self.env.state.grid == TileType.EMPTY)))
        if new_bricks > 0:
            self._match_stats.bricks_destroyed += new_bricks
            # Debuffs can only spawn when bricks are destroyed
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if state_before.powerups[r, c] == PowerUpType.NONE and self.env.state.powerups[r, c] in DEBUFF_TYPES:
                        self._match_stats.debuffs_spawned += 1

        # Track powerups collected
        for pid, player in enumerate(self.env.state.players):
            tr, tc = player.tile
            if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                consumed = state_before.powerups[tr, tc]
                if consumed != PowerUpType.NONE and self.env.state.powerups[tr, tc] == PowerUpType.NONE:
                    if consumed in BUFF_TYPES:
                        if pid == 0:
                            self._match_stats.p0_buffs_collected += 1
                        else:
                            self._match_stats.p1_buffs_collected += 1
                    elif consumed in DEBUFF_TYPES:
                        if pid == 0:
                            self._match_stats.p0_debuffs_collected += 1
                        else:
                            self._match_stats.p1_debuffs_collected += 1

        # Track kicks and chain detonations only when bombs exist
        if state_before.bombs or self.env.state.bombs:
            old_bombs = {ob.id: ob for ob in state_before.bombs}
            for b in self.env.state.bombs:
                old_b = old_bombs.get(b.id)
                if old_b is not None:
                    if (old_b.velocity_r == 0.0 and old_b.velocity_c == 0.0) and (b.velocity_r != 0.0 or b.velocity_c != 0.0):
                        self._match_stats.kicks_executed += 1

            detonated_ids = old_bombs.keys() - {b.id for b in self.env.state.bombs}
            for bid in detonated_ids:
                old_b = old_bombs[bid]
                if old_b.timer > 1:
                    self._match_stats.chain_detonations += 1

        done = terminated or truncated
        match_stats_record = None

        if done:
            # Determine winner and suicide
            winner = env_info.get("winner")
            draw = env_info.get("draw", False)
            self._match_stats.winner = winner
            self._match_stats.draw = draw
            self._match_stats.debuffs_avoided = max(
                0, self._match_stats.debuffs_spawned - self._match_stats.p0_debuffs_collected
            )

            p0 = self.env.state.players[0]
            p1 = self.env.state.players[1]

            if not p0.alive:
                tr, tc = p0.tile
                mask = int(self.env.state.flame_owner_mask[tr, tc]) if (0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE) else 0
                if mask == 0 and 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                    mask = int(state_before.flame_owner_mask[tr, tc])
                if bool(mask & 1) and not bool(mask & 2):
                    self._match_stats.p0_suicide = True

            if not p1.alive:
                tr, tc = p1.tile
                mask = int(self.env.state.flame_owner_mask[tr, tc]) if (0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE) else 0
                if mask == 0 and 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                    mask = int(state_before.flame_owner_mask[tr, tc])
                if bool(mask & 2) and not bool(mask & 1):
                    self._match_stats.p1_suicide = True

            match_stats_record = MatchStats(**self._match_stats.__dict__)

            # Auto-reset environment for the next match
            self.env.reset()
            self.reward_tracker.reset(self.env.state)
            self.sample_opponent(self.past_checkpoint_paths)
            self._match_stats = MatchStats()

        obs_0, obs_1 = self.obs_builder.build_both(self.env.state, copy=False)
        mask_0 = self.action_filter.get_action_mask(self.env.state, 0, ego_perspective=True)
        mask_1 = self.action_filter.get_action_mask(self.env.state, 1, ego_perspective=True)

        info = {
            "opponent_type": self.opponent_type,
            "opponent_checkpoint": self.opponent_checkpoint_path,
            "match_stats": match_stats_record,
            "p0_reward": p0_reward,
            "p1_reward": p1_reward,
            "tick": self.env.state.tick,
        }

        return obs_0, p0_reward, terminated, truncated, info, mask_0, obs_1, mask_1


class SyncVectorEnv:
    """Synchronous in-process vectorized environment running N parallel matches."""

    def __init__(
        self,
        num_envs: int = 4,
        seed: int | None = 42,
        self_damage_scale: float = 0.25,
        safety_filter_prob: float = 1.0,
    ) -> None:
        self.num_envs = num_envs
        self.envs = [
            BombermanMatchEnv(
                seed=None if seed is None else seed + i,
                self_damage_scale=self_damage_scale,
                safety_filter_prob=safety_filter_prob,
            )
            for i in range(num_envs)
        ]

    def set_curriculum(self, self_damage_scale: float, safety_filter_prob: float) -> None:
        for env in self.envs:
            env.set_curriculum(self_damage_scale, safety_filter_prob)

    def set_opponent_pool(self, past_checkpoint_paths: list[str] | None) -> None:
        for env in self.envs:
            env.set_opponent_pool(past_checkpoint_paths)

    def reset(
        self,
        seed: int | None = None,
        past_checkpoint_paths: list[str] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        list[dict[str, Any]],
    ]:
        obs_p0_spatial, obs_p0_scalars = [], []
        masks_p0_move, masks_p0_bomb = [], []
        obs_p1_spatial, obs_p1_scalars = [], []
        masks_p1_move, masks_p1_bomb = [], []
        infos = []

        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            o0, m0, o1, m1, info = env.reset(seed=s, past_checkpoint_paths=past_checkpoint_paths)
            obs_p0_spatial.append(o0["spatial"])
            obs_p0_scalars.append(o0["scalars"])
            masks_p0_move.append(m0["move_mask"])
            masks_p0_bomb.append(m0["bomb_mask"])

            obs_p1_spatial.append(o1["spatial"])
            obs_p1_scalars.append(o1["scalars"])
            masks_p1_move.append(m1["move_mask"])
            masks_p1_bomb.append(m1["bomb_mask"])
            infos.append(info)

        obs_0 = {
            "spatial": np.stack(obs_p0_spatial, axis=0),
            "scalars": np.stack(obs_p0_scalars, axis=0),
        }
        mask_0 = {
            "move_mask": np.stack(masks_p0_move, axis=0),
            "bomb_mask": np.stack(masks_p0_bomb, axis=0),
        }
        obs_1 = {
            "spatial": np.stack(obs_p1_spatial, axis=0),
            "scalars": np.stack(obs_p1_scalars, axis=0),
        }
        mask_1 = {
            "move_mask": np.stack(masks_p1_move, axis=0),
            "bomb_mask": np.stack(masks_p1_bomb, axis=0),
        }
        return obs_0, mask_0, obs_1, mask_1, infos

    def step(
        self,
        actions_p0: np.ndarray | list[Any],
        actions_p1: np.ndarray | list[Any] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        obs_p0_spatial, obs_p0_scalars = [], []
        masks_p0_move, masks_p0_bomb = [], []
        obs_p1_spatial, obs_p1_scalars = [], []
        masks_p1_move, masks_p1_bomb = [], []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, env in enumerate(self.envs):
            act_0 = actions_p0[i]
            act_1 = actions_p1[i] if actions_p1 is not None else None
            o0, r0, term, trunc, info, m0, o1, m1 = env.step(act_0, act_1)

            obs_p0_spatial.append(o0["spatial"])
            obs_p0_scalars.append(o0["scalars"])
            masks_p0_move.append(m0["move_mask"])
            masks_p0_bomb.append(m0["bomb_mask"])

            obs_p1_spatial.append(o1["spatial"])
            obs_p1_scalars.append(o1["scalars"])
            masks_p1_move.append(m1["move_mask"])
            masks_p1_bomb.append(m1["bomb_mask"])

            rewards.append(r0)
            terminateds.append(term)
            truncateds.append(trunc)
            infos.append(info)

        obs_0 = {
            "spatial": np.stack(obs_p0_spatial, axis=0),
            "scalars": np.stack(obs_p0_scalars, axis=0),
        }
        mask_0 = {
            "move_mask": np.stack(masks_p0_move, axis=0),
            "bomb_mask": np.stack(masks_p0_bomb, axis=0),
        }
        obs_1 = {
            "spatial": np.stack(obs_p1_spatial, axis=0),
            "scalars": np.stack(obs_p1_scalars, axis=0),
        }
        mask_1 = {
            "move_mask": np.stack(masks_p1_move, axis=0),
            "bomb_mask": np.stack(masks_p1_bomb, axis=0),
        }
        return (
            obs_0,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            infos,
            mask_0,
            obs_1,
            mask_1,
        )

    def close(self) -> None:
        pass


def _worker_fn(
    pipe: mp.connection.Connection,
    seed: int | None,
    self_damage_scale: float,
    safety_filter_prob: float,
) -> None:
    env = BombermanMatchEnv(
        seed=seed,
        self_damage_scale=self_damage_scale,
        safety_filter_prob=safety_filter_prob,
    )
    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break

        if cmd == "reset":
            past_checkpoints = data.get("past_checkpoints", None)
            o0, m0, o1, m1, info = env.reset(seed=data.get("seed", None), past_checkpoint_paths=past_checkpoints)
            pipe.send((o0, m0, o1, m1, info))
        elif cmd == "step":
            act_0, act_1 = data
            result = env.step(act_0, act_1)
            pipe.send(result)
        elif cmd == "set_curriculum":
            scale, prob = data
            env.set_curriculum(scale, prob)
            pipe.send(True)
        elif cmd == "set_opponent_pool":
            env.set_opponent_pool(data)
            pipe.send(True)
        elif cmd == "close":
            break
    pipe.close()


class AsyncVectorEnv:
    """Asynchronous multi-process vectorized environment running N parallel matches."""

    def __init__(
        self,
        num_envs: int = 4,
        seed: int | None = 42,
        self_damage_scale: float = 0.25,
        safety_filter_prob: float = 1.0,
    ) -> None:
        self.num_envs = num_envs
        self.closed = False
        self.pipes: list[mp.connection.Connection] = []
        self.workers: list[mp.Process] = []

        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")

        for i in range(num_envs):
            parent_pipe, child_pipe = mp.Pipe()
            w_seed = None if seed is None else seed + i
            proc = ctx.Process(
                target=_worker_fn,
                args=(child_pipe, w_seed, self_damage_scale, safety_filter_prob),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.workers.append(proc)

    def set_curriculum(self, self_damage_scale: float, safety_filter_prob: float) -> None:
        for pipe in self.pipes:
            pipe.send(("set_curriculum", (self_damage_scale, safety_filter_prob)))
        for pipe in self.pipes:
            pipe.recv()

    def set_opponent_pool(self, past_checkpoint_paths: list[str] | None) -> None:
        for pipe in self.pipes:
            pipe.send(("set_opponent_pool", past_checkpoint_paths))
        for pipe in self.pipes:
            pipe.recv()

    def reset(
        self,
        seed: int | None = None,
        past_checkpoint_paths: list[str] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        list[dict[str, Any]],
    ]:
        for i, pipe in enumerate(self.pipes):
            s = None if seed is None else seed + i
            pipe.send(("reset", {"seed": s, "past_checkpoints": past_checkpoint_paths}))

        results = [pipe.recv() for pipe in self.pipes]
        obs_p0_spatial = [r[0]["spatial"] for r in results]
        obs_p0_scalars = [r[0]["scalars"] for r in results]
        masks_p0_move = [r[1]["move_mask"] for r in results]
        masks_p0_bomb = [r[1]["bomb_mask"] for r in results]

        obs_p1_spatial = [r[2]["spatial"] for r in results]
        obs_p1_scalars = [r[2]["scalars"] for r in results]
        masks_p1_move = [r[3]["move_mask"] for r in results]
        masks_p1_bomb = [r[3]["bomb_mask"] for r in results]
        infos = [r[4] for r in results]

        obs_0 = {
            "spatial": np.stack(obs_p0_spatial, axis=0),
            "scalars": np.stack(obs_p0_scalars, axis=0),
        }
        mask_0 = {
            "move_mask": np.stack(masks_p0_move, axis=0),
            "bomb_mask": np.stack(masks_p0_bomb, axis=0),
        }
        obs_1 = {
            "spatial": np.stack(obs_p1_spatial, axis=0),
            "scalars": np.stack(obs_p1_scalars, axis=0),
        }
        mask_1 = {
            "move_mask": np.stack(masks_p1_move, axis=0),
            "bomb_mask": np.stack(masks_p1_bomb, axis=0),
        }
        return obs_0, mask_0, obs_1, mask_1, infos

    def step(
        self,
        actions_p0: np.ndarray | list[Any],
        actions_p1: np.ndarray | list[Any] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        for i, pipe in enumerate(self.pipes):
            act_0 = actions_p0[i]
            act_1 = actions_p1[i] if actions_p1 is not None else None
            pipe.send(("step", (act_0, act_1)))

        results = [pipe.recv() for pipe in self.pipes]

        obs_p0_spatial = [r[0]["spatial"] for r in results]
        obs_p0_scalars = [r[0]["scalars"] for r in results]
        rewards = [r[1] for r in results]
        terminateds = [r[2] for r in results]
        truncateds = [r[3] for r in results]
        infos = [r[4] for r in results]

        masks_p0_move = [r[5]["move_mask"] for r in results]
        masks_p0_bomb = [r[5]["bomb_mask"] for r in results]

        obs_p1_spatial = [r[6]["spatial"] for r in results]
        obs_p1_scalars = [r[6]["scalars"] for r in results]

        masks_p1_move = [r[7]["move_mask"] for r in results]
        masks_p1_bomb = [r[7]["bomb_mask"] for r in results]

        obs_0 = {
            "spatial": np.stack(obs_p0_spatial, axis=0),
            "scalars": np.stack(obs_p0_scalars, axis=0),
        }
        mask_0 = {
            "move_mask": np.stack(masks_p0_move, axis=0),
            "bomb_mask": np.stack(masks_p0_bomb, axis=0),
        }
        obs_1 = {
            "spatial": np.stack(obs_p1_spatial, axis=0),
            "scalars": np.stack(obs_p1_scalars, axis=0),
        }
        mask_1 = {
            "move_mask": np.stack(masks_p1_move, axis=0),
            "bomb_mask": np.stack(masks_p1_bomb, axis=0),
        }
        return (
            obs_0,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            infos,
            mask_0,
            obs_1,
            mask_1,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        for proc in self.workers:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
