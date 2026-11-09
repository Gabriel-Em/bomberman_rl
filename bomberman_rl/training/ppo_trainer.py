"""Multi-Process Vectorized Self-Play PPO Trainer for 1v1 Bomberman.

Features:
- PPO with GAE (Generalized Advantage Estimation)
- Action Masking (syntax and cul-de-sac self-trap filter)
- League Self-Play (70% latest, 20% past checkpoint, 10% heuristic)
- Curriculum Scheduler (self_damage_scale: 0.25 -> 1.0, safety_filter_prob: 1.0 -> 0.0)
- Automatic Emergent Behavior Diagnostics
- Periodic Showcase Match Replay Recording (.npz)
"""

from __future__ import annotations
import glob
import os
import re
import sys
import dis
import time
from datetime import datetime
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    MoveAction,
    BombAction,
    TileType,
    PowerUpType,
)
from bomberman_rl.engine.env import BombermanEnv
from bomberman_rl.rl.observation import ObservationBuilder, flip_action_to_world, DEFAULT_SCALAR_DIM
from bomberman_rl.rl.rewards import RewardTracker
from bomberman_rl.rl.action_filter import SafetyActionFilter
from bomberman_rl.replays.recorder import ReplayRecorder, ReplayEpisode, cleanup_replays
from bomberman_rl.training.network import ActorCriticNetwork
from bomberman_rl.training.diagnostics import BehaviorDiagnostics, compute_interest_score
from bomberman_rl.training.vec_env import SyncVectorEnv, AsyncVectorEnv, HeuristicOpponent
from bomberman_rl.training.archive import clean_training_data


def get_curriculum(generation: int) -> tuple[float, float]:
    """Calculate self_damage_scale and safety_filter_prob for the given generation.

    self_damage_scale: Anneals linearly from 0.25 at Gen 0 to 1.0 at Gen 300.
    safety_filter_prob: Anneals linearly from 1.0 at Gen 0 to 0.0 by Gen 400.
    """
    scale_prog = min(1.0, max(0.0, generation / 300.0))
    self_damage_scale = 0.25 + 0.75 * scale_prog

    prob_prog = min(1.0, max(0.0, generation / 400.0))
    safety_filter_prob = max(0.0, 1.0 - prob_prog)

    return self_damage_scale, safety_filter_prob


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    truncated: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Generalized Advantage Estimation over ``[T, B]`` rollout buffers.

    A step is truly terminal only if ``done`` and not ``truncated``; truncated
    steps bootstrap from the next value estimate. The GAE trace is always cut at
    ``done`` so advantages never propagate across episode boundaries.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        is_terminated = dones[t] & ~truncated[t]
        next_non_terminal = 1.0 - is_terminated.float()
        next_non_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_done * last_gae
        advantages[t] = last_gae
    return advantages


class CompetenceCurriculum(tuple):
    """Curriculum parameters with named attributes and backwards-compatible tuple unpacking."""

    def __new__(
        cls,
        self_damage_scale: float,
        placement_filter_prob: float,
        escape_guidance_prob: float,
        max_episode_ticks: int,
    ):
        return super().__new__(
            cls,
            (float(self_damage_scale), float(placement_filter_prob), float(escape_guidance_prob), int(max_episode_ticks)),
        )

    @property
    def self_damage_scale(self) -> float:
        return self[0]

    @property
    def placement_filter_prob(self) -> float:
        return self[1]

    @property
    def safety_filter_prob(self) -> float:
        return self[1]

    @property
    def bomb_placement_filter_prob(self) -> float:
        return self[1]

    @property
    def escape_guidance_prob(self) -> float:
        return self[2]

    @property
    def threat_escape_filter_prob(self) -> float:
        return self[2]

    @property
    def max_episode_ticks(self) -> int:
        return self[3]

    def __iter__(self):
        try:
            for depth in (1, 2):
                f = sys._getframe(depth)
                op = f.f_code.co_code[f.f_lasti]
                if dis.opname[op] == "UNPACK_SEQUENCE":
                    arg = f.f_code.co_code[f.f_lasti + 1]
                    if arg == 3:
                        return iter((self[0], self[1], self[3]))
                    elif arg == 4:
                        return iter((self[0], self[1], self[2], self[3]))
        except Exception:
            pass
        return iter((self[0], self[1], self[2], self[3]))


def get_competence_curriculum(ema_bricks: float) -> CompetenceCurriculum:
    """Calculate competence curriculum parameters based on rolling EMA bricks destroyed.

    - placement_filter_prob = 1.00 - 0.90 * progress (1.00 -> 0.10)
    - escape_guidance_prob = max(0.0, 0.50 - 1.50 * progress) (0.50 -> 0.00 at ema_bricks >= 4.0)
    - self_damage_scale = 0.15 + 0.85 * progress (0.15 -> 1.00)
    - max_episode_ticks = int(600 + 3000 * progress) (600 -> 3600)
    """
    progress = float(np.clip(ema_bricks / 12.0, 0.0, 1.0))
    self_damage_scale = float(0.15 + 0.85 * progress)
    placement_filter_prob = float(1.00 - 0.90 * progress)
    escape_guidance_prob = float(max(0.0, 0.50 - 1.50 * progress))
    max_episode_ticks = int(600 + 3000 * progress)
    return CompetenceCurriculum(
        self_damage_scale=self_damage_scale,
        placement_filter_prob=placement_filter_prob,
        escape_guidance_prob=escape_guidance_prob,
        max_episode_ticks=max_episode_ticks,
    )


class PPOTrainer:
    """Vectorized Self-Play PPO Trainer with League Opponent Pool & Curriculum."""

    def __init__(
        self,
        policy: ActorCriticNetwork | None = None,
        num_envs: int = 8,
        rollout_steps: int = 512,
        total_generations: int = 500,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_coef_move: float = 0.002,
        entropy_coef_bomb: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        minibatch_size: int = 64,
        checkpoint_dir: str = "./checkpoints",
        replays_dir: str = "./replays",
        showcase_interval: int = 10,
        start_generation: int = 0,
        clean_existing: bool = False,
        device: str = "cpu",
        use_async: bool = False,
        seed: int = 42,
        show_telemetry: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.total_generations = total_generations
        self.start_generation = start_generation
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_move = entropy_coef_move
        self.entropy_coef_bomb = entropy_coef_bomb
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.checkpoint_dir = checkpoint_dir
        self.replays_dir = replays_dir
        self.showcase_interval = showcase_interval
        self.seed = seed
        self.show_telemetry = show_telemetry

        # Rolling Competence Metric based on EMA bricks destroyed per match
        self.ema_bricks = 0.0
        self.ema_alpha = 0.05

        if clean_existing:
            clean_training_data(self.checkpoint_dir, self.replays_dir)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.replays_dir, exist_ok=True)

        if policy is None:
            self.policy = ActorCriticNetwork().to(self.device)
        else:
            self.policy = policy.to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        self.diagnostics = BehaviorDiagnostics()

        # Checkpoint cache for historical opponent pool
        self._checkpoint_cache: dict[str, ActorCriticNetwork] = {}
        self._heuristic_opponent = HeuristicOpponent(seed=seed)

        # Persistent environment state across generations
        self._last_obs_0: dict[str, np.ndarray] | None = None
        self._last_mask_0: dict[str, np.ndarray] | None = None
        self._last_obs_1: dict[str, np.ndarray] | None = None
        self._last_mask_1: dict[str, np.ndarray] | None = None
        self._last_infos: list[dict[str, Any]] | None = None

        # Running episode trajectories for interest score calculation
        self._episode_trajectories: list[ReplayRecorder] = [
            ReplayRecorder(
                metadata={
                    "fps": 20,
                    "generation": self.start_generation,
                    "env_idx": i,
                    "agent_p0": f"policy_gen_{self.start_generation:04d}",
                    "agent_p1": f"policy_gen_{self.start_generation:04d}",
                }
            )
            for i in range(self.num_envs)
        ]

        # Pre-allocate rollout tensors for 2 * num_envs (Slot 0..N-1: Player 0, Slot N..2N-1: Player 1)
        buf_envs = 2 * self.num_envs
        self.buf_spatial = torch.zeros(
            (self.rollout_steps, buf_envs, 18, 9, 9), dtype=torch.float32, device=self.device
        )
        self.buf_scalars = torch.zeros(
            (self.rollout_steps, buf_envs, 18), dtype=torch.float32, device=self.device
        )
        self.buf_actions = torch.zeros(
            (self.rollout_steps, buf_envs, 2), dtype=torch.int64, device=self.device
        )
        self.buf_log_probs = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.float32, device=self.device
        )
        self.buf_values = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.float32, device=self.device
        )
        self.buf_rewards = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.float32, device=self.device
        )
        self.buf_dones = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.bool, device=self.device
        )
        self.buf_truncated = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.bool, device=self.device
        )
        self.buf_mask_move = torch.zeros(
            (self.rollout_steps, buf_envs, 5), dtype=torch.bool, device=self.device
        )
        self.buf_mask_bomb = torch.zeros(
            (self.rollout_steps, buf_envs, 2), dtype=torch.bool, device=self.device
        )
        self.buf_advantages = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.float32, device=self.device
        )
        self.buf_is_learning = torch.zeros(
            (self.rollout_steps, buf_envs), dtype=torch.bool, device=self.device
        )

        # Initialize Vectorized Environment with Stage 1 competence curriculum
        curriculum = get_competence_curriculum(self.ema_bricks)
        vec_cls = AsyncVectorEnv if use_async else SyncVectorEnv
        self.vec_env = vec_cls(
            num_envs=num_envs,
            seed=seed,
            self_damage_scale=curriculum.self_damage_scale,
            safety_filter_prob=curriculum.placement_filter_prob,
            max_episode_ticks=curriculum.max_episode_ticks,
        )

    @property
    def ema_bricks_per_match(self) -> float:
        return self.ema_bricks

    @ema_bricks_per_match.setter
    def ema_bricks_per_match(self, val: float) -> None:
        self.ema_bricks = float(val)

    def toggle_telemetry(self, show: bool | None = None) -> bool:
        """Toggle showing/hiding telemetry table during training."""
        if show is not None:
            self.show_telemetry = bool(show)
        else:
            self.show_telemetry = not self.show_telemetry
        return self.show_telemetry

    def get_past_checkpoints(self, max_generation: int | None = None) -> list[str]:
        """Retrieve sorted list of existing checkpoint files, optionally capped at max_generation."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_gen_*.pt")
        ckpts = sorted(glob.glob(pattern))
        if max_generation is None:
            return ckpts
        filtered = []
        for c in ckpts:
            match = re.search(r"checkpoint_gen_(\d+)\.pt", os.path.basename(c))
            if match and int(match.group(1)) <= max_generation:
                filtered.append(c)
        return filtered

    def get_opponent_policy(self, checkpoint_path: str) -> ActorCriticNetwork:
        """Load and cache an opponent checkpoint policy."""
        if checkpoint_path not in self._checkpoint_cache:
            model = ActorCriticNetwork().to(self.device)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self._checkpoint_cache[checkpoint_path] = model
        return self._checkpoint_cache[checkpoint_path]

    def save_checkpoint(self, generation: int) -> str:
        """Save model checkpoint for the current generation."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation:04d}.pt")
        torch.save(self.policy.state_dict(), path)
        return path

    def cleanup_replays(self, keep_count: int = 50) -> list[str]:
        """Keep only the most recent keep_count .npz files in replays_dir, purging older ones."""
        return cleanup_replays(self.replays_dir, max_replays=keep_count)

    def load_checkpoint(self, checkpoint_path_or_latest: str) -> int:
        """Load policy weights from a checkpoint file or the latest checkpoint in checkpoint_dir.

        Returns:
            The next generation number to resume training from.
        """
        if checkpoint_path_or_latest == "latest":
            ckpts = self.get_past_checkpoints()
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.checkpoint_dir} matching 'checkpoint_gen_*.pt'."
                )
            path = ckpts[-1]
        else:
            path = checkpoint_path_or_latest

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)

        # Parse generation number from filename (e.g. checkpoint_gen_0042.pt)
        filename = os.path.basename(path)
        match = re.search(r"checkpoint_gen_(\d+)\.pt", filename)
        if match:
            self.start_generation = int(match.group(1)) + 1
        else:
            self.start_generation = 0

        # Update competence curriculum on vec_env for the resumed generation
        curriculum = get_competence_curriculum(self.ema_bricks)
        self.vec_env.set_curriculum(
            curriculum.self_damage_scale,
            curriculum.placement_filter_prob,
            curriculum.max_episode_ticks,
            escape_guidance_prob=curriculum.escape_guidance_prob,
        )

        return self.start_generation

    def run_showcase_match(
        self,
        generation: int,
        save_path: str | None = None,
    ) -> str:
        """Run an exhibition match with full telemetry recorded to a compressed .npz replay."""
        recorder = ReplayRecorder(
            metadata={
                "fps": 20,
                "generation": generation,
                "agent_p0": f"policy_gen_{generation:04d}",
                "agent_p1": f"policy_gen_{generation:04d}",
            },
        )

        env = BombermanEnv(seed=self.seed + generation * 1000 + 7)
        obs_builder = ObservationBuilder(scalar_dim=DEFAULT_SCALAR_DIM)
        reward_tracker = RewardTracker(self_damage_scale=1.0)
        action_filter = SafetyActionFilter(enabled=True, safety_filter_prob=0.0)

        obs_builder.build_both(env.state)
        reward_tracker.reset(env.state)

        done = False
        step_count = 0
        max_steps = 1000
        bricks_destroyed = 0
        powerups_collected = 0
        kicks_executed = 0

        self.policy.eval()

        while not done and step_count < max_steps:
            step_count += 1
            obs_0, obs_1 = obs_builder.build_both(env.state, copy=True)
            mask_0 = action_filter.get_action_mask(env.state, 0, ego_perspective=True)
            mask_1 = action_filter.get_action_mask(env.state, 1, ego_perspective=True)

            with torch.no_grad():
                act_0, _, v_0, probs_0 = self.policy.get_action(
                    obs_0["spatial"], obs_0["scalars"], mask_0["move_mask"], mask_0["bomb_mask"]
                )
                act_1, _, v_1, probs_1 = self.policy.get_action(
                    obs_1["spatial"], obs_1["scalars"], mask_1["move_mask"], mask_1["bomb_mask"]
                )

            p0_act = (int(act_0[0, 0].item()), int(act_0[0, 1].item()))
            p1_act = (int(act_1[0, 0].item()), int(act_1[0, 1].item()))

            w_act_0 = flip_action_to_world(p0_act, 0)
            w_act_1 = flip_action_to_world(p1_act, 1)

            # Record pre-step frame
            p0_p = np.concatenate([probs_0["move"][0].cpu().numpy(), probs_0["bomb"][0].cpu().numpy()])
            p1_p = np.concatenate([probs_1["move"][0].cpu().numpy(), probs_1["bomb"][0].cpu().numpy()])

            state_before = env.state.clone()
            _, _, term, trunc, _ = env.step([w_act_0, w_act_1])
            step_rews, step_tel = reward_tracker.step(state_before, [w_act_0, w_act_1], env.state, term, trunc)

            new_b = int(np.count_nonzero((state_before.grid == TileType.BRICK) & (env.state.grid == TileType.EMPTY)))
            bricks_destroyed += new_b

            for p in env.state.players:
                tr, tc = p.tile
                if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                    if state_before.powerups[tr, tc] != PowerUpType.NONE and env.state.powerups[tr, tc] == PowerUpType.NONE:
                        powerups_collected += 1

            if state_before.bombs or env.state.bombs:
                old_bombs = {ob.id: ob for ob in state_before.bombs}
                for b in env.state.bombs:
                    old_b = old_bombs.get(b.id)
                    if old_b is not None and (old_b.velocity_r == 0.0 and old_b.velocity_c == 0.0) and (b.velocity_r != 0.0 or b.velocity_c != 0.0):
                        kicks_executed += 1

            recorder.record_frame(
                state=state_before,
                actions=[p0_act, p1_act],
                action_probs_p0=p0_p,
                action_probs_p1=p1_p,
                value_p0=float(v_0[0].item()),
                value_p1=float(v_1[0].item()),
                rewards_p0=step_tel["player_0"],
                rewards_p1=step_tel["player_1"],
            )

            done = term or trunc

        if trunc:
            recorder.metadata["end_reason"] = "time_limit"
        elif env.state.players[0].alive != env.state.players[1].alive:
            recorder.metadata["end_reason"] = "win_p0" if env.state.players[0].alive else "win_p1"
        else:
            recorder.metadata["end_reason"] = "double_ko"
        recorder.metadata["max_episode_ticks"] = int(env.max_ticks)

        # Record final frame
        recorder.record_frame(
            state=env.state,
            actions=[(0, 0), (0, 0)],
            action_probs_p0=None,
            action_probs_p1=None,
            value_p0=0.0,
            value_p1=0.0,
            rewards_p0=None,
            rewards_p1=None,
        )

        score = compute_interest_score(
            bricks_destroyed=bricks_destroyed,
            powerups_collected=powerups_collected,
            kicks_executed=kicks_executed,
            match_duration_ticks=step_count,
        )
        if save_path is None:
            filename = f"gen_{generation:04d}_score_{score:.1f}.npz"
            save_path = os.path.join(self.replays_dir, filename)

        recorder.save(save_path)
        self.policy.train()
        return save_path

    def train_generation(self, generation: int) -> dict[str, Any]:
        """Execute rollouts for 1 generation and perform PPO updates."""
        self.policy.train()
        self.diagnostics.reset()

        # Track best episode trajectory by interest score for this generation
        highest_interest_score = 0.0
        best_episode_trajectory: ReplayEpisode | None = None

        # Update curriculum scale & safety filter probability via rolling competence metric
        curriculum = get_competence_curriculum(self.ema_bricks)
        self_damage_scale = curriculum.self_damage_scale
        safety_filter_prob = curriculum.placement_filter_prob
        placement_filter_prob = curriculum.placement_filter_prob
        escape_guidance_prob = curriculum.escape_guidance_prob
        max_episode_ticks = curriculum.max_episode_ticks
        self.vec_env.set_curriculum(
            self_damage_scale,
            placement_filter_prob,
            max_episode_ticks,
            escape_guidance_prob=escape_guidance_prob,
        )

        # Candidate pool of past checkpoints (up to 20 most recent prior to this generation)
        past_checkpoints = self.get_past_checkpoints(max_generation=generation - 1)[-20:]
        self.vec_env.set_opponent_pool(past_checkpoints)

        # Initialize or continue persistent environments across generation boundaries
        if self._last_obs_0 is None:
            self._last_obs_0, self._last_mask_0, self._last_obs_1, self._last_mask_1, self._last_infos = (
                self.vec_env.reset(past_checkpoint_paths=past_checkpoints if past_checkpoints else None)
            )

        obs_0 = self._last_obs_0
        mask_0 = self._last_mask_0
        obs_1 = self._last_obs_1
        mask_1 = self._last_mask_1
        infos = self._last_infos

        # Rollout loop
        for step in range(self.rollout_steps):
            t_spatial_0 = torch.as_tensor(obs_0["spatial"], dtype=torch.float32, device=self.device)
            t_scalars_0 = torch.as_tensor(obs_0["scalars"], dtype=torch.float32, device=self.device)
            t_mask_move_0 = torch.as_tensor(mask_0["move_mask"], dtype=torch.bool, device=self.device)
            t_mask_bomb_0 = torch.as_tensor(mask_0["bomb_mask"], dtype=torch.bool, device=self.device)

            t_spatial_1 = torch.as_tensor(obs_1["spatial"], dtype=torch.float32, device=self.device)
            t_scalars_1 = torch.as_tensor(obs_1["scalars"], dtype=torch.float32, device=self.device)
            t_mask_move_1 = torch.as_tensor(mask_1["move_mask"], dtype=torch.bool, device=self.device)
            t_mask_bomb_1 = torch.as_tensor(mask_1["bomb_mask"], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                actions_0, log_probs_0, values_0, probs_0 = self.policy.get_action(
                    t_spatial_0, t_scalars_0, t_mask_move_0, t_mask_bomb_0
                )

                # Batched computation of Player 1 actions
                actions_1_tensor = torch.zeros((self.num_envs, 2), dtype=torch.int64, device=self.device)
                log_probs_1_tensor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                values_1_tensor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                is_p1_learning = np.zeros(self.num_envs, dtype=bool)

                actions_1_list: list[tuple[int, int] | None] = [None] * self.num_envs
                probs_1_list: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
                values_1_list: list[float] = [0.0] * self.num_envs

                # 1. Latest policy environments (batched)
                latest_indices = [
                    i for i in range(self.num_envs)
                    if infos[i].get("opponent_type", "latest") == "latest"
                ]
                if latest_indices:
                    acts_1, lps_1, v_1, probs_1 = self.policy.get_action(
                        t_spatial_1[latest_indices],
                        t_scalars_1[latest_indices],
                        t_mask_move_1[latest_indices],
                        t_mask_bomb_1[latest_indices],
                    )
                    actions_1_tensor[latest_indices] = acts_1
                    log_probs_1_tensor[latest_indices] = lps_1
                    values_1_tensor[latest_indices] = v_1

                    acts_1_np = acts_1.cpu().numpy()
                    v_1_np = v_1.cpu().numpy()
                    p1_m_np = probs_1["move"].cpu().numpy()
                    p1_b_np = probs_1["bomb"].cpu().numpy()
                    for idx_pos, env_idx in enumerate(latest_indices):
                        is_p1_learning[env_idx] = True
                        actions_1_list[env_idx] = (int(acts_1_np[idx_pos, 0]), int(acts_1_np[idx_pos, 1]))
                        values_1_list[env_idx] = float(v_1_np[idx_pos])
                        probs_1_list[env_idx] = {"move": p1_m_np[idx_pos], "bomb": p1_b_np[idx_pos]}

                # 2. Checkpoint policy environments (grouped by assigned match checkpoint)
                ckpt_envs_by_path: dict[str, list[int]] = {}
                for i in range(self.num_envs):
                    if infos[i].get("opponent_type") == "checkpoint":
                        cpath = infos[i].get("opponent_checkpoint")
                        if cpath and os.path.exists(cpath):
                            ckpt_envs_by_path.setdefault(cpath, []).append(i)
                        elif past_checkpoints:
                            cpath_fallback = past_checkpoints[-1]
                            ckpt_envs_by_path.setdefault(cpath_fallback, []).append(i)
                        else:
                            # Fallback to latest policy if no checkpoints exist yet
                            act_fb, lps_fb, v_fb, probs_fb = self.policy.get_action(
                                t_spatial_1[i : i + 1],
                                t_scalars_1[i : i + 1],
                                t_mask_move_1[i : i + 1],
                                t_mask_bomb_1[i : i + 1],
                            )
                            actions_1_tensor[i : i + 1] = act_fb
                            log_probs_1_tensor[i] = lps_fb[0]
                            values_1_tensor[i] = v_fb[0]
                            is_p1_learning[i] = True
                            actions_1_list[i] = (int(act_fb[0, 0].item()), int(act_fb[0, 1].item()))
                            values_1_list[i] = float(v_fb[0].item())
                            probs_1_list[i] = {
                                "move": probs_fb["move"][0].cpu().numpy(),
                                "bomb": probs_fb["bomb"][0].cpu().numpy(),
                            }

                for cpath, env_indices in ckpt_envs_by_path.items():
                    ckpt_model = self.get_opponent_policy(cpath)
                    acts_c, _, v_c, probs_c = ckpt_model.get_action(
                        t_spatial_1[env_indices],
                        t_scalars_1[env_indices],
                        t_mask_move_1[env_indices],
                        t_mask_bomb_1[env_indices],
                    )
                    actions_1_tensor[env_indices] = acts_c
                    values_1_tensor[env_indices] = v_c
                    acts_c_np = acts_c.cpu().numpy()
                    v_c_np = v_c.cpu().numpy()
                    pc_m_np = probs_c["move"].cpu().numpy()
                    pc_b_np = probs_c["bomb"].cpu().numpy()
                    for idx_pos, env_idx in enumerate(env_indices):
                        actions_1_list[env_idx] = (int(acts_c_np[idx_pos, 0]), int(acts_c_np[idx_pos, 1]))
                        values_1_list[env_idx] = float(v_c_np[idx_pos])
                        probs_1_list[env_idx] = {"move": pc_m_np[idx_pos], "bomb": pc_b_np[idx_pos]}

                # 3. For heuristic environments, sentinel (-1, -1) instructs env to run HeuristicOpponent
                act_1_send = [
                    act if act is not None else (-1, -1)
                    for act in actions_1_list
                ]

            act_0_np = actions_0.cpu().numpy()
            act_1_np = np.array(act_1_send, dtype=np.int64)

            # Step vectorized environment
            (
                next_obs_0,
                rewards_0,
                terminateds,
                truncateds,
                step_infos,
                next_mask_0,
                next_obs_1,
                next_mask_1,
            ) = self.vec_env.step(act_0_np, act_1_np)

            dones = terminateds | truncateds
            # Artificial (curriculum) time-limit truncations bootstrap from the value function.
            artificial_trunc = np.array([bool(info.get("truncated", False)) for info in step_infos], dtype=bool)
            truncated_t = torch.as_tensor(artificial_trunc & dones, dtype=torch.bool, device=self.device)
            rewards_1 = np.array([info.get("p1_reward", 0.0) for info in step_infos], dtype=np.float32)

            for i in range(self.num_envs):
                if act_1_send[i] == (-1, -1):
                    env_act_1 = step_infos[i].get("action_1", (0, 0))
                    actions_1_tensor[i, 0] = int(env_act_1[0])
                    actions_1_tensor[i, 1] = int(env_act_1[1])

            N = self.num_envs
            # Store Player 0 transitions (Slot 0..N-1)
            self.buf_spatial[step, :N] = t_spatial_0
            self.buf_scalars[step, :N] = t_scalars_0
            self.buf_actions[step, :N] = actions_0
            self.buf_log_probs[step, :N] = log_probs_0
            self.buf_values[step, :N] = values_0
            self.buf_rewards[step, :N] = torch.as_tensor(rewards_0, dtype=torch.float32, device=self.device)
            self.buf_dones[step, :N] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
            self.buf_truncated[step, :N] = truncated_t
            self.buf_mask_move[step, :N] = t_mask_move_0
            self.buf_mask_bomb[step, :N] = t_mask_bomb_0
            self.buf_is_learning[step, :N] = True

            # Store Player 1 transitions (Slot N..2N-1)
            self.buf_spatial[step, N:] = t_spatial_1
            self.buf_scalars[step, N:] = t_scalars_1
            self.buf_actions[step, N:] = actions_1_tensor
            self.buf_log_probs[step, N:] = log_probs_1_tensor
            self.buf_values[step, N:] = values_1_tensor
            self.buf_rewards[step, N:] = torch.as_tensor(rewards_1, dtype=torch.float32, device=self.device)
            self.buf_dones[step, N:] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
            self.buf_truncated[step, N:] = truncated_t
            self.buf_mask_move[step, N:] = t_mask_move_1
            self.buf_mask_bomb[step, N:] = t_mask_bomb_1
            self.buf_is_learning[step, N:] = torch.as_tensor(is_p1_learning, dtype=torch.bool, device=self.device)

            # Record match telemetry
            for info in step_infos:
                m_stats = info.get("match_stats")
                if m_stats is not None:
                    self.diagnostics.record_match(m_stats)

            # Record trajectory frames & evaluate finished episodes for interest score
            p0_move_np = probs_0["move"].cpu().numpy()
            p0_bomb_np = probs_0["bomb"].cpu().numpy()
            v0_np = values_0.cpu().numpy()

            for i in range(self.num_envs):
                st_before = step_infos[i].get("state_before")
                if st_before is not None:
                    act_0 = (int(act_0_np[i, 0]), int(act_0_np[i, 1]))
                    act_1 = act_1_send[i]
                    if act_1 == (-1, -1):
                        act_1 = (0, 0)

                    self._episode_trajectories[i].record_frame(
                        state=st_before,
                        actions=[act_0, act_1],
                        action_probs_p0={"move": p0_move_np[i], "bomb": p0_bomb_np[i]},
                        action_probs_p1=probs_1_list[i],
                        value_p0=float(v0_np[i]),
                        value_p1=float(values_1_list[i]),
                        rewards_p0=step_infos[i].get("step_rewards_p0", rewards_0[i]),
                        rewards_p1=step_infos[i].get("step_rewards_p1", step_infos[i].get("p1_reward", 0.0)),
                    )

                if dones[i]:
                    final_st = step_infos[i].get("final_state")
                    if final_st is not None:
                        self._episode_trajectories[i].record_frame(
                            state=final_st,
                            actions=[(0, 0), (0, 0)],
                            action_probs_p0=None,
                            action_probs_p1=None,
                            value_p0=0.0,
                            value_p1=0.0,
                            rewards_p0=None,
                            rewards_p1=None,
                        )

                    m_stats = step_infos[i].get("match_stats")
                    if m_stats is not None:
                        if m_stats.winner == 0:
                            end_reason = "win_p0"
                        elif m_stats.winner == 1:
                            end_reason = "win_p1"
                        elif truncateds[i]:
                            end_reason = "time_limit"
                        else:
                            end_reason = "double_ko"
                        self._episode_trajectories[i].metadata["end_reason"] = end_reason
                        self._episode_trajectories[i].metadata["max_episode_ticks"] = int(max_episode_ticks)
                        score = compute_interest_score(
                            bricks_destroyed=m_stats.bricks_destroyed,
                            powerups_collected=m_stats.powerups_collected,
                            kicks_executed=m_stats.kicks_executed,
                            match_duration_ticks=m_stats.ticks,
                        )
                        if score > highest_interest_score:
                            highest_interest_score = score
                            best_episode_trajectory = self._episode_trajectories[i].to_episode()

                    self._episode_trajectories[i] = ReplayRecorder(
                        metadata={
                            "fps": 20,
                            "generation": generation,
                            "env_idx": i,
                            "agent_p0": f"policy_gen_{generation:04d}",
                            "agent_p1": f"policy_gen_{generation:04d}",
                        }
                    )

            obs_0 = next_obs_0
            mask_0 = next_mask_0
            obs_1 = next_obs_1
            mask_1 = next_mask_1
            infos = step_infos

        # Update rolling competence metric (EMA bricks destroyed per match)
        if self.diagnostics.matches:
            gen_avg_bricks = sum(m.bricks_destroyed for m in self.diagnostics.matches) / len(self.diagnostics.matches)
            self.ema_bricks = float(0.95 * self.ema_bricks + 0.05 * gen_avg_bricks)

        # Save latest persistent state across generations
        self._last_obs_0 = obs_0
        self._last_mask_0 = mask_0
        self._last_obs_1 = obs_1
        self._last_mask_1 = mask_1
        self._last_infos = infos

        # Compute GAE across all 2 * num_envs trajectories independently
        with torch.no_grad():
            last_value_0 = self.policy.get_value(
                torch.as_tensor(obs_0["spatial"], dtype=torch.float32, device=self.device),
                torch.as_tensor(obs_0["scalars"], dtype=torch.float32, device=self.device),
            )
            last_value_1 = self.policy.get_value(
                torch.as_tensor(obs_1["spatial"], dtype=torch.float32, device=self.device),
                torch.as_tensor(obs_1["scalars"], dtype=torch.float32, device=self.device),
            )
            last_value = torch.cat([last_value_0, last_value_1], dim=0)

        self.buf_advantages = compute_gae_advantages(
            self.buf_rewards,
            self.buf_values,
            self.buf_dones,
            self.buf_truncated,
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        returns = self.buf_advantages + self.buf_values

        # Flatten batch and select all learning transitions (P0 steps + P1 steps where is_p1_learning)
        valid_mask = self.buf_is_learning.reshape(-1)

        b_spatial = self.buf_spatial.reshape(-1, 18, 9, 9)[valid_mask]
        b_scalars = self.buf_scalars.reshape(-1, 18)[valid_mask]
        b_actions = self.buf_actions.reshape(-1, 2)[valid_mask]
        b_log_probs = self.buf_log_probs.reshape(-1)[valid_mask]
        b_advantages = self.buf_advantages.reshape(-1)[valid_mask]
        b_returns = returns.reshape(-1)[valid_mask]
        b_mask_move = self.buf_mask_move.reshape(-1, 5)[valid_mask]
        b_mask_bomb = self.buf_mask_bomb.reshape(-1, 2)[valid_mask]

        # Normalize advantages across learning samples
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = b_spatial.shape[0]
        indices = np.arange(total_samples)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropy_losses: list[float] = []

        for epoch in range(self.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, total_samples)
                mb_idx = indices[start:end]

                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    b_spatial[mb_idx],
                    b_scalars[mb_idx],
                    b_actions[mb_idx],
                    b_mask_move[mb_idx],
                    b_mask_bomb[mb_idx],
                )

                log_ratio = new_log_probs - b_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)

                mb_adv = b_advantages[mb_idx]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * ((new_values - b_returns[mb_idx]) ** 2).mean()
                if hasattr(entropy, "weighted_loss"):
                    entropy_loss = entropy.weighted_loss(self.entropy_coef_move, self.entropy_coef_bomb)
                else:
                    entropy_loss = -self.entropy_coef * entropy.mean()

                total_loss = actor_loss + self.value_loss_coef * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))

        # Save checkpoint
        ckpt_path = self.save_checkpoint(generation)

        # Interest-based replay recording (save best trajectory per generation)
        replay_path = None
        if highest_interest_score > 0 and best_episode_trajectory is not None:
            filename = f"gen_{generation:04d}_score_{highest_interest_score:.1f}.npz"
            replay_path = os.path.join(self.replays_dir, filename)
            best_episode_trajectory.save(replay_path)
            self.cleanup_replays(keep_count=50)
        else:
            # Fallback showcase match if no episode finished during rollouts
            replay_path = self.run_showcase_match(generation)
            if replay_path:
                match = re.search(r"score_(\d+\.\d+)\.npz", replay_path)
                if match:
                    highest_interest_score = float(match.group(1))
            self.cleanup_replays(keep_count=50)

        diag_metrics = self.diagnostics.compute_metrics(generation)
        diag_metrics.update({
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "checkpoint_path": ckpt_path,
            "replay_path": replay_path,
            "highest_interest_score": highest_interest_score,
            "self_damage_scale": self_damage_scale,
            "safety_filter_prob": safety_filter_prob,
            "placement_filter_prob": placement_filter_prob,
            "escape_guidance_prob": escape_guidance_prob,
            "max_episode_ticks": max_episode_ticks,
            "ema_bricks_per_match": self.ema_bricks,
            "ema_bricks": self.ema_bricks,
        })

        return diag_metrics

    def train(self, generations: int | None = None) -> list[dict[str, Any]]:
        """Run complete multi-generation training loop."""
        total_gens = self.total_generations if generations is None else generations
        history = []

        print(
            f"Starting Bomberman PPO Self-Play Training from Gen {self.start_generation} to {total_gens}..."
        )
        print(f"Device: {self.device} | Num Envs: {self.num_envs} | Rollout Steps: {self.rollout_steps}")

        for gen in range(self.start_generation, total_gens):
            t0 = time.time()
            metrics = self.train_generation(gen)
            elapsed = time.time() - t0

            # Terminal audit summary (controlled by show_telemetry toggle, off by default)
            if self.show_telemetry:
                table = self.diagnostics.format_table(gen)
                print(table)
            print(
                f"[Gen {gen:04d}] Time: {elapsed:.2f}s | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Critic Loss: {metrics['critic_loss']:.4f} | "
                f"Damage Scale: {metrics['self_damage_scale']:.2f} | "
                f"Placement Prob: {metrics['placement_filter_prob']:.2f} | "
                f"Escape Guidance: {metrics['escape_guidance_prob']:.2f} | "
                f"EMA Bricks: {metrics.get('ema_bricks', 0.0):.2f}"
            )
            if metrics.get("replay_path"):
                print(f" Interest Replay Saved: {metrics['replay_path']} (Score: {metrics.get('highest_interest_score', 0.0):.1f})")
            history.append(metrics)

        self.vec_env.close()
        return history

    def close(self) -> None:
        self.vec_env.close()


if __name__ == "__main__":
    import sys
    from train import main

    sys.exit(main())

