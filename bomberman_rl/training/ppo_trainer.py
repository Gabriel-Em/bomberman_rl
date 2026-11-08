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
)
from bomberman_rl.engine.env import BombermanEnv
from bomberman_rl.rl.observation import ObservationBuilder, flip_action_to_world, DEFAULT_SCALAR_DIM
from bomberman_rl.rl.rewards import RewardTracker
from bomberman_rl.rl.action_filter import SafetyActionFilter
from bomberman_rl.replays.recorder import ReplayRecorder, ReplayEpisode
from bomberman_rl.training.network import ActorCriticNetwork
from bomberman_rl.training.diagnostics import BehaviorDiagnostics
from bomberman_rl.training.vec_env import SyncVectorEnv, AsyncVectorEnv, HeuristicOpponent


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


class PPOTrainer:
    """Vectorized Self-Play PPO Trainer with League Opponent Pool & Curriculum."""

    def __init__(
        self,
        policy: ActorCriticNetwork | None = None,
        num_envs: int = 8,
        rollout_steps: int = 128,
        total_generations: int = 500,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        minibatch_size: int = 64,
        checkpoint_dir: str = "./checkpoints",
        replays_dir: str = "./replays",
        showcase_interval: int = 10,
        start_generation: int = 0,
        device: str = "cpu",
        use_async: bool = False,
        seed: int = 42,
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
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.checkpoint_dir = checkpoint_dir
        self.replays_dir = replays_dir
        self.showcase_interval = showcase_interval
        self.seed = seed

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

        # Pre-allocate rollout tensors to eliminate per-generation GC overhead
        self.buf_spatial = torch.zeros(
            (self.rollout_steps, self.num_envs, 18, 9, 9), dtype=torch.float32, device=self.device
        )
        self.buf_scalars = torch.zeros(
            (self.rollout_steps, self.num_envs, 18), dtype=torch.float32, device=self.device
        )
        self.buf_actions = torch.zeros(
            (self.rollout_steps, self.num_envs, 2), dtype=torch.int64, device=self.device
        )
        self.buf_log_probs = torch.zeros(
            (self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.buf_values = torch.zeros(
            (self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.buf_rewards = torch.zeros(
            (self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.buf_dones = torch.zeros(
            (self.rollout_steps, self.num_envs), dtype=torch.bool, device=self.device
        )
        self.buf_mask_move = torch.zeros(
            (self.rollout_steps, self.num_envs, 5), dtype=torch.bool, device=self.device
        )
        self.buf_mask_bomb = torch.zeros(
            (self.rollout_steps, self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.buf_advantages = torch.zeros(
            (self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device
        )

        # Initialize Vectorized Environment
        init_scale, init_prob = get_curriculum(0)
        vec_cls = AsyncVectorEnv if use_async else SyncVectorEnv
        self.vec_env = vec_cls(
            num_envs=num_envs,
            seed=seed,
            self_damage_scale=init_scale,
            safety_filter_prob=init_prob,
        )

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

        # Update curriculum on vec_env for the resumed generation
        init_scale, init_prob = get_curriculum(self.start_generation)
        self.vec_env.set_curriculum(init_scale, init_prob)

        return self.start_generation

    def run_showcase_match(
        self,
        generation: int,
        save_path: str | None = None,
    ) -> str:
        """Run an exhibition match with full telemetry recorded to a compressed .npz replay."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.replays_dir, f"gen_{generation:04d}_{timestamp}.npz")

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

        recorder.save(save_path)
        self.policy.train()
        return save_path

    def train_generation(self, generation: int) -> dict[str, Any]:
        """Execute rollouts for 1 generation and perform PPO updates."""
        self.policy.train()
        self.diagnostics.reset()

        # Update curriculum scale & safety filter probability
        self_damage_scale, safety_filter_prob = get_curriculum(generation)
        self.vec_env.set_curriculum(self_damage_scale, safety_filter_prob)

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
            t_spatial = torch.as_tensor(obs_0["spatial"], dtype=torch.float32, device=self.device)
            t_scalars = torch.as_tensor(obs_0["scalars"], dtype=torch.float32, device=self.device)
            t_mask_move = torch.as_tensor(mask_0["move_mask"], dtype=torch.bool, device=self.device)
            t_mask_bomb = torch.as_tensor(mask_0["bomb_mask"], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                actions_0, log_probs_0, values_0, _ = self.policy.get_action(
                    t_spatial, t_scalars, t_mask_move, t_mask_bomb
                )

                # Batched computation of Player 1 actions
                actions_1_list: list[tuple[int, int] | None] = [None] * self.num_envs

                # 1. Latest policy environments (batched)
                latest_indices = [
                    i for i in range(self.num_envs)
                    if infos[i].get("opponent_type", "latest") == "latest"
                ]
                if latest_indices:
                    t_sp_1 = torch.as_tensor(obs_1["spatial"][latest_indices], dtype=torch.float32, device=self.device)
                    t_sc_1 = torch.as_tensor(obs_1["scalars"][latest_indices], dtype=torch.float32, device=self.device)
                    t_mm_1 = torch.as_tensor(mask_1["move_mask"][latest_indices], dtype=torch.bool, device=self.device)
                    t_bm_1 = torch.as_tensor(mask_1["bomb_mask"][latest_indices], dtype=torch.bool, device=self.device)
                    acts_1, _, _, _ = self.policy.get_action(t_sp_1, t_sc_1, t_mm_1, t_bm_1)
                    acts_1_np = acts_1.cpu().numpy()
                    for idx_pos, env_idx in enumerate(latest_indices):
                        actions_1_list[env_idx] = (int(acts_1_np[idx_pos, 0]), int(acts_1_np[idx_pos, 1]))

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
                            t_sp_c = torch.as_tensor(obs_1["spatial"][i : i + 1], dtype=torch.float32, device=self.device)
                            t_sc_c = torch.as_tensor(obs_1["scalars"][i : i + 1], dtype=torch.float32, device=self.device)
                            t_mm_c = torch.as_tensor(mask_1["move_mask"][i : i + 1], dtype=torch.bool, device=self.device)
                            t_bm_c = torch.as_tensor(mask_1["bomb_mask"][i : i + 1], dtype=torch.bool, device=self.device)
                            act_fb, _, _, _ = self.policy.get_action(t_sp_c, t_sc_c, t_mm_c, t_bm_c)
                            actions_1_list[i] = (int(act_fb[0, 0].item()), int(act_fb[0, 1].item()))

                for cpath, env_indices in ckpt_envs_by_path.items():
                    ckpt_model = self.get_opponent_policy(cpath)
                    t_sp_c = torch.as_tensor(obs_1["spatial"][env_indices], dtype=torch.float32, device=self.device)
                    t_sc_c = torch.as_tensor(obs_1["scalars"][env_indices], dtype=torch.float32, device=self.device)
                    t_mm_c = torch.as_tensor(mask_1["move_mask"][env_indices], dtype=torch.bool, device=self.device)
                    t_bm_c = torch.as_tensor(mask_1["bomb_mask"][env_indices], dtype=torch.bool, device=self.device)
                    acts_c, _, _, _ = ckpt_model.get_action(t_sp_c, t_sc_c, t_mm_c, t_bm_c)
                    acts_c_np = acts_c.cpu().numpy()
                    for idx_pos, env_idx in enumerate(env_indices):
                        actions_1_list[env_idx] = (int(acts_c_np[idx_pos, 0]), int(acts_c_np[idx_pos, 1]))

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

            # Store in pre-allocated rollout buffers
            self.buf_spatial[step] = t_spatial
            self.buf_scalars[step] = t_scalars
            self.buf_actions[step] = actions_0
            self.buf_log_probs[step] = log_probs_0
            self.buf_values[step] = values_0
            self.buf_rewards[step] = torch.as_tensor(rewards_0, dtype=torch.float32, device=self.device)
            self.buf_dones[step] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
            self.buf_mask_move[step] = t_mask_move
            self.buf_mask_bomb[step] = t_mask_bomb

            # Record match telemetry
            for info in step_infos:
                m_stats = info.get("match_stats")
                if m_stats is not None:
                    self.diagnostics.record_match(m_stats)

            obs_0 = next_obs_0
            mask_0 = next_mask_0
            obs_1 = next_obs_1
            mask_1 = next_mask_1
            infos = step_infos

        # Save latest persistent state across generations
        self._last_obs_0 = obs_0
        self._last_mask_0 = mask_0
        self._last_obs_1 = obs_1
        self._last_mask_1 = mask_1
        self._last_infos = infos

        # Compute GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            last_value = self.policy.get_value(
                torch.as_tensor(obs_0["spatial"], dtype=torch.float32, device=self.device),
                torch.as_tensor(obs_0["scalars"], dtype=torch.float32, device=self.device),
            )

        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self.buf_dones[t].float()
                next_val = last_value
            else:
                next_non_terminal = 1.0 - self.buf_dones[t].float()
                next_val = self.buf_values[t + 1]

            delta = self.buf_rewards[t] + self.gamma * next_val * next_non_terminal - self.buf_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.buf_advantages[t] = last_gae

        returns = self.buf_advantages + self.buf_values

        # Flatten batch for PPO optimization
        b_spatial = self.buf_spatial.reshape(-1, 18, 9, 9)
        b_scalars = self.buf_scalars.reshape(-1, 18)
        b_actions = self.buf_actions.reshape(-1, 2)
        b_log_probs = self.buf_log_probs.reshape(-1)
        b_advantages = self.buf_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_mask_move = self.buf_mask_move.reshape(-1, 5)
        b_mask_bomb = self.buf_mask_bomb.reshape(-1, 2)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = self.rollout_steps * self.num_envs
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

        # Showcase replay recording
        replay_path = None
        if self.showcase_interval > 0 and (generation % self.showcase_interval == 0):
            replay_path = self.run_showcase_match(generation)

        diag_metrics = self.diagnostics.compute_metrics(generation)
        diag_metrics.update({
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "checkpoint_path": ckpt_path,
            "replay_path": replay_path,
            "self_damage_scale": self_damage_scale,
            "safety_filter_prob": safety_filter_prob,
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

            # Terminal audit summary
            table = self.diagnostics.format_table(gen)
            print(table)
            print(
                f"[Gen {gen:04d}] Time: {elapsed:.2f}s | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Critic Loss: {metrics['critic_loss']:.4f} | "
                f"Damage Scale: {metrics['self_damage_scale']:.2f} | "
                f"Filter Prob: {metrics['safety_filter_prob']:.2f}"
            )
            if metrics.get("replay_path"):
                print(f" Showcase Replay Saved: {metrics['replay_path']}")
            history.append(metrics)

        self.vec_env.close()
        return history

    def close(self) -> None:
        self.vec_env.close()
