"""Interactive Pygame player & replay forensic analyzer for 1v1 Bomberman.

Usage:
    # Play Human vs Agent:
    python -m bomberman_rl.visualizer.play_interactive --mode human_vs_agent

    # Watch Agent vs Agent self-play:
    python -m bomberman_rl.visualizer.play_interactive --mode agent_vs_agent

    # Inspect a recorded replay file:
    python -m bomberman_rl.visualizer.play_interactive --mode replay --replay ./replays/test_match.npz

    # Record live match:
    python -m bomberman_rl.visualizer.play_interactive --mode agent_vs_agent --record ./replays/live_match.npz
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Any
import numpy as np

import pygame

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    TICK_RATE,
    MoveAction,
    BombAction,
    TileType,
    PowerUpType,
)
from bomberman_rl.engine.state import GameState, PlayerState
from bomberman_rl.engine.physics import BombermanPhysics
from bomberman_rl.rl.observation import ObservationBuilder, compute_danger_map
from bomberman_rl.rl.rewards import RewardTracker
from bomberman_rl.rl.action_filter import SafetyActionFilter
from bomberman_rl.replays.recorder import (
    ReplayRecorder,
    ReplayEpisode,
    find_latest_replay,
    list_replays,
)
from bomberman_rl.visualizer.renderer import Renderer


MAX_CATCHUP_TICKS_PER_FRAME: int = 3


class HumanInputLatch:
    """Accumulates human key presses across render frames until a simulation tick consumes them.

    Rendering runs faster than the simulation tick rate, so a quick tap can arrive on a frame
    that does not advance the simulation. The latch keeps the latest move and any bomb press
    until the next tick.
    """

    def __init__(self) -> None:
        self._move: int = int(MoveAction.NOOP)
        self._bomb: int = int(BombAction.NOOP)

    def update(self, action: tuple[int, int]) -> None:
        if int(action[0]) != int(MoveAction.NOOP):
            self._move = int(action[0])
        if int(action[1]) != int(BombAction.NOOP):
            self._bomb = int(action[1])

    def consume(self) -> tuple[int, int]:
        action = (self._move, self._bomb)
        self._move = int(MoveAction.NOOP)
        self._bomb = int(BombAction.NOOP)
        return action


def advance_tick_budget(
    accum: float,
    dt: float,
    speed: float,
    max_ticks: int = MAX_CATCHUP_TICKS_PER_FRAME,
) -> tuple[int, float]:
    """Convert elapsed frame time into simulation ticks to run now.

    Returns ``(ticks, remaining_accumulator)``. After a frame hitch at most ``max_ticks`` run
    and the excess time is discarded so the game never fast-forwards.
    """
    if dt <= 0.0:
        return 0, accum
    accum += dt * (TICK_RATE * speed)
    ticks = int(accum)
    accum -= ticks
    if ticks > max_ticks:
        ticks = max_ticks
    return ticks, accum


class HeuristicAgent:
    """Intelligent rule-based opponent with genuine policy distributions and critic estimates."""

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self.action_filter = SafetyActionFilter(enabled=True)

    def compute_action_and_telemetry(
        self,
        state: GameState,
        danger_map: np.ndarray,
    ) -> tuple[tuple[int, int], np.ndarray, float]:
        """Compute action, 10-dim policy probability distribution, and critic V(s)."""
        player = state.players[self.player_id]
        opp = state.players[1 - self.player_id]

        if not player.alive:
            probs = np.zeros(10, dtype=np.float32)
            probs[0] = 1.0  # NOOP move
            probs[5] = 1.0  # NOOP bomb
            return (int(MoveAction.NOOP), int(BombAction.NOOP)), probs, -1.0

        masks = self.action_filter.get_action_mask(state, self.player_id, ego_perspective=False)
        move_mask = masks["move_mask"]
        bomb_mask = masks["bomb_mask"]

        pr, pc = player.tile
        curr_danger = danger_map[pr, pc] if (0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE) else 0.0

        # Move scores
        move_scores = np.zeros(5, dtype=np.float32)
        move_deltas = {
            MoveAction.NOOP: (0, 0),
            MoveAction.UP: (-1, 0),
            MoveAction.DOWN: (1, 0),
            MoveAction.LEFT: (0, -1),
            MoveAction.RIGHT: (0, 1),
        }

        for m_act, (dr, dc) in move_deltas.items():
            if not move_mask[int(m_act)]:
                move_scores[int(m_act)] = -100.0
                continue

            nr, nc = pr + dr, pc + dc
            score = 0.0
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                # Avoid danger
                d_val = danger_map[nr, nc]
                score -= d_val * 25.0

                # Seek powerups
                if state.powerups[nr, nc] != PowerUpType.NONE:
                    score += 8.0

                # Move toward opponent if safe
                if opp.alive:
                    dist_before = abs(pr - opp.r) + abs(pc - opp.c)
                    dist_after = abs(nr - opp.r) + abs(nc - opp.c)
                    if curr_danger > 0:
                        # Prioritize escaping danger
                        if d_val < curr_danger:
                            score += 15.0
                    else:
                        if dist_after < dist_before:
                            score += 3.0
            move_scores[int(m_act)] = score

        # Softmax for move probabilities
        exp_m = np.exp(move_scores - np.max(move_scores))
        p_move = exp_m / np.sum(exp_m)
        chosen_m = int(np.argmax(p_move))

        # Bomb decision
        bomb_scores = np.array([2.0, 0.0], dtype=np.float32)
        can_place = bomb_mask[int(BombAction.PLACE_BOMB)]
        active_bombs = sum(1 for b in state.bombs if b.owner_id == player.id)

        if can_place and active_bombs < player.max_bombs and curr_danger < 0.2:
            # Check adjacent bricks
            adj_bricks = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = pr + dr, pc + dc
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE and state.grid[rr, cc] == TileType.BRICK:
                    adj_bricks += 1
            if adj_bricks > 0:
                bomb_scores[1] = 4.0 + adj_bricks * 1.5

        exp_b = np.exp(bomb_scores - np.max(bomb_scores))
        p_bomb = exp_b / np.sum(exp_b)
        chosen_b = int(np.argmax(p_bomb))

        full_probs = np.concatenate([p_move, p_bomb]).astype(np.float32)

        # Critic value estimate V(s) in [-1.0, 1.0]
        if not opp.alive:
            v = 1.0
        else:
            base_v = 0.0
            # Powerup advantage
            base_v += (player.max_bombs - opp.max_bombs) * 0.1
            base_v += (player.blast_radius - opp.blast_radius) * 0.1
            base_v += (player.speed - opp.speed) * 0.2
            if player.shield_timer > 0:
                base_v += 0.25
            if curr_danger > 0.5:
                base_v -= 0.6  # Trapped in blast zone!
            v = float(np.clip(base_v, -0.95, 0.95))

        return (chosen_m, chosen_b), full_probs, v


def run_interactive(
    mode: str = "agent_vs_agent",
    replay_path: str | None = None,
    record_path: str | None = None,
    seed: int | None = 42,
    playback_speed: float = 1.0,
    headless: bool = False,
    max_ticks: int = MAX_TICKS,
    replays_dir: str = "./replays",
) -> None:
    """Run interactive visual loop for live gameplay or replay inspection."""
    renderer = Renderer(title=f"Bomberman MARL - {mode.upper()}", headless=headless, mode=mode)
    renderer.max_ticks = max_ticks
    renderer.set_playback_speed(playback_speed)

    # --- Mode: Replay Forensic Analyzer ---
    if mode == "replay":
        if replay_path is None or replay_path == "latest":
            latest = find_latest_replay(replays_dir)
            if latest is None:
                raise FileNotFoundError(f"No replay files (.npz) found in '{replays_dir}'.")
            replay_path = latest

        if not os.path.isfile(replay_path):
            raise FileNotFoundError(f"Replay archive not found: {replay_path}")

        episode = ReplayEpisode.load(replay_path)
        total_frames = len(episode)
        if total_frames == 0:
            print("Replay contains zero frames.")
            return

        curr_frame_idx = 0
        running = True
        subtick_accum = 0.0

        while running:
            dt = renderer.tick_clock(60)
            inp = renderer.handle_input()

            if inp["quit"]:
                break

            # Handle seeking & stepping
            if inp["seek_tick"] is not None:
                ratio = inp["seek_tick"]
                curr_frame_idx = int(np.clip(round(ratio * (total_frames - 1)), 0, total_frames - 1))
                subtick_accum = 0.0
            elif inp["step"] != 0:
                curr_frame_idx = int(np.clip(curr_frame_idx + inp["step"], 0, total_frames - 1))
                subtick_accum = 0.0

            # Playback advance
            if not renderer.paused:
                subtick_accum += dt * (TICK_RATE * renderer.playback_speed)
                steps_to_advance = int(subtick_accum)
                if steps_to_advance > 0:
                    curr_frame_idx = min(total_frames - 1, curr_frame_idx + steps_to_advance)
                    subtick_accum -= steps_to_advance
                    if curr_frame_idx >= total_frames - 1:
                        renderer.paused = True
                        if headless:
                            break

            state = episode.get_state(curr_frame_idx)
            prev_state = episode.get_state(max(0, curr_frame_idx - 1))
            telemetry = episode.get_telemetry(curr_frame_idx)

            renderer.render_frame(
                state=state,
                prev_state=prev_state,
                telemetry=telemetry,
                events=episode.events,
                interp_alpha=float(np.clip(subtick_accum, 0.0, 1.0)),
                total_ticks=total_frames,
                tick_limit=episode.metadata.get("max_episode_ticks"),
                end_reason=episode.metadata.get("end_reason"),
            )

        pygame.quit()
        return

    # --- Modes: Live Simulation (Human vs Agent or Agent vs Agent) ---
    state = GameState.create_initial(seed)
    prev_state = state.clone()
    reward_tracker = RewardTracker()
    reward_tracker.reset(state)
    recorder = ReplayRecorder(metadata={"mode": mode, "seed": seed})

    agent_p0 = HeuristicAgent(0) if mode == "agent_vs_agent" else None
    agent_p1 = HeuristicAgent(1)

    running = True
    terminated = False
    subtick_accum = 0.0
    human_latch = HumanInputLatch()

    # Record initial frame
    danger_map = compute_danger_map(state)
    rec_p0_act = (MoveAction.NOOP, BombAction.NOOP)
    rec_p1_act = (MoveAction.NOOP, BombAction.NOOP)
    p0_probs = np.zeros(10, dtype=np.float32)
    p0_probs[0] = 1.0
    p0_probs[5] = 1.0
    p1_probs = np.zeros(10, dtype=np.float32)
    p1_probs[0] = 1.0
    p1_probs[5] = 1.0
    v0, v1 = 0.0, 0.0

    recorder.record_frame(
        state=state,
        actions=(rec_p0_act, rec_p1_act),
        action_probs_p0=p0_probs,
        action_probs_p1=p1_probs,
        value_p0=v0,
        value_p1=v1,
    )

    episode = recorder.to_episode()
    live_scrub_index = 0

    while running:
        dt = renderer.tick_clock(60)
        inp = renderer.handle_input()

        if inp["quit"]:
            break

        total_frames = len(recorder)

        # Scrubber interaction during live match
        if inp["seek_tick"] is not None:
            renderer.paused = True
            ratio = inp["seek_tick"]
            live_scrub_index = int(np.clip(round(ratio * (total_frames - 1)), 0, total_frames - 1))
        elif inp["step"] != 0 and renderer.paused:
            live_scrub_index = int(np.clip(live_scrub_index + inp["step"], 0, total_frames - 1))

        human_latch.update(inp["human_action"])
        if renderer.paused:
            human_latch.consume()

        # Simulation step when unpaused
        if not renderer.paused and not terminated:
            ticks_due, subtick_accum = advance_tick_budget(subtick_accum, dt, renderer.playback_speed)
            for _ in range(ticks_due):
                if terminated:
                    break
                danger_map = compute_danger_map(state)

                # Compute actions
                if mode == "human_vs_agent":
                    p0_act = human_latch.consume()
                    p0_probs = np.zeros(10, dtype=np.float32)
                    p0_probs[int(p0_act[0])] = 1.0
                    p0_probs[5 + int(p0_act[1])] = 1.0
                    v0 = 0.1
                else:
                    assert agent_p0 is not None
                    p0_act, p0_probs, v0 = agent_p0.compute_action_and_telemetry(state, danger_map)

                p1_act, p1_probs, v1 = agent_p1.compute_action_and_telemetry(state, danger_map)

                state_before = state.clone()
                BombermanPhysics.step(state, [p0_act, p1_act])

                p0 = state.players[0]
                p1 = state.players[1]
                t_term = (not p0.alive or not p1.alive)
                t_trunc = (state.tick >= max_ticks)
                terminated = t_term or t_trunc

                step_rewards, step_telemetry = reward_tracker.step(
                    state_before=state_before,
                    actions=[p0_act, p1_act],
                    state_after=state,
                    terminated=t_term,
                    truncated=t_trunc,
                )

                recorder.record_frame(
                    state=state,
                    actions=(p0_act, p1_act),
                    action_probs_p0=p0_probs,
                    action_probs_p1=p1_probs,
                    value_p0=v0,
                    value_p1=v1,
                    rewards_p0=step_telemetry["player_0"],
                    rewards_p1=step_telemetry["player_1"],
                )

                prev_state = state_before
                live_scrub_index = len(recorder) - 1

                if headless and terminated:
                    running = False

        # Render display
        if renderer.paused and live_scrub_index < len(recorder):
            disp_episode = recorder.to_episode()
            disp_state = disp_episode.get_state(live_scrub_index)
            disp_prev = disp_episode.get_state(max(0, live_scrub_index - 1))
            disp_telemetry = disp_episode.get_telemetry(live_scrub_index)
            renderer.render_frame(
                state=disp_state,
                prev_state=disp_prev,
                telemetry=disp_telemetry,
                events=disp_episode.events,
                interp_alpha=1.0,
                total_ticks=len(recorder),
            )
        else:
            telemetry = {
                "tick": state.tick,
                "actions": {"player_0": p0_act if 'p0_act' in locals() else (0, 0), "player_1": p1_act if 'p1_act' in locals() else (0, 0)},
                "action_probs_p0": p0_probs,
                "action_probs_p1": p1_probs,
                "value_p0": v0,
                "value_p1": v1,
                "rewards_p0": reward_tracker.episode_logs[0].as_dict(),
                "rewards_p1": reward_tracker.episode_logs[1].as_dict(),
                "cumulative_rewards_p0": reward_tracker.episode_logs[0].as_dict(),
                "cumulative_rewards_p1": reward_tracker.episode_logs[1].as_dict(),
            }
            renderer.render_frame(
                state=state,
                prev_state=prev_state,
                telemetry=telemetry,
                events=recorder.events,
                interp_alpha=float(np.clip(subtick_accum, 0.0, 1.0)),
                total_ticks=max(len(recorder), max_ticks),
            )

    if record_path:
        print(f"Saving match trajectory ({len(recorder)} frames) to {record_path}...")
        recorder.save(record_path)
        print("Save completed.")

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bomberman RL Interactive Telemetry Visualizer")
    parser.add_argument(
        "--mode",
        choices=["human_vs_agent", "agent_vs_agent", "replay"],
        default="agent_vs_agent",
        help="Interactive playback mode",
    )
    parser.add_argument("--replay", type=str, default=None, help="Path to .npz replay trajectory (default: latest in replays-dir)")
    parser.add_argument("--replays-dir", type=str, default="./replays", help="Directory to search for replays (default: ./replays)")
    parser.add_argument("--record", type=str, default=None, help="Path to record match trajectory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--max_ticks", type=int, default=MAX_TICKS, help="Maximum ticks")

    args = parser.parse_args()

    run_interactive(
        mode=args.mode,
        replay_path=args.replay,
        record_path=args.record,
        seed=args.seed,
        playback_speed=args.speed,
        headless=args.headless,
        max_ticks=args.max_ticks,
        replays_dir=args.replays_dir,
    )


if __name__ == "__main__":
    main()
