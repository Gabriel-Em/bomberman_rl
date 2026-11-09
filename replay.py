#!/usr/bin/env python3
"""Convenient replay viewer CLI for Bomberman RL.

By default with no arguments, automatically detects and launches the latest replay
in the replays folder.

Usage:
    python replay.py                 # Load and play the latest replay
    python replay.py <path>          # Play a specific replay file
    python replay.py --list          # List all available replays
    python replay.py --speed 2.0     # Play latest replay at 2x speed
"""

from __future__ import annotations
import argparse
import datetime
import os
import sys

from bomberman_rl.replays import find_latest_replay, list_replays, ReplayEpisode
from bomberman_rl.visualizer.play_interactive import run_interactive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bomberman RL Replay Player - plays latest replay by default.",
    )
    parser.add_argument(
        "replay",
        nargs="?",
        default=None,
        help="Optional path to .npz replay file (default: latest in replays directory)",
    )
    parser.add_argument(
        "--replays-dir",
        type=str,
        default="./replays",
        help="Directory to search for replays (default: ./replays)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available replays sorted from newest to oldest and exit",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Initial playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a Pygame window",
    )
    return parser.parse_args()


def handle_list(replays_dir: str) -> int:
    replays = list_replays(replays_dir)
    if not replays:
        print(f"No replays found in '{replays_dir}'.")
        return 0

    print(f"Found {len(replays)} replay(s) in '{replays_dir}':")
    for idx, path in enumerate(replays):
        mtime = os.path.getmtime(path)
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = os.path.getsize(path) / 1024.0
        rel_path = os.path.relpath(path)
        tag = " (latest)" if idx == 0 else ""
        print(f"  [{idx + 1}] {rel_path}{tag}")
        print(f"      Modified: {dt_str} | Size: {size_kb:.1f} KB")
    return 0


def main() -> int:
    args = parse_args()

    if args.list:
        return handle_list(args.replays_dir)

    target_replay = args.replay
    if target_replay is None:
        target_replay = find_latest_replay(args.replays_dir)
        if target_replay is None:
            print(
                f"Error: No replay files (.npz) found in '{args.replays_dir}'.\n"
                f"Run training first to produce showcase replays: python train.py",
                file=sys.stderr,
            )
            return 1
        print(f"Playing latest replay: {os.path.relpath(target_replay)}")
    else:
        if not os.path.isfile(target_replay):
            print(f"Error: Replay file not found: {target_replay}", file=sys.stderr)
            return 1

    run_interactive(
        mode="replay",
        replay_path=target_replay,
        playback_speed=args.speed,
        headless=args.headless,
        replays_dir=args.replays_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
