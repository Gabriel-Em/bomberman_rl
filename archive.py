#!/usr/bin/env python3
"""CLI utility to archive training data (checkpoints and replays) with timestamping."""

from __future__ import annotations
import argparse
import sys

from bomberman_rl.training import archive_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive training checkpoints and replays into timestamped compressed bundles."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Path to checkpoints directory (default: ./checkpoints)",
    )
    parser.add_argument(
        "--replays-dir",
        type=str,
        default="./replays",
        help="Path to showcase replays directory (default: ./replays)",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default="./archives",
        help="Path to destination archive directory (default: ./archives)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Explicit timestamp string (default: current YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip generating compressed .tar.gz bundle (only move folder)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = archive_training_data(
        checkpoint_dir=args.checkpoint_dir,
        replays_dir=args.replays_dir,
        archive_dir=args.archive_dir,
        timestamp=args.timestamp,
        create_compressed=not args.no_compress,
    )

    if not result["archived"]:
        print(f"No training data found in '{args.checkpoint_dir}' or '{args.replays_dir}' to archive.")
        return 0

    print(f"Successfully archived {result['files_count']} training files:")
    print(f"  Archive Directory: {result['archive_dir']}")
    if result["archive_tar"]:
        print(f"  Compressed Bundle: {result['archive_tar']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
