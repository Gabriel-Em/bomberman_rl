#!/usr/bin/env python3
"""CLI Entry point for Bomberman 1v1 Multi-Process Vectorized Self-Play PPO Training."""

from __future__ import annotations
import argparse
import sys
import torch

from bomberman_rl.training import (
    PPOTrainer,
    ActorCriticNetwork,
    archive_training_data,
    clean_training_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 1v1 Bomberman via Multi-Process Vectorized Self-Play PPO."
    )
    parser.add_argument(
        "--total-generations",
        "--generations",
        dest="total_generations",
        type=int,
        default=500,
        help="Total self-play training generations (default: 500)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environment games on CPU (default: 8)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=512,
        help="Rollout steps per environment per generation (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="PPO optimization epochs per generation (default: 4)",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO gradient updates (default: 64)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save snapshot pool checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--replays-dir",
        type=str,
        default="./replays",
        help="Directory to save showcase replay trajectories (default: ./replays)",
    )
    parser.add_argument(
        "--showcase-interval",
        type=int,
        default=10,
        help="Generations between automated showcase replay recordings (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Force synchronous in-process vector env (default: False, uses AsyncVectorEnv)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint file path or 'latest'. If not specified, unarchived training data from stopped runs is removed to start clean (default: None)",
    )
    parser.add_argument(
        "--archive-first",
        action="store_true",
        help="Archive existing checkpoints and replays before starting new training (default: False)",
    )
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="Reset and remove all existing checkpoints and replays, starting fresh from Gen 0 (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--show-telemetry",
        action="store_true",
        default=False,
        help="Show detailed telemetry diagnostics tables during training (default: False, hidden)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.resume is None:
        if args.archive_first:
            res = archive_training_data(
                checkpoint_dir=args.checkpoint_dir,
                replays_dir=args.replays_dir,
            )
            if res["archived"]:
                print(f"Archived previous training data ({res['files_count']} files) to {res['archive_dir']}")
                if res["archive_tar"]:
                    print(f"Compressed bundle: {res['archive_tar']}")
            else:
                print("No previous training data found to archive.")
        else:
            res = clean_training_data(
                checkpoint_dir=args.checkpoint_dir,
                replays_dir=args.replays_dir,
            )
            if res["cleaned"]:
                print(f"Removed previous training data ({res['files_count']} files) to ensure clean directories for new training.")
    else:
        if args.reset_checkpoints:
            args.resume = None
            res = clean_training_data(
                checkpoint_dir=args.checkpoint_dir,
                replays_dir=args.replays_dir,
            )
            if res["cleaned"]:
                print(f"Reset: removed previous training data ({res['files_count']} files).")
        elif args.archive_first:
            res = archive_training_data(
                checkpoint_dir=args.checkpoint_dir,
                replays_dir=args.replays_dir,
            )
            if res["archived"]:
                print(f"Archived previous training data ({res['files_count']} files) to {res['archive_dir']}")
                if res["archive_tar"]:
                    print(f"Compressed bundle: {res['archive_tar']}")

    trainer = PPOTrainer(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_generations=args.total_generations,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        checkpoint_dir=args.checkpoint_dir,
        replays_dir=args.replays_dir,
        showcase_interval=args.showcase_interval,
        device=args.device,
        use_async=not args.sync,
        seed=args.seed,
        show_telemetry=args.show_telemetry,
    )

    if args.resume is not None:
        start_gen = trainer.load_checkpoint(args.resume)
        print(f"Resumed model checkpoint '{args.resume}'. Starting at Generation {start_gen:04d}.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Closing vector environments...")
    finally:
        trainer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
