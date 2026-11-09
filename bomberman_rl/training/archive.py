"""Archive utilities for packaging and moving training checkpoints and replays."""

from __future__ import annotations
import os
import shutil
import tarfile
from datetime import datetime
from typing import Any


def archive_training_data(
    checkpoint_dir: str = "./checkpoints",
    replays_dir: str = "./replays",
    archive_dir: str = "./archives",
    timestamp: str | None = None,
    keep_samples: tuple[str, ...] = ("test_match.npz", ".gitkeep"),
    create_compressed: bool = True,
) -> dict[str, Any]:
    """Archive training snapshots and recorded showcase replays.

    Moves training artifacts out of active working directories into an archive folder,
    creates a timestamped compressed tar.gz bundle, and preserves essential sample fixtures.

    Args:
        checkpoint_dir: Path to model checkpoints directory.
        replays_dir: Path to match replays directory.
        archive_dir: Path to root archive directory (kept out of version control).
        timestamp: Optional explicit timestamp string (format: YYYYMMDD_HHMMSS).
        keep_samples: Filenames to preserve in source directories (e.g., fixtures, .gitkeep).
        create_compressed: Whether to generate a .tar.gz bundle alongside the directory.

    Returns:
        Dict detailing archive status, file count, and paths.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_subfolder_name = f"archive_{ts}"
    target_archive_path = os.path.join(archive_dir, archive_subfolder_name)
    tar_path = os.path.join(archive_dir, f"{archive_subfolder_name}.tar.gz")

    # Discover candidate files in checkpoint_dir
    ckpt_files: list[str] = []
    if os.path.isdir(checkpoint_dir):
        for f in sorted(os.listdir(checkpoint_dir)):
            if f not in keep_samples and not f.startswith("."):
                ckpt_files.append(f)

    # Discover candidate files in replays_dir
    replay_files: list[str] = []
    if os.path.isdir(replays_dir):
        for f in sorted(os.listdir(replays_dir)):
            if f not in keep_samples and not f.startswith("."):
                replay_files.append(f)

    total_files = len(ckpt_files) + len(replay_files)
    if total_files == 0:
        return {
            "archived": False,
            "timestamp": ts,
            "files_count": 0,
            "archive_dir": None,
            "archive_tar": None,
            "files": [],
        }

    # Ensure archive subdirectories exist
    target_ckpt_dir = os.path.join(target_archive_path, "checkpoints")
    target_replays_dir = os.path.join(target_archive_path, "replays")
    os.makedirs(target_ckpt_dir, exist_ok=True)
    os.makedirs(target_replays_dir, exist_ok=True)

    moved_files: list[str] = []

    # Move checkpoints
    for fname in ckpt_files:
        src = os.path.join(checkpoint_dir, fname)
        dst = os.path.join(target_ckpt_dir, fname)
        shutil.move(src, dst)
        moved_files.append(dst)

    # Move replays
    for fname in replay_files:
        src = os.path.join(replays_dir, fname)
        dst = os.path.join(target_replays_dir, fname)
        shutil.move(src, dst)
        moved_files.append(dst)

    # Create compressed tar.gz bundle
    if create_compressed:
        os.makedirs(archive_dir, exist_ok=True)
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(target_archive_path, arcname=archive_subfolder_name)

    return {
        "archived": True,
        "timestamp": ts,
        "files_count": len(moved_files),
        "archive_dir": target_archive_path,
        "archive_tar": tar_path if create_compressed else None,
        "files": moved_files,
    }


def clean_training_data(
    checkpoint_dir: str = "./checkpoints",
    replays_dir: str = "./replays",
    keep_samples: tuple[str, ...] = ("test_match.npz", ".gitkeep"),
) -> dict[str, Any]:
    """Remove training checkpoints and showcase replays to prepare clean directories for a fresh run.

    Deletes past training artifacts from active working directories while preserving essential
    fixtures and version-controlled placeholders.

    Args:
        checkpoint_dir: Path to model checkpoints directory.
        replays_dir: Path to match replays directory.
        keep_samples: Filenames to preserve in source directories (e.g., fixtures, .gitkeep).

    Returns:
        Dict detailing cleaning status, removed file count, and list of removed file paths.
    """
    removed_files: list[str] = []

    for target_dir in (checkpoint_dir, replays_dir):
        if not os.path.isdir(target_dir):
            continue
        for fname in sorted(os.listdir(target_dir)):
            if fname in keep_samples or fname.startswith("."):
                continue
            path = os.path.join(target_dir, fname)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                removed_files.append(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                removed_files.append(path)

    return {
        "cleaned": len(removed_files) > 0,
        "files_count": len(removed_files),
        "checkpoint_dir": checkpoint_dir,
        "replays_dir": replays_dir,
        "files": removed_files,
    }
