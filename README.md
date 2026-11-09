# Bomberman RL

A high-performance, deterministic 1v1 Multi-Agent Reinforcement Learning (MARL) Bomberman simulation platform featuring multi-process self-play Proximal Policy Optimization (PPO), real-time Pygame telemetry visualization, and frame-accurate forensic replay analysis.

---

## Table of Contents

- [Features](#features)
- [Architecture & Package Layout](#architecture--package-layout)
- [Installation & Environment Setup](#installation--environment-setup)
- [Training](#training)
  - [Quickstart Training](#quickstart-training)
  - [Command-Line Arguments](#command-line-arguments)
  - [Self-Play & Checkpointing Workflow](#self-play--checkpointing-workflow)
  - [Archiving Training Runs](#archiving-training-runs)
- [Interactive Visualizer & Replay Player](#interactive-visualizer--replay-player)
  - [Viewing Replays](#viewing-replays)
  - [Live Agent vs Agent](#live-agent-vs-agent)
  - [Human vs Agent](#human-vs-agent)
  - [Interactive Controls & Shortcuts](#interactive-controls--shortcuts)
  - [Visualizer CLI Arguments](#visualizer-cli-arguments)
- [Testing & Verification](#testing--verification)

---

## Features

- **Headless Deterministic Engine:** Pure NumPy/Python 11x11 grid simulation with integer grid coordinates, sub-tick movement interpolation, blast raycasting, chain explosions, powerups (Extra Bomb, Blast Range, Speed Boost), and sudden death ring-of-fire.
- **MARL Vectorized Training:** High-throughput `AsyncVectorEnv` and `SyncVectorEnv` supporting batched CPU multi-process game environments.
- **Self-Play PPO:** Actor-Critic network with residual convolutional spatial encoders, ego-centric rotation invariants, and opponent snapshot pool sampling to mitigate non-stationarity and policy cycling.
- **Real-Time Telemetry & Replay HUD:** 1280x760 60 FPS Pygame telemetry visualizer featuring arena rendering, value function $V(s)$ sparklines, policy distribution bar charts, danger heatmaps, reachable safe-space overlays, and interactive frame scrubbing.
- **Bit-Exact Replay Recording:** Serialized `.npz` replay episodes storing bit-exact game state transitions, action distributions, critic values, and reward telemetry for forensic debugging.

---

## Architecture & Package Layout

```
bomberman_rl/
├── engine/               # Core deterministic game simulation
│   ├── config.py         # Grid dimensions, tick rates, tile types, actions, constants
│   ├── env.py            # Gymnasium-compatible 1v1 Bomberman environment
│   ├── physics.py        # Raycasting, explosions, collision detection, sudden death
│   └── state.py          # Dataclasses: GameState, PlayerState, BombState
├── rl/                   # Reinforcement learning representations
│   ├── action_filter.py  # In-place suicidal bomb filtering and invalid action masking
│   ├── observation.py    # Spatial tensor feature builder and BFS danger maps
│   └── rewards.py        # Reward shaping, dense survival incentives, combat telemetry
├── training/             # PPO self-play pipeline & vectorization
│   ├── archive.py        # Snapshot & replay archival and compression utilities
│   ├── diagnostics.py    # Entropy, KL divergence, value loss tracking, Elo metrics
│   ├── network.py        # Actor-Critic CNN architecture with dual action heads
│   ├── ppo_trainer.py    # Multi-generation self-play PPO trainer
│   └── vec_env.py        # Async multi-process and synchronous vector environments
├── replays/              # Trajectory serialization and inspection
│   └── recorder.py       # ReplayRecorder & ReplayEpisode npz loader/saver
└── visualizer/           # Pygame HUD and playback interface
    ├── play_interactive.py # Unified CLI for human play, bot matches, and replay inspection
    └── renderer.py       # 1280x760 rendering loop, overlays, and scrubber UI
train.py                  # CLI training entrypoint
archive.py                # Standalone CLI training archival utility
replay.py                 # Fast CLI replay runner (plays latest replay by default)
tests/                    # Test suite covering engine, physics, RL, and replays
```

---

## Installation & Environment Setup

### Requirements
- Python 3.10+ (tested with Python 3.12)
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0
- Pygame >= 2.5.0
- Pytest >= 7.0.0

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/bomberman_rl.git
cd bomberman_rl

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

The training pipeline uses multi-process vectorized self-play PPO. The learning agent samples opponents from a historical snapshot pool of previous generations to ensure robust strategy evaluation.

### Quickstart Training

Start self-play training with 8 parallel environments:

```bash
python train.py --num-envs 8 --total-generations 500
```

Resume training from the latest checkpoint:

```bash
python train.py --resume latest --total-generations 500
```

Resume training from a specific checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_gen_0150.pt --total-generations 500
```

Run training on GPU with custom rollout lengths:

```bash
python train.py --device cuda --num-envs 16 --rollout-steps 256 --total-generations 1000
```

Run synchronous in-process training (useful for debugging without multiprocessing):

```bash
python train.py --sync --num-envs 4
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--total-generations` | `int` | `500` | Total self-play generations to run. |
| `--num-envs` | `int` | `8` | Number of parallel game environments on CPU workers. |
| `--rollout-steps` | `int` | `128` | Rollout steps collected per environment per generation. |
| `--lr` | `float` | `3e-4` | Learning rate for Adam optimizer. |
| `--num-epochs` | `int` | `4` | PPO optimization epochs per generation. |
| `--minibatch-size` | `int` | `64` | Minibatch size for gradient descent steps. |
| `--checkpoint-dir` | `str` | `./checkpoints` | Target directory for saving model weight snapshots (`.pt`). |
| `--replays-dir` | `str` | `./replays` | Target directory for automated showcase replays (`.npz`). |
| `--showcase-interval` | `int` | `10` | Frequency (generations) between automated match recordings. |
| `--resume` | `str` | `None` | Resume training from checkpoint file path or `'latest'`. When omitted, unarchived training data from stopped runs is removed to start clean. |
| `--archive-first` | flag | `False` | Automatically archive previous checkpoints and replays to `./archives` before training. |
| `--device` | `str` | `cuda`/`cpu` | Compute device (`cpu` or `cuda`). Auto-detects CUDA when available. |
| `--sync` | flag | `False` | Run synchronous vector env instead of asynchronous multi-process. |
| `--seed` | `int` | `42` | Global random seed for reproducible initialization. |

### Self-Play & Checkpointing Workflow

1. **Snapshot Checkpointing:** At the end of each generation, agent weights are serialized to `--checkpoint-dir` (`checkpoint_gen_XXXX.pt`).
2. **Opponent Sampling:** In each generation, parallel environments dynamically sample opponent models from the snapshot pool (or the current policy) to maintain diverse adversarial play.
3. **Showcase Trajectories:** Every `--showcase-interval` generations, a full match is automatically played out and saved to `--replays-dir` (`gen_XXXX_<timestamp>.npz`) for visual telemetry inspection.

### Archiving Training Runs

To archive checkpoints and generated replays without committing large binary files to Git, run the archive utility:

```bash
# Archive all current checkpoints and replays into a timestamped compressed bundle
python archive.py
```

This:
1. Moves all generated `.pt` models and `.npz` replays into a timestamped folder in `./archives/archive_YYYYMMDD_HHMMSS/`.
2. Creates a compressed `archive_YYYYMMDD_HHMMSS.tar.gz` archive.
3. Keeps baseline sample fixtures (`test_match.npz` and `.gitkeep`) intact in `./replays` and `./checkpoints`.
4. Leaves the archive directory untracked by Git (enforced via `.gitignore`).

You can also archive automatically before starting a new training session:

```bash
python train.py --archive-first --num-envs 8 --total-generations 500
```

Or archive programmatically from Python:

```python
from bomberman_rl.training import archive_training_data

result = archive_training_data(
    checkpoint_dir="./checkpoints",
    replays_dir="./replays",
    archive_dir="./archives",
)
print(f"Archived {result['files_count']} files to {result['archive_tar']}")
```

---

## Interactive Visualizer & Replay Player

The visualizer provides a 1280x760 telemetry dashboard with live arena rendering, policy probability bars, $V(s)$ value critic history, and timeline scrubbing.

### Viewing Replays

To instantly play the most recent replay in `./replays/` without specifying arguments:

```bash
python replay.py
```

List all recorded replays with timestamps and file sizes:

```bash
python replay.py --list
```

Or view a specific replay trajectory:

```bash
python replay.py ./replays/test_match.npz
```

You can also use the full module interface with custom options:

```bash
python -m bomberman_rl.visualizer.play_interactive --mode replay --replay ./replays/test_match.npz
```

### Live Agent vs Agent

Run a live simulation between heuristic agents with real-time telemetry:

```bash
python -m bomberman_rl.visualizer.play_interactive --mode agent_vs_agent
```

Record the live match directly to an `.npz` file for later review:

```bash
python -m bomberman_rl.visualizer.play_interactive --mode agent_vs_agent --record ./replays/my_match.npz
```

### Human vs Agent

Play interactively against an agent:

```bash
python -m bomberman_rl.visualizer.play_interactive --mode human_vs_agent
```

### Interactive Controls & Shortcuts

#### Movement & Combat (Human Mode)
- `W` / `Up Arrow`: Move Up
- `S` / `Down Arrow`: Move Down
- `A` / `Left Arrow`: Move Left
- `D` / `Right Arrow`: Move Right
- `Space`: Place Bomb

#### Playback & Forensic Controls
- `Space`: Pause / Resume playback
- `Left Arrow` (when paused): Step backward 1 tick
- `Right Arrow` (when paused): Step forward 1 tick
- `Up Arrow`: Increase playback speed (0.25x -> 0.5x -> 1.0x -> 2.0x -> 4.0x)
- `Down Arrow`: Decrease playback speed
- `Mouse Click / Drag` on bottom timeline: Seek directly to any match tick
- `Esc`: Exit player

#### Diagnostic Overlays
- `1`: Toggle Danger Heatmap (danger radius & blast projection overlay)
- `2`: Toggle Reachable Safe Space (BFS flood-fill connectivity)
- `3`: Toggle Policy Direction Arrow (visual indicator of predicted move)
- `4`: Clear all diagnostic overlays

### Visualizer CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `choice` | `agent_vs_agent` | Playback mode: `replay`, `agent_vs_agent`, or `human_vs_agent`. |
| `--replay` | `str` | `None` | Path to `.npz` replay file to load (required in `replay` mode). |
| `--record` | `str` | `None` | Path to export recorded match trajectory. |
| `--speed` | `float` | `1.0` | Initial playback speed multiplier. |
| `--seed` | `int` | `42` | Random seed for procedural arena generation. |
| `--max_ticks` | `int` | `800` | Maximum match duration before timeout. |
| `--headless` | flag | `False` | Run in headless mode without opening a Pygame display. |

---

## Testing & Verification

The test suite covers physics edge cases, sudden death mechanics, observation tensor correctness, reward shaping invariants, replay bit-exact roundtrips, and multi-process training rollouts.

Run the full test suite:

```bash
pytest
```

Run specific test modules:

```bash
# Engine physics and edge cases
pytest tests/test_engine_edge_cases.py

# Observation features, danger maps, and reward calculation
pytest tests/test_rl_observations_and_rewards.py

# Replay serialization, bit-exact roundtrip, and renderer HUD
pytest tests/test_replay_and_renderer.py

# Multi-process vectorization, network forward pass, and PPO training loop
pytest tests/test_training_pipeline.py
```
