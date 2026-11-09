"""Vectorized Training and Reinforcement Learning Pipeline for 1v1 Bomberman."""

from bomberman_rl.training.network import (
    ActorCriticNetwork,
    ResidualBlock,
)
from bomberman_rl.training.vec_env import (
    SyncVectorEnv,
    AsyncVectorEnv,
    BombermanMatchEnv,
    HeuristicOpponent,
)
from bomberman_rl.training.diagnostics import (
    BehaviorDiagnostics,
    MatchStats,
    compute_interest_score,
)
from bomberman_rl.training.ppo_trainer import (
    PPOTrainer,
    get_curriculum,
    get_competence_curriculum,
    CompetenceCurriculum,
)
from bomberman_rl.training.archive import archive_training_data, clean_training_data

__all__ = [
    "ActorCriticNetwork",
    "ResidualBlock",
    "SyncVectorEnv",
    "AsyncVectorEnv",
    "BombermanMatchEnv",
    "HeuristicOpponent",
    "BehaviorDiagnostics",
    "MatchStats",
    "compute_interest_score",
    "PPOTrainer",
    "get_curriculum",
    "get_competence_curriculum",
    "CompetenceCurriculum",
    "archive_training_data",
    "clean_training_data",
]
