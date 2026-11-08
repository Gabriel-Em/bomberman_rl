"""Reinforcement Learning infrastructure for 1v1 Bomberman."""

from bomberman_rl.rl.observation import (
    SPATIAL_CHANNELS,
    DEFAULT_SCALAR_DIM,
    FULL_SCALAR_DIM,
    ObservationBuilder,
    compute_danger_map,
    compute_reachable_safe_tiles,
    flip_action_to_world,
    flip_action_to_ego,
)
from bomberman_rl.rl.rewards import (
    CurriculumLethalityScheduler,
    RewardTracker,
    PlayerRewardLog,
    compute_pbrs_potential,
    check_players_connected,
    BRICK_REWARD,
    CONNECTIVITY_REWARD,
    AUXILIARY_REWARD_CAP,
    TERMINAL_WIN,
    TERMINAL_DRAW,
)
from bomberman_rl.rl.action_filter import (
    SafetyActionFilter,
    ActionFilter,
    is_cul_de_sac,
)

__all__ = [
    "SPATIAL_CHANNELS",
    "DEFAULT_SCALAR_DIM",
    "FULL_SCALAR_DIM",
    "ObservationBuilder",
    "compute_danger_map",
    "compute_reachable_safe_tiles",
    "flip_action_to_world",
    "flip_action_to_ego",
    "CurriculumLethalityScheduler",
    "RewardTracker",
    "PlayerRewardLog",
    "compute_pbrs_potential",
    "check_players_connected",
    "BRICK_REWARD",
    "CONNECTIVITY_REWARD",
    "AUXILIARY_REWARD_CAP",
    "TERMINAL_WIN",
    "TERMINAL_DRAW",
    "SafetyActionFilter",
    "ActionFilter",
    "is_cul_de_sac",
]
