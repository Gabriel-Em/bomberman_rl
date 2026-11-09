"""Actor-Critic Neural Network Architecture for 1v1 Bomberman.

Residual CNN for spatial grid representations fused with scalar state embeddings
into MultiDiscrete action distributions (movement: 5, bomb: 2) and scalar value function.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActionEntropy:
    """Encapsulates movement and bomb entropy with custom head weighting."""

    def __init__(self, move_entropy: torch.Tensor, bomb_entropy: torch.Tensor) -> None:
        self.move = move_entropy
        self.bomb = bomb_entropy

    def weighted_loss(self, coef_move: float = 0.005, coef_bomb: float = 0.01) -> torch.Tensor:
        return -(coef_move * self.move.mean() + coef_bomb * self.bomb.mean())

    def mean(self) -> torch.Tensor:
        return (self.move + self.bomb).mean()

    def __add__(self, other: Any) -> torch.Tensor:
        return (self.move + self.bomb) + other

    def __radd__(self, other: Any) -> torch.Tensor:
        return other + (self.move + self.bomb)

    def __mul__(self, other: Any) -> torch.Tensor:
        return (self.move + self.bomb) * other

    def __rmul__(self, other: Any) -> torch.Tensor:
        return other * (self.move + self.bomb)

    def __neg__(self) -> torch.Tensor:
        return -(self.move + self.bomb)

    def __sub__(self, other: Any) -> torch.Tensor:
        return (self.move + self.bomb) - other

    def __rsub__(self, other: Any) -> torch.Tensor:
        return other - (self.move + self.bomb)


class ResidualBlock(nn.Module):
    """Residual convolutional block with 3x3 convs, GroupNorm, and ReLU."""

    def __init__(self, channels: int = 64, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ActorCriticNetwork(nn.Module):
    """Residual CNN + Scalar MLP Fusion Actor-Critic Network.

    Processes 18-channel 9x9 spatial grid features through 3x3 residual blocks
    and 18-dim normalized scalar stats, fused into movement and bomb action heads
    and a state-value critic head.
    """

    def __init__(
        self,
        spatial_channels: int = 18,
        scalar_dim: int = 18,
        num_res_blocks: int = 2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.spatial_channels = spatial_channels
        self.scalar_dim = scalar_dim

        # Spatial Trunk
        self.conv_in = nn.Conv2d(spatial_channels, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        # Spatial reduction via 1x1 conv to 16 channels
        self.conv_1x1 = nn.Conv2d(64, 16, kernel_size=1)
        self.spatial_flatten = nn.Flatten()
        # 16 * 9 * 9 = 1296 features

        # Scalar Trunk
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion Core
        self.fusion_fc1 = nn.Linear(1296 + 64, hidden_dim)  # 1360 -> 256
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim)  # 256 -> 256

        # Heads
        self.movement_head = nn.Linear(hidden_dim, 5)  # NOOP, UP, DOWN, LEFT, RIGHT
        self.bomb_head = nn.Linear(hidden_dim, 2)      # NO_BOMB, PLACE_BOMB
        self.critic_head = nn.Linear(hidden_dim, 1)    # V(s)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output heads specific gain initialization
        nn.init.orthogonal_(self.movement_head.weight, gain=0.01)
        nn.init.orthogonal_(self.bomb_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

        # Bomb head bias initialization: [1.0, -1.0] (~73% NO_BOMB, ~27% PLACE_BOMB)
        with torch.no_grad():
            self.bomb_head.bias.copy_(torch.tensor([1.0, -1.0]))

    @staticmethod
    def _detect_near_brick_safe(spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Detect if player is on a safe tile adjacent to breakable brick with 0 active bombs."""
        B = spatial.shape[0]
        device = spatial.device
        if spatial.shape[1] < 16 or scalars.shape[1] < 5:
            return torch.zeros(B, dtype=torch.bool, device=device)

        zero_bombs = (scalars[:, 4] < 1e-4)
        player_pos = spatial[:, 15]

        # Danger at player position
        if spatial.shape[1] > 8:
            danger = spatial[:, 8]
            safe_tile = ((danger * player_pos).sum(dim=(-2, -1)) < 1e-4)
        else:
            safe_tile = torch.ones(B, dtype=torch.bool, device=device)

        # Adjacent breakable brick
        brick = spatial[:, 2]
        padded = F.pad(brick, (1, 1, 1, 1), mode="constant", value=0.0)
        adj_bricks = (
            padded[:, :-2, 1:-1]
            + padded[:, 2:, 1:-1]
            + padded[:, 1:-1, :-2]
            + padded[:, 1:-1, 2:]
        )
        has_adj_brick = ((adj_bricks * player_pos).sum(dim=(-2, -1)) > 0.5)

        return zero_bombs & safe_tile & has_adj_brick

    def extract_features(self, spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Fuse spatial convolutional representation and scalar embedding."""
        x_spatial = F.relu(self.conv_in(spatial))
        x_spatial = self.res_blocks(x_spatial)
        x_spatial = F.relu(self.conv_1x1(x_spatial))
        flat_spatial = self.spatial_flatten(x_spatial)

        flat_scalar = self.scalar_mlp(scalars)

        fused = torch.cat([flat_spatial, flat_scalar], dim=-1)
        h = F.relu(self.fusion_fc1(fused))
        h = F.relu(self.fusion_fc2(h))
        return h

    def forward(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute unmasked movement logits, bomb logits, and critic state value."""
        h = self.extract_features(spatial, scalars)
        move_logits = self.movement_head(h)
        bomb_logits = self.bomb_head(h)
        values = self.critic_head(h).squeeze(-1)
        return move_logits, bomb_logits, values

    def get_action(
        self,
        spatial: torch.Tensor | np.ndarray,
        scalars: torch.Tensor | np.ndarray,
        move_mask: torch.Tensor | np.ndarray | None = None,
        bomb_mask: torch.Tensor | np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Sample actions under optional action masks and return probabilities and values."""
        device = self.movement_head.weight.device

        if not isinstance(spatial, torch.Tensor):
            spatial = torch.as_tensor(spatial, dtype=torch.float32, device=device)
        else:
            spatial = spatial.to(device=device, dtype=torch.float32)

        if not isinstance(scalars, torch.Tensor):
            scalars = torch.as_tensor(scalars, dtype=torch.float32, device=device)
        else:
            scalars = scalars.to(device=device, dtype=torch.float32)

        if spatial.dim() == 3:
            spatial = spatial.unsqueeze(0)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        move_logits, bomb_logits, values = self.forward(spatial, scalars)

        if move_mask is not None:
            if not isinstance(move_mask, torch.Tensor):
                move_mask = torch.as_tensor(move_mask, dtype=torch.bool, device=device)
            else:
                move_mask = move_mask.to(device=device, dtype=torch.bool)
            if move_mask.dim() == 1:
                move_mask = move_mask.unsqueeze(0)
            move_logits = move_logits.masked_fill(~move_mask, -1e8)

        if bomb_mask is not None:
            if not isinstance(bomb_mask, torch.Tensor):
                bomb_mask = torch.as_tensor(bomb_mask, dtype=torch.bool, device=device)
            else:
                bomb_mask = bomb_mask.to(device=device, dtype=torch.bool)
            if bomb_mask.dim() == 1:
                bomb_mask = bomb_mask.unsqueeze(0)

            # Minimum Bomb Exploration Floor:
            # Set exploration logit floor to -4.0 (~1.8%/tick) normally, and -2.2 (~10%/tick)
            # specifically when player is on a safe tile adjacent to a breakable brick with active_bombs == 0.
            can_bomb = bomb_mask[..., 1]
            if can_bomb.any():
                near_brick_safe = self._detect_near_brick_safe(spatial, scalars)
                floor_offset = torch.where(near_brick_safe, -2.2, -4.0)
                floor_1 = bomb_logits[..., 0] + floor_offset
                bomb_logits_1 = torch.where(can_bomb, torch.maximum(bomb_logits[..., 1], floor_1), bomb_logits[..., 1])
                bomb_logits = torch.stack([bomb_logits[..., 0], bomb_logits_1], dim=-1)

            bomb_logits = bomb_logits.masked_fill(~bomb_mask, -1e8)

        move_dist = Categorical(logits=move_logits)
        bomb_dist = Categorical(logits=bomb_logits)

        if deterministic:
            move_act = torch.argmax(move_logits, dim=-1)
            bomb_act = torch.argmax(bomb_logits, dim=-1)
        else:
            move_act = move_dist.sample()
            bomb_act = bomb_dist.sample()

        log_probs = move_dist.log_prob(move_act) + bomb_dist.log_prob(bomb_act)
        actions = torch.stack([move_act, bomb_act], dim=-1)

        action_probs = {
            "move": move_dist.probs,
            "bomb": bomb_dist.probs,
        }

        return actions, log_probs, values, action_probs

    def evaluate_actions(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        move_mask: torch.Tensor | None = None,
        bomb_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action log probabilities, entropy, and state values for PPO updates."""
        device = self.movement_head.weight.device
        spatial = spatial.to(device=device, dtype=torch.float32)
        scalars = scalars.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device)

        move_logits, bomb_logits, values = self.forward(spatial, scalars)

        if move_mask is not None:
            move_mask = move_mask.to(device=device, dtype=torch.bool)
            if move_mask.dim() == 1:
                move_mask = move_mask.unsqueeze(0)
            move_logits = move_logits.masked_fill(~move_mask, -1e8)

        if bomb_mask is not None:
            bomb_mask = bomb_mask.to(device=device, dtype=torch.bool)
            if bomb_mask.dim() == 1:
                bomb_mask = bomb_mask.unsqueeze(0)

            # Minimum Bomb Exploration Floor
            can_bomb = bomb_mask[..., 1]
            if can_bomb.any():
                near_brick_safe = self._detect_near_brick_safe(spatial, scalars)
                floor_offset = torch.where(near_brick_safe, -2.2, -4.0)
                floor_1 = bomb_logits[..., 0] + floor_offset
                bomb_logits_1 = torch.where(can_bomb, torch.maximum(bomb_logits[..., 1], floor_1), bomb_logits[..., 1])
                bomb_logits = torch.stack([bomb_logits[..., 0], bomb_logits_1], dim=-1)

            bomb_logits = bomb_logits.masked_fill(~bomb_mask, -1e8)

        move_dist = Categorical(logits=move_logits)
        bomb_dist = Categorical(logits=bomb_logits)

        move_act = actions[..., 0]
        bomb_act = actions[..., 1]

        log_probs = move_dist.log_prob(move_act) + bomb_dist.log_prob(bomb_act)
        entropy = ActionEntropy(move_dist.entropy(), bomb_dist.entropy())

        return log_probs, entropy, values

    def get_value(
        self,
        spatial: torch.Tensor | np.ndarray,
        scalars: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Compute scalar state values."""
        device = self.critic_head.weight.device
        if not isinstance(spatial, torch.Tensor):
            spatial = torch.as_tensor(spatial, dtype=torch.float32, device=device)
        else:
            spatial = spatial.to(device=device, dtype=torch.float32)

        if not isinstance(scalars, torch.Tensor):
            scalars = torch.as_tensor(scalars, dtype=torch.float32, device=device)
        else:
            scalars = scalars.to(device=device, dtype=torch.float32)

        if spatial.dim() == 3:
            spatial = spatial.unsqueeze(0)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        h = self.extract_features(spatial, scalars)
        return self.critic_head(h).squeeze(-1)
