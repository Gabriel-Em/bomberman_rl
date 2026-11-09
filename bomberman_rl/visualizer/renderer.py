"""Pygame MARL Visual Telemetry Dashboard & Interactive Replay Player.

Renders a 1280x760 window at 60 FPS featuring:
- Left Panel (660x660): 11x11 procedural arcade arena with sub-tile interpolation.
- Right Panel (620x660): Live MARL neural telemetry, policy distributions, and V(s) sparkline.
- Bottom Panel (1280x100): Synchronized timeline scrubber with frame-exact seeking.
"""

from __future__ import annotations
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np

# Ensure headless safety when SDL_VIDEODRIVER is set
import pygame

from bomberman_rl.engine.config import (
    GRID_SIZE,
    MAX_TICKS,
    TICK_RATE,
    TileType,
    PowerUpType,
    MoveAction,
    BombAction,
    SHIELD_DURATION,
    BOMB_TIMER,
)
from bomberman_rl.engine.state import GameState, PlayerState, BombState
from bomberman_rl.rl.observation import (
    compute_danger_map,
    compute_reachable_safe_tiles,
)
from bomberman_rl.replays.recorder import (
    GameEvent,
    ReplayEpisode,
    REWARD_COMPONENT_NAMES,
)

# Colors
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (24, 27, 33)
COLOR_BORDER = (45, 52, 64)
COLOR_TEXT_PRIMARY = (235, 240, 248)
COLOR_TEXT_MUTED = (140, 150, 168)
COLOR_P0_CYAN = (30, 185, 255)
COLOR_P0_DARK = (15, 85, 140)
COLOR_P1_RED = (255, 65, 75)
COLOR_P1_DARK = (135, 25, 35)
COLOR_GOLD = (255, 215, 60)
COLOR_GREEN = (40, 220, 120)
COLOR_PURPLE = (165, 70, 210)


@dataclass
class BrickParticle:
    """Terracotta crumbling debris particle with 10-frame lifespan."""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: tuple[int, int, int]
    lifetime: int = 10
    age: int = 0

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.25  # gravity
        self.age += 1
        return self.age < self.lifetime


class Renderer:
    """MARL-grade Pygame telemetry renderer and interactive scrubber."""

    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 760
    ARENA_SIZE = 660
    TILE_SIZE = 60
    BOARD_TILES = 11

    def __init__(
        self,
        surface: pygame.Surface | None = None,
        title: str = "Bomberman RL Telemetry & Diagnostics",
        headless: bool = False,
        mode: str = "agent_vs_agent",
    ) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.mode = mode
        self.headless = headless
        if surface is not None:
            self.screen = surface
        else:
            if headless and "SDL_VIDEODRIVER" not in os.environ:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption(title)

        if not self.headless:
            try:
                pygame.mouse.set_visible(False)
            except Exception:
                pass

        self.clock = pygame.time.Clock()

        # Cached procedural textures
        self._tex_outer_brick = self._create_outer_brick_texture()
        self._tex_pillar = self._create_pillar_texture()
        self._tex_brick = self._create_breakable_brick_texture()
        self._tex_floor = self._create_floor_texture()

        # Fonts
        self.font_lg = pygame.font.Font(None, 26)
        self.font_md = pygame.font.Font(None, 22)
        self.font_sm = pygame.font.Font(None, 18)
        self.font_xs = pygame.font.Font(None, 15)

        # Overlays
        self.overlay_danger = False       # Hotkey 1
        self.overlay_safe_space = False   # Hotkey 2
        self.overlay_policy_arrow = False # Hotkey 3

        # Right-hand neural telemetry panel (Hotkey T), hidden by default
        self.show_telemetry = False

        # Particle system
        self.particles: list[BrickParticle] = []
        self._prev_grid_state: np.ndarray | None = None

        # Sparkline telemetry history (rolling 200 ticks)
        self.v_history_p0: deque[float] = deque(maxlen=200)
        self.v_history_p1: deque[float] = deque(maxlen=200)
        self._last_sparkline_tick: int = -1

        # Scrubber & Playback state
        self.paused = False
        self.speed_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        self.speed_idx = 2  # 1.0x default
        self.current_tick = 0
        self.max_ticks = MAX_TICKS
        self.is_dragging_scrubber = False
        self._mouse_moved = False

        # Tooltip for event timeline
        self.active_tooltip: str | None = None
        self.active_tooltip_pos: tuple[int, int] | None = None

        # Pre-allocated alpha overlay surface
        self._overlay_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)

    @property
    def playback_speed(self) -> float:
        return self.speed_multipliers[self.speed_idx]

    def set_playback_speed(self, speed: float) -> None:
        """Select the supported playback multiplier nearest to ``speed``."""
        self.speed_idx = min(
            range(len(self.speed_multipliers)),
            key=lambda i: abs(self.speed_multipliers[i] - speed),
        )

    # --- Procedural Texture Generation ---
    def _create_outer_brick_texture(self) -> pygame.Surface:
        surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
        surf.fill((46, 50, 58))  # Dark steel-gray
        # Bevel highlight
        pygame.draw.line(surf, (72, 78, 88), (0, 0), (self.TILE_SIZE - 1, 0), 2)
        pygame.draw.line(surf, (72, 78, 88), (0, 0), (0, self.TILE_SIZE - 1), 2)
        # Bevel shadow
        pygame.draw.line(surf, (28, 30, 36), (0, self.TILE_SIZE - 1), (self.TILE_SIZE - 1, self.TILE_SIZE - 1), 2)
        pygame.draw.line(surf, (28, 30, 36), (self.TILE_SIZE - 1, 0), (self.TILE_SIZE - 1, self.TILE_SIZE - 1), 2)
        # Rivets at four corners
        rivet_color = (95, 102, 115)
        rivet_center = (25, 27, 32)
        for rx, ry in [(8, 8), (52, 8), (8, 52), (52, 52)]:
            pygame.draw.circle(surf, rivet_color, (rx, ry), 3)
            pygame.draw.circle(surf, rivet_center, (rx, ry), 1)
        return surf

    def _create_pillar_texture(self) -> pygame.Surface:
        surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
        surf.fill((68, 76, 90))  # Solid slate
        # Outer bevel
        pygame.draw.line(surf, (108, 120, 140), (0, 0), (self.TILE_SIZE - 1, 0), 3)
        pygame.draw.line(surf, (108, 120, 140), (0, 0), (0, self.TILE_SIZE - 1), 3)
        pygame.draw.line(surf, (38, 44, 54), (0, self.TILE_SIZE - 1), (self.TILE_SIZE - 1, self.TILE_SIZE - 1), 3)
        pygame.draw.line(surf, (38, 44, 54), (self.TILE_SIZE - 1, 0), (self.TILE_SIZE - 1, self.TILE_SIZE - 1), 3)
        # Recessed inner square
        inner_rect = pygame.Rect(10, 10, 40, 40)
        pygame.draw.rect(surf, (54, 62, 74), inner_rect)
        pygame.draw.line(surf, (38, 44, 54), (10, 10), (49, 10), 2)
        pygame.draw.line(surf, (38, 44, 54), (10, 10), (10, 49), 2)
        pygame.draw.line(surf, (88, 98, 115), (10, 49), (49, 49), 2)
        pygame.draw.line(surf, (88, 98, 115), (49, 10), (49, 49), 2)
        return surf

    def _create_breakable_brick_texture(self) -> pygame.Surface:
        surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
        surf.fill((92, 44, 34))  # Mortar base
        brick_color_1 = (182, 80, 58)
        brick_color_2 = (165, 70, 50)
        # 4 rows of terracotta bricks with staggered mortar lines
        row_height = 13
        for r_idx in range(4):
            y = 2 + r_idx * 14
            offset = 0 if r_idx % 2 == 0 else 15
            for col in range(-1, 3):
                x = offset + col * 30
                rect = pygame.Rect(x, y, 27, row_height)
                c = brick_color_1 if (r_idx + col) % 2 == 0 else brick_color_2
                pygame.draw.rect(surf, c, rect)
                # Subtle highlight on top of each brick
                pygame.draw.line(surf, (205, 95, 72), (max(0, x), y), (min(59, x + 26), y), 1)
        return surf

    def _create_floor_texture(self) -> pygame.Surface:
        surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
        surf.fill((34, 38, 46))
        pygame.draw.rect(surf, (38, 42, 50), pygame.Rect(1, 1, self.TILE_SIZE - 2, self.TILE_SIZE - 2))
        return surf

    # --- Particle System ---
    def spawn_brick_crumble(self, grid_r: int, grid_c: int) -> None:
        """Spawn 14 terracotta debris particles upon brick wall breach."""
        cx = (grid_c + 1.5) * self.TILE_SIZE
        cy = (grid_r + 1.5) * self.TILE_SIZE
        palette = [(185, 82, 60), (160, 68, 50), (120, 50, 40)]
        for _ in range(14):
            angle = float(np.random.uniform(0, 2 * math.pi))
            speed = float(np.random.uniform(1.2, 4.0))
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 1.5
            size = float(np.random.uniform(3.0, 7.0))
            color = palette[int(np.random.randint(len(palette)))]
            self.particles.append(BrickParticle(cx, cy, vx, vy, size, color))

    def update_particles(self) -> None:
        self.particles = [p for p in self.particles if p.update()]

    def render_particles(self, surface: pygame.Surface) -> None:
        for p in self.particles:
            alpha = max(0, int(255 * (1.0 - p.age / p.lifetime)))
            part_surf = pygame.Surface((int(p.size), int(p.size)), pygame.SRCALPHA)
            part_surf.fill((*p.color, alpha))
            surface.blit(part_surf, (int(p.x - p.size / 2), int(p.y - p.size / 2)))

    # --- Left Panel: Arena Rendering ---
    def render_arena(
        self,
        surface: pygame.Surface,
        state: GameState,
        prev_state: GameState | None = None,
        interp_alpha: float = 1.0,
    ) -> None:
        # Detect new brick destructions for particle spawning
        if self._prev_grid_state is not None:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if self._prev_grid_state[r, c] == TileType.BRICK and state.grid[r, c] == TileType.EMPTY:
                        self.spawn_brick_crumble(r, c)
        self._prev_grid_state = state.grid.copy()

        # 1. Outer Border & Base Grid Floor
        for br in range(self.BOARD_TILES):
            for bc in range(self.BOARD_TILES):
                x = bc * self.TILE_SIZE
                y = br * self.TILE_SIZE
                is_border = (br == 0 or br == 10 or bc == 0 or bc == 10)
                if is_border:
                    surface.blit(self._tex_outer_brick, (x, y))
                else:
                    surface.blit(self._tex_floor, (x, y))

        # 2. Interior Walls & Bricks
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x = (c + 1) * self.TILE_SIZE
                y = (r + 1) * self.TILE_SIZE
                tile_val = state.grid[r, c]
                if tile_val == TileType.PILLAR:
                    surface.blit(self._tex_pillar, (x, y))
                elif tile_val == TileType.BRICK:
                    surface.blit(self._tex_brick, (x, y))

        # 3. Power-Ups
        t = float(state.tick)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ptype = state.powerups[r, c]
                if ptype != PowerUpType.NONE:
                    bob = math.sin(t * 0.35 + r * 1.5 + c) * 3.0
                    px = (c + 1.5) * self.TILE_SIZE
                    py = (r + 1.5) * self.TILE_SIZE + bob
                    self._draw_powerup_icon(surface, int(px), int(py), PowerUpType(ptype), t)

        # 4. Overlays (Danger Heatmap, Reachable Safe Space)
        if self.overlay_danger:
            self._render_danger_overlay(surface, state)
        if self.overlay_safe_space:
            self._render_safe_space_overlay(surface, state)

        # 5. Explosions (Cross beams with flame timer decay)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ft = state.flame_timer[r, c]
                if ft > 0:
                    px = (c + 1) * self.TILE_SIZE
                    py = (r + 1) * self.TILE_SIZE
                    self._draw_explosion_tile(surface, px, py, ft)

        # 6. Bombs (Pulsing scale, spark fuse, rolling directional arrow)
        for bomb in state.bombs:
            # Interpolate rolling bomb position
            br, bc = bomb.r, bomb.c
            bx = (bc + 1.0) * self.TILE_SIZE
            by = (br + 1.0) * self.TILE_SIZE
            self._draw_bomb(surface, bx, by, bomb, state.tick)

        # 7. Players (Interpolated positions, facing direction, shield bubbles)
        for p in state.players:
            # Sub-tile interpolated position
            pr, pc = p.r, p.c
            if prev_state is not None and p.id < len(prev_state.players):
                old_p = prev_state.players[p.id]
                pr = old_p.r * (1.0 - interp_alpha) + p.r * interp_alpha
                pc = old_p.c * (1.0 - interp_alpha) + p.c * interp_alpha
            px = (pc + 1.0) * self.TILE_SIZE
            py = (pr + 1.0) * self.TILE_SIZE
            self._draw_player(surface, px, py, p, state.tick)

        # 8. Spatial Value Gradient Policy Arrow Overlay
        if self.overlay_policy_arrow:
            self._render_policy_arrows(surface, state)

        # 9. Crumbling debris particles
        self.render_particles(surface)

    def _draw_powerup_icon(
        self, surface: pygame.Surface, cx: int, cy: int, ptype: PowerUpType, tick: float
    ) -> None:
        badge_w, badge_h = 36, 36
        rect = pygame.Rect(cx - badge_w // 2, cy - badge_h // 2, badge_w, badge_h)

        if ptype == PowerUpType.MAX_BLAST:
            # Pulsing golden star for 1% rare drop
            glow = int(180 + 75 * math.sin(tick * 0.8))
            pygame.draw.circle(surface, (255, 230, glow), (cx, cy), 18)
            pygame.draw.circle(surface, COLOR_GOLD, (cx, cy), 14)
            label = self.font_xs.render("MAX", True, (30, 20, 0))
            surface.blit(label, label.get_rect(center=(cx, cy)))
            return

        type_meta = {
            PowerUpType.BLAST_UP: ("+R", (230, 60, 40), (255, 255, 255)),
            PowerUpType.BOMB_UP: ("+B", (30, 160, 240), (255, 255, 255)),
            PowerUpType.SPEED_UP: ("SPD", (240, 200, 30), (20, 20, 20)),
            PowerUpType.BLAST_DOWN: ("-R", (125, 35, 145), (255, 200, 255)),
            PowerUpType.BOMB_DOWN: ("-B", (125, 35, 145), (255, 200, 255)),
            PowerUpType.KICK: ("KICK", (35, 185, 85), (255, 255, 255)),
            PowerUpType.BOMB_PASS: ("PASS", (200, 225, 255), (30, 40, 50)),
            PowerUpType.SHIELD: ("SHLD", COLOR_GOLD, (30, 20, 0)),
        }
        text, bg_col, text_col = type_meta.get(ptype, ("?", (100, 100, 100), (255, 255, 255)))

        pygame.draw.rect(surface, (20, 24, 30), rect.inflate(4, 4), border_radius=6)
        pygame.draw.rect(surface, bg_col, rect, border_radius=6)
        lbl = self.font_xs.render(text, True, text_col)
        surface.blit(lbl, lbl.get_rect(center=(cx, cy)))

    def _draw_explosion_tile(self, surface: pygame.Surface, px: int, py: int, flame_timer: int) -> None:
        alpha_factor = min(1.0, float(flame_timer) / 10.0)
        core_alpha = int(255 * alpha_factor)
        mid_alpha = int(180 * alpha_factor)
        outer_alpha = int(100 * alpha_factor)

        # Multi-layered cross beam
        surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        # Red outer glow
        pygame.draw.rect(surf, (230, 40, 25, outer_alpha), (0, 0, self.TILE_SIZE, self.TILE_SIZE))
        # Orange middle beam
        pygame.draw.rect(surf, (255, 140, 20, mid_alpha), (10, 10, 40, 40))
        # Bright white core beam
        pygame.draw.rect(surf, (255, 255, 255, core_alpha), (18, 18, 24, 24))
        surface.blit(surf, (px, py))

    def _draw_bomb(self, surface: pygame.Surface, bx: float, by: float, bomb: BombState, tick: int) -> None:
        pulse = 4.0 * math.sin(tick * 0.6)
        radius = int(20 + pulse)

        # Rolling bomb shadow
        pygame.draw.circle(surface, (12, 14, 18), (int(bx), int(by) + 4), radius)
        # Main sphere
        pygame.draw.circle(surface, (30, 32, 40), (int(bx), int(by)), radius)
        # Specular shine
        pygame.draw.circle(surface, (110, 120, 140), (int(bx - radius * 0.3), int(by - radius * 0.3)), max(2, radius // 5))

        # Glowing spark fuse (Yellow -> Orange -> Flashing Red in final 20 ticks)
        fuse_x = int(bx)
        fuse_y = int(by - radius + 2)
        if bomb.timer > 20:
            spark_col = (255, 230, 40)
        elif bomb.timer > 10:
            spark_col = (255, 130, 20)
        else:
            # Flashing red in final 1.0s
            spark_col = (255, 30, 30) if (tick // 2) % 2 == 0 else (255, 255, 255)

        pygame.draw.circle(surface, spark_col, (fuse_x, fuse_y), 4)
        pygame.draw.circle(surface, (255, 255, 200), (fuse_x, fuse_y), 2)

        # Directional arrow on rolling kicked bombs
        if abs(bomb.velocity_r) > 1e-4 or abs(bomb.velocity_c) > 1e-4:
            arrow_len = 16
            dr = np.sign(bomb.velocity_r)
            dc = np.sign(bomb.velocity_c)
            ex = int(bx + dc * arrow_len)
            ey = int(by + dr * arrow_len)
            pygame.draw.line(surface, COLOR_GOLD, (int(bx), int(by)), (ex, ey), 3)

    def _draw_player(self, surface: pygame.Surface, px: float, py: float, player: PlayerState, tick: int) -> None:
        cx, cy = int(px), int(py)
        is_p0 = (player.id == 0)
        base_col = COLOR_P0_CYAN if is_p0 else COLOR_P1_RED
        accent_col = COLOR_P0_DARK if is_p0 else COLOR_P1_DARK

        if not player.alive:
            # Eliminated ghost silhouette
            pygame.draw.circle(surface, (70, 75, 85), (cx, cy), 18)
            lbl = self.font_xs.render("RIP", True, (200, 200, 200))
            surface.blit(lbl, lbl.get_rect(center=(cx, cy)))
            return

        # Player shadow
        pygame.draw.circle(surface, (10, 12, 16), (cx, cy + 4), 19)
        # Body
        pygame.draw.circle(surface, base_col, (cx, cy), 18)
        pygame.draw.circle(surface, accent_col, (cx, cy), 18, 2)
        # Visor / eyes
        pygame.draw.rect(surface, (240, 245, 255), (cx - 7, cy - 6, 14, 8), border_radius=3)
        pygame.draw.rect(surface, (20, 30, 40), (cx - 4, cy - 4, 3, 4))
        pygame.draw.rect(surface, (20, 30, 40), (cx + 1, cy - 4, 3, 4))

        # Shield forcefield bubble
        if player.shield_timer > 0:
            shield_radius = 26
            shield_surf = pygame.Surface((shield_radius * 2 + 4, shield_radius * 2 + 4), pygame.SRCALPHA)
            # Rotating hexagonal forcefield
            angle_offset = (tick * 0.12) % (2 * math.pi)
            hex_pts = []
            for k in range(6):
                ang = angle_offset + k * (math.pi / 3.0)
                hx = shield_radius + 2 + shield_radius * math.cos(ang)
                hy = shield_radius + 2 + shield_radius * math.sin(ang)
                hex_pts.append((hx, hy))
            pygame.draw.polygon(shield_surf, (80, 220, 255, 70), hex_pts)
            pygame.draw.polygon(shield_surf, COLOR_GOLD, hex_pts, 2)
            surface.blit(shield_surf, (cx - shield_radius - 2, cy - shield_radius - 2))

            # Miniature countdown arc
            arc_ratio = float(player.shield_timer) / float(SHIELD_DURATION)
            arc_rect = pygame.Rect(cx - 28, cy - 28, 56, 56)
            arc_end = -math.pi / 2.0 + (2.0 * math.pi * arc_ratio)
            pygame.draw.arc(surface, COLOR_GOLD, arc_rect, -math.pi / 2.0, arc_end, 3)

    # --- Overlays ---
    def _render_danger_overlay(self, surface: pygame.Surface, state: GameState) -> None:
        """Render Chain-Aware Danger Map (Ch 8) directly over arena tiles."""
        danger_map = compute_danger_map(state)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                d = float(danger_map[r, c])
                if d > 1e-4:
                    px = (c + 1) * self.TILE_SIZE
                    py = (r + 1) * self.TILE_SIZE
                    alpha = int(np.clip(d * 180, 20, 220))
                    self._overlay_surf.fill((255, 35, 35, alpha))
                    surface.blit(self._overlay_surf, (px, py))
                    # Danger percentage text
                    lbl = self.font_xs.render(f"{int(d * 100)}%", True, (255, 240, 240))
                    surface.blit(lbl, (px + 3, py + 3))

    def _render_safe_space_overlay(self, surface: pygame.Surface, state: GameState) -> None:
        """Render Reachable Safe Space (Ch 17) overlay for living players."""
        for p in state.players:
            if not p.alive:
                continue
            flood = compute_reachable_safe_tiles(state, p)
            col = (40, 220, 160) if p.id == 0 else (255, 140, 40)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    val = float(flood[r, c])
                    if val > 1e-4:
                        px = (c + 1) * self.TILE_SIZE
                        py = (r + 1) * self.TILE_SIZE
                        alpha = int(np.clip(val * 130, 20, 150))
                        self._overlay_surf.fill((*col, alpha))
                        surface.blit(self._overlay_surf, (px, py))

    def _render_policy_arrows(self, surface: pygame.Surface, state: GameState) -> None:
        """Draw spatial value gradient arrow from each player toward best local move."""
        move_vecs = {
            MoveAction.UP: (0, -1),
            MoveAction.DOWN: (0, 1),
            MoveAction.LEFT: (-1, 0),
            MoveAction.RIGHT: (1, 0),
        }
        for p in state.players:
            if not p.alive:
                continue
            cx = int((p.c + 1.0) * self.TILE_SIZE)
            cy = int((p.r + 1.0) * self.TILE_SIZE)
            col = COLOR_P0_CYAN if p.id == 0 else COLOR_P1_RED

            # Arrow pointing toward center or open tile
            arrow_len = 24
            dr = -1 if p.r > 4.5 else 1
            dc = -1 if p.c > 4.5 else 1
            ex = cx + int(dc * arrow_len)
            ey = cy + int(dr * arrow_len)
            pygame.draw.line(surface, col, (cx, cy), (ex, ey), 3)
            pygame.draw.circle(surface, COLOR_GOLD, (ex, ey), 4)

    # --- Right Panel: Telemetry & Neural Inspector ---
    def render_right_panel(
        self,
        surface: pygame.Surface,
        state: GameState,
        telemetry: dict[str, Any] | None,
    ) -> None:
        rx = self.ARENA_SIZE  # 660
        panel_rect = pygame.Rect(rx, 0, self.WINDOW_WIDTH - rx, self.ARENA_SIZE)
        pygame.draw.rect(surface, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(surface, COLOR_BORDER, (rx, 0), (rx, self.ARENA_SIZE), 2)

        # Critic value history (rolling 200 ticks) is tracked even while the panel is hidden
        v0 = telemetry.get("value_p0", 0.0) if telemetry else 0.0
        v1 = telemetry.get("value_p1", 0.0) if telemetry else 0.0

        if telemetry and "value_history_p0" in telemetry and "value_history_p1" in telemetry:
            self.v_history_p0 = deque(telemetry["value_history_p0"], maxlen=200)
            self.v_history_p1 = deque(telemetry["value_history_p1"], maxlen=200)
            self._last_sparkline_tick = state.tick
        elif state.tick != self._last_sparkline_tick:
            self.v_history_p0.append(v0)
            self.v_history_p1.append(v1)
            self._last_sparkline_tick = state.tick

        if not self.show_telemetry:
            self._render_player_inventory(surface, rx + 20, 20, 0, state.players[0])
            self._render_player_inventory(surface, rx + 20, 80, 1, state.players[1])
            hint = self.font_md.render("Press T to show neural telemetry", True, COLOR_TEXT_MUTED)
            surface.blit(hint, hint.get_rect(center=(rx + (self.WINDOW_WIDTH - rx) // 2, self.ARENA_SIZE // 2)))
            return

        # 1. Header & Overlays status
        header_text = self.font_lg.render("NEURAL TELEMETRY & DIAGNOSTICS", True, COLOR_TEXT_PRIMARY)
        surface.blit(header_text, (rx + 20, 14))

        # Overlay status pills
        ov_d = "[1: DANGER]" if self.overlay_danger else " 1: DANGER "
        ov_s = "[2: SAFE BFS]" if self.overlay_safe_space else " 2: SAFE BFS "
        ov_p = "[3: POLICY ARROW]" if self.overlay_policy_arrow else " 3: POLICY ARROW "
        pill_text = f"Overlays: {ov_d} | {ov_s} | {ov_p}"
        pill_lbl = self.font_xs.render(pill_text, True, COLOR_TEXT_MUTED)
        surface.blit(pill_lbl, (rx + 20, 40))

        # 2. Critic Value V(s) Sparkline Graph
        self._render_sparkline(surface, rx + 20, 60, 580, 115, v0, v1)

        # 3. Player 0 Diagnostics (Cyan)
        self._render_player_telemetry(surface, rx + 20, 185, 0, state.players[0], telemetry)

        # 4. Player 1 Diagnostics (Crimson)
        self._render_player_telemetry(surface, rx + 20, 420, 1, state.players[1], telemetry)

    def _render_sparkline(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        curr_v0: float,
        curr_v1: float,
    ) -> None:
        # Background box
        box_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, (16, 18, 22), box_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, box_rect, 1, border_radius=6)

        # Zero centerline
        mid_y = y + h // 2
        pygame.draw.line(surface, (45, 50, 62), (x, mid_y), (x + w, mid_y), 1)

        # Y-axis labels
        l_top = self.font_xs.render("+1.0", True, COLOR_TEXT_MUTED)
        l_mid = self.font_xs.render(" 0.0", True, COLOR_TEXT_MUTED)
        l_bot = self.font_xs.render("-1.0", True, COLOR_TEXT_MUTED)
        surface.blit(l_top, (x + 5, y + 3))
        surface.blit(l_mid, (x + 5, mid_y - 6))
        surface.blit(l_bot, (x + 5, y + h - 14))

        # Legend & Live values
        v0_lbl = self.font_xs.render(f"P0 V(s): {curr_v0:+.2f}", True, COLOR_P0_CYAN)
        v1_lbl = self.font_xs.render(f"P1 V(s): {curr_v1:+.2f}", True, COLOR_P1_RED)
        surface.blit(v0_lbl, (x + w - 190, y + 4))
        surface.blit(v1_lbl, (x + w - 95, y + 4))

        # Plot line curves
        n0 = len(self.v_history_p0)
        if n0 > 1:
            pts0 = []
            pts1 = []
            for i in range(n0):
                px = x + 35 + int(float(i) / 200.0 * (w - 45))
                # Map [-1, 1] to [y + h - 6, y + 6]
                val0 = np.clip(self.v_history_p0[i], -1.0, 1.0)
                val1 = np.clip(self.v_history_p1[i], -1.0, 1.0)
                py0 = mid_y - int(val0 * ((h - 14) / 2.0))
                py1 = mid_y - int(val1 * ((h - 14) / 2.0))
                pts0.append((px, py0))
                pts1.append((px, py1))

            pygame.draw.lines(surface, COLOR_P0_CYAN, False, pts0, 2)
            pygame.draw.lines(surface, COLOR_P1_RED, False, pts1, 2)

    def _render_inventory_line(self, surface: pygame.Surface, x: int, y: int, player: PlayerState) -> None:
        """Draw the bombs / radius / speed / perks summary for one player."""
        b_text = f"Bombs: {player.max_bombs} max | Radius: {player.blast_radius} | Speed: {player.speed:.1f}"
        perks = []
        if player.can_kick:
            perks.append("[KICK]")
        if player.bomb_pass:
            perks.append("[PASS]")
        if player.shield_timer > 0:
            perks.append(f"[SHLD {player.shield_timer / 20.0:.1f}s]")
        perk_str = " ".join(perks) if perks else "[NONE]"
        inv_lbl = self.font_sm.render(f"{b_text} | Perks: {perk_str}", True, COLOR_TEXT_PRIMARY)
        surface.blit(inv_lbl, (x, y))

    def _render_player_inventory(
        self, surface: pygame.Surface, x: int, y: int, pid: int, player: PlayerState
    ) -> None:
        """Compact per-player header plus inventory, shown while the full telemetry is hidden."""
        is_p0 = pid == 0
        theme_col = COLOR_P0_CYAN if is_p0 else COLOR_P1_RED
        name = "PLAYER 0 (CYAN)" if is_p0 else "PLAYER 1 (CRIMSON)"
        status = "ALIVE" if player.alive else "ELIMINATED"
        status_col = COLOR_GREEN if player.alive else (220, 50, 50)
        surface.blit(self.font_md.render(name, True, theme_col), (x, y))
        surface.blit(self.font_md.render(f"[{status}]", True, status_col), (x + 180, y))
        self._render_inventory_line(surface, x, y + 22, player)

    def _render_player_telemetry(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        pid: int,
        player: PlayerState,
        telemetry: dict[str, Any] | None,
    ) -> None:
        is_p0 = (pid == 0)
        theme_col = COLOR_P0_CYAN if is_p0 else COLOR_P1_RED
        name = "PLAYER 0 (CYAN)" if is_p0 else "PLAYER 1 (CRIMSON)"
        status = "ALIVE" if player.alive else "ELIMINATED"
        status_col = COLOR_GREEN if player.alive else (220, 50, 50)

        # Header bar
        title_lbl = self.font_md.render(name, True, theme_col)
        stat_lbl = self.font_md.render(f"[{status}]", True, status_col)
        surface.blit(title_lbl, (x, y))
        surface.blit(stat_lbl, (x + 180, y))

        # 1. Live Inventory Bar
        self._render_inventory_line(surface, x, y + 22, player)

        # 2. Action Probability Bars
        probs = None
        chosen_act = (0, 0)
        if telemetry:
            probs = telemetry.get(f"action_probs_p{pid}")
            actions = telemetry.get("actions", {})
            chosen_act = actions.get(f"player_{pid}", (0, 0))

        # 5 Move action bars
        move_names = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
        bar_x = x
        bar_y = y + 42
        bar_w = 90
        bar_h = 14

        m_chosen = chosen_act[0]
        b_chosen = chosen_act[1]

        for m_idx, m_name in enumerate(move_names):
            p_val = 0.2
            if probs is not None and len(probs) > m_idx:
                p_val = float(probs[m_idx])

            bx = bar_x + m_idx * (bar_w + 10)
            rect = pygame.Rect(bx, bar_y + 14, bar_w, bar_h)
            fill_w = int(np.clip(p_val * bar_w, 0, bar_w))
            fill_rect = pygame.Rect(bx, bar_y + 14, fill_w, bar_h)

            pygame.draw.rect(surface, (28, 32, 40), rect, border_radius=3)
            pygame.draw.rect(surface, theme_col, fill_rect, border_radius=3)

            # Highlight chosen action
            if m_idx == m_chosen:
                pygame.draw.rect(surface, COLOR_GOLD, rect, 2, border_radius=3)

            name_lbl = self.font_xs.render(m_name, True, COLOR_TEXT_MUTED)
            pct_lbl = self.font_xs.render(f"{int(p_val * 100)}%", True, COLOR_TEXT_PRIMARY)
            surface.blit(name_lbl, (bx, bar_y))
            surface.blit(pct_lbl, (bx + bar_w - 24, bar_y + 2))

        # 2 Bomb action bars
        bomb_y = bar_y + 34
        bomb_names = ["NOOP BOMB", "PLACE BOMB"]
        for b_idx, b_name in enumerate(bomb_names):
            p_val = 0.5
            p_idx = 5 + b_idx
            if probs is not None and len(probs) > p_idx:
                p_val = float(probs[p_idx])

            bx = bar_x + b_idx * 140
            rect = pygame.Rect(bx, bomb_y + 14, 130, bar_h)
            fill_w = int(np.clip(p_val * 130, 0, 130))
            fill_rect = pygame.Rect(bx, bomb_y + 14, fill_w, bar_h)

            pygame.draw.rect(surface, (28, 32, 40), rect, border_radius=3)
            pygame.draw.rect(surface, (240, 140, 30), fill_rect, border_radius=3)

            if b_idx == b_chosen:
                pygame.draw.rect(surface, COLOR_GOLD, rect, 2, border_radius=3)

            lbl = self.font_xs.render(f"{b_name}: {int(p_val * 100)}%", True, COLOR_TEXT_PRIMARY)
            surface.blit(lbl, (bx, bomb_y))

        # 3. Cumulative & Instantaneous Reward Breakdown
        rew = telemetry.get(f"rewards_p{pid}", {}) if telemetry else {}
        cum_rew = telemetry.get(f"cumulative_rewards_p{pid}", {}) if telemetry else {}

        tot = cum_rew.get("total", 0.0)
        w_l = cum_rew.get("win", 0.0) + cum_rew.get("loss", 0.0) + cum_rew.get("draw", 0.0)
        brk = cum_rew.get("brick_destruction", 0.0)
        pbr = cum_rew.get("pbrs_stats", 0.0)
        con = cum_rew.get("connectivity", 0.0)

        step_tot = rew.get("total", 0.0)

        r_y = bomb_y + 36
        r_line1 = f"Reward: {tot:+.3f} (Step: {step_tot:+.3f}) | Terminal: {w_l:+.2f}"
        r_line2 = f"Bricks: {brk:+.3f} | PBRS: {pbr:+.3f} | Conn: {con:+.3f}"
        lbl_r1 = self.font_sm.render(r_line1, True, COLOR_TEXT_PRIMARY)
        lbl_r2 = self.font_sm.render(r_line2, True, COLOR_TEXT_MUTED)
        surface.blit(lbl_r1, (x, r_y))
        surface.blit(lbl_r2, (x, r_y + 18))

    # --- Event Markers ---
    _EVENT_SHAPES: dict[str, str] = {
        "brick_destroyed": "diamond",
        "powerup_pickup": "circle",
        "shield_activated": "square",
        "player_death": "cross",
    }

    def event_marker_shape(self, event_type: str) -> str:
        """Marker shape encodes the event type."""
        return self._EVENT_SHAPES.get(event_type, "triangle")

    def event_marker_color(self, ev: GameEvent) -> tuple[int, int, int]:
        """Marker color encodes the acting player (cyan = P0, red = P1, gold = unattributed)."""
        if ev.player_id == 0:
            return COLOR_P0_CYAN
        if ev.player_id == 1:
            return COLOR_P1_RED
        return COLOR_GOLD

    def _draw_event_marker(
        self, surface: pygame.Surface, mx: int, my: int, shape: str, color: tuple[int, int, int]
    ) -> None:
        outline = (10, 12, 16)
        if shape == "diamond":
            pts = [(mx, my - 7), (mx + 5, my), (mx, my + 7), (mx - 5, my)]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, outline, pts, 1)
        elif shape == "circle":
            pygame.draw.circle(surface, color, (mx, my), 5)
            pygame.draw.circle(surface, outline, (mx, my), 5, 1)
        elif shape == "square":
            rect = pygame.Rect(mx - 4, my - 4, 8, 8)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, outline, rect, 1)
        elif shape == "cross":
            pygame.draw.line(surface, outline, (mx - 5, my - 5), (mx + 5, my + 5), 5)
            pygame.draw.line(surface, outline, (mx - 5, my + 5), (mx + 5, my - 5), 5)
            pygame.draw.line(surface, color, (mx - 5, my - 5), (mx + 5, my + 5), 3)
            pygame.draw.line(surface, color, (mx - 5, my + 5), (mx + 5, my - 5), 3)
        else:
            pts = [(mx, my - 6), (mx + 5, my + 5), (mx - 5, my + 5)]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, outline, pts, 1)

    # --- Bottom Panel: Timeline Scrubber ---
    def render_bottom_panel(
        self,
        surface: pygame.Surface,
        tick: int,
        total_ticks: int,
        events: list[GameEvent] | None = None,
        tick_limit: int | None = None,
        end_reason: str | None = None,
    ) -> None:
        panel_y = self.ARENA_SIZE  # 660
        panel_rect = pygame.Rect(0, panel_y, self.WINDOW_WIDTH, self.WINDOW_HEIGHT - panel_y)
        pygame.draw.rect(surface, (14, 16, 20), panel_rect)
        pygame.draw.line(surface, COLOR_BORDER, (0, panel_y), (self.WINDOW_WIDTH, panel_y), 2)

        # 1. Exact Match Time & Tick counters
        time_sec = tick / float(TICK_RATE)
        m = int(time_sec // 60)
        s = int(time_sec % 60)
        ms = int((time_sec * 100) % 100)
        time_str = f"{m:02d}:{s:02d}.{ms:02d} / 03:00.00"

        tot_display = tick_limit if tick_limit is not None else max(total_ticks, MAX_TICKS)
        tick_str = f"Tick {tick:04d} / {tot_display:04d}"
        if end_reason is not None and tick >= total_ticks - 1:
            tick_str += f" [END: {end_reason}]"

        time_lbl = self.font_md.render(time_str, True, COLOR_TEXT_PRIMARY)
        tick_lbl = self.font_md.render(tick_str, True, COLOR_TEXT_MUTED)
        surface.blit(time_lbl, (40, panel_y + 12))
        surface.blit(tick_lbl, (220, panel_y + 12))

        # Status & Speed indicator
        status_text = "[PAUSED]" if self.paused else "[PLAYING]"
        status_col = COLOR_GOLD if self.paused else COLOR_GREEN
        stat_lbl = self.font_md.render(status_text, True, status_col)
        surface.blit(stat_lbl, (410, panel_y + 12))

        speed_str = f"Speed: {self.playback_speed:.2f}x"
        speed_lbl = self.font_md.render(speed_str, True, COLOR_TEXT_PRIMARY)
        surface.blit(speed_lbl, (520, panel_y + 12))

        if getattr(self, "mode", "agent_vs_agent") == "human_vs_agent":
            help_str = "Controls: WASD: Move | SPACE: Bomb | P: Pause | LEFT/RIGHT: Jump ±5s | UP/DOWN: Speed | T: Telemetry | 1-3: Overlays"
        else:
            help_str = "Controls: SPACE/P: Pause | LEFT/RIGHT: Jump ±5s (Step ±1 when paused) | UP/DOWN: Speed | T: Telemetry | 1-3: Overlays"
        help_lbl = self.font_xs.render(help_str, True, COLOR_TEXT_MUTED)
        surface.blit(help_lbl, (660, panel_y + 14))

        # 2. Scrubber Track & Synchronization Invariant
        track_x = 40
        track_y = panel_y + 42
        track_w = 1200
        track_h = 16

        # Progress ratio strictly computed as tick / total_recorded_ticks
        progress = float(tick) / float(max(1, total_ticks))
        progress = float(np.clip(progress, 0.0, 1.0))

        track_rect = pygame.Rect(track_x, track_y, track_w, track_h)
        fill_rect = pygame.Rect(track_x, track_y, int(progress * track_w), track_h)

        pygame.draw.rect(surface, (28, 32, 40), track_rect, border_radius=8)
        pygame.draw.rect(surface, (45, 125, 205), fill_rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_BORDER, track_rect, 1, border_radius=8)

        # 3. Discrete Event Markers
        mouse_pos = pygame.mouse.get_pos()
        self.active_tooltip = None

        if events:
            for ev in events:
                ev_ratio = float(ev.tick) / float(max(1, total_ticks))
                if 0.0 <= ev_ratio <= 1.0:
                    mx = track_x + int(ev_ratio * track_w)
                    my = track_y + track_h // 2

                    m_col = self.event_marker_color(ev)
                    self._draw_event_marker(surface, mx, my, self.event_marker_shape(ev.event_type), m_col)

                    # Hover detection for tooltip
                    if abs(mouse_pos[0] - mx) < 6 and abs(mouse_pos[1] - my) < 12:
                        self.active_tooltip = f"[Tick {ev.tick}] {ev.description}"
                        self.active_tooltip_pos = (mx, track_y - 22)

        # 4. Draggable Playhead Thumb
        thumb_x = track_x + int(progress * track_w)
        thumb_y = track_y + track_h // 2
        pygame.draw.circle(surface, (10, 12, 16), (thumb_x, thumb_y), 11)
        pygame.draw.circle(surface, (255, 255, 255), (thumb_x, thumb_y), 9)
        pygame.draw.circle(surface, (45, 140, 235), (thumb_x, thumb_y), 5)

        # Render Tooltip if hovered
        if self.active_tooltip and self.active_tooltip_pos:
            tx, ty = self.active_tooltip_pos
            tip_surf = self.font_xs.render(self.active_tooltip, True, (255, 255, 255))
            tip_rect = tip_surf.get_rect(center=(tx, ty))
            bg_rect = tip_rect.inflate(12, 6)
            pygame.draw.rect(surface, (20, 24, 32), bg_rect, border_radius=4)
            pygame.draw.rect(surface, COLOR_GOLD, bg_rect, 1, border_radius=4)
            surface.blit(tip_surf, tip_rect)

    # --- Interaction & Event Handling ---
    def handle_input(self, mode: str | None = None) -> dict[str, Any]:
        """Handle keyboard shortcuts, overlays, and mouse dragging."""
        effective_mode = mode if mode is not None else getattr(self, "mode", "agent_vs_agent")
        events_processed = {
            "quit": False,
            "toggle_pause": False,
            "step": 0,
            "seek_tick": None,
            "speed_change": 0,
            "human_action": (MoveAction.NOOP, BombAction.NOOP),
        }

        # Human keyboard input state
        keys = pygame.key.get_pressed()
        move_act = MoveAction.NOOP
        bomb_act = BombAction.NOOP

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_act = MoveAction.UP
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_act = MoveAction.DOWN
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move_act = MoveAction.LEFT
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move_act = MoveAction.RIGHT

        if keys[pygame.K_SPACE] or keys[pygame.K_b] or keys[pygame.K_RETURN]:
            bomb_act = BombAction.PLACE_BOMB

        events_processed["human_action"] = (move_act, bomb_act)

        for event in pygame.event.get():
            if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                self._mouse_moved = True

            if event.type == pygame.QUIT:
                events_processed["quit"] = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events_processed["quit"] = True
                elif event.key in (pygame.K_p, getattr(pygame, "K_PAUSE", -1)):
                    self.paused = not self.paused
                    events_processed["toggle_pause"] = True
                elif event.key == pygame.K_SPACE:
                    if effective_mode == "human_vs_agent":
                        bomb_act = BombAction.PLACE_BOMB
                        events_processed["human_action"] = (move_act, bomb_act)
                    else:
                        self.paused = not self.paused
                        events_processed["toggle_pause"] = True
                elif event.key in (pygame.K_b, pygame.K_RETURN) and effective_mode == "human_vs_agent":
                    bomb_act = BombAction.PLACE_BOMB
                    events_processed["human_action"] = (move_act, bomb_act)
                elif event.key == pygame.K_LEFT:
                    if effective_mode == "human_vs_agent":
                        move_act = MoveAction.LEFT
                        events_processed["human_action"] = (move_act, bomb_act)
                    else:
                        events_processed["step"] = -1 if self.paused else -5 * TICK_RATE
                elif event.key == pygame.K_RIGHT:
                    if effective_mode == "human_vs_agent":
                        move_act = MoveAction.RIGHT
                        events_processed["human_action"] = (move_act, bomb_act)
                    else:
                        events_processed["step"] = 1 if self.paused else 5 * TICK_RATE
                elif event.key == pygame.K_UP:
                    if effective_mode == "human_vs_agent":
                        move_act = MoveAction.UP
                        events_processed["human_action"] = (move_act, bomb_act)
                    else:
                        self.speed_idx = min(len(self.speed_multipliers) - 1, self.speed_idx + 1)
                        events_processed["speed_change"] = 1
                elif event.key == pygame.K_DOWN:
                    if effective_mode == "human_vs_agent":
                        move_act = MoveAction.DOWN
                        events_processed["human_action"] = (move_act, bomb_act)
                    else:
                        self.speed_idx = max(0, self.speed_idx - 1)
                        events_processed["speed_change"] = -1
                elif event.key == pygame.K_w and effective_mode == "human_vs_agent":
                    move_act = MoveAction.UP
                    events_processed["human_action"] = (move_act, bomb_act)
                elif event.key == pygame.K_s and effective_mode == "human_vs_agent":
                    move_act = MoveAction.DOWN
                    events_processed["human_action"] = (move_act, bomb_act)
                elif event.key == pygame.K_a and effective_mode == "human_vs_agent":
                    move_act = MoveAction.LEFT
                    events_processed["human_action"] = (move_act, bomb_act)
                elif event.key == pygame.K_d and effective_mode == "human_vs_agent":
                    move_act = MoveAction.RIGHT
                    events_processed["human_action"] = (move_act, bomb_act)
                # Overlays (Keys 1, 2, 3, 4)
                elif event.key == pygame.K_1:
                    self.overlay_danger = not self.overlay_danger
                elif event.key == pygame.K_2:
                    self.overlay_safe_space = not self.overlay_safe_space
                elif event.key == pygame.K_3:
                    self.overlay_policy_arrow = not self.overlay_policy_arrow
                elif event.key == pygame.K_t:
                    self.show_telemetry = not self.show_telemetry
                elif event.key == pygame.K_4:
                    self.overlay_danger = False
                    self.overlay_safe_space = False
                    self.overlay_policy_arrow = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if 690 <= my <= 745 and 40 <= mx <= 1240:
                    self.is_dragging_scrubber = True
                    ratio = np.clip((mx - 40) / 1200.0, 0.0, 1.0)
                    events_processed["seek_tick"] = ratio

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.is_dragging_scrubber = False

            elif event.type == pygame.MOUSEMOTION and self.is_dragging_scrubber:
                mx, _ = event.pos
                ratio = np.clip((mx - 40) / 1200.0, 0.0, 1.0)
                events_processed["seek_tick"] = ratio

        return events_processed

    def render_frame(
        self,
        state: GameState,
        prev_state: GameState | None = None,
        telemetry: dict[str, Any] | None = None,
        events: list[GameEvent] | None = None,
        interp_alpha: float = 1.0,
        total_ticks: int = MAX_TICKS,
        tick_limit: int | None = None,
        end_reason: str | None = None,
    ) -> None:
        """Compose and flip the full 1280x760 frame."""
        self.screen.fill(COLOR_BG)

        # Arena
        self.render_arena(self.screen, state, prev_state, interp_alpha)

        # Right Panel Diagnostics
        self.render_right_panel(self.screen, state, telemetry)

        # Bottom Timeline Scrubber
        self.render_bottom_panel(self.screen, state.tick, total_ticks, events, tick_limit, end_reason)

        self.update_particles()

        # In-game custom cursor (resolves WSL invisible hardware cursor)
        self.render_cursor(self.screen)

        if not self.headless:
            pygame.display.flip()

    _CURSOR_OFFSETS: tuple[tuple[int, int], ...] = (
        (0, 0),
        (0, 17),
        (5, 13),
        (8, 20),
        (11, 19),
        (8, 12),
        (13, 12),
    )

    def _draw_cursor(self, surface: pygame.Surface, mx: int, my: int) -> None:
        """Draw high-contrast hardware-style cursor arrow with dark border and drop shadow."""
        pts = [(mx + dx, my + dy) for dx, dy in self._CURSOR_OFFSETS]
        shadow_pts = [(x + 1, y + 2) for x, y in pts]
        pygame.draw.polygon(surface, (10, 12, 16), shadow_pts)
        pygame.draw.polygon(surface, (250, 252, 255), pts)
        pygame.draw.polygon(surface, (15, 18, 24), pts, 1)

    def render_cursor(self, surface: pygame.Surface) -> None:
        """Render custom software cursor to ensure visibility across all platforms (e.g. WSL)."""
        if self.headless:
            return
        if not self._mouse_moved and not pygame.mouse.get_focused():
            return
        mx, my = pygame.mouse.get_pos()
        if not (0 <= mx < self.WINDOW_WIDTH and 0 <= my < self.WINDOW_HEIGHT):
            return
        self._draw_cursor(surface, mx, my)

    def tick_clock(self, fps: int = 60) -> float:
        return self.clock.tick(fps) / 1000.0
