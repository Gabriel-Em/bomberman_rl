"""Automated Emergent Behavior Audit & Telemetry Logging for 1v1 Bomberman.

Tracks win/loss/draw rates, self-inflicted suicide frequency, match durations,
brick destruction rates, power-up buffs vs debuffs ratios, kicks executed,
and recursive chain reaction detonations across training generations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class MatchStats:
    """Individual match outcome telemetry for emergent behavior auditing."""
    winner: int | None = None
    draw: bool = False
    p0_suicide: bool = False
    p1_suicide: bool = False
    ticks: int = 0
    bricks_destroyed: int = 0
    p0_buffs_collected: int = 0
    p0_debuffs_collected: int = 0
    p1_buffs_collected: int = 0
    p1_debuffs_collected: int = 0
    debuffs_spawned: int = 0
    debuffs_avoided: int = 0
    kicks_executed: int = 0
    chain_detonations: int = 0


class BehaviorDiagnostics:
    """Aggregates and formats emergent behavior metrics across training generations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset accumulated generation statistics."""
        self.matches: list[MatchStats] = []

    def record_match(self, stats: MatchStats) -> None:
        """Record statistics of a completed match."""
        self.matches.append(stats)

    @property
    def total_matches(self) -> int:
        return len(self.matches)

    def compute_metrics(self, generation: int = 0) -> dict[str, Any]:
        """Compute aggregated metrics for the current generation."""
        n = len(self.matches)
        if n == 0:
            return {
                "generation": generation,
                "matches": 0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "draw_rate": 0.0,
                "suicide_rate": 0.0,
                "avg_match_length": 0.0,
                "avg_bricks_destroyed": 0.0,
                "buffs_collected": 0,
                "debuffs_collected": 0,
                "debuffs_avoided": 0,
                "buff_debuff_ratio": 0.0,
                "avg_kicks": 0.0,
                "avg_chain_detonations": 0.0,
            }

        wins = sum(1 for m in self.matches if m.winner == 0)
        losses = sum(1 for m in self.matches if m.winner == 1)
        draws = sum(1 for m in self.matches if m.draw or m.winner is None)
        suicides = sum(1 for m in self.matches if m.p0_suicide)

        total_ticks = sum(m.ticks for m in self.matches)
        total_bricks = sum(m.bricks_destroyed for m in self.matches)
        total_buffs = sum(m.p0_buffs_collected for m in self.matches)
        total_debuffs = sum(m.p0_debuffs_collected for m in self.matches)
        total_debuffs_avoided = sum(
            m.debuffs_avoided if m.debuffs_avoided > 0 else max(0, m.debuffs_spawned - m.p0_debuffs_collected)
            for m in self.matches
        )
        total_kicks = sum(m.kicks_executed for m in self.matches)
        total_chains = sum(m.chain_detonations for m in self.matches)

        # Ratio of buffs collected vs debuffs encountered/avoided
        buff_ratio = float(total_buffs) / max(1.0, float(total_debuffs))

        return {
            "generation": generation,
            "matches": n,
            "win_rate": wins / n,
            "loss_rate": losses / n,
            "draw_rate": draws / n,
            "suicide_rate": suicides / n,
            "avg_match_length": total_ticks / n,
            "avg_bricks_destroyed": total_bricks / n,
            "buffs_collected": total_buffs,
            "debuffs_collected": total_debuffs,
            "debuffs_avoided": total_debuffs_avoided,
            "buff_debuff_ratio": buff_ratio,
            "avg_kicks": total_kicks / n,
            "avg_chain_detonations": total_chains / n,
        }

    def format_table(self, generation: int = 0) -> str:
        """Format metrics into a clean terminal report table."""
        metrics = self.compute_metrics(generation)
        n = metrics["matches"]
        if n == 0:
            return f"Generation {generation:04d}: No matches completed."

        win_pct = metrics["win_rate"] * 100.0
        loss_pct = metrics["loss_rate"] * 100.0
        draw_pct = metrics["draw_rate"] * 100.0
        suicide_pct = metrics["suicide_rate"] * 100.0
        avg_len = metrics["avg_match_length"]
        avg_sec = avg_len * 0.05
        avg_bricks = metrics["avg_bricks_destroyed"]
        buffs = metrics["buffs_collected"]
        debuffs = metrics["debuffs_collected"]
        avoided = metrics["debuffs_avoided"]
        ratio = metrics["buff_debuff_ratio"]
        avg_kicks = metrics["avg_kicks"]
        avg_chains = metrics["avg_chain_detonations"]

        lines = [
            "=" * 72,
            f" Generation {generation:04d} Emergent Behavior Audit (Matches: {n})",
            "-" * 72,
            f" Win / Loss / Draw Rate:     {win_pct:5.1f}% / {loss_pct:5.1f}% / {draw_pct:5.1f}%",
            f" P0 Suicide Rate:            {suicide_pct:5.1f}%",
            f" Avg Match Duration:         {avg_len:5.1f} ticks ({avg_sec:4.1f}s)",
            f" Bricks Destroyed / Match:   {avg_bricks:5.1f}",
            f" Power-ups Buff/Debuff:      {ratio:5.2f}x ({buffs} buffs vs {debuffs} debuffs, {avoided} avoided)",
            f" Kicks Executed / Match:     {avg_kicks:5.2f}",
            f" Chain Detonations / Match:  {avg_chains:5.2f}",
            "=" * 72,
        ]
        return "\n".join(lines)
