"""Small, dependency-free experiment accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class ProportionEstimate:
    successes: int
    trials: int
    proportion: float
    lower_95: float
    upper_95: float


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> ProportionEstimate:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require trials > 0 and 0 <= successes <= trials")
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    margin = z * sqrt(estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials**2))
    margin /= denominator
    return ProportionEstimate(successes, trials, estimate, center - margin, center + margin)


@dataclass(frozen=True)
class ExperimentAllocation:
    engineering_acceptance: int = 10
    pilot_crossed_pair: int = 20
    pilot_sparse_3_to_5: int = 20
    pilot_moderate_5_to_10: int = 20
    continuous_clear_scenes: int = 20
    comparison_geometry: int = 20
    comparison_visual_servo: int = 20
    comparison_act: int = 20

    @property
    def total(self) -> int:
        return sum(vars(self).values())
