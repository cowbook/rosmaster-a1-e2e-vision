from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class SteeringFilter:
    max_abs: float
    low_pass_alpha: float
    max_rate_per_sec: float
    _previous: float = 0.0
    _has_previous: bool = False

    def reset(self) -> None:
        self._previous = 0.0
        self._has_previous = False

    def apply(self, raw_steering: float, dt_sec: float) -> float:
        target = clamp(raw_steering, -self.max_abs, self.max_abs)
        alpha = clamp(self.low_pass_alpha, 0.0, 1.0)

        if not self._has_previous:
            self._previous = target
            self._has_previous = True
            return self._previous

        filtered = alpha * target + (1.0 - alpha) * self._previous
        if dt_sec > 0.0 and self.max_rate_per_sec > 0.0:
            max_delta = self.max_rate_per_sec * dt_sec
            delta = clamp(filtered - self._previous, -max_delta, max_delta)
            filtered = self._previous + delta

        self._previous = clamp(filtered, -self.max_abs, self.max_abs)
        return self._previous
