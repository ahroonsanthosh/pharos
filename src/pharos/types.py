"""Plain data types passed between pipeline stages. Angles are radians."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# The aircraft's height above the ground at world positions (..., 2), its own elevation
# being what it is: the image scale over uneven ground. Not positive where it would be
# underground.
HeightField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Telemetry:
    """What the aircraft reports alongside a frame.

    ``altitude`` is height above ground in metres, or with ``above_takeoff``
    height above the takeoff point (barometric), which a terrain model turns
    into height above ground (:mod:`pharos.terrain`). Either may be biased.
    ``yaw`` is the compass heading if one is available; it may carry a large
    bias, so the localizer only trusts its frame-to-frame changes and uses its
    absolute value as a loose prior. ``pitch``/``roll`` are the camera tilt
    from nadir (see :mod:`pharos.geometry`).
    """

    altitude: float
    yaw: float | None = None
    pitch: float = 0.0
    roll: float = 0.0
    above_takeoff: bool = False

    @property
    def tilt(self) -> float:
        """Angle between the optical axis and straight down."""
        return float(np.arccos(np.clip(np.cos(self.pitch) * np.cos(self.roll), -1.0, 1.0)))


@dataclass(frozen=True)
class Frame:
    t: float
    image: np.ndarray  # BGR uint8 or gray uint8
    telemetry: Telemetry


@dataclass(frozen=True)
class GroundTruth:
    """True aircraft state at one frame (simulation or a reference GNSS/INS)."""

    t: float
    x: float
    y: float
    altitude: float
    heading: float
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class Fix:
    """One absolute position hypothesis produced by a matcher."""

    x: float
    y: float
    heading: float
    score: float  # similarity of the best match, in [0, 1]
    distinctiveness: float  # 1 - second_best / best, in [0, 1]
    scale: float = 1.0  # correction factor applied to the reported altitude
    excess: float = 0.0  # peak height above its window's noise (see correlate.peak_excess)
    # Position ambiguity of the peak, m^2 (east-east, east-north, north-north): large
    # along a ridge, e.g. when flying along a straight field edge (correlate.peak_covariance)
    spread: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # -1 or 1 when the lowest or highest altitude hypothesis won: ``scale`` is then only
    # a bound (the best fit may lie further out), not a measurement
    scale_edge: int = 0


class Status(StrEnum):
    LOST = "lost"  # no position; searching the whole region
    CONVERGING = "converging"  # initialized, not yet confirmed by enough frames
    TRACKING = "tracking"  # confirmed; estimate is trustworthy
    COASTING = "coasting"  # confirmed earlier, but no recent match: dead reckoning only


@dataclass(frozen=True)
class Estimate:
    t: float
    status: Status
    x: float = float("nan")
    y: float = float("nan")
    heading: float = float("nan")
    sigma_xy: float = float("inf")  # 1-sigma horizontal uncertainty, metres
    fix: Fix | None = None  # the measurement accepted this frame, if any
    latency_ms: float = 0.0
