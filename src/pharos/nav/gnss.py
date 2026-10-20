"""Watching the aircraft's GNSS against its visual position: spoofing and jamming.

A jammed receiver says so: it loses its fix. A spoofed one does not: it reports a
confident position that walks or jumps away from where the aircraft really is.
While pharos trusts its own position, a GNSS position further off than both
uncertainties allow, for longer than a glitch lasts, is flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pharos.types import Estimate, Status


@dataclass
class GnssCheck:
    gnss_sigma_m: float = 5.0  # a healthy receiver's 1-sigma error
    margin_m: float = 10.0  # map registration and mount calibration
    hold_s: float = 3.0  # disagreement lasting this long is not a glitch
    _since: float | None = None

    def update(self, estimate: Estimate, gnss_xy: tuple[float, float] | None) -> str | None:
        """``"agrees"`` or ``"disagrees"`` while pharos trusts its position, else None."""
        if gnss_xy is None or estimate.status is not Status.TRACKING:
            self._since = None
            return None
        off = math.hypot(estimate.x - gnss_xy[0], estimate.y - gnss_xy[1])
        limit = 3.0 * math.hypot(estimate.sigma_xy, self.gnss_sigma_m) + self.margin_m
        if off <= limit:
            self._since = None
            return "agrees"
        if self._since is None:
            self._since = estimate.t
        return "disagrees" if estimate.t - self._since >= self.hold_s else "agrees"
