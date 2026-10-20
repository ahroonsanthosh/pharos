"""A complete, seeded simulation scenario: world, appearances, flight, sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pharos.config import from_dict
from pharos.sim.appearance import Look
from pharos.sim.trajectory import TrajectorySpec
from pharos.sim.world import ChangeSpec, TerrainSpec, WorldSpec


@dataclass(frozen=True)
class CameraSpec:
    width: int = 640
    height: int = 480
    hfov_deg: float = 70.0
    blur: float = 0.6  # optics, Gaussian sigma in pixels
    motion_blur_px: float = 2.0  # streak length along the direction of flight
    noise: float = 0.012  # sensor noise (0..1 scale)
    vignetting: float = 0.25  # brightness loss in the corners
    exposure_jitter: float = 0.08  # frame-to-frame auto-exposure wander
    jpeg_quality: int = 85
    # Unexpected trouble: (start s, duration s) spans where the video freezes on one image,
    # or where frames are lost altogether; raindrops stuck on the lens.
    freezes: tuple[tuple[float, float], ...] = ()
    dropped: tuple[tuple[float, float], ...] = ()
    raindrops: int = 0


@dataclass(frozen=True)
class TelemetrySpec:
    """Errors on what the aircraft reports. Biases get a random sign."""

    compass: bool = True
    compass_bias_deg: float = 10.0
    compass_drift_deg_per_min: float = 2.0
    compass_noise_deg: float = 1.0
    altitude_bias: float = 0.04  # fraction of true height (baro vs terrain)
    altitude_noise_m: float = 1.0
    altitude_offset_m: float = 0.0  # additive baro error (terrain height change since takeoff)
    attitude_noise_deg: float = 0.5
    attitude_bias_deg: float = 0.0  # constant camera mount misalignment, random direction
    latency_s: float = 0.0  # telemetry describes the aircraft this long before the frame
    takeoff_known: bool = True  # over hills: the localizer is told where the aircraft took off
    # (start s, duration s) spans where the telemetry stops updating (repeats its last
    # reading), and (start s, duration s, degrees) spans the compass is pulled off by
    # magnetic interference.
    dropouts: tuple[tuple[float, float], ...] = ()
    compass_jumps: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str = "nominal"
    seed: int = 0
    map_gsd: float = 1.0  # metres per pixel of the reference map
    geo_origin: tuple[float, float] = (46.95, 7.45)  # lat, lon of world (0, 0)
    world: WorldSpec = field(default_factory=WorldSpec)
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    changes: ChangeSpec = field(default_factory=ChangeSpec)
    map_look: Look = field(default_factory=Look)
    flight_look: Look = field(default_factory=lambda: Look(seed=1))
    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    telemetry: TelemetrySpec = field(default_factory=TelemetrySpec)
    # GNSS lost at the start, its last fix known to within this many metres (None: a cold
    # start, nothing known). The fix given is the truth off by that much, at random.
    known_start_m: float | None = None
    map_shift_m: tuple[float, float] = (0.0, 0.0)  # the map's georeference is this far off


SCENARIO_DIR = Path(__file__).parent / "scenarios"


def builtin_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenario(name_or_path: str | Path) -> Scenario:
    """Load a built-in scenario by name (e.g. ``"winter_snow"``) or any YAML file.

    A file with ``base: <scenario>`` takes that scenario and changes only what it lists.
    """
    return from_dict(Scenario, _scenario_data(name_or_path))


def _scenario_data(name_or_path: str | Path) -> dict:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            raise ValueError(f"unknown scenario {name_or_path!r}; built-in: {builtin_scenarios()}")
    data = yaml.safe_load(path.read_text()) or {}
    if (base := data.pop("base", None)) is not None:
        data = _merged(_scenario_data(base), data)
    return data


def _merged(base: dict, changes: dict) -> dict:
    """``base`` with ``changes`` applied, nested mappings merged key by key."""
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            value = _merged(out[key], value)
        out[key] = value
    return out
