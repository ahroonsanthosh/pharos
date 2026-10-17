from pathlib import Path

import numpy as np
import pytest

from pharos.config import from_dict, load_yaml
from pharos.geometry import Camera
from pharos.nav.localizer import LocalizerConfig
from pharos.recording import Recording, load_recording, save_recording
from pharos.sim.scenario import Scenario
from pharos.types import Frame, GroundTruth, Telemetry

SCENARIOS = sorted((Path(__file__).parents[1] / "configs" / "scenarios").glob("*.yaml"))


def test_nested_config_and_tuples():
    sc = from_dict(Scenario, {"flight_look": {"season": "snow", "white_balance": [1, 1, 0.9]}})
    assert sc.flight_look.season == "snow"
    assert sc.flight_look.white_balance == (1, 1, 0.9)
    cfg = from_dict(LocalizerConfig, {"matcher": {"scales": [1.0]}, "max_misses": 3})
    assert cfg.matcher.scales == (1.0,)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        from_dict(Scenario, {"wrold": {}})


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenarios_load(path):
    assert load_yaml(Scenario, path).name == path.stem


def test_recording_round_trip(tmp_path):
    cam = Camera.from_fov(64, 48, 60)
    img = np.full((48, 64, 3), 128, np.uint8)
    frames = [
        Frame(0.0, img, Telemetry(100.0, 0.5, 0.01, -0.02)),
        Frame(0.5, img, Telemetry(101.0, None)),
    ]
    truth = [GroundTruth(0.0, 1.0, 2.0, 100.0, 0.4), GroundTruth(0.5, 3.0, 4.0, 101.0, 0.5)]
    save_recording(Recording(cam, frames, truth), tmp_path)
    rec = load_recording(tmp_path)
    assert rec.camera == cam and len(rec) == 2
    loaded = list(rec)
    assert np.isclose(loaded[0].telemetry.yaw, 0.5)
    assert np.isclose(loaded[0].telemetry.roll, -0.02)
    assert loaded[1].telemetry.yaw is None
    assert rec.truth is not None and np.isclose(rec.truth[1].heading, 0.5)
