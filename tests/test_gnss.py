"""GNSS against the visual position: a glitch passes, a spoofed walk-off is flagged."""

from pharos.nav.gnss import GnssCheck
from pharos.types import Estimate, Status


def trusted(t, x=0.0, sigma=5.0):
    return Estimate(t, Status.TRACKING, x, 0.0, 0.0, sigma)


def test_gnss_that_walks_away_from_a_trusted_position_is_flagged():
    check = GnssCheck()
    assert check.update(trusted(0.0), (8.0, 0.0)) == "agrees"
    assert check.update(trusted(0.5), (120.0, 0.0)) == "agrees"  # one glitch: not yet
    assert check.update(trusted(1.0), (2.0, 0.0)) == "agrees"
    verdicts = [check.update(trusted(2.0 + 0.5 * k), (60.0 + 5 * k, 0.0)) for k in range(10)]
    assert verdicts[0] == "agrees" and verdicts[-1] == "disagrees"
    assert check.update(trusted(9.0), (3.0, 0.0)) == "agrees"  # back in agreement


def test_without_trust_or_gnss_there_is_no_verdict():
    check = GnssCheck()
    assert check.update(trusted(0.0), None) is None
    assert check.update(Estimate(0.5, Status.COASTING, 0.0, 0.0, 0.0, 30.0), (500.0, 0.0)) is None
