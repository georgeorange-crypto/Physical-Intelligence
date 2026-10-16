import time

from robot_vla.safety.watchdog import Watchdog


def test_watchdog_allows_first_command_then_expires_after_missing_heartbeat():
    watchdog = Watchdog(timeout_s=0.01)
    time.sleep(0.02)
    assert not watchdog.expired
    watchdog.heartbeat()
    assert not watchdog.expired
    time.sleep(0.02)
    assert watchdog.expired
