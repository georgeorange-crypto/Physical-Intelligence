import time


class Watchdog:
    def __init__(self, timeout_s: float = 1.0) -> None:
        self.timeout_s = timeout_s
        self.last_heartbeat = time.monotonic()
        self._started = False

    def heartbeat(self) -> None:
        self.last_heartbeat = time.monotonic()
        self._started = True

    @property
    def expired(self) -> bool:
        return self._started and time.monotonic() - self.last_heartbeat > self.timeout_s
