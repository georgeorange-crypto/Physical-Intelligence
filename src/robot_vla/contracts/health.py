from dataclasses import dataclass


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    backend: str
    controller_version: str
    message: str = ""
