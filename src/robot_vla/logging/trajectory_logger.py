import json
import time
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        row = {"timestamp_ns": time.time_ns(), **record}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
