import numpy as np


class RateLimiter:
    def __init__(self, max_delta_deg: float = 5.0) -> None:
        if max_delta_deg <= 0:
            raise ValueError("max_delta_deg must be positive")
        self.max_delta_deg = float(max_delta_deg)

    def apply(self, current_deg: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
        return current_deg + np.clip(
            target_deg - current_deg, -self.max_delta_deg, self.max_delta_deg
        )
