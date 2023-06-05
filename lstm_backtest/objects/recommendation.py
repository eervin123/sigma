from dataclasses import dataclass

import numpy as np

from objects.action_type import ActionType


@dataclass
class Recommendation:
    long: np.ndarray
    short: np.ndarray
    indx_hi: int
    indx_low: int
    action: ActionType
    rec_id: str
