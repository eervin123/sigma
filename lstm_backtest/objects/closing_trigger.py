from enum import Enum


class ClosingTrigger(str, Enum):
    QUICK_PROFIT = "QUICK_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    END_OF_DAY = "END_OF_DAY"
    OPEN_TOO_LONG = "OPEN_TOO_LONG"
