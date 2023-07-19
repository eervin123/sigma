from enum import Enum

class ActionType(str, Enum):
  OPEN_SHORT          = "open-short"
  OPEN_LONG           = "open-long"
  CLOSE_LONG          = "close-long"
  CLOSE_SHORT         = "close-short"
  LEGACY_OPEN_LONG    = "open_long"
  LEGACY_OPEN_SHORT   = "open_short"
  LEGACY_CLOSE_LONG   = "close_long"
  LEGACY_CLOSE_SHORT  = "close_short"
  NOOP                = "no_op"

BASELINE_BACKTEST_RESULT_KEY = "baseline_backtest_result"
