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

LSTM_REVERSAL_EXITS_BACKTEST_RESULT_KEY           = "LSTM_only_reversal_exits"
LSTM_PREDICTION_WINDOW_EXITS_BACKTEST_RESULT_KEY  = "LSTM_only_prediction_window_exits"

BACKTEST_NAME               = "Name"
ENTRY_SLOPE_THRESHOLD       = "Entry Slope Threshold"
SHORT_ENTRY_SLOPE_THRESHOLD = "Short Entry Slope Threshold"
VBT_TOTAL_RETURN            = "Total Return [%]"
VBT_WIN_RATE                = "Win Rate [%]"
VBT_TOTAL_TRADES            = "Total Trades"

