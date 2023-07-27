import pandas as pd
import vectorbtpro as vbt
import numpy as np
from typing import List
from lstm_results_utils import store_backtest_results
from lstm_analysis_constants import (ActionType, EntryType, LSTM_REVERSAL_EXITS_BACKTEST_RESULT_KEY
                                     , LSTM_PREDICTION_WINDOW_EXITS_BACKTEST_RESULT_KEY)

def run_backtest_lstm_recommendations_reversal_exits(df: pd.DataFrame, results_as_list: List):
  # Exits are from reversals, as our LSTM model doesn't produce exit signals
  entries         = pd.Series(np.where((df['recommendations'] == ActionType.OPEN_LONG  ), True, False))
  exits           = pd.Series(np.where((df['recommendations'] == ActionType.CLOSE_LONG ), True, False))
  short_entries   = pd.Series(np.where((df['recommendations'] == ActionType.OPEN_SHORT ), True, False))
  short_exits     = pd.Series(np.where((df['recommendations'] == ActionType.CLOSE_SHORT), True, False))

  pf = vbt.Portfolio.from_signals(
      high                = df['BTCUSDT_High'],
      low                 = df['BTCUSDT_Low'],
      open                = df['BTCUSDT_Open'],
      close               = df['BTCUSDT_Close'],
      entries             = entries, # commented out for a short only backtest
      exits               = exits,
      short_entries       = short_entries,
      short_exits         = short_exits,    
      time_delta_format   = 'Rows', # Use the row index to calculate the time delta    
      )

  store_backtest_results(LSTM_REVERSAL_EXITS_BACKTEST_RESULT_KEY, pf, results_as_list, None)




def run_backtests_lstm_recommendations_prediction_size_exit(df: pd.DataFrame, results_as_list: List, prediction_window: int):
  # Exits are controlled by td_stop
  entries         = pd.Series(np.where((df['recommendations'] == ActionType.OPEN_LONG  ), True, False))
  short_entries   = pd.Series(np.where((df['recommendations'] == ActionType.OPEN_SHORT ), True, False))

  pf = vbt.Portfolio.from_signals(
      high                = df['BTCUSDT_High'],
      low                 = df['BTCUSDT_Low'],
      open                = df['BTCUSDT_Open'],
      close               = df['BTCUSDT_Close'],
      entries             = entries, # commented out for a short only backtest    
      short_entries       = short_entries,     
      td_stop             = prediction_window,
      time_delta_format   = 'Rows', # Use the row index to calculate the time delta    
      )
  
  store_backtest_results(LSTM_PREDICTION_WINDOW_EXITS_BACKTEST_RESULT_KEY, pf, results_as_list, None)