import pandas as pd
import vectorbtpro as vbt
import numpy as np
from typing import List, Tuple
from lstm_results_utils import store_backtest_results
from prediction_window_slopes import PredictionWindowSlopes
from lstm_analysis_constants import EntryType


def _calculate_long_minus_short_thresholds(df: pd.DataFrame, threshold_increment: float) -> List:
  long_minus_short_min = df['long_minus_short'].min()
  long_minus_short_max = df['long_minus_short'].max()

  return [x for x in np.arange(long_minus_short_min, long_minus_short_max, threshold_increment)]




def run_backtest_long_minus_short_entry_type_long_only(df: pd.DataFrame, results_as_list: List, prediction_window: int, threshold_increment: float, min_num_entries: int):
  long_minus_short_thresholds = _calculate_long_minus_short_thresholds(df, threshold_increment)

  for threshold in long_minus_short_thresholds:
    entries = pd.Series(np.where((df['long_minus_short'] < threshold), True, False))

    num_entries = (entries == True).sum()

    if num_entries > min_num_entries:
        pf = vbt.Portfolio.from_signals(
            high              = df['BTCUSDT_High'],
            low               = df['BTCUSDT_Low'],
            open              = df['BTCUSDT_Open'],
            close             = df['BTCUSDT_Close'],
            entries           = entries, # commented out for a short only backtest    
            td_stop           = prediction_window, # Hold on to the position for 8 bars
            time_delta_format = 'Rows', # Use the row index to calculate the time delta    
            accumulate        = False,    
            )
        
        key = f"Long minus short"    
        slopes = PredictionWindowSlopes(None, None, None, None, threshold, EntryType.LONG_ONLY)
        store_backtest_results(key, pf, results_as_list, slopes)



def run_backtest_long_minus_short_entry_type_short_only(df: pd.DataFrame, results_as_list: List, prediction_window: int, threshold_increment: float, min_num_entries: int):
  long_minus_short_thresholds = _calculate_long_minus_short_thresholds(df, threshold_increment)

  for threshold in long_minus_short_thresholds:
    short_entries = pd.Series(np.where((df['long_minus_short'] < threshold), True, False))

    num_entries = (short_entries == True).sum()

    if num_entries > min_num_entries:
        pf = vbt.Portfolio.from_signals(
            high              = df['BTCUSDT_High'],
            low               = df['BTCUSDT_Low'],
            open              = df['BTCUSDT_Open'],
            close             = df['BTCUSDT_Close'],
            short_entries     = short_entries, # commented out for a short only backtest    
            td_stop           = prediction_window, # Hold on to the position for 8 bars
            time_delta_format = 'Rows', # Use the row index to calculate the time delta    
            accumulate        = False,    
            )
        
        key = f"Long minus short"    
        slopes = PredictionWindowSlopes(None, None, None, None, threshold, EntryType.SHORT_ONLY)
        store_backtest_results(key, pf, results_as_list, slopes)




def run_backtest_long_minus_short_entry_type_long_short(df: pd.DataFrame, results_as_list: List, prediction_window: int, threshold_increment: float, min_num_entries: int, quantiles: np.ndarray):
  #long_minus_short_thresholds = _calculate_long_minus_short_thresholds(df, threshold_increment)
  long_minus_short_quantiles  = np.linspace(0, 1, num=101)
  long_minus_short_thresholds = [entry for entry in df["long_minus_short"].quantile(long_minus_short_quantiles)]
  long_slope_thresholds       = [entry for entry in df["long_slope"].quantile(quantiles)]
  short_slope_thresholds      = [entry for entry in df["short_slope"].quantile(quantiles)]

  # long_minus_short_thresholds =np.arange(df['long_minus_short'].min(), df['long_minus_short'].max(), threshold_increment)
  # long_slope_thresholds       =np.arange(df["long_slope"].min(), df["long_slope"].max(), threshold_increment)
  # short_slope_thresholds      =np.arange(df["short_slope"].min(), df["short_slope"].max(), threshold_increment)

  for threshold in long_minus_short_thresholds:  
    for long_slope_threshold in long_slope_thresholds:
      for short_slope_threshold in short_slope_thresholds:
        entries       = pd.Series(np.where((df['long_minus_short'] < threshold) & (df['long_slope'] > long_slope_threshold), True, False))
        short_entries = pd.Series(np.where((df['long_minus_short'] < threshold) & (df['short_slope'] < short_slope_threshold), True, False))

        num_entries = (entries == True).sum() + (short_entries == True).sum()

        if num_entries > min_num_entries:
          pf = vbt.Portfolio.from_signals(
              high              = df['BTCUSDT_High'],
              low               = df['BTCUSDT_Low'],
              open              = df['BTCUSDT_Open'],
              close             = df['BTCUSDT_Close'],
              entries           = entries, # commented out for a short only backtest    
              short_entries     = short_entries,
              td_stop           = prediction_window, # Hold on to the position for 8 bars
              time_delta_format = 'Rows', # Use the row index to calculate the time delta    
              accumulate        = False,    
              )

          key = f"Long minus short with slopes"    
          slopes = PredictionWindowSlopes(long_slope_threshold, short_slope_threshold, None, None, threshold, EntryType.LONG_SHORT)
          store_backtest_results(key, pf, results_as_list, slopes)



#run_backtest_long_minus_short_entry_type_long_short(df, results_as_list, prediction_window, threshold_increment, min_num_entries, quantiles)

# loop 1 is split into 100
# for threshold in long_minus_short_thresholds:  
#   for long_slope_threshold in long_slope_quantile_values:
#     for short_slope_threshold in short_slope_quantile_values:
#       entries       = pd.Series(np.where((df['long_minus_short'] < threshold) & (df['long_slope'] > long_slope_threshold), True, False))
#       short_entries = pd.Series(np.where((df['long_minus_short'] < threshold) & (df['short_slope'] < short_slope_threshold), True, False))

#       num_entries = (entries == True).sum() + (short_entries == True).sum()

#       if num_entries > min_num_entries:
#         pf = vbt.Portfolio.from_signals(
#             high              = df['BTCUSDT_High'],
#             low               = df['BTCUSDT_Low'],
#             open              = df['BTCUSDT_Open'],
#             close             = df['BTCUSDT_Close'],
#             entries           = entries, # commented out for a short only backtest    
#             short_entries     = short_entries,
#             td_stop           = prediction_window, # Hold on to the position for 8 bars
#             time_delta_format = 'Rows', # Use the row index to calculate the time delta    
#             accumulate        = False,    
#             )

#         key = f"Long minus short with slopes - type 3"    
#         slopes = PredictionWindowSlopes(long_slope_threshold, short_slope_threshold, threshold, EntryType.LONG_SHORT)
#         store_backtest_results(key, pf, results_as_list, slopes)