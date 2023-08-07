import pandas as pd
import vectorbtpro as vbt
import numpy as np
from typing import List, Tuple
from lstm_results_utils import store_backtest_results
from prediction_window_slopes import PredictionWindowSlopes
from lstm_analysis_constants import EntryType


def _generate_slope_thresholds(df: pd.DataFrame, threshold_increment: float) -> Tuple[List]:
  min_long_slope  = df["long_slope"].min()
  max_long_slope  = df["long_slope"].max()
  min_short_slope = df["short_slope"].min()
  max_short_slope = df["short_slope"].max()

  entry_slope_threshold       = [x for x in np.arange(min_long_slope , max_long_slope , threshold_increment)]
  short_entry_slope_threshold = [x for x in np.arange(min_short_slope, max_short_slope, threshold_increment)]

  return entry_slope_threshold, short_entry_slope_threshold




def run_backtest_long_slope_short_slope_prediction_size_exit(df: pd.DataFrame, results_as_list: List, prediction_window: int, threshold_increment: float, min_num_entries: int):  
  entry_slope_threshold, short_entry_slope_threshold = _generate_slope_thresholds(df, threshold_increment)

  for t1 in entry_slope_threshold:    
    for t2 in short_entry_slope_threshold:        
      entries       = pd.Series(np.where((df['long_slope' ] > t1 ), True, False))
      short_entries = pd.Series(np.where((df['short_slope'] < t2 ), True, False))        

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
            # sl_stop = 0.005,
            )    
        
        key = f"Long slope and short slope entries, prediction window exits"
        slopes = PredictionWindowSlopes(t1, t2, None, None, None, EntryType.LONG_SHORT)
        store_backtest_results(key, pf, results_as_list, slopes)



def run_backtest_long_slope_short_slope_fractional_exits(df: pd.DataFrame, results_as_list: List, threshold_increment: float, min_num_entries: int):  
  entry_slope_threshold, short_entry_slope_threshold = _generate_slope_thresholds(df, threshold_increment)
  exit_percentages = [0.25, 0.50, 0.75]

  for t1 in entry_slope_threshold:      
    exit_t1_thresholds = [t1 * entry for entry in exit_percentages]

    for t2 in short_entry_slope_threshold:      
      exit_t2_thresholds = [t2 * entry for entry in exit_percentages]
      entries       = pd.Series(np.where((df['long_slope' ] > t1 ), True, False))
      short_entries = pd.Series(np.where((df['short_slope'] < t2 ), True, False))

      for exit_t1_threshold in exit_t1_thresholds:
        for exit_t2_threshold in exit_t2_thresholds:
          exits         = pd.Series(np.where((df['long_slope' ] < exit_t1_threshold ), True, False)) 
          short_exits   = pd.Series(np.where((df['short_slope'] > exit_t2_threshold ), True, False))

          num_entries = (entries == True).sum() + (short_entries == True).sum()

          if num_entries > min_num_entries:    
            pf = vbt.Portfolio.from_signals(
                high              = df['BTCUSDT_High'],
                low               = df['BTCUSDT_Low'],
                open              = df['BTCUSDT_Open'],
                close             = df['BTCUSDT_Close'],
                entries           = entries, # commented out for a short only backtest          
                exits             = exits,
                short_entries     = short_entries,              
                short_exits       = short_exits,
                time_delta_format = 'Rows', # Use the row index to calculate the time delta              
                accumulate        = False,
                # sl_stop = 0.005,
                )    
            
            key = f"Long slope and short slope entries, fractional slope exits"
            slopes = PredictionWindowSlopes(t1, t2, exit_t1_threshold, exit_t2_threshold, None, EntryType.LONG_SHORT)
            store_backtest_results(key, pf, results_as_list, slopes)      