from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import pandas as pd
import vectorbtpro as vbt
import numpy as np
from abc import ABC, abstractmethod

DEFAULT_NUM_INCREMENTS              = 30
MINIMUM_NUM_TRADES                  = 100
OUTTERMOST_LOOP_BACKTEST_CHUNK_SIZE = 1       # Controls the size of the outer loop
MIDDLE_LOOP_BACKTEST_CHUNK_SIZE     = 5       # Controls the size of the middle loop


# Single RID column names
SINGLE_CLOSE_COL_NAME            = 'BTCUSDT_Close'
SINGLE_HIGH_COL_NAME             = 'BTCUSDT_High'
SINGLE_LOW_COL_NAME              = 'BTCUSDT_Low'
SINGLE_OPEN_COL_NAME             = 'BTCUSDT_Open'
SINGLE_LONG_MINUS_SHORT_COL_NAME = "long_minus_short"
SINGLE_LONG_SLOPE_COL_NAME       = "long_slope"
SINGLE_SHORT_SLOPE_COL_NAME      = "short_slope"

# Merged RID column names
MERGED_CLOSE_COL_NAME            = 'close'
MERGED_HIGH_COL_NAME             = 'high'
MERGED_LOW_COL_NAME              = 'low'
MERGED_OPEN_COL_NAME             = 'open'
MERGED_LONG_MINUS_SHORT_COL_NAME = "long_minus_short_avg"
MERGED_LONG_SLOPE_COL_NAME       = "long_slope_avg"
MERGED_SHORT_SLOPE_COL_NAME      = "short_slope_avg"

@dataclass
class DataFrameColNames:
  close_col_name            : str
  high_col_name             : str
  low_col_name              : str
  open_col_name             : str
  long_minus_short_col_name : str
  long_slope_col_name       : str
  short_slope_col_name      : str

  
class DataFrameFormat(Enum):
  SINGLE = 1                      # The dataframe was generated using the function process_pickle_files()
  MERGED = 2                      # The dataframe was generated using one of the concretions of BaseDataFrameMerger


DATAFRAME_FORMAT_MAPPING = {
  DataFrameFormat.SINGLE: DataFrameColNames(SINGLE_CLOSE_COL_NAME, SINGLE_HIGH_COL_NAME, SINGLE_LOW_COL_NAME, SINGLE_OPEN_COL_NAME, SINGLE_LONG_MINUS_SHORT_COL_NAME, SINGLE_LONG_SLOPE_COL_NAME, SINGLE_SHORT_SLOPE_COL_NAME),
  DataFrameFormat.MERGED: DataFrameColNames(MERGED_CLOSE_COL_NAME, MERGED_HIGH_COL_NAME, MERGED_LOW_COL_NAME, MERGED_OPEN_COL_NAME, MERGED_LONG_MINUS_SHORT_COL_NAME, MERGED_LONG_SLOPE_COL_NAME, MERGED_SHORT_SLOPE_COL_NAME)
}


def lms_with_slopes_indicator_func(  long_minus_short, long_slope, short_slope            # input names
                                   , lms_threshold, long_slope_thresh, short_slope_thresh # param names
                                  ):
  entries       = pd.Series(np.where((long_minus_short < lms_threshold) & (long_slope  > long_slope_thresh ), True, False))
  short_entries = pd.Series(np.where((long_minus_short < lms_threshold) & (short_slope < short_slope_thresh), True, False))
  
  return entries, short_entries



def create_indicator():
  indicator = vbt.IndicatorFactory(
    class_name    = 'LongMinusShortwithSlopes',                                   # name of the class
    short_name    = 'LMSWithSlopes',                                              # name of the indicator
    input_names   = ['long_minus_short', 'long_slope', 'short_slope'],            # names of input arguments
    param_names   = ['lms_threshold', 'long_slope_thresh', 'short_slope_thresh'], # names of parameters
    output_names  = ['entries', 'short_entries'],                                 # names of output values
  ).with_apply_func(
    lms_with_slopes_indicator_func,                                               # function to apply
    takes_1d            = True,                                                   # whether the function takes 1-dim. arrays as input
    lms_threshold       = 0.5,                                                    # default value for parameter 'lms_threshold'
    long_slope_thresh   = 0.0,                                                    # default value for parameter 'long_slope_thresh'
    short_slope_thresh  = 0.0,                                                    # default value for parameter 'short_slope_thresh'
  )

  return indicator



def create_strategy(df: pd.DataFrame,
                    lms_thresholds      : np.array,
                    long_thresholds     : np.array,
                    short_thresholds    : np.array,
                    lms_col_name        : str, 
                    long_slope_col_name : str, 
                    short_slope_col_name: str): 
  indicator = create_indicator()    
      
  strategy = indicator.run(
      long_minus_short    = df[lms_col_name],
      long_slope          = df[long_slope_col_name],
      short_slope         = df[short_slope_col_name],
      lms_threshold       = lms_thresholds,
      long_slope_thresh   = long_thresholds,
      short_slope_thresh  = short_thresholds,
      param_product       = True, # True: all combinations of parameters, False: only one combination for each parameter
  )

  return strategy

  

def extract_metrics_from_result(portfolios) -> pd.DataFrame:
  stats_df              = None
  min_num_trade_filter  = portfolios.trades.count() > MINIMUM_NUM_TRADES    

  if min_num_trade_filter.any():
    filtered_pf           = portfolios.loc[:, min_num_trade_filter]
    
    metrics_dict = {
      'total_return'    : filtered_pf.total_return,
      'win_rate'        : filtered_pf.trades.win_rate,
      'sharpe_ratio'    : filtered_pf.sharpe_ratio,
      'sortino_ratio'   : filtered_pf.sortino_ratio,
      'max_drawdown'    : filtered_pf.max_drawdown,
      'profit_factor'   : filtered_pf.trades.profit_factor,
      'long_count'      : filtered_pf.trades.direction_long.count(),
      'short_count'     : filtered_pf.trades.direction_short.count(),
      'long_pnl_sum'    : filtered_pf.trades.direction_long.pnl.sum(),
      'short_pnl_sum'   : filtered_pf.trades.direction_short.pnl.sum()
    }

    stats_df = pd.concat(list(metrics_dict.values()), axis=1, keys=list(metrics_dict.keys()))
  
  return stats_df



class BaseVbtBackTestProcessor(ABC):
  def __init__(self, df: pd.DataFrame, prediction_window_size: int, dataframe_format: DataFrameFormat):
    self.df                     = df
    self.prediction_window_size = prediction_window_size
    self.format_mapping         = DATAFRAME_FORMAT_MAPPING.get(dataframe_format)    



  def run_backtest(self):
    # No tp_stop, no sl_stop
    all_stats = None
    outer_loop_thresholds, middle_loop_thresholds, short_thresholds = self._generate_thresholds()    

    for entry in outer_loop_thresholds:
      for middle_entry in middle_loop_thresholds:
        strategy            = create_strategy(self.df, entry, middle_entry, short_thresholds, self.format_mapping.long_minus_short_col_name, self.format_mapping.long_slope_col_name, self.format_mapping.short_slope_col_name)
        current_portfolios  = vbt.Portfolio.from_signals(
            close               = self.df[self.format_mapping.close_col_name],
            high                = self.df[self.format_mapping.high_col_name],
            low                 = self.df[self.format_mapping.low_col_name],
            open                = self.df[self.format_mapping.open_col_name],
            entries             = strategy.entries,
            short_entries       = strategy.short_entries,
            td_stop             = self.prediction_window_size,
            time_delta_format   = 'Rows',
            accumulate          = False,
        )
        curr_stats = extract_metrics_from_result(current_portfolios)    

        if all_stats is None:
          all_stats = curr_stats
        elif curr_stats is not None:
          # concat curr_stats to all_stats
          all_stats = pd.concat([all_stats, curr_stats], axis=0)    

    return all_stats
          
  


  def _generate_raw_thresholds(self) -> Tuple[np.array, np.array, np.array]:
    long_slope_min              = self.df[self.format_mapping.long_slope_col_name].min()
    long_slope_max              = self.df[self.format_mapping.long_slope_col_name].max()    

    short_slope_min             = self.df[self.format_mapping.short_slope_col_name].min()
    short_slope_max             = self.df[self.format_mapping.short_slope_col_name].max()    

    long_minus_short_min        = self.df[self.format_mapping.long_minus_short_col_name].min()
    long_minus_short_max        = self.df[self.format_mapping.long_minus_short_col_name].max()    

    lms_threshold       = np.linspace(long_minus_short_min, long_minus_short_max, DEFAULT_NUM_INCREMENTS)
    long_slope_thresh   = np.linspace(long_slope_min      , long_slope_max      , DEFAULT_NUM_INCREMENTS)
    short_slope_thresh  = np.linspace(short_slope_min     , short_slope_max     , DEFAULT_NUM_INCREMENTS)

    return lms_threshold, long_slope_thresh, short_slope_thresh
  


  @abstractmethod
  def _generate_thresholds(self) -> Tuple[List[np.array], List[np.array], np.array]:
    pass



class VbtBackTestProcessorNoMemoryConstraint(BaseVbtBackTestProcessor):
  def _generate_thresholds(self) -> Tuple[List[np.array], List[np.array], np.array]:
    lms_thresholds, long_thresholds, short_thresholds = self._generate_raw_thresholds()

    return [lms_thresholds], [long_thresholds], short_thresholds
  



class VbtBackTestProcessorWithMemoryConstraint(BaseVbtBackTestProcessor):  
  def _split_into_chunks(self, array: np.array, chunk_size: int) -> List[np.array]:
    return [array[i:i+chunk_size] for i in range(0, len(array), chunk_size)]




class VbtBackTestProcessorOneLoopMemoryConstraint(VbtBackTestProcessorWithMemoryConstraint):
  def _generate_thresholds(self) -> Tuple[List[np.array], List[np.array], np.array]:
    lms_thresholds, long_thresholds, short_thresholds = self._generate_raw_thresholds()

    return self._split_into_chunks(lms_thresholds, OUTTERMOST_LOOP_BACKTEST_CHUNK_SIZE), [long_thresholds], short_thresholds
    
  



class VbtBackTestProcessorTwoLoopMemoryConstraint(VbtBackTestProcessorOneLoopMemoryConstraint):
  def _generate_thresholds(self) -> Tuple[List[np.array], List[np.array], np.array]:
    lms_thresholds, long_thresholds, short_thresholds = self._generate_raw_thresholds()    

    return self._split_into_chunks(lms_thresholds, OUTTERMOST_LOOP_BACKTEST_CHUNK_SIZE), self._split_into_chunks(long_thresholds, MIDDLE_LOOP_BACKTEST_CHUNK_SIZE), short_thresholds
  