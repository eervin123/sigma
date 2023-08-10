from dataclasses import dataclass
from enum import Enum
import pandas as pd
import vectorbtpro as vbt
import numpy as np

DEFAULT_NUM_INCREMENTS            = 10
MINIMUM_NUM_TRADES                = 100


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
  SINGLE = 1
  MERGED = 2


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



def generate_thresholds(df: pd.DataFrame,
                        lms_col_name        : str,
                        long_slope_col_name : str,
                        short_slope_col_name: str):
  long_slope_min              = df[long_slope_col_name].min()
  long_slope_max              = df[long_slope_col_name].max()
  long_slope_increment        = abs(long_slope_max - long_slope_min) / DEFAULT_NUM_INCREMENTS

  short_slope_min             = df[short_slope_col_name].min()
  short_slope_max             = df[short_slope_col_name].max()
  short_slope_increment       = abs(short_slope_max - short_slope_min) / DEFAULT_NUM_INCREMENTS

  long_minus_short_min        = df[lms_col_name].min()
  long_minus_short_max        = df[lms_col_name].max()
  long_minus_short_increment  = abs(long_minus_short_max - long_minus_short_min) / DEFAULT_NUM_INCREMENTS

  lms_threshold       = np.arange(long_minus_short_min, long_minus_short_max, long_minus_short_increment)
  long_slope_thresh   = np.arange(long_slope_min, long_slope_max, long_slope_increment)
  short_slope_thresh  = np.arange(short_slope_min, short_slope_max, short_slope_increment)

  return lms_threshold, long_slope_thresh, short_slope_thresh



#### UNLIMITED MEMORY - Begin ####
def create_strategy(df: pd.DataFrame,
                    lms_col_name        : str, 
                    long_slope_col_name : str, 
                    short_slope_col_name: str): 
  indicator = create_indicator()  
  lms_thresholds, long_thresholds, short_thresholds = generate_thresholds(df, lms_col_name, long_slope_col_name, short_slope_col_name)
      
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



def run_vbt_backtest(df: pd.DataFrame, prediction_window_size: int, dataframe_format: DataFrameFormat):
  # No tp_stop, no sl_stop
  format_mapping  = DATAFRAME_FORMAT_MAPPING.get(dataframe_format)
  strategy        = create_strategy(df, format_mapping.long_minus_short_col_name, format_mapping.long_slope_col_name, format_mapping.short_slope_col_name)

  multiple_pf = vbt.Portfolio.from_signals(
      close               = df[format_mapping.close_col_name],
      high                = df[format_mapping.high_col_name],
      low                 = df[format_mapping.low_col_name],
      open                = df[format_mapping.open_col_name],
      entries             = strategy.entries,
      short_entries       = strategy.short_entries,
      td_stop             = prediction_window_size,
      time_delta_format   = 'Rows',
      accumulate          = False,
  )

  return multiple_pf
#### UNLIMITED MEMORY - End ####


  

def extract_metrics_from_result(portfolios, output_file_path: str):
  min_num_trade_filter  = portfolios.trades.count() > MINIMUM_NUM_TRADES
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
  stats_df.to_csv(output_file_path)

