from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, List, Optional, Tuple
from dataframes_merger import BaseDataFrameMerger, DataFrameMergerUtils
from settings_and_params import extract_prediction_window_sizes, extract_run_id, get_data_frame_file_path, get_results_file_path
from parameter_optimization import MERGED_CLOSE_COL_NAME, MERGED_HIGH_COL_NAME, MERGED_LOW_COL_NAME, MERGED_OPEN_COL_NAME, DataFrameFormat, Thresholds, extract_metrics_from_single_result, generate_index_list
from parameter_optimization_factory import VbtBackTestProcessorFactory, VbtBackTestProcessorType





class MultiModelBacktest(ABC):
  def __init__(self, merger: BaseDataFrameMerger, model_names: List[str]):
    self.merger                 = merger
    self.model_names            = model_names
    self.prediction_window_size = None



  def run(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if self._validate():      
      self.model_names.sort()

      merged_df = self.merger.process([get_data_frame_file_path(entry) for entry in self.model_names])
      result    = self._run_backtest(merged_df)

      return merged_df, result
    
    return None, None



  def _validate(self) -> bool:
    values = extract_prediction_window_sizes(self.model_names)

    if len(values) == 1:
      self.prediction_window_size = values.pop()

      return True
    else:
      raise Exception("The prediction window sizes are not all the same")     


  @abstractmethod
  def _run_backtest(self, merged_df: pd.DataFrame) -> pd.DataFrame:
    pass
    


class AverageMultiModelBacktest(MultiModelBacktest):  
  def _run_backtest(self, merged_df: pd.DataFrame) -> pd.DataFrame:
    return VbtBackTestProcessorFactory.create(VbtBackTestProcessorType.WITH_MEMORY_CONSTRAINT_TWO_LOOPS, merged_df, self.prediction_window_size, DataFrameFormat.MERGED).run_backtest()
  


NUM_TOP_COMBINATIONS_TO_SELECT = 5





def calculate_majority_count(num_cols: int) -> int:  
  return (num_cols // 2) + 1



def calculate_entries_using_majority_rule(df: pd.DataFrame, thresholds: List[Thresholds]):
  entries_series_list       = []
  short_entries_series_list = []

  for model_thresholds in thresholds:
    lms_col_name          = DataFrameMergerUtils.get_long_minus_short_col_name(model_thresholds.model_id)
    long_slope_col_name   = DataFrameMergerUtils.get_long_slope_col_name(model_thresholds.model_id)
    short_slope_col_name  = DataFrameMergerUtils.get_short_slope_col_name(model_thresholds.model_id)

    entries_series        = pd.Series(np.where((df[lms_col_name] < model_thresholds.long_minus_short) & (df[long_slope_col_name ] > model_thresholds.long_slope ), True, False))
    short_entries_series  = pd.Series(np.where((df[lms_col_name] < model_thresholds.long_minus_short) & (df[short_slope_col_name] < model_thresholds.short_slope), True, False))

    entries_series_list.append(entries_series)
    short_entries_series_list.append(short_entries_series)

  num_true_entries        = sum(entries_series_list)
  num_true_short_entries  = sum(short_entries_series_list)
  majority_count          = calculate_majority_count(len(thresholds))

  majority_entries        = num_true_entries >= majority_count
  majority_short_entries  = num_true_short_entries >= majority_count

  return majority_entries, majority_short_entries



def select_top_n_combinations_from_one_backtest(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df["combined_weight"]         = df["total_return"] * df["win_rate"] * df["sharpe_ratio"] * df["sortino_ratio"] * (1 + df["max_drawdown"]) * df["profit_factor"]
    df["long_short_count"]        = df["long_count"] + df["short_count"]
    df["long_vs_short_diff_pct"]  = abs((df["long_count"] / df["long_short_count"]) - (df["short_count"] / df["long_short_count"]))

    filtered_df = df[(df["long_vs_short_diff_pct"] < 0.5)].nlargest(n, "combined_weight")

    return filtered_df



class MajorityMultiModelBacktest(MultiModelBacktest):
  def _run_backtest(self, merged_df: pd.DataFrame) -> pd.DataFrame:
    # Steps:
    # 1. Load the individual backtest result for each model - so that we don't have to re-run them
    full_backtest_results = self._load_result_files()
    # 2. Select the top N combinations for each model
    top_N_combinations = self._select_top_performing_combinations(full_backtest_results, NUM_TOP_COMBINATIONS_TO_SELECT)
    top_N_thresholds   = self._extract_thresholds_from_models(top_N_combinations)
    combination_list   = self._generate_flattened_combination_list(top_N_thresholds)
    # 3. Run the backtest for each combination in the list
    return self._run_all_backtests(merged_df, combination_list)    
  


  def _load_result_files(self) -> Dict[str, pd.DataFrame]:
    return {extract_run_id(model_name): pd.read_csv(get_results_file_path(model_name)) for model_name in self.model_names}
  


  def _select_top_performing_combinations(self, full_results: Dict[str, pd.DataFrame], n: int) -> Dict[str, pd.DataFrame]:
    return {key: select_top_n_combinations_from_one_backtest(value, n) for key, value in full_results.items()}
    



  def _extract_thresholds_from_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Thresholds]]:
    result = {}

    for key, value in data.items():
      result[key] = self._extract_thresholds(value, key)

    return result
  


  def _extract_thresholds(self, df: pd.DataFrame, model_id: str) -> List[Thresholds]:
    return [Thresholds( model_id         = model_id,
                        long_minus_short = row['LMSWithSlopes_lms_threshold'],
                        long_slope       = row['LMSWithSlopes_long_slope_thresh'],
                        short_slope      = row['LMSWithSlopes_short_slope_thresh'])
                        
            for _, row in df.iterrows()]
  


  def _generate_flattened_combination_list(self, data: Dict[str, List[Thresholds]]) -> List[List[Thresholds]]:
    # Get the values from the dictionary
    values = list(data.values())

    # Generate all possible combinations
    combinations = list(itertools.product(*values))

    return [list(entry) for entry in combinations]
  
  

  def _run_all_backtests(self, merged_df: pd.DataFrame, combinations: List[List[Thresholds]]):
    all_stats = None
    
    for combination in combinations:
      entries, short_entries = calculate_entries_using_majority_rule(merged_df, combination)

      num_entries = (entries == True).sum() + (short_entries == True).sum()

      if num_entries > 100:    
        current_portfolio = vbt.Portfolio.from_signals(
            high              = merged_df[MERGED_HIGH_COL_NAME],
            low               = merged_df[MERGED_LOW_COL_NAME],
            open              = merged_df[MERGED_OPEN_COL_NAME],
            close             = merged_df[MERGED_CLOSE_COL_NAME],
            entries           = entries, 
            short_entries     = short_entries,
            td_stop           = self.prediction_window_size, 
            time_delta_format = 'Rows', # Use the row index to calculate the time delta              
            accumulate        = False,            
            )    
        
        curr_stats = extract_metrics_from_single_result(combination, current_portfolio)    

        if all_stats is None:
          all_stats = curr_stats
        elif curr_stats is not None:
          # concat curr_stats to all_stats
          all_stats = pd.concat([all_stats, curr_stats], ignore_index=True)
    
    if all_stats is not None:
      index_list_of_lists = [generate_index_list(entry.model_id) for entry in combinations[0]]
      flattened_list = [item for sublist in index_list_of_lists for item in sublist]
      all_stats.set_index(flattened_list, inplace=True)

    return all_stats
  
  


def run_one_multi_model_backtest_majority_rule(merger: BaseDataFrameMerger, model_names: List[str], combination: List[Thresholds]):
  model_names.sort()
  prediction_window_size = extract_prediction_window_sizes(model_names).pop()
  merged_df = merger.process([get_data_frame_file_path(entry) for entry in model_names])
  entries, short_entries = calculate_entries_using_majority_rule(merged_df, combination)

  current_portfolio = vbt.Portfolio.from_signals(
            high              = merged_df[MERGED_HIGH_COL_NAME],
            low               = merged_df[MERGED_LOW_COL_NAME],
            open              = merged_df[MERGED_OPEN_COL_NAME],
            close             = merged_df[MERGED_CLOSE_COL_NAME],
            entries           = entries, 
            short_entries     = short_entries,
            td_stop           = prediction_window_size, 
            time_delta_format = 'Rows', # Use the row index to calculate the time delta              
            accumulate        = False,            
            )
  
  return current_portfolio
  



  



  



  