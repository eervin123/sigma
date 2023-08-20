import pandas as pd
from typing import List
from multiple_models_backtesting import select_top_n_combinations_from_one_backtest
from settings_and_params import extract_model_name_from_csv_for_excel_output_file_path, generate_csv_for_excel_output_file_path, get_all_result_csv_full_file_paths, get_default_results_dir_absolute_path

MODEL_NAME_COLUMN       = "model_name"
MODEL_NAME_COLUMN_INDEX = 0

class BacktestResultsAggregator:
  def __init__(self, num_top_combinations_to_select: int):
    self.num_top_combinations_to_select = num_top_combinations_to_select



  def aggregate_from_list(self, model_names: List[str]) -> pd.DataFrame:
    combined_df = None

    for model_name in model_names:
      curr_df = self._extract_top_combinations_from_one_backtest(model_name)

      if combined_df is None:
        combined_df = curr_df
      else:
        combined_df = pd.concat([combined_df, curr_df], ignore_index=True)
    

    return combined_df
  


  def aggregate_from_dir(self, dir_path: str = get_default_results_dir_absolute_path()) -> pd.DataFrame:
    paths       = get_all_result_csv_full_file_paths(dir_path)
    model_names = [extract_model_name_from_csv_for_excel_output_file_path(path) for path in paths]

    return self.aggregate_from_list(model_names)



  def _extract_top_combinations_from_one_backtest(self, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(generate_csv_for_excel_output_file_path(model_name))
    df = select_top_n_combinations_from_one_backtest(df, self.num_top_combinations_to_select)

    df.insert(MODEL_NAME_COLUMN_INDEX, MODEL_NAME_COLUMN, model_name)

    return df
    