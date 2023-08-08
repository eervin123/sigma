from typing import List, Optional
import pandas as pd
from dataclasses import dataclass
from settings_and_params import extract_prediction_window_size, extract_run_id


@dataclass
class DataFrameInfo:
  name                  : str
  df                    : pd.DataFrame
  prediction_window_size: int


class DataFrameMerger:
  def process(self, csv_list: List[str]) -> Optional[pd.DataFrame]:
    df_info = self._read_all_dataframe_csv_files(csv_list)

    if self._validate_dataframes(df_info):
      merged_df = self._create_merge_dataframe(df_info)
      self._calculate_values(merged_df)

      return merged_df
    else:
      print("The CSV files have mismatching prediction window sizes, and/or mismatching number of rows.")          
      return None



  def _read_dataframe_csv(self, file_path: str) -> DataFrameInfo:        
    file_name               = file_path.split('/')[-1].split('.')[0]
    prediction_window_size  = extract_prediction_window_size(file_name)
    run_id                  = extract_run_id(file_name)
    df                      = pd.read_csv(file_path)
        
    columns_to_keep = ['close_time', 'BTCUSDT_Open', 'BTCUSDT_High', 'BTCUSDT_Low', 'BTCUSDT_Close', 'long_minus_short', 'long_slope', 'short_slope']
    df = df[columns_to_keep]

    column_name_change = {
        'BTCUSDT_Open'      : "open"
      , 'BTCUSDT_High'      : "high"
      , 'BTCUSDT_Low'       : "low"
      , 'BTCUSDT_Close'     : "close"
      , 'long_minus_short'  : f"long_minus_short_{run_id}"
      , 'long_slope'        : f"long_slope_{run_id}"
      , 'short_slope'       : f"short_slope_{run_id}"
    }
    df.rename(columns=column_name_change, inplace=True)

    df.set_index("close_time", inplace=True)

    return DataFrameInfo(file_name, df, prediction_window_size)



  def _read_all_dataframe_csv_files(self, csv_list: List[str]) -> List[DataFrameInfo]:
    return [self._read_dataframe_csv(csv_file) for csv_file in csv_list]  
  


  def _validate_dataframes(self, df_list: List[DataFrameInfo]):
    first_entry = df_list[0]

    return all((entry.prediction_window_size == first_entry.prediction_window_size) and
               (len(entry.df) == len(first_entry.df))
                for entry in df_list)
  


  def _create_merge_dataframe(self, df_list: List[DataFrameInfo]) -> pd.DataFrame:
    merged_df = df_list[0].df

    if len(df_list) > 1:
      columns_to_drop = ['open', 'high', 'low', 'close']

      for df_info in df_list[1:]:
        df_info.df.drop(columns=columns_to_drop, inplace=True)
        merged_df = merged_df.merge(df_info.df, on="close_time")

    return merged_df
  


  def _calculate_avg_value(self, df: pd.DataFrame, col_prefix: str):
    col_names     = [col for col in df.columns if col.startswith(col_prefix)]
    new_col_name  = f"{col_prefix}_avg"
    df[new_col_name] = df[col_names].apply(lambda row: row.mean(), axis=1)


  
  def _calculate_values(self, df: pd.DataFrame):
    self._calculate_avg_value(df, "long_slope")
    self._calculate_avg_value(df, "short_slope")
    self._calculate_avg_value(df, "long_minus_short")