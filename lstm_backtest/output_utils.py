import os
import pandas as pd
from lstm_analysis_utils import process_pickle_files
from parameter_optimization import DataFrameFormat
from parameter_optimization_factory import VbtBackTestProcessorType, VbtBackTestProcessorFactory
import vectorbtpro as vbt

from settings_and_params import extract_prediction_window_size, generate_csv_for_excel_output_file_path, generate_dataframe_csv_output_file_path

PICKLE_FILES_INPUT_PREFIX = "RID"

def export_raw_dataframe_to_csv(df: pd.DataFrame, output_file_path: str):  
  copy_df         = df.copy(deep=True)
  columns_to_drop = copy_df.filter(like="BTCUSDT_Open_").columns

  copy_df.drop(columns_to_drop, axis=1, inplace=True)

  copy_df.to_csv(output_file_path)


# joe_df = df.copy(deep=True)

# output_df           = joe_df[['BTCUSDT_Open', 'BTCUSDT_High', 'BTCUSDT_Low','BTCUSDT_Close']]
# entries.index       = df.index
# short_entries.index = df.index
# output_df.loc[:, "entries"] = entries
# output_df.loc[:, "short_entries"] = short_entries

# Generates all the dataframe CSV files from the given dirs
def export_all_raw_dataframes_to_csv(input_path: str, output_path: str):  
  entries = os.scandir(input_path)
  sorted_entries = sorted(entries, key=lambda entry: entry.name)
  
  for entry in sorted_entries:
    if entry.is_dir() and PICKLE_FILES_INPUT_PREFIX in entry.name:
      try:
        print(f"Processing {entry.name}....")
        prediction_window               = extract_prediction_window_size(entry.name)       
        dataframe_csv_output_file_name  = generate_dataframe_csv_output_file_path(entry.name, output_path)
        df = process_pickle_files(entry.path, prediction_window)    
        export_raw_dataframe_to_csv(df, dataframe_csv_output_file_name)     
      except Exception as e:
        pass



# Do the full analysis of each input folder: read, process, run backtests, and export CSV files
def perform_full_analysis_on_all_input_dirs(input_path: str, output_path: str):  
  entries = os.scandir(input_path)
  sorted_entries = sorted(entries, key=lambda entry: entry.name)

  for entry in sorted_entries:
    if entry.is_dir() and PICKLE_FILES_INPUT_PREFIX in entry.name:
      vbt.settings.wrapping ["freq"]                = "1m"
      vbt.settings.portfolio['init_cash']           = 10000
      
      try:
        print(f"Performing analysis on {entry.name}....")

        model_name                      = entry.name
        prediction_window               = extract_prediction_window_size(model_name)
        csv_for_excel_output_file_name  = generate_csv_for_excel_output_file_path(model_name, output_path)
        dataframe_csv_output_file_name  = generate_dataframe_csv_output_file_path(model_name, output_path)
                  
        df        = process_pickle_files(entry.path, prediction_window)    
        processor = VbtBackTestProcessorFactory.create(VbtBackTestProcessorType.WITH_MEMORY_CONSTRAINT_TWO_LOOPS, df, prediction_window, DataFrameFormat.SINGLE)

        if processor:
          result = processor.run_backtests()

          if result is not None:
            result.to_csv(csv_for_excel_output_file_name)

          export_raw_dataframe_to_csv(df, dataframe_csv_output_file_name)     
      except Exception as e:
        print("---Failed to analyze - exception: ", e)
        
        




if __name__ == '__main__':
  perform_full_analysis_on_all_input_dirs("/home/htram/onramp/sigma/data", "/home/htram/onramp/sigma/results")
