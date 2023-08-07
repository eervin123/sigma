import os
import pandas as pd
from lstm_analysis_utils import process_pickle_files

from settings_and_params import INPUT_DIR, OUTPUT_DIR, extract_prediction_window_size, generate_dataframe_csv_output_file_path

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


def export_all_raw_dataframes_to_csv(input_path: str, output_path: str):  
  with os.scandir(input_path) as entries:
    for entry in entries:
      if entry.is_dir() and PICKLE_FILES_INPUT_PREFIX in entry.name:
        try:
          print(f"Processing {entry.name}....")
          prediction_window               = extract_prediction_window_size(entry.name)       
          dataframe_csv_output_file_name  = generate_dataframe_csv_output_file_path(entry.name)
          df = process_pickle_files(entry.path, prediction_window)    
          export_raw_dataframe_to_csv(df, output_path + f"/{dataframe_csv_output_file_name}")     
        except Exception as e:
          pass
        
        




if __name__ == '__main__':
  export_all_raw_dataframes_to_csv("/home/htram/onramp/sigma/data", "/home/htram/onramp/sigma/results")
