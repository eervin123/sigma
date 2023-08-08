import re


INPUT_DIR   = "../data/"
OUTPUT_DIR  = "../results/"


def extract_prediction_window_size(model_name: str) -> int:
  pattern = r'_pw(\d+)_'
  match   = re.search(pattern, model_name)

  if match:
    return int(match.group(1))
  else:
    return None
  


def extract_run_id(model_name: str) -> str:
  parts = model_name.split("_")
  
  return parts[0]



def generate_excel_output_file_path(model_name: str) -> str:
  return OUTPUT_DIR + f"{model_name + '.xlsx'}"



def generate_dataframe_csv_output_file_path(model_name: str) -> str:
  return OUTPUT_DIR + f"{model_name + '.csv'}"
