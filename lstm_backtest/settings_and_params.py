import re
import os
from typing import List, Set, Tuple


INPUT_DIR         = "../data"
OUTPUT_DIR        = "../results"
RESULT_CSV_SUFFIX = "_convert_to_excel"


def extract_prediction_window_size(model_name: str) -> int:
  pattern = r'_pw(\d+)_'
  match   = re.search(pattern, model_name)

  if match:
    return int(match.group(1))
  else:
    return None
  


def extract_file_name_and_extension(file_path: str) -> Tuple[str, str]:
  base_name = os.path.basename(file_path)

  return os.path.splitext(base_name)

  


def extract_run_id(model_name: str) -> str:
  parts = model_name.split("_")
  
  return parts[0]



def extract_run_id_from_file_path(file_path: str) -> str:
  file_name, _ = extract_file_name_and_extension(file_path)
  return extract_run_id(file_name)




def extract_prediction_window_sizes(group: List[str]) -> Set[int]:
  return set([extract_prediction_window_size(entry) for entry in group])



def extract_run_ids(group: List[str]) -> List[str]:
  return [extract_run_id_from_file_path(file_path) for file_path in group]




def generate_excel_output_file_path(model_name: str, output_dir: str = OUTPUT_DIR) -> str:
  return output_dir + f"/{model_name + '.xlsx'}"



def generate_csv_for_excel_output_file_path(model_name: str, output_dir: str = OUTPUT_DIR) -> str:
  return output_dir + f"/{model_name + RESULT_CSV_SUFFIX + '.csv'}"


def generate_dataframe_csv_output_file_path(model_name: str, output_dir: str = OUTPUT_DIR) -> str:
  return output_dir + f"/{model_name + '.csv'}"




def construct_output_file_name_no_ext(group: List[str]) -> str:
  run_ids = extract_run_ids(group)
  run_ids.sort()
  file_name = "_".join(run_ids)

  return f"{file_name}"



def generate_multiple_models_backtest_output_file_name_no_ext(group: List[str], method: str) -> str:
  return f"{construct_output_file_name_no_ext(group)}_{method}"



  
def get_data_frame_file_path(model_name: str) -> str:
  relative_path = f"{OUTPUT_DIR}/{model_name}" + '.csv'

  return os.path.abspath(relative_path)



def get_results_file_path(model_name: str) -> str:
  relative_path = f"{OUTPUT_DIR}/{model_name + RESULT_CSV_SUFFIX }" + '.csv'

  return os.path.abspath(relative_path)