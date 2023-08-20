import re
import os
import glob
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



def extract_model_name_from_csv_for_excel_output_file_path(file_path: str) -> str:
  name, _ = extract_file_name_and_extension(file_path)
  name = name.replace(RESULT_CSV_SUFFIX, "")

  return name




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



def generate_aggregated_top_N_output_file_path(n: int, output_dir: str = OUTPUT_DIR) -> str:
  return output_dir + f"/aggregated_results_top{n}" + '.csv'


  
def get_data_frame_file_path(model_name: str) -> str:
  relative_path = f"{OUTPUT_DIR}/{model_name}" + '.csv'

  return os.path.abspath(relative_path)



def get_results_file_path(model_name: str) -> str:
  relative_path = f"{OUTPUT_DIR}/{model_name + RESULT_CSV_SUFFIX }" + '.csv'

  return os.path.abspath(relative_path)



def get_default_results_dir_absolute_path() -> str:
  return os.path.abspath(OUTPUT_DIR)



def get_all_result_csv_full_file_paths(path: str = get_default_results_dir_absolute_path()) -> List[str]:
  all_csv_files   = glob.glob(os.path.join(path, "*.csv"))
  filtered_files  = [file_name for file_name in all_csv_files if RESULT_CSV_SUFFIX in file_name]

  filtered_files.sort()

  return filtered_files


