import pandas as pd
import glob
from typing import List
import numpy as np
import vectorbtpro as vbt



def _read_pickle_files(path: str) -> List:
  pickle_files  = glob.glob(path)
  data          = []

  pickle_files.sort()

  for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
      file_contents = pd.read_pickle(f)
      data.append(file_contents)

  return data  




def _generate_df(pickle_file_contents: List) -> pd.DataFrame:
  # Define an empty list to store the concatenated dataframes
  merged_data = []

  # Loop over each item in the data list
  for item in pickle_file_contents:
    # Create the initial dataframe with 'Time', the prices, and 'recommendations' along with 'prediction_details'
    data_df = pd.DataFrame(item['price_data']['close_time'], columns=['close_time'])
    data_df[['BTCUSDT_Open', 'BTCUSDT_High', 'BTCUSDT_Low', 'BTCUSDT_Close']] = item['price_data'][['open', 'high', 'low', 'close']]
    
    try:
      data_df['recommendations'] = item['recommendations']
    except Exception as e:
      print(f"No recommendations found in file {item}")

    # Create the dataframe from 'prediction_details'
    predictions_df = pd.DataFrame(item['prediction_details'], columns=['long', 'short', 'indx_hi', 'indx_low', 'action'])

    # Reset the indexes of both dataframes
    data_df.reset_index(drop=True, inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)

    # Concatenate the dataframes horizontally
    merged_df = pd.concat([data_df, predictions_df], axis=1)

    # Append the merged dataframe to the list
    merged_data.append(merged_df)



  # Concatenate the dataframes in merged_data vertically
  result_df = pd.concat(merged_data, axis=0)

  # Reset the index of the final concatenated dataframe
  result_df.reset_index(drop=True, inplace=True)

  # Sort the dataframe by the 'Time' column in ascending order
  result_df.sort_values('close_time', inplace=True)

  # Reset the index again after sorting
  result_df.reset_index(drop=True, inplace=True)

  return result_df




def read_pickle_files_into_df(path: str) -> pd.DataFrame:
  data  = _read_pickle_files(path)
  df    = _generate_df(data)

  return df



def add_forward_prices_to_df(df: pd.DataFrame, prediction_window: int):  
  for i in range(1, prediction_window + 1):
    column_name = f"BTCUSDT_Open_{i}"
    df[column_name] = df['BTCUSDT_Open'].shift(-i)



def generate_fwd_actual_column(df: pd.DataFrame):  
  column_names = [entry for entry in df.columns if 'BTCUSDT_Open' in entry]

  # Calculate the highest high price among the forward prices  
  highest_high = df[column_names].max(axis=1)
  #print(highest_high.shape)
  # Calculate the percentage of each forward price relative to the current open price to the highest high
  ranked_forward_prices = df[column_names[:-1]].div(highest_high, axis=0)
  #print(ranked_forward_prices.shape)
  # Normalize the ranked forward prices to ensure the sum is equal to 1
  ranked_forward_prices_normalized = ranked_forward_prices.div(ranked_forward_prices.sum(axis=1), axis=0)
  #print(ranked_forward_prices_normalized.shape)
  # Convert the normalized ranked forward prices to a numpy array
  array = ranked_forward_prices_normalized.to_numpy()
  #print(array.shape)
  #print(array)

  # Add the column as 'fwd_8_actual' in result_df
  df['fwd_actual'] = array.tolist() # TODO: this may also need to be modified to handle the different forecast periods

  

# Define a function to calculate Euclidean distance
def _calc_euclidean_distance(arr1, arr2):
  return np.sqrt(np.sum((np.array(arr1) - np.array(arr2)) ** 2))



def generate_df_with_euclidean_distances(df: pd.DataFrame, prediction_window: int) -> pd.DataFrame:
  new_df = df[:-prediction_window].copy()

  # Apply this function to the 'long' and 'fwd_8_actual' columns
  new_df['long_distance_to_actual']  = new_df.apply(lambda row: _calc_euclidean_distance(row['long'], row['fwd_actual']), axis=1)
  new_df['short_distance_to_actual'] = new_df.apply(lambda row: _calc_euclidean_distance(row['short'], row['fwd_actual']), axis=1)
  new_df['long_minus_short']         = new_df.apply(lambda row: _calc_euclidean_distance(row['long'], row['short']), axis=1) # this is the difference between the long and short predictions
  new_df.index = new_df['close_time']

  return new_df


# Function to calculate the slope of the weights
def _calculate_slope(weights):
    t = np.arange(len(weights))
    slope, intercept = np.polyfit(t, weights, 1)
    return slope



def calculate_slopes(df: pd.DataFrame):
  df['long_slope']    = df['long'].apply(_calculate_slope)
  df['short_slope']   = df['short'].apply(_calculate_slope)
  df['actual_slope']  = df['fwd_actual'].apply(_calculate_slope)
  # format floats to 5 decimal places
  pd.options.display.float_format = '{:.5f}'.format
  df[['long_slope', 'short_slope', 'actual_slope']].describe()



def calculate_correlation_slopes(df: pd.DataFrame):
  correlation = df['long_distance_to_actual'].corr(df['short_distance_to_actual'])

  print("Correlation between Euclidean distance between long array and short array and future actual results: ", correlation)

  correlation_short = df['long_distance_to_actual'].corr(df['long_minus_short'])

  print("Correlation between difference in long minus short predictions and future actual results for longs: ", correlation_short)

  correlation_difference = df['short_distance_to_actual'].corr(df['long_minus_short'])

  print("Correlation between difference in long minus short predictions and future actual results for shorts: ", correlation_difference)

  long_correlation_slope = df['long_slope'].corr(df['actual_slope'])  
  short_correlation_slope = df['short_slope'].corr(df['actual_slope'])  
  
  print(f"Correlation between long slopes and future results: {long_correlation_slope}")
  print(f"Correlation between short slopes and future results: {short_correlation_slope}")




def process_pickle_files(path: str, prediction_window: int):
  df = read_pickle_files_into_df(path)

  add_forward_prices_to_df(df, prediction_window)
  df = df.copy()  # for large prediction_window size, the copy() call eliminates the fragmented dataframe warning

  generate_fwd_actual_column(df)

  df = generate_df_with_euclidean_distances(df, prediction_window)
  calculate_slopes(df)
  calculate_correlation_slopes(df)

  df.index = pd.to_datetime(df["close_time"], utc=True, unit="s")

  return df








  