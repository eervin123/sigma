from enum import Enum
import os
import argparse
import time
from typing import List
import pandas as pd
from collections import defaultdict

DATA_DIR              = "data"

class PriceFileFormatType(Enum):
  FORMAT_TYPE_1 = 1
  FORMAT_TYPE_2 = 2


FORMAT_TYPE_1_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]



def _get_absolute_path_to_data_file(file_name: str) -> str:
  return os.path.join(os.getcwd(), DATA_DIR, file_name)



def _convert_chunk_df_to_correct_format(chunk_df: pd.DataFrame, format: PriceFileFormatType) -> pd.DataFrame:
  if format == PriceFileFormatType.FORMAT_TYPE_1:
    chunk_df = chunk_df.rename(columns={  'open_time'             : 'Open time'
                                        , 'open'                  : 'Open'
                                        , 'high'                  : 'High'
                                        , 'low'                   : 'Low'
                                        , 'close'                 : 'Close'
                                        , 'volume'                : 'Volume'
                                        , 'quote_volume'          : 'Quote volume'
                                        , 'count'                 : 'Trade count'
                                        , 'taker_buy_volume'      : 'Taker base volume'
                                        , 'taker_buy_quote_volume': 'Taker quote volume'})
    
    chunk_df.index = pd.to_datetime(chunk_df['Open time'], unit='ms', utc=True)
    chunk_df = chunk_df.drop(columns=['ignore', 'close_time', 'Open time'])    
  else:
    chunk_df = chunk_df.rename(columns={  'Number of trades'      : 'Trade count'
                                       })
    chunk_df.index = pd.to_datetime(chunk_df['Open time'])
    chunk_df = chunk_df.drop(columns=['Open time', 'Close time'])    

  return chunk_df



def _read_csv_format1_no_header(file_path: str) -> pd.DataFrame:
  return pd.read_csv(file_path, header=None, names=FORMAT_TYPE_1_COLUMNS)



def _read_monthly_csv_format1(file_names: str) -> pd.DataFrame:
  combined_df = None

  for file_name in file_names:
    df = _read_csv_format1_no_header(_get_absolute_path_to_data_file(file_name))

    if df is not None:
      if combined_df is None:
        combined_df = df
      else:
        combined_df = pd.concat([combined_df, df])

  return combined_df



def _convert_format1_files(files_for_2021: List[str], file2: str) -> pd.DataFrame:
  monthly_df  = _read_monthly_csv_format1(files_for_2021)  
  part2_df    = pd.read_csv(_get_absolute_path_to_data_file(file2))

  combined_df = pd.concat([monthly_df, part2_df])
  combined_df = _convert_chunk_df_to_correct_format(combined_df, PriceFileFormatType.FORMAT_TYPE_1)

  return combined_df



def _convert_format2_file(file: str) -> pd.DataFrame:
  df = pd.read_csv(_get_absolute_path_to_data_file(file))
  df = _convert_chunk_df_to_correct_format(df, PriceFileFormatType.FORMAT_TYPE_2)

  return df



def _combine_into_one_df(files_for_2021: List[str], file2: str, file3: str = None) -> pd.DataFrame:
  part1_df = _convert_format1_files(files_for_2021, file2)

  if file3 is not None:
    part2_df = _convert_format2_file(file3)
    combined_df = pd.concat([part1_df, part2_df])
  else:
    combined_df = part1_df

  return combined_df



def _convert_btc_files():
  files_for_2021 = [
    "BTCUSDT-1s-2019-01.csv"
  , "BTCUSDT-1s-2019-02.csv"
  , "BTCUSDT-1s-2019-03.csv"
  , "BTCUSDT-1s-2019-04.csv"
  , "BTCUSDT-1s-2019-05.csv"
  , "BTCUSDT-1s-2019-06.csv"
  , "BTCUSDT-1s-2019-07.csv"
  , "BTCUSDT-1s-2019-08.csv"
  ]
  file2 = "secbtcusdtm_20192020.csv"
  file3 = "btcsec2021-23.csv"

  df = _combine_into_one_df(files_for_2021, file2, file3)
  df.to_csv(_get_absolute_path_to_data_file("BTCUSDT-1s-2019-202304.csv"))



def _convert_eth_files():
  files_for_2021 = [
    "ETHUSDT-1s-2019-01.csv"
  , "ETHUSDT-1s-2019-02.csv"
  , "ETHUSDT-1s-2019-03.csv"
  , "ETHUSDT-1s-2019-04.csv"
  , "ETHUSDT-1s-2019-05.csv"
  , "ETHUSDT-1s-2019-06.csv"
  , "ETHUSDT-1s-2019-07.csv"
  , "ETHUSDT-1s-2019-08.csv"
  ]
  file2 = "ETHUSDT-1s-201909-202308.csv"  

  df = _combine_into_one_df(files_for_2021, file2, None)
  df.to_csv(_get_absolute_path_to_data_file("ETHUSDT-1s-2019-202308.csv"))



def _perform_file_conversion():
  _convert_eth_files()  

  
  
  
  
if __name__ == '__main__':
  _perform_file_conversion()