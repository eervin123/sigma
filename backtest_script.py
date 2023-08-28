from enum import Enum
import os
import argparse
import time
import pandas as pd
from collections import defaultdict
from backtesting import Strategy, Backtest

DEFAULT_CHUNK_SIZE    = 5000000         # The price file will be very large - so we will process it in chunks
NO_CHUNK_SIZE_VALUE   = -1              # Use this value to process the entire file in one chunk
NO_CHUNK_INDEX_VALUE  = "NoChunk"
DATA_DIR              = "data"  
RESULTS_DIR           = "results"     

BTC_PRICE_FILE_PART1  = "secbtcusdtm_20192020.csv"
BTC_PRICE_FILE_PART2  = "btcsec2021-23.csv"
ALL_BTC_PRICE_FILES   = "all_btc_prices"
ETH_PRICE_FILE        = "ETHUSDT-1s-201909-202308.csv"
DEFAULT_PRICE_FILE    = BTC_PRICE_FILE_PART1
DEFAULT_TRADE_FILE    = "signal - yosemite btc - slow.csv"

NUM_PRICES_PER_HOUR   = 60 * 60         # When using minute data, it should be 60.  Seconds data should be 60 x 60

# The price files currently come in 2 different formats...
class PriceFileFormatType(Enum):
  FORMAT_TYPE_1 = 1
  FORMAT_TYPE_2 = 2



def _extract_file_name_no_ext(absolute_path: str) -> str:
  return os.path.splitext(os.path.basename(absolute_path))[0]



PRICE_FILE_TO_FORMAT_TYPE_DICT = defaultdict(lambda: PriceFileFormatType.FORMAT_TYPE_1)
PRICE_FILE_TO_FORMAT_TYPE_DICT[ _extract_file_name_no_ext(BTC_PRICE_FILE_PART1)        ] = PriceFileFormatType.FORMAT_TYPE_1
PRICE_FILE_TO_FORMAT_TYPE_DICT[ _extract_file_name_no_ext(BTC_PRICE_FILE_PART2)        ] = PriceFileFormatType.FORMAT_TYPE_2
# ALL_BTC_PRICE_FILES is a hybrid, so it doesn't have its own entry
PRICE_FILE_TO_FORMAT_TYPE_DICT[ _extract_file_name_no_ext(ETH_PRICE_FILE      )        ] = PriceFileFormatType.FORMAT_TYPE_1



class pos_manager(Strategy):
    
    def init(self):
        super().init()
        self.signal = self.I(lambda: self.data.signal)

        self.last_trade_entry_bar = None

    def bars_since_prev_trade(self):
        if self.last_trade_entry_bar is not None:
            return len(self.data.Close) - self.last_trade_entry_bar
        return 0

    def next(self):
        super().next()

        # Check for new trades
        trades_len = len(self.trades)
        if trades_len:
            self.last_trade_entry_bar = self.trades[-1].entry_bar

        bars_since_prev_trade = self.bars_since_prev_trade()

        if not self.position:
            if self.signal[-1] == 1:
                self.buy(size=0.4)

            if self.signal[-1] == 2:
                self.sell(size=0.3)

        elif self.position.size > 0:
            if self.position.pl_pct > 0.003:
                self.position.close()
           
            elif self.position.pl_pct < -0.00 and self.signal[-1] == 2:
                self.position.close()

            elif self.position.pl_pct < -0.05:
                self.position.close()
                   
            elif trades_len and self.trades[-1].pl_pct < -0.0040 and bars_since_prev_trade > NUM_PRICES_PER_HOUR and trades_len < 5:
                self.buy(size=0.2)

        elif self.position.size < 0:
            if self.position.pl_pct > 0.003:
                self.position.close()
           
            elif self.position.pl_pct < -0.00 and (self.signal[-1] == 1 or self.signal[-1] ==2):
                self.position.close()

            elif trades_len and self.trades[-1].pl_pct < -0.0040 and bars_since_prev_trade > NUM_PRICES_PER_HOUR and trades_len < 5:
                self.sell(size=0.2)




def _read_trade_file(file_path: str) -> pd.DataFrame:          
  df_trades               = pd.read_csv(file_path)
  df_trades['Date/Time']  = pd.to_datetime(df_trades['Date/Time']).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
  df_trades               = df_trades.set_index('Date/Time')
  df_trades.index         = pd.to_datetime(df_trades.index)

  return df_trades



def _convert_chunk_df_to_correct_format(chunk_df: pd.DataFrame, price_file_name: str) -> pd.DataFrame:
  if PRICE_FILE_TO_FORMAT_TYPE_DICT[price_file_name] == PriceFileFormatType.FORMAT_TYPE_1:
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
    chunk_df['datetime'] = chunk_df.index
  else:
    chunk_df = chunk_df.rename(columns={  'Number of trades'      : 'Trade count'
                                       })
    chunk_df.index = pd.to_datetime(chunk_df['Open time'])
    chunk_df = chunk_df.drop(columns=['Open time', 'Close time'])
    chunk_df['datetime'] = chunk_df.index        

  return chunk_df



def _merge_price_and_trade(df_prices: pd.DataFrame, df_trades: pd.DataFrame) -> pd.DataFrame:
  df = pd.merge(df_prices, df_trades, left_on='Open time', right_on='Date/Time', how='left')
  df['signal']=df['signal'].fillna(0)
  df = df.set_index('datetime')

  return df



def _process_backtest_chunk(df_prices: pd.DataFrame, df_trades: pd.DataFrame):
  merged_df = _merge_price_and_trade(df_prices, df_trades)
  bt = Backtest(  merged_df
                , pos_manager
                , cash              = 100_000_000
                , trade_on_close    = True
                , commission        = .0014
                , exclusive_orders  = False
                , margin            = 0.25, # Set this to 0.5 for 2x leverage, 0.25 for 4x leverage, 0.125 for 8x leverage, etc.
              )
  stat = bt.run()

  return stat





def _generate_output_file_prefix(price_file: str, trade_file: str, chunk_index: int) -> str:
  price_name = _extract_file_name_no_ext(price_file)
  trade_name = _extract_file_name_no_ext(trade_file)

  return f'{price_name}_{trade_name}_chunk{chunk_index}'



def _generate_output_file_name(file_prefix: str, name: str) -> str:
  return RESULTS_DIR + f"/{file_prefix + '_' + name }" + ".csv"



def _output_to_files(stat: pd.DataFrame, file_prefix: str):
  trades_stat = stat['_trades']
  trades_csv_file_name = _generate_output_file_name(file_prefix, 'trades')
  trades_stat.to_csv(trades_csv_file_name)
  
  stat_csv_file_name = _generate_output_file_name(file_prefix, 'stat')
  stat.to_csv(stat_csv_file_name)

  equity_curve_stat = stat['_equity_curve']
  
  equity = equity_curve_stat # equity_curve_stat.resample('1D').last()  
  equity_csv_file_name = _generate_output_file_name(file_prefix, 'equity')
  equity.to_csv(equity_csv_file_name)




def _process_one_chunk(chunk_df: pd.DataFrame, price_file: str, trade_file: str, chunk_index_as_str: str, df_trades: pd.DataFrame):
  start_time  = time.time()
  chunk_df    = _convert_chunk_df_to_correct_format(chunk_df, _extract_file_name_no_ext(price_file))
  stat        = _process_backtest_chunk(chunk_df, df_trades.copy())
  file_prefix = _generate_output_file_prefix(price_file, trade_file, chunk_index_as_str)
  _output_to_files(stat, file_prefix)
  end_time    = time.time()
  print(f'      Processed chunk {chunk_index_as_str} in {end_time - start_time} seconds.')



def _combine_and_convert_all_btc_prices() -> pd.DataFrame:
  btc_df1 = pd.read_csv(os.path.join(os.getcwd(), DATA_DIR, BTC_PRICE_FILE_PART1))
  btc_df1 = _convert_chunk_df_to_correct_format(btc_df1, _extract_file_name_no_ext(BTC_PRICE_FILE_PART1))

  btc_df2 = pd.read_csv(os.path.join(os.getcwd(), DATA_DIR, BTC_PRICE_FILE_PART2))
  btc_df2 = _convert_chunk_df_to_correct_format(btc_df2, _extract_file_name_no_ext(BTC_PRICE_FILE_PART2))

  combined_df = pd.concat([btc_df1, btc_df2])
  combined_df.sort_index(inplace=True)

  return combined_df



def _process_all_btc_prices(trade_file: str, df_trades: pd.DataFrame):
  start_time  = time.time()
  
  chunk_df    = _combine_and_convert_all_btc_prices()  
  stat        = _process_backtest_chunk(chunk_df, df_trades.copy())
  file_prefix = _generate_output_file_prefix(ALL_BTC_PRICE_FILES, trade_file, NO_CHUNK_INDEX_VALUE)
  _output_to_files(stat, file_prefix)
  end_time    = time.time()

  print(f'      Processed chunk {NO_CHUNK_INDEX_VALUE} in {end_time - start_time} seconds.')
  




def _process_backtest(price_file: str, trade_file: str, chunk_size: int, do_all_btc_prices: bool):
  price_file_msg = ALL_BTC_PRICE_FILES if do_all_btc_prices else price_file

  print(f'Processing backtest using price file "{price_file_msg}" and trade file "{trade_file}"...')
  df_trades = _read_trade_file(trade_file)

  if do_all_btc_prices:
    _process_all_btc_prices(trade_file, df_trades)
    
  elif chunk_size == NO_CHUNK_SIZE_VALUE:
    chunk_df = pd.read_csv(price_file)
    _process_one_chunk(chunk_df, price_file, trade_file, NO_CHUNK_INDEX_VALUE, df_trades) 
  else:
    for chunk_index, chunk_df in enumerate(pd.read_csv(price_file, chunksize=chunk_size)):
      _process_one_chunk(chunk_df, price_file, trade_file, str(chunk_index), df_trades)    
  
  print("Done processing backtest.")



if __name__ == '__main__':  
  parser = argparse.ArgumentParser(description="Process price and trade files.")
  parser.add_argument("--price_file"      , default=DEFAULT_PRICE_FILE, help=f"Path to price file (default: {DEFAULT_PRICE_FILE})")
  parser.add_argument("--trade_file"      , default=DEFAULT_TRADE_FILE, help=f"Path to trade file (default: {DEFAULT_TRADE_FILE})")
  parser.add_argument("--chunk_size"      , default=DEFAULT_CHUNK_SIZE, help=f"Chunk size (default: {DEFAULT_CHUNK_SIZE}, use -1 to process entire file)")

  parser.add_argument("--all_btc_prices"  , default=0                 , help=f"1: combine the two hardcoded BTC price files (default: 0) and do nochunk")
  args = parser.parse_args()

  price_file_path = os.path.join(os.getcwd(), DATA_DIR, args.price_file)
  trade_file_path = os.path.join(os.getcwd(), DATA_DIR, args.trade_file)

  _process_backtest(price_file_path, trade_file_path, int(args.chunk_size), bool(args.all_btc_prices))