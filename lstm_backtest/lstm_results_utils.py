import pandas as pd
import vectorbtpro as vbt
from typing import Dict, List
from lstm_analysis_constants import BACKTEST_NAME, VBT_TOTAL_RETURN, VBT_WIN_RATE, VBT_TOTAL_TRADES

def _print_one_backtest_results(values: Dict):  
  print(f"--------{values[BACKTEST_NAME]}--------")
  print(f"{VBT_TOTAL_RETURN}: {values[VBT_TOTAL_RETURN] :.2f}")
  print(f"{VBT_WIN_RATE}: {values[VBT_WIN_RATE] :.2f}")
  print(f"{VBT_TOTAL_TRADES}: {values[VBT_TOTAL_TRADES]}")



def create_empty_results_df() -> pd.DataFrame:
  data = {
    BACKTEST_NAME     : [],
    VBT_TOTAL_RETURN  : [],
    VBT_WIN_RATE      : [],
    VBT_TOTAL_TRADES  : []
  }

  return pd.DataFrame(data)



def store_backtest_results(name: str, pf: vbt.Portfolio, results: List[Dict[str, any]]):
  values: Dict[str, any]    = {}

  values[BACKTEST_NAME    ]  = name
  values[VBT_TOTAL_RETURN ]  = pf.total_return * 100.0
  values[VBT_WIN_RATE     ]  = pf.trades.win_rate * 100.0
  values[VBT_TOTAL_TRADES ]  = pf.trades.count()

  results.append(values)
      
  _print_one_backtest_results(values)



def export_results(results: List[Dict[str, any]]) -> pd.DataFrame:
  return pd.DataFrame(results)

  