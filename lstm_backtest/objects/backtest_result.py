from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class BacktestResult:
    trade_df: pd.DataFrame
    training_details: List
    win_rate: float
    balance: float
    suspicious_trades: List

    def __str__(self):
        return f"BacktestResult(win_rate={self.win_rate}, balance={self.balance}， suspicious_trades={len(self.suspicious_trades)})"
