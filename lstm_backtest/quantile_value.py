import pandas as pd
from dataclasses import dataclass

@dataclass
class QuantileValue:
    quantile: float
    value   : float


@dataclass
class QuantileBand:
    lower_bound: QuantileValue
    upper_bound: QuantileValue


def generate_quantile_bands(values: pd.Series) -> QuantileBand:
  quantile_count = len(values) - 1
  return [QuantileBand(QuantileValue(values.index[i  ], values.iloc[i  ]), 
                       QuantileValue(values.index[i+1], values.iloc[i+1])) 
          for i in range(len(values)-1)][:quantile_count]
    