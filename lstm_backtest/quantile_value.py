from typing import List
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


def generate_quantile_bands(values: pd.Series) -> List[QuantileBand]:
  quantile_count = len(values) - 1
  return [QuantileBand(QuantileValue(values.index[i  ], values.iloc[i  ]), 
                       QuantileValue(values.index[i+1], values.iloc[i+1])) 
          for i in range(len(values)-1)][:quantile_count]



def extract_boundary_values_from_quantile_bands(quantile_bands: List[QuantileBand]) -> List:
  values = list(set([entry.lower_bound.value for entry in quantile_bands] + [entry.upper_bound.value for entry in quantile_bands]))

  values.sort()

  return values
                                                
    