from enum import Enum
import pandas as pd
from typing import List, Optional, Tuple
from dataframes_merger import BaseDataFrameMerger
from settings_and_params import extract_prediction_window_sizes
from parameter_optimization import DataFrameFormat
from parameter_optimization_factory import VbtBackTestProcessorFactory, VbtBackTestProcessorType

class MultiModelBacktestMethod(Enum):
  AVERAGE         = "average"
  INDIVIDUAL      = "individual"



class MultiModelBacktest:
  def __init__(self, merger: BaseDataFrameMerger, model_files: List[str]):
    self.merger                 = merger
    self.model_files            = model_files
    self.prediction_window_size = None



  def run(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if self._validate():      
      df      = self.merger.process(self.model_files)
      result  = VbtBackTestProcessorFactory.create(VbtBackTestProcessorType.WITH_MEMORY_CONSTRAINT_TWO_LOOPS, df, self.prediction_window_size, DataFrameFormat.MERGED).run_backtest()

      return df, result
    
    return None, None



  def _validate(self) -> bool:
    values = extract_prediction_window_sizes(self.model_files)

    if len(values) == 1:
      self.prediction_window_size = values.pop()

      return True
    else:
      raise Exception("The prediction window sizes are not all the same")        

  



  