from typing import Optional
import pandas as pd
from enum import Enum
from parameter_optimization import BaseVbtBackTestProcessor, DataFrameFormat, VbtBackTestProcessorNoMemoryConstraint, VbtBackTestProcessorOneLoopMemoryConstraint, VbtBackTestProcessorTwoLoopsMemoryConstraint


class VbtBackTestProcessorType(Enum):
  NO_MEMORY_CONSTRAINT              = 0
  WITH_MEMORY_CONSTRAINT_ONE_LOOP   = 1
  WITH_MEMORY_CONSTRAINT_TWO_LOOPS  = 2  

  

class VbtBackTestProcessorFactory:
  @staticmethod
  def create(type: VbtBackTestProcessorType, df: pd.DataFrame, prediction_window_size: int, dataframe_format: DataFrameFormat) -> Optional[BaseVbtBackTestProcessor]:
    instance = None

    if type == VbtBackTestProcessorType.NO_MEMORY_CONSTRAINT:
      instance = VbtBackTestProcessorNoMemoryConstraint(df, prediction_window_size, dataframe_format)
    elif type == VbtBackTestProcessorType.WITH_MEMORY_CONSTRAINT_ONE_LOOP:
      instance = VbtBackTestProcessorOneLoopMemoryConstraint(df, prediction_window_size, dataframe_format)
    elif type == VbtBackTestProcessorType.WITH_MEMORY_CONSTRAINT_TWO_LOOPS:
      instance = VbtBackTestProcessorTwoLoopsMemoryConstraint(df, prediction_window_size, dataframe_format)    
    
    return instance


    