from enum import Enum
from typing import List, Optional

from dataframes_merger import BaseDataFrameMerger
from multiple_models_backtesting import MultiModelBacktest, AverageMultiModelBacktest, MajorityMultiModelBacktest


class MultiModelBacktestMethod(Enum):
  AVERAGE         = "average"
  MAJORITY        = "majority"


class MultiModelBacktestFactory:
  @staticmethod
  def create(method: MultiModelBacktestMethod, merger: BaseDataFrameMerger, model_files: List[str]) -> Optional[MultiModelBacktest]:
    instance = None

    if method == MultiModelBacktestMethod.AVERAGE:
      instance = AverageMultiModelBacktest(merger, model_files)
    elif method == MultiModelBacktestMethod.MAJORITY:
      instance = MajorityMultiModelBacktest(merger, model_files)

    return instance