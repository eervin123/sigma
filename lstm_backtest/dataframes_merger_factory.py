from enum import Enum
from typing import Optional

from dataframes_merger import BaseDataFrameMerger, IntersectionDataFrameMerger, UnionDataFrameMerger


class DataFrameMergerType(Enum):
  INTERSECTION  = 0
  UNION         = 1



class DataFrameMergerFactory:
  @staticmethod
  def create(type: DataFrameMergerType) -> Optional[BaseDataFrameMerger]:
    instance = None

    if type == DataFrameMergerType.INTERSECTION:
      instance = IntersectionDataFrameMerger()
    elif type == DataFrameMergerType.UNION:
      instance = UnionDataFrameMerger()

    return instance