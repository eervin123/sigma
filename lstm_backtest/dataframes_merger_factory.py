from typing import Optional

from dataframes_merger import BaseDataFrameMerger, DataFrameMergerType, IntersectionDataFrameMerger, UnionDataFrameMerger



class DataFrameMergerFactory:
  @staticmethod
  def create(type: DataFrameMergerType) -> Optional[BaseDataFrameMerger]:
    instance = None

    if type == DataFrameMergerType.INTERSECTION:
      instance = IntersectionDataFrameMerger()
    elif type == DataFrameMergerType.UNION:
      instance = UnionDataFrameMerger()

    return instance