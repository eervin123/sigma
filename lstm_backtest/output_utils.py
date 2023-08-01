import pandas as pd

def export_raw_dataframe_to_csv(df: pd.DataFrame, output_file_path: str):  
  copy_df         = df.copy(deep=True)
  columns_to_drop = copy_df.filter(like="BTCUSDT_Open_").columns

  copy_df.drop(columns_to_drop, axis=1, inplace=True)

  copy_df.to_csv(output_file_path)


# joe_df = df.copy(deep=True)

# output_df           = joe_df[['BTCUSDT_Open', 'BTCUSDT_High', 'BTCUSDT_Low','BTCUSDT_Close']]
# entries.index       = df.index
# short_entries.index = df.index
# output_df.loc[:, "entries"] = entries
# output_df.loc[:, "short_entries"] = short_entries
