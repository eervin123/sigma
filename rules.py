#%%

import backtesting as bt
import pandas as pd
import numpy as np
import vectorbtpro as vbt

# Import the data
# Read in the 1 minute data from my HDF5 file
data = vbt.Data.from_hdf('data.hdf5')
print(type(data)) # <class 'vectorbt.base.data.Data'>

# The data is a vbt object, run the .get() method to get the underlying pandas dataframe
btc_1m = data.get()
print(type(btc_1m)) # <class 'pandas.core.frame.DataFrame'>
# Set columns to lowercase
btc_1m.columns = btc_1m.columns.str.lower()
# Print out the tail for sanity check
btc_1m.tail()

# Update the settings
vbt.settings.plotting["layout"]["template"] = "vbt_dark"
vbt.settings.plotting["layout"]["width"] = 800
vbt.settings.plotting['layout']['height'] = 350
vbt.settings.wrapping["freq"] = "1m"
vbt.settings.portfolio['init_cash'] = 10000
# %% Create the strategy
# Create a grid trading strategy class for backtesting.py
class grid_trading(bt.Strategy):
    # Define the parameters
    params = dict(
        # The size of the grid
        grid_size = 0.05,
        # The minimum price to start the grid
        min_price = 5000,
        # The maximum price to start the grid
        max_price = 6000,
        # The amount of leverage to use
        leverage = 1,
    )

        

