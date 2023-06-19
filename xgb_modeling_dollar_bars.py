import numpy as np
import pandas as pd
from numba import njit
import vectorbtpro as vbt
vbt.settings.set_theme("dark")
vbt.settings.plotting["layout"]["width"] = 800
vbt.settings.plotting['layout']['height'] = 200
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42) # random forest classifier

data_path = '/home/joel/dev/data/minute_data/BTCUSDT_1m_futures.pkl' 
futures_1m = vbt.BinanceData.load(data_path)

def dollar_bar_func(ohlc_df, dollar_bar_size):
    # Calculate dollar value traded for each row
    ohlc_df['DollarValue'] = ohlc_df['Close'] * ohlc_df['Volume']
    
    # Calculate cumulative dollar value
    ohlc_df['CumulativeDollarValue'] = ohlc_df['DollarValue'].cumsum()
    
    # Determine the number of dollar bars
    num_bars = int(ohlc_df['CumulativeDollarValue'].iloc[-1] / dollar_bar_size)
    
    # Generate index positions for dollar bars
    bar_indices = [0]
    cumulative_value = 0
    for i in range(1, len(ohlc_df)):
        cumulative_value += ohlc_df['DollarValue'].iloc[i]
        if cumulative_value >= dollar_bar_size:
            bar_indices.append(i)
            cumulative_value = 0
    
    # Create a new dataframe with dollar bars
    dollar_bars = []
    for i in range(len(bar_indices) - 1):
        start_idx = bar_indices[i]
        end_idx = bar_indices[i + 1]
        
        dollar_bar = {
            'Open': ohlc_df['Open'].iloc[start_idx],
            'High': ohlc_df['High'].iloc[start_idx:end_idx].max(),
            'Low': ohlc_df['Low'].iloc[start_idx:end_idx].min(),
            'Close': ohlc_df['Close'].iloc[end_idx],
            'Volume': ohlc_df['Volume'].iloc[start_idx:end_idx].sum(),
            'Quote volume': ohlc_df['Quote volume'].iloc[start_idx:end_idx].sum(),
            'Trade count': ohlc_df['Trade count'].iloc[start_idx:end_idx].sum(),
            'Taker base volume': ohlc_df['Taker base volume'].iloc[start_idx:end_idx].sum(),
            'Taker quote volume': ohlc_df['Taker quote volume'].iloc[start_idx:end_idx].sum()
        }
        
        if isinstance(ohlc_df.index, pd.DatetimeIndex):
            dollar_bar['Open Time'] = ohlc_df.index[start_idx]
            dollar_bar['Close Time'] = ohlc_df.index[end_idx] - pd.Timedelta(milliseconds=1)
        elif 'Open Time' in ohlc_df.columns:
            dollar_bar['Open Time'] = ohlc_df['Open Time'].iloc[start_idx]
            dollar_bar['Close Time'] = ohlc_df['Open Time'].iloc[end_idx] - pd.Timedelta(milliseconds=1)
        
        dollar_bars.append(dollar_bar)
    
    dollar_bars_df = pd.concat([pd.DataFrame([bar]) for bar in dollar_bars], ignore_index=True)
    
    return dollar_bars_df

# Create a simple function to simplify the number so we can use it in our column names
def simplify_number(num):
    """
    Simplifies a large number by converting it to a shorter representation with a suffix (K, M, B).
    simplify_number(1000) -> 1K
    """
    suffixes = ['', 'K', 'M', 'B']
    suffix_index = 0

    while abs(num) >= 1000 and suffix_index < len(suffixes) - 1:
        num /= 1000.0
        suffix_index += 1

    suffix = suffixes[suffix_index] if suffix_index > 0 else ''
    simplified_num = f'{int(num)}{suffix}'

    return simplified_num

def merge_and_fill_dollar_bars(original_df, dollar_bars_df, dollar_bar_size):
    # Add prefix to column names in dollar bars dataframe
    dollar_bar_prefix = f'db_{simplify_number(dollar_bar_size)}_'
    dollar_bars_df_renamed = dollar_bars_df.add_prefix(dollar_bar_prefix)

    # Convert 'Open Time' columns to pandas datetime format and set them as index
    dollar_bars_df_renamed.index = pd.to_datetime(dollar_bars_df_renamed[dollar_bar_prefix + 'Open Time'])

    # Merge the dataframes on the index
    merged_df = original_df.merge(dollar_bars_df_renamed, how='left', left_index=True, right_index=True)

    # Set the flag for a new dollar bar with prefix
    merged_df[dollar_bar_prefix + 'NewDBFlag'] = ~merged_df[dollar_bar_prefix + 'Close'].isna()

    # Forward fill the NaN values for all columns except the new dollar bar flag
    columns_to_ffill = [col for col in merged_df.columns if col != dollar_bar_prefix + 'NewDBFlag']
    merged_df[columns_to_ffill] = merged_df[columns_to_ffill].fillna(method='ffill')

    # Fill the remaining NaN values in the new dollar bar flag column with False
    merged_df[dollar_bar_prefix + 'NewDBFlag'] = merged_df[dollar_bar_prefix + 'NewDBFlag'].fillna(False)
    
    # Assign the renamed 'Open Time' column back to the dataframe
    merged_df[dollar_bar_prefix + 'Open Time'] = merged_df[dollar_bar_prefix + 'Open Time']

    return merged_df


# dollar_bar_size = 90_000_000
# btc_dollar_bars = dollar_bar_func(futures_1m.get(), dollar_bar_size=dollar_bar_size)
# btc_dollar_bars.index = pd.to_datetime(btc_dollar_bars['Open Time'])
# btc_dollar_bars.shape

btc_90M_db_vbt = vbt.BinanceData.load('btc_90M_db_vbt.pkl')
data = btc_90M_db_vbt['2021-01-01':'2021-01-31']

# Generate the features (X) using TA-Lib indicators
X = data.run("talib")

# add trend label as a feature
#.5 == 50% increase in price, 0.2 == 20% decrease in price
X['trend'] = data.run("trendlb", .5, 0.2, mode="binary").labels 

# X['dayofweek'] = data['Open Time'].dayofweek # add day of week as a feature

# Now we are trying to generate future price predictions so we will set the y labels to the price change n periods in the future
n = 150 # number of periods in the future to predict
y = (data.close.shift(-n) / data.close - 1).rolling(n).mean() # future price change we use rolling mean to smooth the data

# Preprocessing steps to handle NaNs
X = X.replace([-np.inf, np.inf], np.nan) # replace inf with nan
invalid_column_mask = X.isnull().all(axis=0) | (X.nunique() == 1) # drop columns that are all nan or have only one unique value
X = X.loc[:, ~invalid_column_mask] # drop invalid columns
invalid_row_mask = X.isnull().any(axis=1) | y.isnull() # drop rows that have nan in any column or in y

# Drop invalid rows in X and y
X = X.loc[~invalid_row_mask]
y = y.loc[~invalid_row_mask]

# Construct the pipeline
steps = [
    ('imputation', SimpleImputer(strategy='mean')),  # Imputation replaces missing values
    ('scaler', StandardScaler()),  # StandardScaler normalizes the data
    ('pca', PCA(n_components=15)),  # PCA reduces dimensionality
    
    # Choose one of the following models
    # ('model', Ridge())  # Ridge regression is used as the prediction model
    # ('model', LinearRegression())  # Linear regression is used as the prediction model
    # ('model', LogisticRegression())  # Logistic regression is used as the prediction model
    # ('model', Lasso())  # Lasso regression is used as the prediction model
    # ('model', ElasticNet())  # ElasticNet regression is used as the prediction model
    # ('model', SVR())  # Support Vector Regression is used as the prediction model
    ('model', XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist'))  # XGBoost regression is used as the prediction model
]

#XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')
pipeline = Pipeline(steps)

# Cross-validate
cv = vbt.SKLSplitter(
    "from_expanding",
    min_length=1000,
    offset=10,
    split=-10,
    set_labels=["train", "test"]
)

cv_splitter = cv.get_splitter(X)
# Plot the cross-validation splits
cv_splitter.plot().show_svg()


# Use your pipeline to compress features and fit the model for predictions
print(f'Pipeline Steps :{pipeline.steps}')
pipeline.fit(X, y)  # Fit the pipeline on the entire dataset    
print(f'Pipeline Score :{pipeline.score(X, y)}')  # Score the pipeline on the entire dataset of training data
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1) # how well the model generalizes to unseen data
average_score = np.mean(scores)
print(f'Average cross-validation score: {average_score}')

# Predictions
X_slices = cv_splitter.take(X)
y_slices = cv_splitter.take(y)

test_labels = []
test_preds = []
for split in X_slices.index.unique(level="split"):  
    X_train_slice = X_slices[(split, "train")]  
    y_train_slice = y_slices[(split, "train")]
    X_test_slice = X_slices[(split, "test")]
    y_test_slice = y_slices[(split, "test")]
    slice_pipeline = pipeline.fit(X_train_slice, y_train_slice)  
    test_pred = slice_pipeline.predict(X_test_slice)  
    test_pred = pd.Series(test_pred, index=y_test_slice.index)
    test_labels.append(y_test_slice)
    test_preds.append(test_pred)

test_labels = pd.concat(test_labels).rename("labels")  
test_preds = pd.concat(test_preds).rename("preds")

# Show the accuracy of the predictions
# Assuming test_labels and test_preds are your true and predicted values
mse = mean_squared_error(test_labels, test_preds)
rmse = np.sqrt(mse)  # or use mean_squared_error with squared=False
mae = mean_absolute_error(test_labels, test_preds)
r2 = r2_score(test_labels, test_preds)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# Visualize the predictions as a heatmap plotted against the price
data.close.vbt.overlay_with_heatmap(test_preds).show_svg()

pf = vbt.Portfolio.from_signals(
    data.close[test_preds.index], # use only the test set
    test_preds > 0.05, # long when probability of price increase is greater than 2%
    test_preds < -0.05, # short when probability prediction is less than -5%
    direction="both" # long and short
)
print(pf.stats())

pf.plot().show_svg()
# Show first period
# pf['2018':'2021'].plot().show_svg()
# Show second period
# pf['2021':'2023'].plot().show_svg()

pf.stats().to_csv('stats-jan-2021-EOD.csv')


