
import numpy as np
import pandas as pd
from numba import njit
import vectorbtpro as vbt
vbt.settings.set_theme("dark")
vbt.settings.plotting["layout"]["width"] = 800
vbt.settings.plotting['layout']['height'] = 200
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from collections import Counter


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42) # random forest classifier
from joblib import dump, load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt


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
            'Close': ohlc_df['Close'].iloc[end_idx-1],
            'Volume': ohlc_df['Volume'].iloc[start_idx:end_idx].sum(),
            'Quote volume': ohlc_df['Quote volume'].iloc[start_idx:end_idx].sum(),
            'Trade count': ohlc_df['Trade count'].iloc[start_idx:end_idx].sum(),
            'Taker base volume': ohlc_df['Taker base volume'].iloc[start_idx:end_idx].sum(),
            'Taker quote volume': ohlc_df['Taker quote volume'].iloc[start_idx:end_idx].sum()
        }
        
        if isinstance(ohlc_df.index, pd.DatetimeIndex):
            dollar_bar['Open Time'] = ohlc_df.index[start_idx]
            dollar_bar['Close Time'] = ohlc_df.index[end_idx-1] - pd.Timedelta(milliseconds=1)
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

def _add_pivot_trends(X, data, pivot_up_th, pivot_down_th, pivot_up_th2, pivot_down_th2, pivot_up_th3, pivot_down_th3):
    pivot_info = data.run("pivotinfo", up_th=pivot_up_th, down_th=pivot_down_th)
    binary_pivot_labels = np.where(data.close > pivot_info.conf_value,1,0) # Create binary labels for pivot points
    X['trend'] = binary_pivot_labels # add pivot label as a feature
    
    pivot_info2 = data.run("pivotinfo", up_th=pivot_up_th2, down_th=pivot_down_th2)
    binary_pivot_labels2 = np.where(data.close > pivot_info2.conf_value,1,0) # Create binary labels for pivot points
    X['trend2'] = binary_pivot_labels2 # add pivot label as a feature
    
    pivot_info3 = data.run("pivotinfo", up_th=pivot_up_th3, down_th=pivot_down_th3)
    binary_pivot_labels3 = np.where(data.close > pivot_info3.conf_value,1,0) # Create binary labels for pivot points
    X['trend3'] = binary_pivot_labels3 # add pivot label as a feature
    
    return X

def _add_ta_features(X, data, lookback_window):

    # Add some TA features
    X['supert'] = data.run("supertrend", period=lookback_window).supert
    X['supert_cross_up'] = data.close.vbt.crossed_above(data.run('supertrend', period=lookback_window).supert)
    X['supert_cross_down'] = data.close.vbt.crossed_below(data.run('supertrend', period=lookback_window).supert)
    X['vwap'] = data.run("VWAP").vwap
    X['rsi'] = data.run("rsi", window=lookback_window).rsi
    X['rsi_overbought'] = pd.Series(np.where(X['rsi'] > 60, 1, 0), index=X.index)
    X['rsi_oversold'] = pd.Series(np.where(X['rsi'] < 40, 1, 0), index=X.index)
    X['bb_width'] = data.run("bbands", window=lookback_window).bandwidth
    X['bb_width_pct'] = data.run("bbands", window=lookback_window).percent_b
    X['fast_k'] = data.run("stoch", fast_k_window=lookback_window, slow_k_window=lookback_window*2, slow_d_window=lookback_window*2).fast_k
    X['slow_k'] = data.run("stoch", fast_k_window=lookback_window, slow_k_window=lookback_window*2, slow_d_window=lookback_window*2).slow_k
    X['slow_k_trending_up'] = X['slow_k'] > X['slow_k'].shift(lookback_window)
    X['slow_k_trending_down'] = X['slow_k'] < X['slow_k'].shift(lookback_window)
    X['slow_d'] = data.run("stoch", fast_k_window=lookback_window, slow_k_window=lookback_window*2, slow_d_window=lookback_window*2).slow_d
    X['slow_k_over_slow_d'] = X['slow_k'] > X['slow_d']
    X['slow_k_under_slow_d'] = X['slow_k'] < X['slow_d']
    return X

def _add_historical_returns(X, data, lookback_window):
    # Add in historical returns
    X['pct_change_1'] = data.close.pct_change(1)
    X['pct_change_5'] = data.close.pct_change(5)
    X['pct_change_10'] = data.close.pct_change(10)
    X['pct_change_20'] = data.close.pct_change(20)
    X['pct_change_40'] = data.close.pct_change(40)
    X['pct_change_60'] = data.close.pct_change(60)
    X['pct_change_100'] = data.close.pct_change(100)
    X['pct_change_160'] = data.close.pct_change(160)
    X['pct_change_260'] = data.close.pct_change(260)
    X['pct_change_420'] = data.close.pct_change(420)
    
    # Add in the relative change as a boolean
    X['yesterday_up'] = np.where(X['pct_change_160'] > 0, 1, 0) # Using 160 bar lookback as a proxy for yesterday
    X['yesterday_down'] = np.where(X['pct_change_160'] < 0, 1, 0) # Using 160 bar lookback as a proxy for yesterday
    X['up_down_run_160'] = np.sign(data.close.diff(160)).rolling(lookback_window).sum()
    X['up_down_run_1'] = np.sign(data.close.diff(1)).rolling(lookback_window).sum()
    # Add in runs of up/down days TODO: Need to make this a function and do it properly
    # X['up_day_count'] = data.close.vbt.rolling_count(data.close > data.close.shift(160), window=lookback_window)
    
    X['mid_range_momentum']= pd.Series(np.where(X['pct_change_100'] > X['pct_change_420'], True, False), index=X.index)
    X['short_range_momentum']= pd.Series(np.where(X['pct_change_20'] > X['pct_change_40'], True, False), index=X.index)
    X['short_over_long_momentum'] = pd.Series(np.where(X['pct_change_20'] > X['pct_change_420'], True, False), index=X.index)
    X['momentum_trending'] = pd.Series(np.where(X['pct_change_20'] > X['pct_change_20'].shift(lookback_window), True, False), index=X.index)
    # Label large moves
    X['large_move_up'] = np.where(data.close > data.close.shift(lookback_window) * 1.05, 1, 0)
    X['large_move_down'] = np.where(data.close < data.close.shift(lookback_window) * 0.95, 1, 0)  
    # Drop the time columns
    return X

def _add_time_features(X):
    X['dayofmonth']  = X.index.day
    X['month']       = X.index.month
    X['year']        = X.index.year
    X['hour']        = X.index.hour
    X['minute']      = X.index.minute
    X['dayofweek']   = X.index.dayofweek   
    return X

def _handle_missing_data(df):
    df = df.replace([-np.inf, np.inf], np.nan) # replace inf with nan
    invalid_column_mask = df.isnull().all(axis=0)
    df = df.loc[:, ~invalid_column_mask] # drop invalid columns
    invalid_row_mask = df.isnull().any(axis=1) # drop rows that have nan in any column
    df = df.loc[~invalid_row_mask]
    return df

def _create_target(X, periods_future, base_predictions=None, meta=False):
    # Now we are trying to generate future price predictions so we will set the y labels to the price change n periods in the future
    y = (X.Close.shift(-periods_future) / X.Close - 1) # future price change

    if base_predictions is not None:
        # If base predictions are available, add them as a column to the data
        y = y.to_frame('future return')
        y['base prediction'] = base_predictions

    # if meta, we want to predict if the price change will be positive or negative
    if meta:
        y = (y > 0).astype(int)
    
    return y

def _generate_features(data, lookback_window, pivot_up_th, pivot_down_th, drop_cols):
    pivot_up_th2 = pivot_up_th * 1.5
    pivot_down_th2 = pivot_down_th * 1.5
    pivot_up_th3 = pivot_up_th * 2
    pivot_down_th3 = pivot_down_th * 2
    
    # Generate the features (X)
    X = data.get()

    # Add all of the World Quant Alphas
    alphas = data.run(["wqa101_%d" % i for i in range(1, 102)], missing_index="drop") # 101 strategies
    X = pd.concat([X, alphas], axis=1)
    # Replace NaNs with 0s
    X = X.fillna(0)
    # Add pivot trends
    X = _add_pivot_trends(X, data, pivot_up_th, pivot_down_th, pivot_up_th2, pivot_down_th2, pivot_up_th3, pivot_down_th3)
    # Add TA features
    X = _add_ta_features(X, data, lookback_window)
    # Add historical returns
    X = _add_historical_returns(X, data, lookback_window)
    # Add time features
    X = _add_time_features(X)

    return X

def prepare_data(data, base_predictions=None, meta=False, pivot_up_th=0.10, pivot_down_th=0.10, periods_future=150, drop_cols=[]):
    lookback_window = 14*periods_future  # Number of dollar bars we are predicting into the future times the typical RSI lookback window of 14
    X = _generate_features(data, lookback_window, pivot_up_th, pivot_down_th, drop_cols)
    
    # Create y using cleaned X data
    y = _create_target(X, periods_future, base_predictions, meta)

    # Adjust X to match the length of y by removing the last rows
    X = X.iloc[:-periods_future]

    # Handle missing data in both X and y
    X = _handle_missing_data(X)

    # Convert column names to string
    X.columns = X.columns.astype(str)
    
    # Reindex y based on X's index to ensure they match
    y = y.reindex(X.index)

    assert len(X) == len(y)  # This will raise an error if X and y are not the same length
    return X, y



def create_pipeline(X, model='xgb', task='regression', class_weight=None):
    """
    Create a scikit-learn pipeline.

    Parameters:
    model (str): The model to use in the pipeline. Default is 'xgb' (XGBoost).
    class_weight (dict, optional): Class weights for classification tasks.

    Returns:
    pipeline (Pipeline): The scikit-learn pipeline.
    """
    X_shape = X.shape
    # Construct the pipeline
    steps = [
        ('imputation', SimpleImputer(strategy='mean')),  # Imputation replaces missing values
        ('scaler', StandardScaler()),  # StandardScaler normalizes the data
    ]
    if task == 'classification':
        if model == 'xgb':
            steps.append(('model', XGBClassifier(class_weight=class_weight)))  # XGBoost classification
        elif model == 'logistic':
            steps.append(('model', LogisticRegression(class_weight=class_weight)))  # Logistic regression
        elif model == 'svc':
            steps.append(('model', SVC(class_weight=class_weight)))  # Support Vector Classification
        # Add more classification models as needed
        else:
            raise ValueError("Invalid model name for classification. Choose from 'xgb', 'logistic', 'svc'.")
    elif task == 'regression':
        if model == 'xgb':
            steps.append(('model', XGBRegressor(objective='reg:squarederror')))  # XGBoost regression is used as the prediction model
        elif model == 'ridge':
            steps.append(('model', Ridge()))  # Ridge regression
        elif model == 'linear':
            steps.append(('model', LinearRegression()))  # Linear regression
        elif model == 'logistic':
            steps.append(('model', LogisticRegression()))  # Logistic regression for regression tasks won't have class weights
        elif model == 'lasso':
            steps.append(('model', Lasso()))  # Lasso regression
        elif model == 'elasticnet':
            steps.append(('model', ElasticNet()))  # ElasticNet regression
        elif model == 'svr':
            steps.append(('model', SVR()))  # Support Vector Regression
        else:
            raise ValueError("Invalid model name. Choose from 'xgb', 'ridge', 'linear', 'logistic', 'lasso', 'elasticnet', 'svr'.")
    else:
        raise ValueError("Invalid task. Choose from 'classification', 'regression'.")

    pipeline = Pipeline(steps)
    
    return pipeline

def create_cv(X, min_length=600, offset=200, split=-200, set_labels=["train", "test"]):
    """
    Create a cross-validation splitter.

    Parameters:
    X (DataFrame): The feature matrix.
    min_length (int): The minimum length of a sample for cross-validation.
    offset (int): The offset used in cross-validation splitting.
    split (int): Index at which to split the data in cross-validation.
    set_labels (list): Labels for the train and test sets in cross-validation.

    Returns:
    cv_splitter (SKLSplitter): The cross-validation splits created from cv.get_splitter(X).
    cv (SKLSplitter): The cross-validation object.
    """

    # Cross-validate Creates a cross-validation object with all the indexes for each cv split
    cv = vbt.SKLSplitter("from_expanding", min_length=min_length, offset=offset, split=split, set_labels=set_labels)
    cv_splitter = cv.get_splitter(X)
    
    return cv_splitter, cv

def create_cv_with_gap(X, min_length=600, test_amount=200, gap = 150, set_labels=["train", "test"]):
    """
    Create a cross-validation splitter.

    Parameters:
    X (DataFrame): The feature matrix.
    min_length (int): The minimum length of a sample for cross-validation.
    offset (int): The offset used in cross-validation splitting.
    split (int): Index at which to split the data in cross-validation.
    set_labels (list): Labels for the train and test sets in cross-validation.

    Returns:
    cv_splitter (SKLSplitter): The cross-validation splits created from cv.get_splitter(X).
    cv (SKLSplitter): The cross-validation object.
    """

    # Cross-validate Creates a cross-validation object with all the indexes for each cv split
    cv = vbt.SKLSplitter("from_expanding", 
                         min_length=min_length, 
                         offset=test_amount, 
                         split=(1.0, vbt.RelRange(length=gap, is_gap=True), test_amount), 
                         set_labels=set_labels,
                         split_range_kwargs=dict(backwards=True)
                         )
    cv_splitter = cv.get_splitter(X)
    
    return cv_splitter, cv

def create_rolling_cv(X, length=2000, split=0.90, offset=True, offsetlen=0, set_labels=["train", "test"]):
    """
    Create a cross-validation splitter.

    Parameters:
    X (DataFrame): The feature matrix.
    min_length (int): The minimum length of a sample for cross-validation.
    split (float): percent of window to split training vs testing.
    set_labels (list): Labels for the train and test sets in cross-validation.
    offset (bool): Whether to offset the splits, True shifts the window forward by only the test number.

    Returns:
    cv_splitter (SKLSplitter): The cross-validation splits created from cv.get_splitter(X).
    cv (SKLSplitter): The cross-validation object.
    """
    if offset:
        offsetlen = 2*(length * split) - length
        cv = vbt.SKLSplitter("from_rolling", length=length, split=split, offset=-offsetlen, offset_anchor="prev_end", set_labels=set_labels)
        cv_splitter = cv.get_splitter(X) 
        return cv_splitter, cv
    # Cross-validate Creates a cross-validation object with all the indexes for each cv split
    else:
        cv = vbt.SKLSplitter("from_rolling", length=length, split=split, set_labels=set_labels) # offset=-offsetlen, offset_anchor="prev_end",
        cv_splitter = cv.get_splitter(X) 
        return cv_splitter, cv
    
def create_rolling_cv_with_gap(X, length=500, split=0.70, gap=150, set_labels=["train", "test"]):
    """
    Create a cross-validation splitter.

    Parameters:
    X (DataFrame): The feature matrix.
    length (int): The length of a sample for cross-validation.
    split (float): The percent of the sample to use for training.
    gap (int): The gap between the training and test sets.
    set_labels (list): Labels for the train and test sets in cross-validation.

    Returns:
    cv_splitter (SKLSplitter): The cross-validation splits created from cv.get_splitter(X).
    cv (SKLSplitter): The cross-validation object.
    """
    assert length > gap, "Length must be greater than gap"

    split_size = int((length - gap) * split) # Total length of the set minus the gap times the split percent is training set
    test_size = length - split_size - gap # Total length minus the training set minus the gap is the test set
    offset = -(split_size-test_size) # Offset the split by the difference between the training and test set this gets the next test set to start where the last one ended

    cv = vbt.SKLSplitter("from_rolling", 
                        length=length,
                        split=(split_size, vbt.RelRange(length=gap, is_gap=True), 1.0),
                        offset=offset,
                        set_labels=set_labels)
    cv_splitter = cv.get_splitter(X)
    return cv_splitter, cv

def cross_validate_and_train(pipeline, X, y, cv_splitter, model_name="", verbose_interval=10, n_clusters=6, clustering=False):
    # Predictions
    X_slices = cv_splitter.take(X)
    y_slices = cv_splitter.take(y)
    
    # Print total number of splits
    total_splits = len(X_slices.index.unique(level="split"))
    print(f"Total number of cross-validation splits: {total_splits}")

    test_labels = []
    test_preds = []
    for split in X_slices.index.unique(level="split"):  
        X_train_slice= X_slices[(split, "train")]  
        y_train_slice= y_slices[(split, "train")] 
        X_test_slice = X_slices[(split, "test")]
        y_test_slice = y_slices[(split, "test")]
        
        # If clustering is enabled
        if clustering:
            # Fit the KMeans clustering algorithm on the training data using only the original columns in X_slices
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(X_train_slice)
            
            # Get the cluster labels for the training data using the KMeans clustering algorithm fitted on the training data
            train_cluster_labels = kmeans.predict(X_train_slice)
            
            # Add the "cluster" column to the training data using the cluster labels obtained above
            X_train_slice["cluster"] = train_cluster_labels

            # Get the cluster labels for the test data using the KMeans clustering algorithm fitted on the training data
            test_cluster_labels = kmeans.predict(X_test_slice)
            
            # Get the cluster labels and their counts for the test data
            test_cluster_counts = Counter(test_cluster_labels)

            # Add the "cluster" column to the test data using the cluster labels obtained above
            X_test_slice["cluster"] = test_cluster_labels

        # Fit the pipeline on the training data
        pipeline.fit(X_train_slice, y_train_slice)
        
        # Make predictions on the test data
        test_pred = pipeline.predict(X_test_slice)  
        test_pred = pd.Series(test_pred, index=y_test_slice.index)
        test_labels.append(y_test_slice)
        test_preds.append(test_pred)

        # Only print the MSE every 'verbose_interval' splits
        if split % verbose_interval == 0:
            print(f"{model_name} Split {split} Mean Squared Error: {mean_squared_error(y_test_slice, test_pred)}")

            if clustering:
                # Print the cluster labels and their counts
                print(f"Cluster Sizes:")
                for label, count in test_cluster_counts.items():
                    print(f"Cluster {label}: {count}")

    # Concatenate the test labels and predictions into a single Series
    test_labels = pd.concat(test_labels).rename("labels")  
    test_preds = pd.concat(test_preds).rename("preds")
    
    # Drop Duplicates
    test_labels = test_labels[~test_labels.index.duplicated(keep='first')]
    test_preds = test_preds[~test_preds.index.duplicated(keep='first')]
    
    return pipeline, test_labels, test_preds

def evaluate_predictions(test_labels, test_preds, model_name="", meta=False):
    if meta:  # Classification metrics for the metamodel
        acc = accuracy_score(test_labels, test_preds)
        prec = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        auc_roc = roc_auc_score(test_labels, test_preds)

        print(f"{model_name} Accuracy: {acc}")
        print(f"{model_name} Precision: {prec}")
        print(f"{model_name} Recall: {recall}")
        print(f"{model_name} F1 Score: {f1}")
        print(f"{model_name} AUC-ROC: {auc_roc}")

        return acc, prec, recall, f1, auc_roc

    else:  # Regression metrics for the original model
        mse = mean_squared_error(test_labels, test_preds)
        rmse = np.sqrt(mse)  # or use mean_squared_error with squared=False
        mae = mean_absolute_error(test_labels, test_preds)
        r2 = r2_score(test_labels, test_preds)

        print(f"{model_name} Mean Squared Error (MSE): {mse}")
        print(f"{model_name} Root Mean Squared Error (RMSE): {rmse}")
        print(f"{model_name} Mean Absolute Error (MAE): {mae}")
        print(f"{model_name} R-squared: {r2}")

        return mse, rmse, mae, r2

def extract_feature_importance(pipeline, X, clustering=False, top_n=None):
    fitted_model = pipeline.named_steps['model']
    feature_names = X.columns.tolist()
    
    if clustering:
        feature_names.append('cluster')

    # Create a DataFrame using a Dictionary
    feature_names_series = pd.Series(feature_names, name='feature_names')
    feature_importance_series = pd.Series(fitted_model.feature_importances_, name='feature_importance')

    fi_df = pd.concat([feature_names_series, feature_importance_series], axis=1)

    # Check if there are any NaNs in the DataFrame
    missing_values = fi_df.isnull().sum()
    if missing_values.any():
        print("Found missing feature importance. Features are:")
        print(fi_df[fi_df.isnull().any(axis=1)])
    else:
        print("No missing feature importance found.")

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Select top_n features
    if top_n is not None:
        fi_df = fi_df.head(top_n)

    # Define size of bar plot
    plt.figure(figsize=(12,8))
    # Plot bar chart
    plt.barh(fi_df['feature_names'], fi_df['feature_importance'], align='center')
    # Add chart labels
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

    # Print feature names and importance
    for index, row in fi_df.iterrows():
        print(f"{row['feature_names']}: {row['feature_importance']}")
   
def plot_prediction_vs_actual(x, y, pos_threshold=0.09, neg_threshold=-0.09, title='Test Predictions vs Actual Results'):
    """
    Plots predictions against actual results and calculates the line of best fit.
    Also draws vertical lines at the provided positive and negative thresholds.

    Parameters:
    x (Series): Predictions
    y (Series): Actual results
    pos_threshold (float): Optimal positive threshold
    neg_threshold (float): Optimal negative threshold
    title (str): Title of the plot

    Returns:
    None
    """

    # Create condition masks for different data types
    tp_condition = (x > 0) & (y > 0)  # True positives condition
    tn_condition = (x < 0) & (y < 0)  # True negatives condition
    fp_condition = (x > 0) & (y < 0)  # False positives condition
    fn_condition = (x < 0) & (y > 0)  # False negatives condition

    # Calculate percent in each condition
    tp_percent = (tp_condition.sum() / len(x)) * 100
    tn_percent = (tn_condition.sum() / len(x)) * 100
    fp_percent = (fp_condition.sum() / len(x)) * 100
    fn_percent = (fn_condition.sum() / len(x)) * 100

    # Create scatter plots for each condition with different colors
    plt.scatter(x[tp_condition], y[tp_condition], color='green', alpha=0.5, label='True Positives', s=10)
    plt.scatter(x[tn_condition], y[tn_condition], color='pink', alpha=0.5, label='True Negatives', s=10)
    plt.scatter(x[fp_condition], y[fp_condition], color='grey', alpha=0.1, label='False Positives', s=10)
    plt.scatter(x[fn_condition], y[fn_condition], color='grey', alpha=0.1, label='False Negatives', s=10)

    # Calculate the line of best fit
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)

    # Generate y-values based on the polynomial
    y_fit = polynomial(x)

    # Plot the line of best fit
    plt.plot(x, y_fit, color='black', label='Line of Best Fit')

    # Draw black dotted lines at x=0 and y=0
    plt.axhline(0, color='black', linestyle='dotted')
    plt.axvline(0, color='black', linestyle='dotted')

    # Draw vertical lines at the optimal thresholds if they are not None
    if pos_threshold is not None:
        plt.axvline(x=pos_threshold, color='green', linestyle='--', label='Positive Threshold')
    if neg_threshold is not None:
        plt.axvline(x=neg_threshold, color='red', linestyle='--', label='Negative Threshold')

    # Add title and labels to the axes
    plt.title(title)
    plt.xlabel('Predictions')
    plt.ylabel('Actual Results')

    # Add a legend
    plt.legend()

    # Print the equation of the line
    slope, intercept = coefficients
    print(f"The equation of the regression line is: y = {slope:.3f}x + {intercept:.3f}")

    # Print the percent in each quadrant
    print(f"\nPercentage of True Positives: {tp_percent:.2f}%")
    print(f"Percentage of True Negatives: {tn_percent:.2f}%")
    print(f"Percentage of False Positives: {fp_percent:.2f}%")
    print(f"Percentage of False Negatives: {fn_percent:.2f}%")

    # Show the plot
    plt.show()
   
def simulate_pf(data, test_preds, open_long_th=0.01, close_long_th=0.0, close_short_th=0.0, open_short_th=-0.01, plot=True):
    # Simulate a portfolio making trades based on predictions
    insample_pf = vbt.Portfolio.from_signals(
    data.close[test_preds.index],  # use only the test set
    entries         = test_preds > open_long_th,  # long when probability of price increase is greater than 2%
    exits           = test_preds < close_long_th,  # long when probability of price increase is greater than 2%
    short_entries   = test_preds < open_short_th,  # long when probability of price increase is greater than 2%
    short_exits     = test_preds > close_short_th,  # short when probability prediction is less than -5%
    # direction="both" # long and short
)
    print(insample_pf.stats())
    if plot==True:
        insample_pf.plot().show()
    return insample_pf

def find_optimal_thresholds(y_true, y_pred, target_percent_pos=70, target_percent_neg=70):
    # Combine y_true and y_pred into a DataFrame
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # Initialize thresholds to None
    pos_threshold = None
    neg_threshold = None

    # Sort the DataFrame by y_pred in ascending order
    data.sort_values(by='y_pred', ascending=True, inplace=True)

    # For positive threshold
    for index, row in data.iterrows():
        # Slice the DataFrame from the current prediction upwards
        slice_pos = data[data['y_pred'] >= row['y_pred']]

        # Calculate the percentage of true positives in the slice
        percent_true_positives = (slice_pos['y_true'] > 0).sum() / len(slice_pos)

        # If the percentage of true positives is equal to or greater than the target, store the current prediction
        if percent_true_positives >= target_percent_pos / 100:
            pos_threshold = row['y_pred']
            break

    # Sort the DataFrame by y_pred in descending order
    data.sort_values(by='y_pred', ascending=False, inplace=True)

    # For negative threshold
    for index, row in data.iterrows():
        # Slice the DataFrame from the current prediction downwards
        slice_neg = data[data['y_pred'] <= row['y_pred']]

        # Calculate the percentage of true negatives in the slice
        percent_true_negatives = (slice_neg['y_true'] < 0).sum() / len(slice_neg)

        # If the percentage of true negatives is equal to or greater than the target, store the current prediction
        if percent_true_negatives >= target_percent_neg / 100:
            neg_threshold = row['y_pred']
            break

    # Return the positive and negative thresholds
    return pos_threshold, neg_threshold

def _compute_atr(data, period=75*14):
    """Compute Average True Range (ATR)"""
    hl = data['High'] - data['Low']
    hc = np.abs(data['High'] - data['Close'].shift())
    lc = np.abs(data['Low'] - data['Close'].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(period).mean()

def apply_triple_barrier(data, atr_multiplier=2, prediction_window=75, use_four_labels=True):
    """
    Label the data using the triple barrier method.
    
    If use_four_labels is True:
    - 0: Lower barrier breach
    - 1: No barrier breach, but slightly down
    - 2: Upper barrier breach
    - 3: No barrier breach, but slightly up
    
    If use_four_labels is False:
    - 0: Lower barrier breach
    - 1: No barrier breach
    - 2: Upper barrier breach
    """
    data['atr'] = _compute_atr(data, period=prediction_window*14)
    
    # Define barriers
    data['upper_barrier'] = data['Close'] + (data['atr'] * atr_multiplier)
    data['lower_barrier'] = data['Close'] - (data['atr'] * atr_multiplier)

    # Initialize labels column
    data['label'] = 1 if not use_four_labels else 0  # Default to no-barrier (3-label system) or time-barrier (4-label system)

    for i in range(0, len(data) - prediction_window):
        window_highs = data['High'].iloc[i+1: i+1+prediction_window].max()
        window_lows = data['Low'].iloc[i+1: i+1+prediction_window].min()

        # Check upper barrier
        if window_highs >= data['upper_barrier'].iloc[i]:
            data['label'].iloc[i] = 2  # upper barrier breach
        # Check lower barrier
        elif window_lows <= data['lower_barrier'].iloc[i]:
            data['label'].iloc[i] = 0  # lower barrier breach
        elif use_four_labels:
            data['label'].iloc[i] = 3 if data['Close'].iloc[i+1] > data['Close'].iloc[i] else 1

    # Drop unnecessary columns for clarity
    data.drop(columns=['atr', 'upper_barrier', 'lower_barrier'], inplace=True)
    
    return data




def prepare_meta_data(data, atr_multiplier=1, prediction_window=75):
    """
    Prepare data for the meta-model. 
    Adds the triple barrier labels and selects the relevant features.
    """
    # Apply triple barrier to get labels
    data_labeled = apply_triple_barrier(data, atr_multiplier, prediction_window)
    
    # Select relevant columns as features
    X = data_labeled[['long_minus_short', 'long_slope', 'short_slope', 'Close']]
    
    # Target variable
    y = data_labeled['label']
    
    return X, y

