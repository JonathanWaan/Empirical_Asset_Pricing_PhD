import pandas as pd
import statsmodels.api as sm
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats

import torch

def load_data():
    # Load datasets
    crsp_data = pd.read_sas("crsp_data.sas7bdat", format="sas7bdat")
    comp_data = pd.read_sas("comp_data.sas7bdat", format="sas7bdat")

    # Combine year and month into a single date column
    crsp_data['date'] = pd.to_datetime(crsp_data[['year', 'month']].assign(day=1))
    comp_data['date'] = pd.to_datetime(comp_data[['year', 'month']].assign(day=1))

    # Ensure PERMNO column exists in both datasets
    comp_data["PERMNO"] = comp_data["permno"]

    # Drop NaN values
    comp_data = comp_data.dropna(subset=["PERMNO", "date"])
    crsp_data = crsp_data.dropna(subset=["PERMNO", "date"])

    # Explicitly sort both datasets by PERMNO and date in ascending order
    crsp_data = crsp_data.sort_values(by=[ "date","PERMNO"], ascending=[True, True]).reset_index(drop=True)
    comp_data = comp_data.sort_values(by=[ "date","PERMNO"], ascending=[True, True]).reset_index(drop=True)

    # Check sorted datasets
    print("\nCRSP Data:")
    print(crsp_data.head())

    print("\nCompustat Data:")
    print(comp_data.head())

    return crsp_data, comp_data


def prepare_data(crsp_data, target_var):



    print(f"Target Variable: {target_var}")
    # Prepare the data
    # grab feature and target variable dataset
    # feature_col = ['B12m',  'RU12', 'ME']
    crsp_data[target_var+"_feat"] = crsp_data[target_var]
    feature_col = crsp_data.columns.difference([target_var,"month","year","date","PERMNO"])
    feature_data = crsp_data[list(feature_col) + ["date","PERMNO"]]
    target_data = crsp_data[["date",target_var,"PERMNO"]]
    target_data['date'] = target_data['date'] - pd.DateOffset(months=int(target_var[1:-1]))
    # remove nan in target data
    target_data = target_data.dropna()
    # Merge the target and feature data
    merged_data = pd.merge(target_data, feature_data, on=["date","PERMNO"], how="left")
    merged_data_na = merged_data.dropna()

# PERMNO * date * feat

    return merged_data_na, feature_col

def prepare_data_multi_day(crsp_data, target_var, n_days=5):

    print(f"Target Variable: {target_var}")
    # Prepare the data
    # grab feature and target variable dataset
    # feature_col = ['B12m',  'RU12', 'ME']
    crsp_data = crsp_data[crsp_data['year']>1980]
    crsp_data[target_var+"_feat"] = crsp_data[target_var]
    feature_col = crsp_data.columns.difference([target_var,"month","year","date","PERMNO"])
    feature_data = crsp_data[list(feature_col) + ["date","PERMNO"]]
    target_data = crsp_data[["date",target_var,"PERMNO"]]
    target_data['date'] = target_data['date'] - pd.DateOffset(months=int(target_var[1:-1]))
    # remove nan in target data
    target_data = target_data.dropna()
    # Merge the target and feature data
    merged_data = pd.merge(target_data, feature_data, on=["date","PERMNO"], how="left")

    # merged_data = merged_data.dropna(subset=[feature_col[0]])
    # PERMNO * date * feat
    # Step 0: Ensure 'date' is datetime
    merged_data['date'] = pd.to_datetime(merged_data['date'])

    # Step 1: Get all unique PERMNOs and all unique dates
    permnos = merged_data['PERMNO'].unique()
    dates = merged_data['date'].unique()

    # Step 2: Create full grid of all PERMNO-date combinations
    full_grid = pd.MultiIndex.from_product(
        [permnos, dates], names=['PERMNO', 'date']
    ).to_frame(index=False)

    # Step 3: Left join the actual data to the full grid
    merged_full = pd.merge(full_grid, merged_data, on=['PERMNO', 'date'], how='left')

    # Step 4: Sort by PERMNO then date
    merged_full = merged_full.sort_values(['PERMNO', 'date'])
    merged_full = merged_full.dropna(subset=feature_col, how='all')
    dataset = []
    dates = np.sort(merged_full['date'].unique())  # ensure sorted

    for i in range(len(dates) - n_days + 1):
        # 1. Select window dates
        if i % 10 == 9:
            print("day",i,"out of", len(dates) - n_days + 1)

        window_dates = dates[i:i + n_days]

        # 2. Slice original data for this window
        window_df = merged_full[merged_full['date'].isin(window_dates)]

        # 3. Find PERMNOs that have at least one row with non-NaN values (excluding PERMNO and date cols)
        # feature_cols = window_df.columns.difference(['PERMNO', 'date'])
        valid_permnos = window_df.dropna(subset=feature_col, how='all')['PERMNO'].unique()

        if len(valid_permnos) == 0:
            continue  # skip this window if no PERMNO has valid data

        # 4. Create full grid using only valid PERMNOs
        full_grid = pd.MultiIndex.from_product(
            [valid_permnos, window_dates],
            names=['PERMNO', 'date']
        ).to_frame(index=False)

        # 5. Merge full grid with actual data
        merged_window = pd.merge(full_grid, window_df, on=['PERMNO', 'date'], how='left')

        # # 6. Drop if any NaNs (or use fillna if desired)
        # if merged_window.isnull().any(axis=None):
        #     continue

        # 7. Sort and reshape
        merged_window = merged_window.sort_values(['PERMNO', 'date'])
        
        sample_matrix = merged_window.to_numpy()
        n_features = sample_matrix.shape[1]
        sample_matrix = sample_matrix.reshape(len(valid_permnos), n_days, n_features)

        # 8. Append
        dataset.append(sample_matrix)

    # Final result: (num_windows, n_permnos, n_days, n_features)



    return dataset

    # # Step 6: Reshape into 3D: (PERMNOs, dates, features)
    # n_permnos = len(permnos)
    # n_dates = len(dates)
    # n_features = merged_full.shape[1]

    # matrix_3d = matrix.reshape(n_permnos, n_dates, n_features)    
    # return matrix_3d, feature_col


def prepare_data_lag(crsp_data, target_var, n_lags=5):
    """
    Prepare data by creating lagged features per PERMNO and aligning target.

    Parameters:
    crsp_data: Original dataframe with all dates
    target_var: String, e.g. "B36m"
    n_lags: Number of lag periods to include as features (default=5)

    Returns:
    merged_data_na: final DataFrame ready for modeling
    feature_cols_final: list of all feature column names (original + lagged)
    """
    print(f"Target Variable: {target_var}, using {n_lags} lagged periods")

    # Extract base features
    crsp_data[target_var+"_feat"] = crsp_data[target_var]
    base_feature_cols = crsp_data.columns.difference([target_var, "month", "year", "date", "PERMNO"])
    lagged_feature_cols = []

    crsp_lagged = crsp_data.copy()

    # Generate lagged features for each base feature
    for col in base_feature_cols:
        for lag in range(1, n_lags + 1):
            lagged_col = f"{col}_lag{lag}"
            crsp_lagged[lagged_col] = crsp_lagged.groupby("PERMNO")[col].shift(lag)
            lagged_feature_cols.append(lagged_col)

    # Shift target if it's future-looking (e.g., B36m means outcome is 36 months later)
    target_df = crsp_lagged[["PERMNO", "date", target_var]].copy()
    target_df['date'] = target_df['date'] - pd.DateOffset(months=int(target_var[1:-1]))  # align with past features

    # Combine current features + lagged features
    full_feature_df = crsp_lagged[["PERMNO", "date"] + list(base_feature_cols) + lagged_feature_cols]

    # Merge target and feature sets
    merged_data = pd.merge(target_df, full_feature_df, on=["PERMNO", "date"], how="inner")
    merged_data_na = merged_data.dropna()

    # Final feature list
    feature_cols_final = list(base_feature_cols) + lagged_feature_cols

    return merged_data_na, feature_cols_final


def grab_training_data_for_NN(merged_data_na, lookback_period, target_var):
    # process data to be fit for temporatl NN model
    # merged_data_na dimension: date, feature
    # resulting data dimention: batch_size, time_step, feature
    # method: 
    # sort by date
    merged_data_na = merged_data_na.sort_values(by=["date"])
    merged_data_na_np = merged_data_na.to_numpy()[:,1:] # remove date column
    merged_data_na_np = np.array(merged_data_na_np, dtype=np.float64)
    # create 3d array
    # 1. create empty array
    # 2. iterate each date, grab the 2d array of data
    
    training_data = []

    for i in range(len(merged_data_na_np) - lookback_period):

        batch = merged_data_na_np[i:i + lookback_period]
        training_data.append(torch.tensor(batch))

    training_data = torch.stack(training_data, dim=0)
    X = training_data[:, :, 1:]
    y = training_data[:, -1, 0]
    return X, y

    

def run_regression(merged_data_na, target_var):
    # Group by date for Fama-MacBeth approach
    grouped = merged_data_na.groupby('date')

    # Initialize lists to store coefficients and R-squared values
    coefficients = []
    r_squared_values = []
    dates = []

    # Run separate regressions for each time period
    for date, group in grouped:
        # Skip dates with too few observations
        if len(group) < 10:  # Minimum threshold for reliable regression
            continue
            
        # Prepare data for this time period
        X = group[group.columns.difference([target_var,"month","year","date","PERMNO"])]
        X = sm.add_constant(X)
        y = group[target_var]
        
        # Run regression
        try:
            model = sm.OLS(y, X).fit()
            
            # Store coefficients and R-squared
            coefficients.append(model.params)
            r_squared_values.append(model.rsquared)
            dates.append(date)
            
        except Exception as e:
            print(f"Error for date {date}: {e}")
            continue

    # Convert lists to DataFrames for easier analysis
    coef_df = pd.DataFrame(coefficients, index=dates)
    r_squared_df = pd.DataFrame(r_squared_values, index=dates, columns=['R_squared'])

    # Calculate Fama-MacBeth statistics
    mean_coefficients = coef_df.mean()
    std_coefficients = coef_df.std()
    t_statistics = mean_coefficients / (std_coefficients / np.sqrt(len(coef_df)))
    p_values = 2 * (1 - stats.t.cdf(abs(t_statistics), df=len(coef_df)-1))

    # Create summary DataFrame
    fama_macbeth_results = pd.DataFrame({
        'Coefficient': mean_coefficients,
        'Std Error': std_coefficients / np.sqrt(len(coef_df)),
        't-statistic': t_statistics,
        'p-value': p_values
    })
    # Print results
    print("\nFama-MacBeth Regression Results:")
    print(fama_macbeth_results)

    # Print average R-squared
    mean_r_squared = r_squared_df['R_squared'].mean()
    print(f"\nAverage R-squared: {mean_r_squared:.4f}")

    # Additional evaluation metrics
    print(f"Number of time periods: {len(coef_df)}")
    print(f"Total observations: {len(merged_data_na)}")

    return fama_macbeth_results, coef_df, r_squared_df







def plot_coefficients(coef_df, feature_col):
    # Plot time series of coefficients for key variables
    plt.figure(figsize=(12, 8))
    for col in feature_col:
        plt.plot(coef_df.index, coef_df[col], label=col)
    plt.title("Time Series of Coefficients")
    plt.xlabel("Date")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_r_squared(r_squared_df, model_name = "LR"):
    # Plot time series of R-squared
    plt.figure(figsize=(10, 6))
    plt.plot(r_squared_df.index, r_squared_df['R_squared'])
    plt.title(f"Time Series of R-squared Values {model_name}")
    plt.xlabel("Date")
    plt.ylabel("R-squared")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def prepare_data_b_ru(crsp_data, target_var, n_lags=5):
    """
    Prepare data by creating lagged features per PERMNO and aligning target.

    Parameters:
    crsp_data: Original dataframe with all dates
    target_var: String, e.g. "B36m"
    n_lags: Number of lag periods to include as features (default=5)

    Returns:
    merged_data_na: final DataFrame ready for modeling
    feature_cols_final: list of all feature column names (original + lagged)
    """
    print(f"Target Variable: {target_var}, using {n_lags} lagged periods")

    # Extract base features
    crsp_data[target_var+"_feat"] = crsp_data[target_var]
    base_feature_cols = crsp_data.columns.difference([target_var, "month", "year", "date", "PERMNO"])
    lagged_feature_cols = []

    crsp_lagged = crsp_data.copy()
    crsp_lagged['RU_future'] = crsp_lagged["RU12"]
    
    # Generate lagged features for each base feature
    for col in base_feature_cols:
        for lag in range(1, n_lags + 1):
            lagged_col = f"{col}_lag{lag}"
            crsp_lagged[lagged_col] = crsp_lagged.groupby("PERMNO")[col].shift(lag)
            lagged_feature_cols.append(lagged_col)

    # Shift target if it's future-looking (e.g., B36m means outcome is 36 months later)
    target_df = crsp_lagged[["PERMNO", "date", target_var,'RU_future']].copy()
    target_df['date'] = target_df['date'] - pd.DateOffset(months=int(target_var[1:-1]))  # align with past features

    # Combine current features + lagged features
    full_feature_df = crsp_lagged[["PERMNO", "date"] + list(base_feature_cols) + lagged_feature_cols]

    # Merge target and feature sets
    merged_data = pd.merge(target_df, full_feature_df, on=["PERMNO", "date"], how="inner")
    merged_data_na = merged_data.dropna()

    # Final feature list
    feature_cols_final = list(base_feature_cols) + lagged_feature_cols

    return merged_data_na, feature_cols_final