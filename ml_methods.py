import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import statsmodels.api as sm
from utils import *
from sklearn.neural_network import MLPRegressor

# def run_model_by_date(merged_data_na, target_var, model_func, model_name):
#     """
#     Run a specified model for each time period
    
#     Parameters:
#     merged_data_na: DataFrame with merged data
#     target_var: String, name of the target variable
#     model_func: Function that returns a fitted model and predictions
#     model_name: String, name of the model for display
    
#     Returns:
#     r_squared_df: DataFrame with R-squared values by date
#     avg_time: Average execution time per date
#     """
#     grouped = merged_data_na.groupby('date')
    
#     r_squared_values = []
#     mse_values = []
#     dates = []
#     execution_times = []
    
#     for date, group in grouped:
#         if len(group) < 10:
#             continue
            
#         # Prepare data for this time period
#         X = group[group.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
#         y = group[target_var]
        
#         # # Split data for evaluation
#         # X_train, X_test, y_train, y_test = train_test_split(
#         #     X, y, test_size=0.2, random_state=42
#         # )
        
#         # Split data for evaluation
#         X_train, X_test, y_train, y_test = X, X, y, y

#         try:
#             start_time = time.time()
            
#             # Train and evaluate model
#             r2, mse = model_func(X_train, X_test, y_train, y_test)
            
#             end_time = time.time()
#             execution_time = end_time - start_time
            
#             r_squared_values.append(r2)
#             mse_values.append(mse)
#             dates.append(date)
#             execution_times.append(execution_time)
            
#         except Exception as e:
#             print(f"Error for date {date} with {model_name}: {e}")
#             continue
    
#     # Create results DataFrame
#     r_squared_df = pd.DataFrame({
#         'R_squared': r_squared_values,
#         'MSE': mse_values
#     }, index=dates)
#     # Calculate average metrics
#     mean_r_squared = np.mean(r_squared_values)
#     mean_mse = np.mean(mse_values)
#     avg_time = np.mean(execution_times)
    
#     print(f"Average R-squared ({model_name}): {mean_r_squared:.4f}")
#     print(f"Average MSE ({model_name}): {mean_mse:.4f}")
#     print(f"Average execution time: {avg_time:.4f} seconds")
    
#     return r_squared_df, mean_r_squared,mean_mse, avg_time




def run_model_by_date(merged_data_na, target_var, model_func, model_name):
    """
    Run a specified model training on current date and testing on next date.
    
    Parameters:
    merged_data_na: DataFrame with merged data
    target_var: String, name of the target variable
    model_func: Function that returns a fitted model and predictions
    model_name: String, name of the model for display
    
    Returns:
    r_squared_df: DataFrame with R-squared and MSE values by date
    mean_r_squared: Average R-squared
    mean_mse: Average MSE
    avg_time: Average execution time per date
    """
    grouped = merged_data_na.groupby('date')
    sorted_dates = sorted(grouped.groups.keys())

    r_squared_values = []
    mse_values = []
    dates = []
    execution_times = []

    for i in range(len(sorted_dates) - 1):
        date_train = sorted_dates[i]
        date_test = sorted_dates[i + 1]

        group_train = grouped.get_group(date_train)
        group_test = grouped.get_group(date_test)

        if len(group_train) < 10 or len(group_test) < 10:
            continue

        X_train = group_train[group_train.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
        y_train = group_train[target_var]

        X_test = group_test[group_test.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
        y_test = group_test[target_var]

        try:
            start_time = time.time()

            r2, mse = model_func(X_train, X_test, y_train, y_test)

            end_time = time.time()
            execution_time = end_time - start_time

            r_squared_values.append(r2)
            mse_values.append(mse)
            dates.append(f"{date_train} → {date_test}")
            execution_times.append(execution_time)

        except Exception as e:
            print(f"Error for {date_train} → {date_test} with {model_name}: {e}")
            continue

    # Create results DataFrame
    r_squared_df = pd.DataFrame({
        'R_squared': r_squared_values,
        'MSE': mse_values
    }, index=dates)

    mean_r_squared = np.mean(r_squared_values)
    mean_mse = np.mean(mse_values)
    avg_time = np.mean(execution_times)

    print(f"Average R-squared ({model_name}): {mean_r_squared:.4f}")
    print(f"Average MSE ({model_name}): {mean_mse:.4f}")
    print(f"Average execution time: {avg_time:.4f} seconds")

    return r_squared_df, mean_r_squared, mean_mse, avg_time




def run_model_by_date_ensemble(merged_data_na, target_var, model_func, model_name, window_size=5):
    """
    Train models on past `window_size` periods and average predictions on the next (6th) period.
    
    Parameters:
    merged_data_na: DataFrame with merged data
    target_var: String, name of the target variable
    model_func: Function that returns a fitted model and predictions
    model_name: String, name of the model for display
    window_size: Number of periods to use for training
    
    Returns:
    r_squared_df: DataFrame with R-squared and MSE values by test date
    """
    grouped = merged_data_na.groupby('date')
    sorted_dates = sorted(grouped.groups.keys())

    r_squared_values = []
    mse_values = []
    dates = []
    execution_times = []

    for i in range(window_size, len(sorted_dates)):
        train_dates = sorted_dates[i - window_size:i]
        test_date = sorted_dates[i]

        try:
            group_test = grouped.get_group(test_date)
            if len(group_test) < 10:
                continue

            X_test = group_test[group_test.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
            y_test = group_test[target_var]

            # Collect predictions from each model trained on one of the past 5 periods
            all_preds = []

            start_time = time.time()

            for train_date in train_dates:
                group_train = grouped.get_group(train_date)
                if len(group_train) < 10:
                    continue

                X_train = group_train[group_train.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
                y_train = group_train[target_var]

                _, _,y_pred = model_func(X_train, X_test, y_train, y_test)
                all_preds.append(y_pred)

            if len(all_preds) == 0:
                continue

            # Average predictions
            y_pred_avg = np.mean(all_preds, axis=0)

            r2 = r2_score(y_test, y_pred_avg)
            mse = mean_squared_error(y_test, y_pred_avg)

            end_time = time.time()

            r_squared_values.append(r2)
            mse_values.append(mse)
            dates.append(f"avg({train_dates[0]}→{train_dates[-1]}) → {test_date}")
            execution_times.append(end_time - start_time)

        except Exception as e:
            print(f"Error for ensemble up to {test_date} with {model_name}: {e}")
            continue

    r_squared_df = pd.DataFrame({
        'R_squared': r_squared_values,
        'MSE': mse_values
    }, index=dates)

    mean_r_squared = np.mean(r_squared_values)
    mean_mse = np.mean(mse_values)
    avg_time = np.mean(execution_times)

    print(f"Average R-squared ({model_name}): {mean_r_squared:.4f}")
    print(f"Average MSE ({model_name}): {mean_mse:.4f}")
    print(f"Average execution time: {avg_time:.4f} seconds")

    return r_squared_df, mean_r_squared, mean_mse, avg_time




def run_model_by_date_concat_train(merged_data_na, target_var, model_func, model_name, window_size=5):
    """
    Train a model using the concatenated data from the past `window_size` periods
    and predict on the current period (6th).
    
    Parameters:
    merged_data_na: DataFrame with merged data
    target_var: String, name of the target variable
    model_func: Function that returns r2 and y_pred
    model_name: String, name of the model for display
    window_size: Number of previous periods to use as training data
    
    Returns:
    r_squared_df: DataFrame with R-squared and MSE values by test date
    """
    grouped = merged_data_na.groupby('date')
    sorted_dates = sorted(grouped.groups.keys())

    r_squared_values = []
    mse_values = []
    dates = []
    execution_times = []

    for i in range(window_size, len(sorted_dates)):
        train_dates = sorted_dates[i - window_size:i]
        test_date = sorted_dates[i]

        try:
            # Combine training data from past `window_size` periods
            train_groups = [grouped.get_group(d) for d in train_dates if len(grouped.get_group(d)) >= 10]
            if len(train_groups) < window_size:
                continue

            train_data = pd.concat(train_groups)
            test_data = grouped.get_group(test_date)

            if len(test_data) < 10:
                continue

            X_train = train_data[train_data.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
            y_train = train_data[target_var]

            X_test = test_data[test_data.columns.difference([target_var, "month", "year", "date", "PERMNO"])]
            y_test = test_data[target_var]

            start_time = time.time()

            _,_, y_pred = model_func(X_train, X_test, y_train, y_test)
            # _,_, y_pred = model_func(X_train, X_train, y_train, y_train)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            end_time = time.time()

            r_squared_values.append(r2)
            mse_values.append(mse)
            dates.append(f"{train_dates[0]}→{train_dates[-1]} → {test_date}")
            execution_times.append(end_time - start_time)
            if i % 15 == 0:
                print(f"Processed {train_dates[0]}→{train_dates[-1]} → {test_date} with {model_name}: R2={r2:.4f}, MSE={mse:.4f}")

        except Exception as e:
            print(f"Error for combined training on {train_dates} to predict {test_date} with {model_name}: {e}")
            continue

    r_squared_df = pd.DataFrame({
        'R_squared': r_squared_values,
        'MSE': mse_values
    }, index=dates)

    mean_r_squared = np.mean(r_squared_values)
    mean_mse = np.mean(mse_values)
    avg_time = np.mean(execution_times)

    print(f"Average R-squared ({model_name}): {mean_r_squared:.4f}")
    print(f"Average MSE ({model_name}): {mean_mse:.4f}")
    print(f"Average execution time: {avg_time:.4f} seconds")

    return r_squared_df, mean_r_squared, mean_mse, avg_time



# Define model functions
def linear_regression_model(X_train, X_test, y_train, y_test):
    """Linear Regression model function"""
    # Add constant for statsmodels
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # Fit model
    model = sm.OLS(y_train, X_train_sm).fit()
    
    # Predict and evaluate
    y_pred = model.predict(X_test_sm)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, y_pred

def sklearn_linear_regression_model(X_train, X_test, y_train, y_test):
    """Scikit-learn Linear Regression model function"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2,mean_squared_error(y_test, y_pred),y_pred

def xgboost_model(X_train, X_test, y_train, y_test):
    """XGBoost model function"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2,mean_squared_error(y_test, y_pred),y_pred

def mlp_model(X_train, X_test, y_train, y_test):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, y_pred


def random_forest_model(X_train, X_test, y_train, y_test):
    """Random Forest model function"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2,mean_squared_error(y_test, y_pred), y_pred

def svr_model(X_train, X_test, y_train, y_test):
    """Support Vector Regression model function"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    return r2,mean_squared_error(y_test, y_pred), y_pred

def kernel_ridge_model(X_train, X_test, y_train, y_test):
    """Kernel Ridge Regression model function"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    return r2,mean_squared_error(y_test, y_pred), y_pred

def gaussian_process_model(X_train, X_test, y_train, y_test):
    """Gaussian Process Regression model function"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Limit number of samples for computational efficiency if needed
    max_samples = 500
    if len(X_train_scaled) > max_samples:
        indices = np.random.choice(len(X_train_scaled), max_samples, replace=False)
        X_train_scaled = X_train_scaled[indices]
        y_train = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
    
    # Define kernel
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    return r2,mean_squared_error(y_test, y_pred), y_pred

def compare_ml_methods(merged_data_na, target_var,ensemble_size = None, train_period = 1):
    """
    Compare different ML methods on the same financial data
    
    Parameters:
    merged_data_na: DataFrame with merged data
    target_var: String, name of target variable
    
    Returns:
    results_df: DataFrame with performance metrics for all models
    """
    print(f"Comparing ML methods for predicting {target_var}")
    print("=" * 50)
    
    # Define models to compare
    models = {

        "Linear Regression (statsmodels)": linear_regression_model,
        # "MLP": mlp_model,
        # "Linear Regression (sklearn)": sklearn_linear_regression_model,
        "XGBoost": xgboost_model,
        "Random Forest": random_forest_model,
        
        # "SVR": svr_model,
        
        # "Kernel Ridge": kernel_ridge_model,
        # "Gaussian Process": gaussian_process_model
    }
    
    # Run all models
    results = {}
    results_df = pd.DataFrame(columns=[
        'Model', 'Avg R-squared', 'Avg Execution Time (s)', 'Improvement over LR (%)'
    ])
    
    baseline_r2 = None
    
    for name, model_func in models.items():
        start_time = time.time()
        print(f"\nRunning {name}...")
        if ensemble_size:
            r2, avg_r2, avg_mse, avg_time = run_model_by_date_ensemble(merged_data_na, target_var, model_func, name, ensemble_size)
        else:
            r2, avg_r2, avg_mse, avg_time = run_model_by_date_concat_train(merged_data_na, target_var, model_func, name, train_period)
        results[name] = {
            'r2': avg_r2,
            'mse': avg_mse,
            'time': avg_time
        }
        
        # Set baseline for comparison (first linear regression model)
        if baseline_r2 is None and "Linear Regression" in name:
            baseline_r2 = avg_r2
        
        # Calculate improvement
        if baseline_r2 is not None:
            improvement = ((avg_r2 - baseline_r2) / abs(baseline_r2)) * 100
        else:
            improvement = 0
            
        # Add to results dataframe
        results_df = results_df._append({
            'Model': name,
            'Avg R-squared': avg_r2,
            'Avg MSE': avg_mse,
            'Avg Execution Time (s)': avg_time,
            'Improvement over LR (%)': improvement,
            "r2": r2
        }, ignore_index=True)
        end_time = time.time()
        print(f"{name} completed in {end_time - start_time:.4f} seconds")
    
    # Sort by R-squared
    results_df = results_df.sort_values('Avg R-squared', ascending=False)
    
    print("\n" + "=" * 50)
    print("Results Summary:")
    print(results_df)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Bar chart of R-squared values
    plt.subplot(2, 1, 1)
    bars = plt.barh(results_df['Model'], results_df['Avg R-squared'])
    plt.xlabel('Average R-squared')
    plt.title('Model Performance Comparison')
    plt.xlim(min(0, results_df['Avg R-squared'].min() - 0.05), 
             max(1, results_df['Avg R-squared'].max() + 0.05))
    
    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width >= 0 else width - 0.05
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                 va='center', ha='left' if width >= 0 else 'right')
    
    # Plot execution times (logarithmic scale)
    plt.subplot(2, 1, 2)
    bars = plt.barh(results_df['Model'], results_df['Avg Execution Time (s)'])
    plt.xlabel('Average Execution Time (seconds)')
    plt.title('Model Execution Time Comparison')
    plt.xscale('log')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.1, bar.get_y() + bar.get_height()/2, f'{width:.3f}s',
                 va='center')
    
    plt.tight_layout()
    plt.show()
    
    return results_df


def run_model_by_date_multi_day(dataset, model_func):
    r_square = []
    mse = []

    for i in range(len(dataset) - 1):
        # Extract and reshape inputs
        if i %10 == 0:
            print(f"Processing dataset {i} / {len(dataset) - 1}")
        X_train = dataset[i][:, :, 3:].reshape(dataset[i].shape[0], -1)
        y_train = dataset[i][:, -1, 2]
        X_test = dataset[i + 1][:, :, 3:].reshape(dataset[i + 1].shape[0], -1)
        y_test = dataset[i + 1][:, -1, 2]

        # Convert to DataFrame/Series for easy NaN handling
        X_train_df = pd.DataFrame(X_train)
        y_train_series = pd.Series(y_train)
        X_test_df = pd.DataFrame(X_test)
        y_test_series = pd.Series(y_test)

        # Drop rows with NaNs in train
        mask_train = X_train_df.notna().all(axis=1) & y_train_series.notna()
        X_train_clean = X_train_df[mask_train]
        y_train_clean = y_train_series[mask_train]

        # Drop rows with NaNs in test
        mask_test = X_test_df.notna().all(axis=1) & y_test_series.notna()
        X_test_clean = X_test_df[mask_test]
        y_test_clean = y_test_series[mask_test]

        # Skip if too few samples remain
        if len(X_train_clean) < 10 or len(X_test_clean) < 10:
            continue

        try:
            X_train_clean = X_train_clean.apply(pd.to_numeric, errors='coerce').astype(float)
            y_train_clean = y_train_clean.apply(pd.to_numeric, errors='coerce').astype(float)

            X_test_clean = X_test_clean.apply(pd.to_numeric, errors='coerce').astype(float)
            y_test_clean = y_test_clean.apply(pd.to_numeric, errors='coerce').astype(float)
            _, _, y_pred = model_func(X_train_clean, X_test_clean, y_train_clean, y_test_clean)
            r_square.append(r2_score(y_test_clean, y_pred))
            mse.append(mean_squared_error(y_test_clean, y_pred))
        except Exception as e:
            print(f"Error at step {i}: {e}")
            continue

    return r_square, mse, np.mean(r_square), np.mean(mse)

def compare_ml_methods_multi_day(dataset):
    models = {
        "Linear Regression (statsmodels)": linear_regression_model,
        "XGBoost": xgboost_model,
        "Random Forest": random_forest_model,
        "MLP": mlp_model,
        "SVR": svr_model,
       
        # Add others if desired
    }

    results_df = pd.DataFrame(columns=[
        'Model', 'Avg R-squared', 'Avg MSE', 'Avg Execution Time (s)', 'Improvement over LR (%)'
    ])

    baseline_r2 = None
    r2_lists = []
    mse_lists = []
    for name, model_func in models.items():
        print(f"\nRunning {name}...")
        r2_list, mse_list, r2_avg, mse_avg = run_model_by_date_multi_day(dataset, model_func)
        r2_lists.append(r2_list)
        mse_lists.append(mse_list)
        if baseline_r2 is None and "Linear Regression" in name:
            baseline_r2 = r2_avg

        improvement = ((r2_avg - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 is not None else 0

        results_df = results_df._append({
            'Model': name,
            'Avg R-squared': r2_avg,
            'Avg MSE': mse_avg,

            'Improvement over LR (%)': improvement
        }, ignore_index=True)

        print(f"{name} completed | Avg R2: {r2_avg:.4f} | Avg MSE: {mse_avg:.4f} ")

    results_df = results_df.sort_values('Avg R-squared', ascending=False)

    # Plotting
    plt.figure(figsize=(12, 10))

    # R-squared
    plt.subplot(2, 1, 1)
    bars = plt.barh(results_df['Model'], results_df['Avg R-squared'])
    plt.xlabel('Average R-squared')
    plt.title('Model Performance Comparison')

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', ha='left')

    # Execution Time
    plt.subplot(2, 1, 2)
    bars = plt.barh(results_df['Model'], results_df['Avg Execution Time (s)'])
    plt.xlabel('Average Execution Time (seconds)')
    plt.title('Execution Time (log scale)')
    plt.xscale('log')

    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', va='center')

    plt.tight_layout()
    plt.show()

    return results_df, r2_lists, mse_lists



# Example usage
if __name__ == "__main__":
    # This would be called from your main script
    # compare_ml_methods(merged_data_na, "B36m")
    pass





