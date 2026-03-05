import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer

def aggregate_duplicate_columns(df):
    """
    Aggregate columns with identical names by summing their values for each hadm_id,
    considering only columns whose names start with 'C' followed by a digit (CUI code).

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential duplicate column names.

    Returns:
    pd.DataFrame: The DataFrame with aggregated columns.
    """
    columns_starting_with_C = df.columns[df.columns.str.match(r'^C\d')]
    duplicate_columns = columns_starting_with_C[columns_starting_with_C.duplicated()].unique()
    aggregated_columns = {}
    for col in duplicate_columns:
        cols_to_aggregate = df.loc[:, df.columns == col]
        aggregated_col = cols_to_aggregate.sum(axis=1)
        aggregated_columns[col] = aggregated_col
    df = df.drop(columns=duplicate_columns)
    aggregated_df = pd.concat([df, pd.DataFrame(aggregated_columns)], axis=1)
    return aggregated_df

def one_hot_encode_non_numeric(df):
    """
    One-hot encode all columns containing non-numeric data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    non_numeric_columns = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=non_numeric_columns)
    return df_encoded

def harmonize_race(race):
    race = str(race).upper()
    if 'BLACK' in race or 'AFRICAN' in race:
        return 'Black/African American'
    elif 'WHITE' in race or 'PORTUGUESE' in race or 'RUSSIAN' in race:
        return 'White'
    elif 'HISPANIC' in race or 'LATINO' in race or 'SOUTH AMERICAN' in race or 'CENTRAL AMERICAN' in race or 'BRAZILIAN' in race:
        return 'Hispanic/Latino'
    elif 'ASIAN' in race or 'CHINESE' in race or 'KOREAN' in race or 'INDIAN' in race:
        return 'Asian'
    else:
        return 'Other/Unknown'

def is_numeric_column(series):
    """
    Check if a pandas Series can be converted to numeric values.

    Parameters:
    series (pd.Series or array-like): The input Series or array-like object.

    Returns:
    bool: True if the Series can be converted to numeric values, False otherwise.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    return pd.to_numeric(series, errors='coerce').notnull().all()

def process_numerical_columns(df, divisor_column=None):
    """
    Convert numerical columns to float and optionally divide them by a given column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    divisor_column (str, optional): The column name to divide by. Expected values are 'los' or 'admission_numbers'. Default is None.

    Returns:
    pd.DataFrame: The DataFrame with processed numerical columns.
    """
    expected_columns = ['los', 'admissions_number']
    exclude_columns = ['age'] + [col for col in df.columns if col.startswith(('race', 'gender', 'marital_status'))]

    if divisor_column is not None:
        if divisor_column not in expected_columns:
            raise ValueError(f"Divisor column '{divisor_column}' is not one of the expected columns: {expected_columns}.")
        if divisor_column not in df.columns:
            raise ValueError(f"Divisor column '{divisor_column}' not found in DataFrame.")
        divisor_series = df[divisor_column].copy() # Use .copy() to avoid SettingWithCopyWarning
        # Replace 0s in divisor_series with 1 to prevent division by zero errors / inf values.
        # NaNs should have been handled by impute_features_data prior to this function.
        divisor_series.replace(0, 1, inplace=True) 
    else:
        divisor_series = None

    for column in df.columns:
        if column in exclude_columns:
            #print(f"Skipping excluded column: {column}")
            continue
        if df[column].ndim != 1:
            #print(f"Skipping non 1-dimensional column: {column}")
            continue
        if is_numeric_column(df[column]):
            df[column] = df[column].astype(float)
            if divisor_series is not None:
                df[column] = df[column].div(divisor_series, axis=0)

    # Remove the divisor column and the other expected columns from the DataFrame
    df = df.drop(columns=[col for col in expected_columns if col in df.columns])

    return df

def impute_features_data(X, method='mean', imputer_to_use=None):
    """
    Impute missing values in the DataFrame X using the specified method or a pre-fitted imputer.

    Parameters:
    - X (DataFrame): The input features with potential missing values.
    - method (str): The imputation method ('mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative'). 
                  Used only if imputer_to_use is None.
    - imputer_to_use (sklearn.impute object, optional): A pre-fitted imputer. If provided, 'method' is ignored.

    Returns:
    - X_imputed (DataFrame): The DataFrame with imputed values.
    - fitted_imputer (sklearn.impute object or None): The imputer that was fitted, if a new one was created. 
                                                    None if imputer_to_use was provided.
    """
    if X.empty:
        print("Input DataFrame X to impute_features_data is empty. Returning empty DataFrame and None imputer.")
        return X.copy(), None

    # Summarize null values before imputation
    total_nulls_before = X.isnull().sum().sum()
    columns_with_nulls_before = (X.isnull().sum() > 0).sum()
    if total_nulls_before > 0:
        print(f"Total null values before imputation: {total_nulls_before}")
        print(f"Number of columns with null values before imputation: {columns_with_nulls_before}")
    else:
        print("No null values to impute.")
        # If imputer_to_use was passed but there are no nulls, still return it for consistency if it was just for transform step
        # If no imputer_to_use and no nulls, no new imputer is fitted.
        return X.copy(), imputer_to_use if imputer_to_use else None 

    # Create a copy to avoid modifying the original DataFrame passed to the function
    X_processed = X.copy()

    # Special handling for 'CHART' columns (fillna(0)) - applied before other imputers
    chart_columns = [col for col in X_processed.columns if col.startswith('CHART')]
    if chart_columns:
        X_processed[chart_columns] = X_processed[chart_columns].fillna(0)
        print(f"Filled NaNs with 0 for {len(chart_columns)} CHART columns.")

    fitted_imputer_return_val = None

    if imputer_to_use:
        print(f"Using pre-fitted imputer: {type(imputer_to_use).__name__}")
        # Ensure all columns X_processed has are expected by imputer_to_use if it has feature_names_in_
        if hasattr(imputer_to_use, 'feature_names_in_'):
            missing_in_X = set(imputer_to_use.feature_names_in_) - set(X_processed.columns)
            extra_in_X = set(X_processed.columns) - set(imputer_to_use.feature_names_in_)
            if missing_in_X:
                print(f"Warning: Columns expected by pre-fitted imputer are missing in X: {missing_in_X}. Adding them with NaNs.")
                for col in missing_in_X:
                    X_processed[col] = np.nan
            if extra_in_X:
                 print(f"Warning: Columns in X are not in pre-fitted imputer: {extra_in_X}. Dropping them before transform.")
                 X_processed = X_processed.drop(columns=list(extra_in_X))
            # Reorder X_processed to match imputer's feature_names_in_
            X_processed = X_processed[imputer_to_use.feature_names_in_]

        X_imputed_np = imputer_to_use.transform(X_processed)
        X_imputed = pd.DataFrame(X_imputed_np, columns=X_processed.columns, index=X_processed.index)
        fitted_imputer_return_val = None # We used a provided imputer, don't return it as if newly fitted
    else:
        print(f"Fitting new imputer with method: {method}")
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
        elif method == 'zero':
            imputer = SimpleImputer(strategy='constant', fill_value=0)
        elif method == 'knn':
            print("Using KNNImputer. This might take a while for large datasets...")
            imputer = KNNImputer(n_neighbors=7, weights='distance') # Default n_neighbors=5 or 7
        elif method == 'iterative':
            print("Using IterativeImputer. This might take a while...")
            imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='median', n_nearest_features=30 if X_processed.shape[1] > 30 else None)
        else:
            raise ValueError(f"Unsupported imputation method: {method}")
        
        # Fit and transform only on columns that still have NaNs after CHART handling
        cols_with_nans_after_chart = X_processed.columns[X_processed.isnull().any()].tolist()
        if not cols_with_nans_after_chart:
            print("No NaNs remaining after CHART column handling. Skipping main imputer fitting.")
            X_imputed = X_processed.copy()
            fitted_imputer_return_val = None # No imputer was actually fitted to other cols
        else:
            print(f"Applying {method} imputation to {len(cols_with_nans_after_chart)} columns with NaNs.")
            X_imputed_subset_np = imputer.fit_transform(X_processed[cols_with_nans_after_chart])
            X_imputed_subset_df = pd.DataFrame(X_imputed_subset_np, columns=cols_with_nans_after_chart, index=X_processed.index)
            
            # Combine with columns that had no NaNs (or were only CHART columns)
            X_imputed = X_processed.copy() # Start fresh from X_processed (which has CHART imputed)
            X_imputed[cols_with_nans_after_chart] = X_imputed_subset_df
            fitted_imputer_return_val = imputer # This is the newly fitted imputer

    # Explicitly replace any inf values with NaN that might arise from imputers like KNN
    X_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check if replacing inf created new NaNs that need imputation
    current_nulls = X_imputed.isnull().sum()
    columns_needing_reimputation = current_nulls[current_nulls > 0].index.tolist()

    if columns_needing_reimputation:
        print(f"Re-imputing NaNs in {len(columns_needing_reimputation)} columns that resulted from inf replacement or failed imputation (using fill_value=0).")
        # This final pass ensures no NaNs are left, using a simple constant imputer.
        final_pass_imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_imputed[columns_needing_reimputation] = final_pass_imputer.fit_transform(X_imputed[columns_needing_reimputation])

    # Summarize null values after all imputation steps
    total_nulls_after_final = X_imputed.isnull().sum().sum()
    columns_with_nulls_after = (X_imputed.isnull().sum() > 0).sum()
    if total_nulls_after_final > 0:
        print(f"Warning: Total null values after final imputation pass: {total_nulls_after_final}")
        print(f"Warning: Number of columns with null values after final imputation pass: {columns_with_nulls_after}")
    else:
        print("All null values successfully imputed.")

    return X_imputed, fitted_imputer_return_val
