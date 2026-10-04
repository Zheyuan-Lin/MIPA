import pandas as pd
import numpy as np
import datetime as dt
from scipy.sparse import coo_matrix
from scripts.utils import *
from scripts.featureselection.medcat_parser import *

def add_count_column(df, column_to_count, id_column):
    """
    Adds a count column to the DataFrame for each group by the specified column and ID.

    Parameters:
    - df: The DataFrame to modify.
    - column_to_count: The name of the column to count.
    - id_column: The name of the ID column to group by.

    Returns:
    - The modified DataFrame with an additional count column.
    """
    # Create a dummy column filled with 1s for counting purposes
    df['dummy'] = 1

    # Group by the specified column and ID, then count the occurrences within each group
    count_series = df.groupby([column_to_count, id_column])['dummy'].transform('sum')

    # Add the count as a new column to the DataFrame
    df['count'] = count_series

    # Drop the dummy column as it's no longer needed
    df.drop('dummy', axis=1, inplace=True)

    return df


### PIVOT FUNCTIONS ###
def pivot_cohort(df: pd.DataFrame, df_type, index_col: str, target_col: str, values='count', max_features=None):
    """
    Pivots long_format data into a DataFrame with specified index and target columns.

    Parameters:
    - df: The DataFrame containing the data.
    - index_col: The column name to use as the DataFrame index.
    - target_col: The column name to pivot on.
    - values: The column name for the values to aggregate.
    - max_features: The maximum number of features to include in the pivoted DataFrame.

    Returns:
    - The pivoted DataFrame with the index_col as the first column and target_col values as other columns.

    Explanation of Aggregation Function:
    - The aggregation function used in the pivot table is determined by the encoding method.
    """
    df = df.dropna(subset=[target_col]) # safety check

    # Add count
    df = add_count_column(df, target_col, index_col)

    # Time column
    # df_count = df_count.rename(columns={'start_hours_from_admit': 'timedelta'})

    # Pivot
    if max_features:
        top_features = df[target_col].value_counts().index[:max_features]
        df = df[df[target_col].isin(top_features)] #filter out the top max_features

    # Build the pivot directly as a pandas Sparse[int64]/Sparse[float64] DataFrame via a
    # scipy COO->CSR matrix, instead of df.pivot_table(..., fill_value=0). pivot_table builds
    # a dense (rows x columns) array before anything else happens -- for a high-cardinality
    # column like DIAG's icd_code (thousands of distinct codes) across the full admission
    # population, that's tens of billions of cells (pandas itself warns about this -- e.g.
    # "may generate 6252133755 cells" was observed for one phenotype's DIAG table), which
    # OOMs well before it can ever be saved. The actual data is highly sparse (each admission
    # only has a handful of the possible codes), so building straight from the sparse (row,
    # col, value) triplets avoids ever materializing that dense array at all. Confirmed
    # equivalent to the old aggfunc='max' pivot_table: 'count' (from add_count_column above)
    # is already the per-(index_col, target_col) aggregated value, broadcast to every row in
    # that group, so any one row per (index_col, target_col) pair carries the correct value --
    # drop_duplicates before building the matrix reproduces aggfunc='max' exactly without
    # needing a true duplicate-combining aggregation.
    df_dedup = df.drop_duplicates(subset=[index_col, target_col])
    row_cat = pd.Categorical(df_dedup[index_col])
    col_cat = pd.Categorical(df_dedup[target_col])
    coo = coo_matrix(
        (df_dedup[values].values, (row_cat.codes, col_cat.codes)),
        shape=(len(row_cat.categories), len(col_cat.categories)),
    ).tocsr()
    pivot_df = pd.DataFrame.sparse.from_spmatrix(coo, index=row_cat.categories, columns=col_cat.categories)
    pivot_df.index.name = index_col

    # Rename features
    pivot_df.columns = [df_type + '_' + str(col) for col in pivot_df.columns]

    # Reset the index to make index_col a column in the DataFrame
    pivot_df.reset_index(inplace=True)

    return pivot_df

def pivot_cohort_timeseries(df: pd.DataFrame, index_col: str, target_col: str, timedelta_col: str, values='presence',
                            ohe=True, agg_window='D'):
    """
    Converts drug administration data into a time series DataFrame.

    Args:
    df (DataFrame): Input data.
    index_col (str): Identifier column (e.g., 'hadm_id').
    target_col (str): Column to pivot on (e.g., 'ndc').
    timedelta_col (str): Time data column (e.g., 'start_hours_from_admit').
    values (str): Type of values to aggregate, defaults to 'presence'.
    ohe (bool): Whether to perform one-hot encoding.
    agg_window (str): Aggregation window ('D' for daily).

    Returns:
    DataFrame: Pivoted DataFrame with time series data.
    """
    #TODO: verify if this function is working correctly + add feature renaming
    # 'max' (string) instead of the np.max callable -- see pivot_cohort's comment above.
    aggfunc = 'max'
    df = df.dropna(subset=[target_col, timedelta_col])  # Drop rows with NaN in crucial columns

    # Convert timedelta column to timedelta if it's in string format
    if df[timedelta_col].dtype == 'object':
        df[timedelta_col] = pd.to_timedelta(df[timedelta_col])

    # One-hot encoding if specified
    if ohe:
        dummies = pd.get_dummies(df[target_col], prefix='', prefix_sep='')
        df = pd.concat([df, dummies], axis=1)
        values = dummies.columns.tolist()  # Columns to aggregate

    # Set the timedelta_col as the sole index for resampling
    df.set_index([index_col, timedelta_col], inplace=True)

    # Resample by the specified aggregation window
    df = df.resample(agg_window, level=1).sum()  # Resample on the level of timedelta_col

    # Reset index to bring index_col back as a column
    df.reset_index(inplace=True)

    # Ensure index_col is not duplicated by checking before setting it as index again
    if index_col in df.columns:
        df.set_index([index_col, timedelta_col], inplace=True)
    else:
        df[index_col] = df.index.get_level_values(0)
        df.set_index([index_col, timedelta_col], inplace=True)

    # Pivot the DataFrame if needed
    pivot_df = df.pivot_table(index=[index_col, timedelta_col], values=values, aggfunc=aggfunc)

    return pivot_df


def generate_pivot(phenotype, pivot_type='simple', train=True, noisy_label=True,
                   notes=False, NLP = False, agg_window='D', debug=None, use_shared_paths=False):
    """
    Generate pivoted data and corresponding dictionaries for various data types.

    Parameters:
    - phenotype: The phenotype to process. Used for phenotype-specific paths unless use_shared_paths is True.
    - pivot_type: Type of pivot to apply ('simple' or 'timeseries').
    - train: A boolean indicating whether the data is for training.
    - noisy_label: A boolean indicating whether the data includes silver labels.
    - notes: A boolean indicating whether to include notes data types.
    - NLP: A boolean indicating whether to process notes with NLP techniques.
    - agg_window: Aggregation window for timeseries pivot.
    - debug: If specified, only process this data type.
    - use_shared_paths: If True and train is False, use shared paths for test/validation data.
    """

    # This check might be redundant if called from mimic_parser_2.py where phenotype could be None for shared paths
    # if phenotype not in PHENOTYPES and not (not train and use_shared_paths):
    #     raise ValueError(f"Phenotype '{phenotype}' is not recognized. Valid phenotypes are: {PHENOTYPES}")

    print("[GENERATING PIVOTED DATA AND DICTIONARIES]")

    df_types = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
    if notes:
        df_types.extend(['RADNOTES', 'DSNOTES'])
    if debug is not None:
        df_types = [dt for dt in df_types if dt == debug] # Corrected debug type check

    current_phenotype_for_paths = None if use_shared_paths else phenotype
    current_noisy_label_for_paths = False if use_shared_paths else noisy_label
    # For dictionary path generation (non-NLP), the 'label' parameter corresponds to noisy_label
    current_label_for_dict_path = current_noisy_label_for_paths

    target_columns = {
        'DIAG': 'icd_code',
        'MEDS': 'ndc',
        'PROC': 'icd_code',
        'CHART': 'itemid',
        'LABS': 'itemid'
        # NLP notes (RADNOTES, DSNOTES) use 'nlp_codes' which is handled differently below
    }

    target_labels = {
            'DIAG'    : 'long_title',
            'MEDS'    : 'nonproprietaryname',
            'PROC'    : 'long_title',
            'CHART'   : 'label',
            'LABS'    : 'label'
    }

    for df_type in df_types:
        # Determine input path for pivot generation
        if use_shared_paths:
            if df_type in ['RADNOTES', 'DSNOTES'] and NLP:
                # For shared NLP pivot, read the output of NLP feature selection
                input_path = generate_nlp_file_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, use_shared_paths)
            else:
                input_path = generate_file_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, long=True, use_shared_paths=use_shared_paths)
        else: # Training paths or non-shared test/val (if any)
            if df_type in ['RADNOTES', 'DSNOTES'] and NLP:
                input_path = generate_nlp_file_path(phenotype, df_type, train, noisy_label, use_shared_paths=False)
            else:
                input_path = generate_file_path(phenotype, df_type, train, noisy_label, long=True, use_shared_paths=False)

        if not path_exists(input_path):
            print(f"[SKIPPING PIVOT FOR {df_type.upper()}] File does not exist: {input_path}")
            continue

        try:
            df = read_gzip_or_parts(input_path)
            if df.empty:
                print(f"[SKIPPING PIVOT FOR {df_type.upper()}] Loaded dataframe is empty from {input_path}")
                continue

            # Determine target column for pivoting (not applicable for NLP notes here, handled by pivot_and_aggregate)
            target_col = target_columns.get(df_type)

            pivoted_df = pd.DataFrame() # Initialize to ensure it has a value

            if pivot_type == 'simple':
                if df_type in ['DSNOTES', 'RADNOTES']:
                    if NLP:
                        # Assumes df read from input_path (generate_nlp_file_path) has 'hadm_id' and 'nlp_codes'
                        pivoted_df = pivot_and_aggregate(df, id_column='hadm_id', cui_column='nlp_codes')
                    else:
                        print(f"[SKIPPING PIVOT FOR {df_type.upper()}] NLP is False, no simple pivot defined for non-NLP notes.")
                        continue
                elif target_col:
                    pivoted_df = pivot_cohort(df, df_type=df_type, index_col='hadm_id', target_col=target_col, max_features=None)
                else:
                    print(f"[SKIPPING PIVOT FOR {df_type.upper()}] Target column not defined for simple pivot.")
                    continue

            elif pivot_type == 'timeseries':
                if df_type in ['DIAG']:
                    if target_col: # Should always be true for DIAG
                        pivoted_df = pivot_cohort(df, df_type=df_type, index_col='hadm_id', target_col=target_col, max_features=None)
                    else:
                        print(f"[SKIPPING PIVOT FOR {df_type.upper()}] Target column not defined for timeseries DIAG pivot.")
                        continue
                elif df_type in ['DSNOTES', 'RADNOTES']:
                    if NLP:
                        pivoted_df = pivot_and_aggregate(df, id_column='hadm_id', cui_column='nlp_codes')
                    else:
                        print(f"[SKIPPING PIVOT FOR {df_type.upper()}] NLP is False, no timeseries pivot defined for non-NLP notes.")
                        continue
                elif target_col: # For MEDS, LABS, CHART etc.
                    # Ensure 'start_hours_from_admit' column exists for timeseries pivot
                    if 'start_hours_from_admit' not in df.columns:
                        print(f"[SKIPPING TIMESERIES PIVOT FOR {df_type.upper()}] Missing 'start_hours_from_admit' column.")
                        continue
                    pivoted_df = pivot_cohort_timeseries(df, 'hadm_id', target_col, 'start_hours_from_admit', agg_window=agg_window)
                else:
                    print(f"[SKIPPING PIVOT FOR {df_type.upper()}] Target column not defined for timeseries pivot.")
                    continue
            else:
                print(f"[SKIPPING PIVOT FOR {df_type.upper()}] Unknown pivot_type: {pivot_type}")
                continue

            if pivoted_df.empty:
                print(f"[SKIPPING SAVE FOR {df_type.upper()}] Pivoted dataframe is empty.")
                continue

            save_output_path = generate_pivot_path(current_phenotype_for_paths, df_type, train, current_noisy_label_for_paths, use_shared_paths)
            actual_save_path = save_pivot(pivoted_df, save_output_path)
            print(f"[SUCCESSFULLY SAVED {df_type.upper()} PIVOTED DATA] {actual_save_path}")

            # Generate and save dictionary (only for non-NLP data types, NLP dicts are done in feature_selection)
            if df_type not in ['DSNOTES', 'RADNOTES'] and target_col:
                target_label_name = target_labels.get(df_type)
                if target_label_name and target_label_name in df.columns and target_col in df.columns:
                    df_clean = df.dropna(subset=[target_col, target_label_name])
                    if not df_clean.empty:
                        unique_pairs = df_clean.drop_duplicates(subset=[target_col, target_label_name])
                        dict_df = unique_pairs[[target_col, target_label_name]].reset_index(drop=True)
                        dict_df[target_col] = df_type + '_' + dict_df[target_col].astype(str)
                        dict_output_path = generate_dict_path(current_phenotype_for_paths, df_type=df_type,
                                                          train=train, label=current_label_for_dict_path,
                                                          use_shared_paths=use_shared_paths)
                        save_csv(dict_df, dict_output_path)
                        print(f"[SUCCESSFULLY SAVED {df_type.upper()} DICTIONARY] {dict_output_path}")
                    else:
                        print(f"[SKIPPING DICTIONARY FOR {df_type.upper()}] No data after dropping NaNs for {target_col} and {target_label_name}.")
                else:
                    print(f"[SKIPPING DICTIONARY FOR {df_type.upper()}] Target label '{target_label_name}' or target column '{target_col}' not found in DataFrame columns.")
            elif df_type not in ['DSNOTES', 'RADNOTES']:
                 print(f"[SKIPPING DICTIONARY FOR {df_type.upper()}] Target column not defined.")

        except ValueError as ve:
            print(f"[ERROR PROCESSING PIVOT FOR {df_type.upper()}] ValueError: {ve}")
            continue
        except Exception as e:
            print(f"[ERROR PROCESSING PIVOT FOR {df_type.upper()} DATA] {e}")
            continue

    print("[PIVOTED DATA AND DICTIONARY GENERATION COMPLETE]")






