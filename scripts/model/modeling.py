from scripts.constants import *
from scripts.utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, f1_score
import numpy as np
import re
import os

from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV # Added for ElasticNetCV


from scripts.model.evaluate import evaluate_all, evaluate_classification_model
from scripts.model.feature_utils import aggregate_duplicate_columns, one_hot_encode_non_numeric, process_numerical_columns, impute_features_data
from scripts.utils import load_baseline
import scripts.constants as k_const
from scripts.model.reporting import generate_performance_report

### this file contains functions for training models - semi-supervised and supervised
def to_snake_case(name):
    """Convert phenotype name to lower snake_case for folder consistency. Collapse multiple underscores."""
    # Handle CamelCase boundaries first
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    # Replace spaces and hyphens with underscores
    name = name.replace(' ', '_').replace('-', '_')
    # Collapse any repeated underscores that may have been introduced (e.g. "Alcohol Abuse" -> "alcohol__abuse")
    name = re.sub('_+', '_', name)
    return name.lower()


def load_and_filter_train(phenotype, notes=True, nlp=True, train=True, noisy_label=True, corrector_column='los', method='zero'):
    """
    Load all pivot tables and filter them based on hadm_id.
    Prioritizes keeping all hadm_ids from the initial y_train cohort.

    Parameters:
    - phenotype (str): The phenotype to load.
    - notes (bool): Whether to include notes data. Default is True.
    - nlp (bool): Whether to include NLP data. Default is True.
    - train (bool): Whether to load training data. Default is True.
    - noisy_label (bool): Whether to include noisy labels. Default is True.
    - corrector_column (str): The column name to use for normalization. Expected values are 'los' or 'admission_numbers'. Default is 'los'.
    - method (str): The imputation method ('mean', 'median', 'most_frequent', 'constant', 'knn', 'zero'). Default is 'zero'.

    Returns:
    - y (pd.DataFrame): The target variable with hadm_id as index.
    - combined_features (pd.DataFrame): The combined features with hadm_id as index.
    """
    # Loading baseline: y_train, baseline_features_train, corrector_train
    # y_train.index will serve as our master list of hadm_ids
    y_train, baseline_features_train, corrector_train = load_baseline(phenotype, train, noisy_label, use_shared_paths=False)
    master_hadm_ids = y_train.index
    print(f"Master list: {len(master_hadm_ids)} hadm_ids from y_train.index for phenotype {phenotype}")

    # Filter baseline_features and corrector_train to master_hadm_ids
    # This is implicitly handled if load_baseline already ensures y, features, and corrector are aligned.
    # If load_baseline could return more baseline_features_train than in y_train, explicit reindex/filter here would be good.
    # Assuming load_baseline's inner joins mean y_train.index, baseline_features_train.index, and corrector_train.index are already aligned.
    
    # Initialize a list to store dataframes for merging, starting with baseline features
    all_feature_dfs = [baseline_features_train[baseline_features_train.index.isin(master_hadm_ids)]]

    # Loading pivot tables
    df_types = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
    if notes and nlp:
        df_types.extend(['RADNOTES', 'DSNOTES'])

    for df_type in df_types:
        path = generate_pivot_path(phenotype, df_type, train, noisy_label, use_shared_paths=False)
        if not path_exists(path):
            print(f"[SKIPPING {df_type.upper()} DATA] File does not exist: {path}")
            continue
        df = read_gzip_or_parts(path)
        print(f"Loaded {df_type} data, shape: {df.shape}")

        if 'hadm_id' not in df.columns:
            print(f"Warning: 'hadm_id' not found in {df_type} data columns. Skipping this df_type.")
            continue
        df.set_index('hadm_id', inplace=True)
        all_feature_dfs.append(df)

    print(f"[ALL FEATURES DATA FOR PHENOTYPE {phenotype} LOADED FOR POTENTIAL MERGE]")

    # Concatenate all feature dataframes horizontally using an outer join
    # This brings together all columns for all hadm_ids present in any table
    if not all_feature_dfs:
        print(f"No feature dataframes (not even baseline) to concatenate for {phenotype}. Returning empty structures.")
        # Ensure y_train is also filtered to potentially empty master_hadm_ids, though y_train.index *is* master_hadm_ids
        return y_train, pd.DataFrame(index=master_hadm_ids)

    temp_combined_features = pd.concat(all_feature_dfs, axis=1, join='outer')
    
    # Reindex to ensure all master_hadm_ids are present and in the correct order;
    # drops hadm_ids not in master_hadm_ids, adds rows of NaNs for master_hadm_ids not in temp_combined_features (if any).
    combined_features = temp_combined_features.reindex(master_hadm_ids)
    print(f"Shape of combined_features after outer concat and reindex to master_hadm_ids: {combined_features.shape}")

    # Select 'los' and 'admissions_number' from corrector, convert to float.
    # Ensure corrector_train is also aligned with master_hadm_ids.
    # load_baseline should ensure corrector_train.index aligns with y_train.index (master_hadm_ids)
    los_admissions = corrector_train.loc[corrector_train.index.isin(master_hadm_ids), ['los', 'admissions_number']].astype(float)

    # Harmonize features based on LOS from corrector
    combined_features = aggregate_duplicate_columns(combined_features)

    # One-hot encode non-numeric columns
    combined_features = one_hot_encode_non_numeric(combined_features)

    # Now, concatenate los_admissions with combined_features
    # Reindex los_admissions to combined_features.index to ensure alignment before concat
    combined_features = pd.concat([combined_features, los_admissions.reindex(combined_features.index)], axis=1, join='outer')


    # Impute missing values in features
    combined_features, fitted_imputer = impute_features_data(combined_features, method=method)
    combined_features = process_numerical_columns(combined_features, corrector_column)

    # Remove ignored features
    features_to_drop_actually_present = [col for col in k_const.FEATURES_TO_IGNORE if col in combined_features.columns]
    if features_to_drop_actually_present:
        combined_features = combined_features.drop(columns=features_to_drop_actually_present)
        print(f"Removed {len(features_to_drop_actually_present)} ignored features: {features_to_drop_actually_present}")

    # y_train is already indexed by master_hadm_ids and aligned.
    # Ensure combined_features and y_train have perfectly aligned indices before returning.
    common_ids = y_train.index.intersection(combined_features.index)
    y_train_final = y_train.loc[common_ids]
    combined_features_final = combined_features.loc[common_ids]
    
    print(f"Final shape of y_train: {y_train_final.shape}, combined_features: {combined_features_final.shape}")

    # Return the target variable, combined features : [both have hadm_id as index]
    # AND the fitted imputer if one was created
    return y_train_final, combined_features_final, fitted_imputer

def load_and_filter_val(phenotype, train_features, notes=True, nlp=True, train=False, noisy_label=True, corrector_column='los', method='zero', fitted_imputer_obj=None):
    """
    Load all pivot tables and filter them based on hadm_id for validation data from shared paths.
    Ensure that all the features in the validation set are the same as the training set.
    Y and hadm_id selection for validation set come from a phenotype-specific index file.
    Prioritizes keeping all hadm_ids from the index file.
    """
    # 1. Load hadm_ids and y for the validation set from the phenotype-specific index file
    index_file_path = os.path.join(k_const.INDEXES_PATH, f"{phenotype}_indexes.csv")
    if not path_exists(index_file_path):
        print(f"[SKIPPING VALIDATION DATA] Index file does not exist: {index_file_path}")
        empty_y = pd.DataFrame(columns=['y'], index=pd.Index([], name='hadm_id'))
        empty_X = pd.DataFrame(columns=train_features, index=pd.Index([], name='hadm_id'))
        return empty_y, empty_X

    index_df = load_csv(index_file_path)
    id_col_name = 'hadm_id'
    set_col_name = 'set'

    if id_col_name not in index_df.columns:
        raise ValueError(f"Index file {index_file_path} must contain an '{id_col_name}' column.")
    if set_col_name not in index_df.columns:
        alternatives = ['split', 'dataset_type', 'fold_id', 'partition']
        for alt in alternatives:
            if alt in index_df.columns:
                print(f"Warning: '{set_col_name}' column not found in index file {index_file_path}, using '{alt}' column instead.")
                set_col_name = alt
                break
        if set_col_name == 'set':
             raise ValueError(f"Index file {index_file_path} must contain a '{set_col_name}' column (or a recognized alternative) to identify validation records.")
    if 'y' not in index_df.columns:
        raise ValueError(f"Index file {index_file_path} must contain a 'y' column for the target variable.")

    val_index_df = index_df[index_df[set_col_name].astype(str).str.lower() == 'validation']
    if not val_index_df.empty and val_index_df[id_col_name].duplicated().any():
        print(f"Warning: Duplicate '{id_col_name}' entries found in 'validation' set of {index_file_path}. Keeping first occurrence.")
        val_index_df = val_index_df.drop_duplicates(subset=[id_col_name], keep='first')
        
    if val_index_df.empty:
        print(f"No 'validation' entries found in '{set_col_name}' column of {index_file_path} for phenotype {phenotype}.")
        empty_y = pd.DataFrame(columns=['y'], index=pd.Index([], name=id_col_name))
        empty_X = pd.DataFrame(columns=train_features, index=pd.Index([], name=id_col_name))
        return empty_y, empty_X

    indexed_hadm_ids = pd.Index(val_index_df[id_col_name].unique(), name=id_col_name)
    y_val_initial = val_index_df[[id_col_name, 'y']].set_index(id_col_name).reindex(indexed_hadm_ids)
    print(f"Initial set from index: {len(indexed_hadm_ids)} validation hadm_ids for phenotype {phenotype}.")

    # 2. Load shared baseline features and corrector
    try:
        features_from_shared_baseline, corrector_from_shared_baseline = load_baseline(
            phenotype, train=False, noisy_label=noisy_label, use_shared_paths=True
        )
    except ValueError as e:
        print(f"Error loading shared baseline for {phenotype}: {e}. Check file existence and format.")
        # Return y_val (indexed by initial list) and empty X, as we can't form core_hadm_ids
        empty_X = pd.DataFrame(columns=train_features, index=indexed_hadm_ids)
        return y_val_initial, empty_X

    # 3. Core hadm_id Set Definition
    core_hadm_ids = indexed_hadm_ids.intersection(features_from_shared_baseline.index).intersection(corrector_from_shared_baseline.index)
    if core_hadm_ids.empty:
        print(f"No common hadm_ids after intersecting index file with baseline and corrector for {phenotype}. Returning empty DataFrames.")
        empty_y = pd.DataFrame(columns=['y'], index=pd.Index([], name=id_col_name))
        empty_X = pd.DataFrame(columns=train_features, index=pd.Index([], name=id_col_name))
        return empty_y, empty_X

    y_val = y_val_initial.reindex(core_hadm_ids) # Filter y_val to core_hadm_ids
    print(f"Core set: {len(core_hadm_ids)} validation hadm_ids after ensuring presence in baseline/corrector.")
    print(f"Shape of y_val after filtering to core_hadm_ids: {y_val.shape}")

    # 4. Prepare Baseline for Merging (reindexed to core_hadm_ids)
    baseline_features_core = features_from_shared_baseline.reindex(core_hadm_ids)
    corrector_val_core = corrector_from_shared_baseline.reindex(core_hadm_ids)

    # 5. & 6. Load All Other Pivot Tables and Initialize list for DataFrames
    all_feature_dfs = [baseline_features_core] # Start with baseline features for core IDs
    
    df_types = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
    if notes and nlp:
        df_types.extend(['RADNOTES', 'DSNOTES'])

    for df_type in df_types:
        path = generate_pivot_path(phenotype, df_type, train=False, noisy_label=noisy_label, use_shared_paths=True)
        if not path_exists(path):
            print(f"[SKIPPING SHARED {df_type.upper()} DATA] File does not exist: {path}")
            continue
        df = read_gzip_or_parts(path)
        print(f"Loaded shared {df_type} data, shape: {df.shape}")
        if id_col_name not in df.columns:
            print(f"Warning: '{id_col_name}' not found in {df_type} data columns. Skipping this df_type.")
            continue
        df.set_index(id_col_name, inplace=True)
        all_feature_dfs.append(df) # Add to list for outer concatenation (not yet filtered by core_hadm_ids)

    # 7. Concatenate ALL Feature Tables using an outer join
    if not all_feature_dfs: # Should at least have baseline_features_core
        print(f"No feature dataframes to merge for phenotype {phenotype}. Returning empty X.")
        return y_val, pd.DataFrame(columns=train_features, index=core_hadm_ids)

    temp_combined_features = pd.concat(all_feature_dfs, axis=1, join='outer')
    
    # 8. Final Reindex to core_hadm_ids
    combined_val_features = temp_combined_features.reindex(core_hadm_ids)
    print(f"Shape of combined_val_features after outer concat and final reindex to core_hadm_ids: {combined_val_features.shape}")

    # --- Start: Sparsity Filters for Validation Set ---
    # 1. Row Sparsity Filter: Drop HADM_IDs with too many missing values
    # Only apply this to validation set to keep it dense for feature selection
    row_missing_threshold = 0.30 # Drop row if more than 30% of its values are NaN
    null_frac_per_row = combined_val_features.isnull().mean(axis=1)
    ids_to_keep_after_row_filter = null_frac_per_row[null_frac_per_row <= row_missing_threshold].index
    
    if len(ids_to_keep_after_row_filter) < len(combined_val_features):
        print(f"Validation Set: Dropped {len(combined_val_features) - len(ids_to_keep_after_row_filter)} HADM_IDs due to >{row_missing_threshold*100}% missing features.")
        combined_val_features = combined_val_features.loc[ids_to_keep_after_row_filter]
        y_val = y_val.loc[ids_to_keep_after_row_filter] # Filter y_val accordingly
        print(f"Shape of combined_val_features after row sparsity filter: {combined_val_features.shape}")
        print(f"Shape of y_val after row sparsity filter: {y_val.shape}")
    else:
        print(f"Validation Set: No HADM_IDs dropped by row sparsity filter (threshold >{row_missing_threshold*100}% missing).")

    if combined_val_features.empty:
        print("Validation set became empty after row sparsity filter. Returning empty DataFrames.")
        empty_y = pd.DataFrame(columns=['y'], index=pd.Index([], name=id_col_name))
        empty_X = pd.DataFrame(columns=train_features, index=pd.Index([], name=id_col_name))
        return empty_y, empty_X

    # 2. Column Sparsity Filter: Drop columns with too many missing values
    col_min_observed_threshold = 0.20 # Keep column if at least 20% of its values are non-NaN (was 0.05)
    observed_frac_per_col = combined_val_features.notnull().mean()
    cols_to_keep_after_col_filter = observed_frac_per_col[observed_frac_per_col >= col_min_observed_threshold].index
    
    if len(cols_to_keep_after_col_filter) < len(combined_val_features.columns):
        print(f"Validation Set: Dropped {len(combined_val_features.columns) - len(cols_to_keep_after_col_filter)} columns due to <{col_min_observed_threshold*100}% observed values.")
        combined_val_features = combined_val_features[cols_to_keep_after_col_filter]
        print(f"Shape of combined_val_features after column sparsity filter: {combined_val_features.shape}")
    else:
        print(f"Validation Set: No columns dropped by column sparsity filter (threshold >={col_min_observed_threshold*100}% observed).")
    # --- End: Sparsity Filters for Validation Set ---

    # 9. Perform processing steps:
    combined_val_features = aggregate_duplicate_columns(combined_val_features)
    combined_val_features = one_hot_encode_non_numeric(combined_val_features)

    missing_cols = [col for col in train_features if col not in combined_val_features.columns]
    for col in missing_cols:
        combined_val_features[col] = 0 
    combined_val_features = combined_val_features.reindex(columns=train_features)
    print(f"Shape of combined_val_features after OHE and aligning with train_features: {combined_val_features.shape}")

    # Impute missing values - use fitted_imputer_obj if provided
    # Pass combined_val_features directly to imputation, as 'los' and 'admissions_number'
    # might not be part of the imputer's vocabulary and should be added post-imputation.
    combined_features_imputed, _ = impute_features_data(combined_val_features, method=method, imputer_to_use=fitted_imputer_obj) # Pass imputer_to_use
    
    # Prepare 'los' and 'admissions_number' for normalization from corrector_val_core
    # These columns are added *after* imputation to ensure they are present for process_numerical_columns
    # and are not dropped by an imputer that wasn't trained on them.
    los_admissions_for_norm = corrector_val_core[['los', 'admissions_number']].astype(float).reindex(combined_features_imputed.index)

    if 'los' not in combined_features_imputed.columns and 'los' in los_admissions_for_norm.columns:
        combined_features_imputed['los'] = los_admissions_for_norm['los']
    if 'admissions_number' not in combined_features_imputed.columns and 'admissions_number' in los_admissions_for_norm.columns:
        combined_features_imputed['admissions_number'] = los_admissions_for_norm['admissions_number']
    
    combined_features_processed = process_numerical_columns(combined_features_imputed, corrector_column)
    final_val_features = combined_features_processed.reindex(columns=train_features, fill_value=0)

    # Final alignment of y_val and features by index
    # y_val is already aligned to core_hadm_ids. final_val_features should also be.
    common_final_ids = core_hadm_ids.intersection(final_val_features.index).sort_values()

    final_val_features = final_val_features.loc[common_final_ids]
    final_y_val = y_val.loc[common_final_ids] # y_val was already on core_hadm_ids, further filtering if final_val_features dropped some

    print(f"Final shape of y_val: {final_y_val.shape}, final_val_features: {final_val_features.shape}")
    if not final_val_features.index.equals(final_y_val.index):
        print("Warning: Index mismatch between final features and y_val despite efforts. Further alignment check needed if this occurs.")

    if final_val_features.empty and not final_y_val.empty :
        print(f"Warning: final_val_features is empty but final_y_val is not for {phenotype}. Check data pipeline.")
    elif not final_val_features.empty and final_y_val.empty:
         print(f"Warning: final_val_features is not empty but final_y_val is for {phenotype}. Check data pipeline.")

    return final_y_val, final_val_features


def train_model(X, y, params=None):
    """
    Train a logistic regression model with L1 regularization and hyperparameter tuning using 10-fold cross-validation.

    Parameters:
    - X (DataFrame): The input features.
    - y (Series): The target variable.
    - params (dict, optional): Dictionary containing model parameters. Expected keys:
        - 'C' (list of float): Inverse of regularization strength. Smaller values specify stronger regularization.
          Typical values: [0.001, 0.01, 0.1, 1, 10, 100]
        - 'max_iter' (list of int): Maximum number of iterations taken for the solvers to converge.
          Typical values: [100, 200, 500, 1000, 5000]

    Returns:
    - best_model: The best model after tuning.
    - best_score: The best score from the cross-validation (log loss).
    - non_zero_features (list): List of non-zero features.

    Example:
    params = {
        'C': [0.1, 1, 10],
        'max_iter': [100, 200, 500]
    }
    best_model, best_score, non_zero_features = train_model(X, y, params)
    """

    print("Setting up the Logistic Regression model with L1 regularization.")

    # Default parameter grid
    default_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 200, 500, 1000, 5000]
    }

    # Use provided params or default params
    param_grid = params if params is not None else default_param_grid

    model = LogisticRegression(penalty='l1', solver='liblinear')

    # Create a custom cross-validation iterator with tqdm
    cv = StratifiedKFold(n_splits=10)

    try:
        # Wrap the cross-validation process with tqdm
        with tqdm(total=cv.get_n_splits() * len(param_grid['C']) * len(param_grid['max_iter']), desc="GridSearchCV") as pbar:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1', verbose=0)
            grid_search.fit(X, y.values.ravel())
            pbar.update(1)

        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_  # F1 score (no need to negate)

        # Evaluate the best model on the training data
        predictions = best_model.predict(X)
        accuracy = accuracy_score(y, predictions)

        # Extract non-zero features
        non_zero_features = X.columns[best_model.coef_[0] != 0].tolist()

        print(f"Best Model: {best_model}")
        print(f"Best F1 Score (CV): {best_score}")
        print(f"Training Accuracy: {accuracy}")
        print(f"Non-zero Features: {non_zero_features}")

        return best_model, best_score, non_zero_features
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, []

def supervised_feature_reduction(model, X_val, y_val):
    """
    Perform feature reduction using a trained logistic regression model with L1 regularization.

    Parameters:
    - model: The trained Logistic Regression model with L1 regularization.
    - X_val (DataFrame): The validation features.
    - y_val (Series): The validation target variable.

    Returns:
    - adjusted_model: The model with adjusted parameters.
    - important_features_names: Names of the predictive features.
    """
    assert isinstance(model, LogisticRegression), "The model must be a Logistic Regression model with L1 regularization."
    assert model.get_params()['penalty'] == 'l1', "The model must use L1 regularization."

    try:
        # Identify non-zero coefficients (features selected by L1 regularization)
        important_features_mask = model.coef_[0] != 0
        selected_features_with_duplicates = X_val.columns[important_features_mask]

        # Ensure important_features_names is unique, preserving order of first appearance
        important_features_names = list(dict.fromkeys(selected_features_with_duplicates))

        if not important_features_names:
            print("No features selected by L1 regularization. Cannot proceed with feature reduction.")
            # Fallback: use all columns from X_val if no features are selected, or handle as error
            # For now, returning None, None as per original behavior if reduction fails or leads to no features
            # Consider if a different fallback is desired, e.g., using X_val.columns or top N by original coef magnitude
            return None, None

        X_important = X_val.loc[:, important_features_names]

        # Retrain model on important features with the validation set
        # Create a new model instance for retraining to avoid modifying the original L1 model's state
        # if it's used elsewhere or if L1 properties are crucial for it.
        # However, the original PheCAP paper implies refitting the same model object.
        # For safety and clarity, let's assume a new instance or re-fitting is acceptable.
        # If strict refitting of the *exact same* model object (with L1) on reduced features is needed,
        # this part might need adjustment, but typical supervised reduction implies training a new model
        # (often without penalty or with a different one) on the selected subset.
        # Given the current structure, the original model object `model` is being refit.
        adjusted_model = LogisticRegression(penalty='l1', solver='liblinear', C=model.get_params()['C'], max_iter=model.get_params()['max_iter'])
        adjusted_model.fit(X_important, y_val.values.ravel())  # Convert y_val to NumPy array and use ravel()
        # The original code was: model.fit(X_important, y_val.values.ravel()); adjusted_model = model
        # This is changed to fit a new instance to ensure the returned model truly reflects the reduced feature space training.

        # Evaluate the adjusted model
        predictions = adjusted_model.predict_proba(X_important)[:, 1]
        ll = log_loss(y_val, predictions)

        print(f"Adjusted Model Log Loss: {ll}")
        print(f"Predictive Features (unique): {list(important_features_names)}")

        return adjusted_model, list(important_features_names)
    except Exception as e:
        print(f"An error occurred during feature reduction: {e}")
        return None, None

def fetch_feature_labels(phenotype, important_features_list, train=True, noisy_label=True, notes=True, nlp=True):
    """
    Fetch the corresponding labels for a list of important feature names based on the phenotype.

    Parameters:
    - phenotype (str): The phenotype for which to fetch feature labels.
    - important_features_list (list of tuples): List of important feature names and their types (e.g., [('itemid', 123), ('cui', 'C0020538')]).
    - train (bool): Whether the data is for training.
    - noisy_label (bool): Whether the data includes noisy labels.
    - notes (bool): Whether to include notes data types.
    - nlp (bool): Whether to include NLP processed data.

    Returns:
    - feature_labels_df (DataFrame): DataFrame containing the important feature names and their corresponding labels.
    """
    # Define the data types and potentially extend them based on notes and NLP
    df_types = ['DIAG', 'MEDS', 'PROC', 'CHART', 'LABS']
    if notes and nlp:
        df_types.extend(['RADNOTES', 'DSNOTES'])

    # Load dictionaries for all data types and concatenate them
    dictionaries_df = []
    for df_type in df_types:
        # TODO : change back to gcs
        dict_path = generate_dict_path(phenotype, train=train, label=noisy_label, df_type=df_type)
        feature_dict_df = read_gzip_or_parts(dict_path)
        feature_dict_df.columns = ['id', 'label']  # Standardize column names after loading
        feature_dict_df['id'] = feature_dict_df['id'].astype(str)
        dictionaries_df.append(feature_dict_df)

    concatenated_dictionaries_df = pd.concat(dictionaries_df, ignore_index=True)

    # Create a DataFrame from the list of important feature names and types
    important_features_df = pd.DataFrame(important_features_list, columns=['id'])
    important_features_df['id'] = important_features_df['id'].astype(str)

    # Perform a left join to fetch the corresponding labels
    feature_labels_df_with_duplicates = pd.merge(important_features_df, concatenated_dictionaries_df, on='id', how='left')

    # Ensure unique labels per feature ID by taking the first encountered label
    # Group by 'id' and take the first non-null label. If all are null for an id, it remains NaN.
    if not feature_labels_df_with_duplicates.empty:
        feature_labels_df = feature_labels_df_with_duplicates.groupby('id', as_index=False).first()
    else:
        feature_labels_df = pd.DataFrame(columns=['id', 'label']) # Ensure correct columns for empty case

    # Re-ensure the order of important_features_df if groupby changed it (it shouldn't if 'id' is unique in important_features_df)
    # However, if important_features_list itself had duplicates, the groupby on 'id' would already make them unique.
    # The `list(dict.fromkeys(important_features_list))` upstream in train_all_* should handle duplicates in input `important_features_list`.
    # If `important_features_df` is guaranteed to have unique IDs, then the order from .first() on groupby('id') should be fine
    # relative to the unique IDs, but not necessarily the order of `important_features_list` if that list had duplicates.
    # To be safe, and ensure output matches the order of (now unique) input features:
    # Create a mapping from the original (now unique) important_features_list
    unique_input_ids = pd.DataFrame(list(dict.fromkeys(important_features_list)), columns=['id'])
    unique_input_ids['id'] = unique_input_ids['id'].astype(str)

    if not feature_labels_df.empty:
        feature_labels_df = pd.merge(unique_input_ids, feature_labels_df, on='id', how='left')
    else: # If feature_labels_df was empty from the start or after groupby
        feature_labels_df = unique_input_ids.copy()
        feature_labels_df['label'] = pd.NA # Add label column with NAs

    return feature_labels_df

def generate_val_set_path(phenotype):
    """
    Generate a path for saving the validation set distribution for a given phenotype.
    """
    base_path = k_const.MODEL_PATH
    file_path = os.path.join(base_path, f"{phenotype}/{phenotype}_val_set_distribution.csv")
    return file_path

def generate_train_set_distribution_path(phenotype):
    """
    Generate a path for saving the training set distribution for a given phenotype.
    """
    base_path = k_const.MODEL_PATH
    file_path = os.path.join(base_path, f"{phenotype}/{phenotype}_train_set_distribution.csv")
    return file_path

def run_feature_reduction(model, X_val, y_val, method='l1'):
    """
    Modular helper to run supervised feature reduction.
    method: 'l1' for L1-based, 'lr' for standard logistic regression.
    Returns (adjusted_model, features)
    """
    if method == 'l1':
        return supervised_feature_reduction(model, X_val, y_val)
    elif method == 'lr':
        return supervised_feature_reduction_lr(model, X_val, y_val)
    else:
        raise ValueError(f"Unknown feature reduction method: {method}")

# Define the phenotype list in one place
PHENOTYPE_LIST = ['dementia'] # Add all your phenotypes here

def train_all_l1(feature_reduction=True, phenotypes=PHENOTYPE_LIST):
    models = {}
    for phenotype in phenotypes:
        y_train, X_train, _ = load_and_filter_train(phenotype)
        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)

        # Export training set distribution
        if not y_train.empty:
            y_train_flat = y_train['y'] if isinstance(y_train, pd.DataFrame) and 'y' in y_train.columns else y_train.squeeze()
            n_train = len(y_train_flat)
            n_train_0 = (y_train_flat == 0).sum()
            n_train_1 = (y_train_flat == 1).sum()
            pct_train_1 = round(n_train_1 / n_train * 100, 2) if n_train > 0 else 0.0
            train_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n_train],
                'y=0': [n_train_0],
                'y=1': [n_train_1],
                '% y=1': [pct_train_1]
            })
            train_dist_path = generate_train_set_distribution_path(phenotype)
            save_csv(train_dist_df, train_dist_path)
            print(f"Saved training set distribution for {phenotype} to {train_dist_path}")

        y_val, X_val = load_and_filter_val(phenotype, train_features=X_train.columns)
        # Export validation set distribution
        if not y_val.empty:
            y_val_flat = y_val['y'] if 'y' in y_val.columns else y_val.squeeze()
            n = len(y_val_flat)
            n_0 = (y_val_flat == 0).sum()
            n_1 = (y_val_flat == 1).sum()
            pct_1 = round(n_1 / n * 100, 2) if n > 0 else 0.0
            val_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n],      # Total samples
                'y=0': [n_0],       # Count of y=0
                'y=1': [n_1],       # Count of y=1
                '% y=1': [pct_1]   # Percentage of y=1
            })
            val_dist_path = generate_val_set_path(phenotype)
            save_csv(val_dist_df, val_dist_path)
            print(f"Saved validation set distribution for {phenotype} to {val_dist_path}")

        initial_model, _, _ = train_model(X_train, y_train, params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 1000]})
        current_model_for_evaluation = initial_model
        current_features = list(X_train.columns)

        if initial_model is None:
            print(f"Skipping feature reduction and saving for {phenotype} due to initial model training failure.")
            models[phenotype] = (None, [])
            continue

        if feature_reduction:
            if y_val.empty or X_val.empty:
                print(f"Skipping feature reduction for {phenotype} due to empty validation set.")
                # current_model_for_evaluation remains initial_model
                # current_features remain X_train.columns
            else:
                reduced_model_on_gold, features_from_reduction = run_feature_reduction(initial_model, X_val, y_val, method='l1')
                if reduced_model_on_gold is not None and features_from_reduction is not None and features_from_reduction:
                    current_model_for_evaluation = reduced_model_on_gold
                    current_features = features_from_reduction
                else:
                    print(f"Feature reduction failed or yielded no features for {phenotype}. Using initial model and its features.")
                    # current_model_for_evaluation remains initial_model
                    # current_features remain X_train.columns (or features from initial_model if it selected some)
                    # If initial_model is L1, its .coef_ would reflect features selected on X_train.
                    # For consistency, if reduction fails, we should use features of the model being evaluated.
                    if hasattr(initial_model, 'coef_'):
                        initial_model_feature_mask = initial_model.coef_[0] != 0
                        current_features = X_train.columns[initial_model_feature_mask].tolist()
                        if not current_features: # if L1 zeroed out everything
                            current_features = list(X_train.columns)

        unique_features = list(dict.fromkeys(current_features))
        if not unique_features:
            print(f"No features selected for {phenotype}. Skipping saving and storing model.")
            models[phenotype] = (None, []) # Store None if no features
            continue

        # Ensure coefs correspond to the unique_features list
        # The current_model_for_evaluation was trained on these unique_features (or a subset if L1/ElasticNet)
        if hasattr(current_model_for_evaluation, 'coef_'):
            raw_coefs = current_model_for_evaluation.coef_[0] if current_model_for_evaluation.coef_.ndim == 2 else current_model_for_evaluation.coef_

            # We need to align raw_coefs with unique_features carefully.
            # If current_model_for_evaluation is initial_model (L1 trained on X_train) and reduction was skipped/failed,
            # its coefficients correspond to X_train.columns. We need to select coefs for unique_features.
            # If current_model_for_evaluation is reduced_model_on_gold, its coefs correspond to features_from_reduction.
            # unique_features should be identical to features_from_reduction if reduction was successful.
            # If features_from_reduction was empty, this block isn't hit due to earlier checks.
            aligned_raw_coefs = raw_coefs # Coefficients already align with features_from_reduction (which is unique_features here)
            model_knows_features = unique_features

            if len(aligned_raw_coefs) == len(model_knows_features) and model_knows_features: # model_knows_features implies aligned_raw_coefs is also non-empty
                feature_coef_map = dict(zip(model_knows_features, aligned_raw_coefs))
                try:
                    odds_ratios = np.exp(np.array(aligned_raw_coefs, dtype=float)) # Ensure float for np.exp
                    feature_or_map = dict(zip(model_knows_features, odds_ratios))
                except OverflowError:
                    print(f"OverflowError calculating odds ratios for {phenotype}. Some coefficients might be too large.")
                    feature_or_map = {f: np.inf for f in model_knows_features} # or np.nan
            else:
                print(f"Warning: Coefficient alignment failed for {phenotype}. Lengths: aligned_raw_coefs={len(aligned_raw_coefs)}, model_knows_features={len(model_knows_features)}. Coefficients/ORs may be incorrect.")
                feature_coef_map = {f: None for f in unique_features} # Fallback based on unique_features from overall logic
                feature_or_map = {f: None for f in unique_features}
        else:
            print(f"Warning: current_model_for_evaluation has no coefficients. Using initial model features.")

        feature_labels = fetch_feature_labels(phenotype, unique_features) # Pass unique_features
        feature_labels['coefficient'] = feature_labels['id'].map(feature_coef_map)
        feature_labels['odds_ratio'] = feature_labels['id'].map(feature_or_map)

        # Sort by absolute coefficient magnitude and take top 30
        if 'coefficient' in feature_labels.columns:
            # Ensure 'coefficient' is numeric for calculating absolute value and sorting.
            feature_labels['coefficient_numeric'] = pd.to_numeric(feature_labels['coefficient'], errors='coerce')
            feature_labels['abs_coef'] = feature_labels['coefficient_numeric'].abs()
            feature_labels = feature_labels.sort_values(by='abs_coef', ascending=False, na_position='last')
            feature_labels = feature_labels.drop(columns=['abs_coef', 'coefficient_numeric']) # Clean up helper columns
        else:
            # Fallback: if coefficient column is missing, sort by ID or keep original order.
            print(f"Warning: 'coefficient' column not found for {phenotype} for sorting by magnitude. Falling back to sorting by ID.")
            if 'id' in feature_labels.columns:
                 feature_labels = feature_labels.sort_values(by='id')

        if len(feature_labels) > 30:
            feature_labels = feature_labels.head(30)

        path = generate_model_path(phenotype)
        feature_label_path = os.path.join(os.path.dirname(path), f"{phenotype}_selected_features.csv")
        save_csv(feature_labels, feature_label_path)
        print(f"Saved selected features for {phenotype} to {feature_label_path}")
        models[phenotype] = (current_model_for_evaluation, unique_features) # Store unique_features
    return models

def train_all_lr(feature_reduction=True, imputation_method='iterative', phenotypes=PHENOTYPE_LIST):
    models = {}
    fitted_imputers_cache = {} # Cache for fitted imputers per phenotype

    for phenotype in phenotypes:
        print(f"--- Training LR for Phenotype: {phenotype} with imputation: {imputation_method} ---")
        y_train, X_train, fitted_imputer = load_and_filter_train(phenotype, method=imputation_method) # Get fitted_imputer
        if fitted_imputer:
            fitted_imputers_cache[phenotype] = fitted_imputer # Store it
            print(f"Fitted imputer {type(fitted_imputer).__name__} for {phenotype} using method '{imputation_method}'.")
        else:
            print(f"No imputer was fitted for {phenotype} during training data load (possibly no NaNs or specific config). Proceeding without a stored imputer for validation.")

        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
        phenotype_snake = to_snake_case(phenotype)
        model_dir = os.path.join(MODEL_PATH, phenotype_snake, 'lr')
        os.makedirs(model_dir, exist_ok=True)
        root_dir = os.path.join(MODEL_PATH, phenotype_snake)
        os.makedirs(root_dir, exist_ok=True)

        # Export training set distribution (root only)
        if not y_train.empty:
            y_train_flat = y_train['y'] if isinstance(y_train, pd.DataFrame) and 'y' in y_train.columns else y_train.squeeze()
            n_train = len(y_train_flat)
            n_train_0 = (y_train_flat == 0).sum()
            n_train_1 = (y_train_flat == 1).sum()
            pct_train_1 = round(n_train_1 / n_train * 100, 2) if n_train > 0 else 0.0
            train_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n_train],
                'y=0': [n_train_0],
                'y=1': [n_train_1],
                '% y=1': [pct_train_1]
            })
            train_dist_path = os.path.join(root_dir, "train_set_distribution.csv")
            save_csv(train_dist_df, train_dist_path)
            print(f"Saved training set distribution for {phenotype} to {train_dist_path}")

        # Pass the fitted imputer to load_and_filter_val
        phenotype_specific_imputer = fitted_imputers_cache.get(phenotype, None)
        y_val, X_val = load_and_filter_val(phenotype, train_features=X_train.columns, 
                                         method=imputation_method, # method still passed for consistency if fitted_imputer is None
                                         fitted_imputer_obj=phenotype_specific_imputer)
        
        # Export validation set distribution (root only)
        if not y_val.empty:
            y_val_flat = y_val['y'] if 'y' in y_val.columns else y_val.squeeze()
            n = len(y_val_flat)
            n_0 = (y_val_flat == 0).sum()
            n_1 = (y_val_flat == 1).sum()
            pct_1 = round(n_1 / n * 100, 2) if n > 0 else 0.0
            val_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n],      # Total samples
                'y=0': [n_0],       # Count of y=0
                'y=1': [n_1],       # Count of y=1
                '% y=1': [pct_1]   # Percentage of y=1
            })
            val_dist_path = os.path.join(root_dir, "val_set_distribution.csv")
            save_csv(val_dist_df, val_dist_path)
            print(f"Saved validation set distribution for {phenotype} to {val_dist_path}")

        initial_model, _, initial_model_features = train_model_lr(X_train, y_train) # train_model_lr returns (model, accuracy, features_list)
        current_model_for_evaluation = initial_model
        current_features = initial_model_features # These are X_train.columns

        if initial_model is None:
            print(f"Skipping feature reduction and saving for {phenotype} due to initial model training failure.")
            models[phenotype] = (None, [])
            continue

        if feature_reduction:
            # For train_all_lr, the feature reduction is more complex now.
            # Stage 1: Top 50 features from initial_model (silver standard)
            if hasattr(initial_model, 'coef_'):
                abs_coefs_initial = np.abs(initial_model.coef_[0])
                sorted_indices_initial = np.argsort(abs_coefs_initial)[::-1] # Descending

                num_features_initial_model = len(X_train.columns)
                top_k_silver = min(50, num_features_initial_model) # Reverted to 50

                top_k_silver_features = X_train.columns[sorted_indices_initial[:top_k_silver]].tolist()
                top_k_silver_features = list(dict.fromkeys(top_k_silver_features)) # Ensure unique

                if not top_k_silver_features:
                    print(f"No features selected from initial_model for {phenotype}. Using initial model.")
                elif y_val.empty or X_val.empty:
                    print(f"Skipping ElasticNet feature reduction for {phenotype} due to empty validation set.")
                else:
                    print(f"Selected top {len(top_k_silver_features)} features from silver-standard model: {top_k_silver_features}")
                    X_val_top_k = X_val.reindex(columns=top_k_silver_features, fill_value=0) # Align X_val to these features

                    # Stage 2: Elastic-Net on gold standard (X_val_top_k, y_val)
                    elastic_net_cv_model = LogisticRegressionCV(
                        Cs=[0.01, 0.1, 1, 10, 50], # List of C values to try
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), # 5-fold stratified CV
                        penalty='elasticnet',
                        solver='saga',
                        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9], # List of l1_ratios to try
                        max_iter=2000, # Increased max_iter for saga
                        scoring='f1', # Changed from neg_log_loss to f1
                        n_jobs=-1, 
                        random_state=42,
                        class_weight='balanced',
                    )
                    
                    print(f"Starting LogisticRegressionCV for ElasticNet on {X_val_top_k.shape[1]} features optimizing for F1 score...")
                    try:
                        elastic_net_cv_model.fit(X_val_top_k, y_val.values.ravel())
                        reduced_model_on_gold = elastic_net_cv_model # The fitted model is the best one
                        best_c = reduced_model_on_gold.C_[0] if isinstance(reduced_model_on_gold.C_, np.ndarray) else reduced_model_on_gold.C_
                        best_l1_ratio = reduced_model_on_gold.l1_ratio_[0] if isinstance(reduced_model_on_gold.l1_ratio_, np.ndarray) else reduced_model_on_gold.l1_ratio_
                        print(f"Best ElasticNet params from CV: C={best_c}, l1_ratio={best_l1_ratio}")
                        elastic_net_selected_mask = reduced_model_on_gold.coef_[0] != 0
                        final_selected_gold_features = X_val_top_k.columns[elastic_net_selected_mask].tolist()
                        final_selected_gold_features = list(dict.fromkeys(final_selected_gold_features))

                        if final_selected_gold_features:
                            current_model_for_evaluation = reduced_model_on_gold
                            current_features = list(X_val_top_k.columns) 
                            print(f"ElasticNet (tuned) effectively selected {len(final_selected_gold_features)} features on gold data: {final_selected_gold_features}")
                            print(f"Model will expect all {len(current_features)} features it was fit on: {current_features[:5]}...etc")
                        else:
                            print(f"ElasticNet (tuned) selected 0 features for {phenotype} on gold data. Using initial model and its features.")
                    except Exception as e:
                        print(f"Error during ElasticNet LogisticRegressionCV for {phenotype}: {e}. Using initial model.")
            else:
                print(f"Initial model for {phenotype} has no coefficients. Using initial model features.")

        unique_features = list(dict.fromkeys(current_features))
        if not unique_features:
            print(f"No features selected for {phenotype}. Skipping saving and storing model.")
            models[phenotype] = (None, []) # Store None if no features
            continue

        # Ensure coefs correspond to the unique_features list
        if hasattr(current_model_for_evaluation, 'coef_'):
            raw_coefs = current_model_for_evaluation.coef_[0] if current_model_for_evaluation.coef_.ndim == 2 else current_model_for_evaluation.coef_
            model_feature_columns = unique_features 
            if model_feature_columns and len(raw_coefs) == len(model_feature_columns):
                temp_feature_map = dict(zip(model_feature_columns, raw_coefs))
                aligned_raw_coefs = [temp_feature_map.get(f) for f in unique_features]
            else:
                 print(f"Warning: Coefficient/feature name mismatch for {phenotype} when creating coef map. Coef len: {len(raw_coefs)}, Model features len: {len(model_feature_columns)}. Unique features for output: {len(unique_features)}")
                 if len(raw_coefs) == len(unique_features):
                    aligned_raw_coefs = raw_coefs 
                 else:
                    aligned_raw_coefs = [0] * len(unique_features) 
            if len(aligned_raw_coefs) == len(unique_features):
                feature_coef_map = dict(zip(unique_features, aligned_raw_coefs))
                try:
                    odds_ratios = np.exp(np.array(aligned_raw_coefs, dtype=float))
                    feature_or_map = dict(zip(unique_features, odds_ratios))
                except OverflowError:
                    print(f"OverflowError calculating odds ratios for {phenotype}. Some coefficients might be too large.")
                    feature_or_map = {f: np.inf for f in unique_features}
            else:
                print(f"Warning: Final coefficient alignment failed for {phenotype}. Coefficients/ORs will be None.")
                feature_coef_map = {f: None for f in unique_features}
                feature_or_map = {f: None for f in unique_features}
        else:
            print(f"Warning: current_model_for_evaluation has no coefficients. Using initial model features.")

        # Fetch feature labels and join with features
        feature_labels = fetch_feature_labels(phenotype, unique_features) # Pass unique_features
        feature_labels['coefficient'] = feature_labels['id'].map(feature_coef_map)
        feature_labels['odds_ratio'] = feature_labels['id'].map(feature_or_map)
        if 'coefficient' in feature_labels.columns and feature_labels['coefficient'].dtype.kind in 'fc':
            feature_labels['coefficient'] = feature_labels['coefficient'].round(4)
        if 'odds_ratio' in feature_labels.columns and feature_labels['odds_ratio'].dtype.kind in 'fc':
            feature_labels['odds_ratio'] = feature_labels['odds_ratio'].round(4)
        # Sort by absolute coefficient magnitude and take top 30
        if 'coefficient' in feature_labels.columns:
            feature_labels['coefficient_numeric'] = pd.to_numeric(feature_labels['coefficient'], errors='coerce')
            feature_labels['abs_coef'] = feature_labels['coefficient_numeric'].abs()
            feature_labels = feature_labels.sort_values(by='abs_coef', ascending=False, na_position='last')
            feature_labels = feature_labels.drop(columns=['abs_coef', 'coefficient_numeric']) # Clean up helper columns
        else:
            if 'id' in feature_labels.columns:
                 feature_labels = feature_labels.sort_values(by='id')
        if len(feature_labels) > 30:
            feature_labels = feature_labels.head(30)
        # Save as features.csv in the model subfolder
        feature_label_path = os.path.join(model_dir, 'features.csv')
        save_csv(feature_labels, feature_label_path)
        print(f"Saved selected features for LR model of {phenotype} to {feature_label_path}")
        models[phenotype] = (current_model_for_evaluation, unique_features) # Store unique_features
    return models

# Other learning modules:

def train_model_multiple(X, y, model_type, params=None):
    """
    Train various types of models with hyperparameter tuning using 10-fold cross-validation.

    Parameters:
    - X (DataFrame): The input features.
    - y (Series): The target variable.
    - model_type (str): Type of model to train ('logistic_l1', 'logistic_l2', 'random_forest', 'gbm', 'svm').
    - params (dict, optional): Dictionary containing model parameters specific to the model type.

    Returns:
    - best_model: The best model after tuning.
    - best_score: The best score from the cross-validation.
    """
    models = {
        'logistic_l1'  : LogisticRegression(penalty='l1', solver='liblinear'),
        'logistic_l2'  : LogisticRegression(penalty='l2', solver='liblinear'),
        'random_forest': RandomForestClassifier(),
        'gbm'          : GradientBoostingClassifier(),
        'svm'          : SVC(probability=True)
    }

    default_params = {
        'logistic_l1': {'C': [0.1, 1, 10], 'max_iter': [100, 200, 500]},
        'logistic_l2': {'C': [0.1, 1, 10], 'max_iter': [100, 200, 500]},
        'random_forest': {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20]},
        'gbm': {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2]},
        'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(models.keys())}.")

    model = models[model_type]
    if params is None:
        params = default_params[model_type]
    else:
        # Update the provided params with the default ones if not specified
        default_params[model_type].update(params)
        params = default_params[model_type]

    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=10, scoring='neg_log_loss')

    try:
        grid_search.fit(X, y.values.ravel())  # Convert y to NumPy array and use ravel()
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  # Convert from negative log loss to log loss

        print(f"Best Model: {best_model}")
        print(f"Best Log Loss Score: {best_score}")

        return best_model, best_score
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def supervised_feature_reduction_multiple(model, X_val, y_val):
    """
    Perform feature reduction using a trained model and evaluate the reduced model.

    Parameters:
    - model: The trained model, which should be one of the supported types with feature importance or coefficients.
    - X_val (DataFrame): The validation features.
    - y_val (Series): The validation target variable.

    Returns:
    - adjusted_model: The model retrained on the important features.
    - important_features_names: Names of the predictive features.
    """
    # Check for model attributes and extract important features
    if hasattr(model, 'coef_') and model.coef_.ndim == 1:
        # Logistic Regression or Linear SVM
        important_features = model.coef_ != 0
    elif hasattr(model, 'coef_') and model.coef_.ndim > 1:
        # Multiclass scenario
        important_features = np.any(model.coef_ != 0, axis=0)
    elif hasattr(model, 'feature_importances_'):
        # Random Forest or GBM
        important_features = model.feature_importances_ > 0
    else:
        raise ValueError("Model does not have a method for identifying important features.")

    important_features_names = X_val.columns[important_features]
    X_important = X_val.loc[:, important_features_names]

    try:
        # Retrain model on important features with the validation set
        model.fit(X_important, y_val)
        predictions = model.predict_proba(X_important)[:, 1]
        ll = log_loss(y_val, predictions)

        print(f"Adjusted Model Log Loss: {ll}")
        print(f"Predictive Features: {list(important_features_names)}")

        return model, list(important_features_names)
    except Exception as e:
        print(f"An error occurred during feature reduction: {e}")
        return None, None

def train_model_lr(X, y, max_iter=5000):
    """
    Train a standard (no penalty) logistic regression model using sklearn.

    Parameters:
    - X (DataFrame): The input features.
    - y (Series): The target variable.
    - max_iter (int): Maximum number of iterations for the solver. Default is 5000.

    Returns:
    - model: The fitted LogisticRegression model.
    - accuracy: Training accuracy on X, y.
    - features: List of all feature names (since no regularization is used).
    """
    print("Training standard (no penalty) logistic regression model.")
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=max_iter, class_weight='balanced')
    model.fit(X, y.values.ravel())
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    features = list(X.columns)
    print(f"Training Accuracy: {accuracy}")
    print(f"Number of features used: {len(features)}")
    return model, accuracy, features

def supervised_feature_reduction_lr(model, X_val, y_val, top_k=30, coef_threshold=None):
    """
    Perform feature reduction using a trained standard (no penalty) logistic regression model (lbfgs).

    Parameters:
    - model: The trained Logistic Regression model (penalty='none').
    - X_val (DataFrame): The validation features.
    - y_val (Series): The validation target variable.
    - top_k (int, optional): If set, select the top_k features by absolute coefficient magnitude.
    - coef_threshold (float, optional): If set, select features with abs(coef) >= threshold.

    Returns:
    - adjusted_model: The model retrained on the selected features.
    - important_features_names: Names of the selected features.
    """
    assert isinstance(model, LogisticRegression), "The model must be a Logistic Regression model."
    assert model.get_params()['penalty'] == None, "The model must use no penalty (penalty='none')."

    try:
        coefs = model.coef_[0]
        abs_coefs = np.abs(coefs)

        # Ensure X_val.columns has the same number of elements as coefs
        if len(X_val.columns) != len(coefs):
            print(f"Mismatch between number of columns in X_val ({len(X_val.columns)}) and model coefficients ({len(coefs)}). Skipping feature reduction.")
            # Attempt to use X_val.columns directly if it matches X_train, otherwise this is a critical error
            # For now, returning the original model and all its features as a fallback.
            # This situation ideally shouldn't happen if X_val is correctly aligned to X_train during loading.
            return model, list(X_val.columns) # Or handle error more strictly

        feature_names = X_val.columns

        if coef_threshold is not None:
            important_mask = abs_coefs >= coef_threshold
        elif top_k is not None:
            if top_k > len(abs_coefs):
                top_k = len(abs_coefs)
            # Get indices of top_k largest absolute coefficients
            top_k_indices = np.argsort(abs_coefs)[-top_k:]
            important_mask = np.zeros_like(abs_coefs, dtype=bool)
            important_mask[top_k_indices] = True
        else:
            important_mask = abs_coefs != 0
            if not important_mask.any():
                top_k_default = min(20, len(abs_coefs))
                if top_k_default > 0:
                    top_k_indices = np.argsort(abs_coefs)[-top_k_default:]
                    important_mask[top_k_indices] = True
                else:
                    print("No features to select (all coefficients are zero and no columns available). Returning original model.")
                    return model, list(feature_names) # Or an empty list if that's preferred

        selected_features_with_duplicates = feature_names[important_mask]

        # Ensure important_features_names is unique, preserving order of first appearance
        important_features_names = list(dict.fromkeys(selected_features_with_duplicates))

        if not important_features_names:
            print("No features selected after applying criteria. Returning original model and its features.")
            # Fallback to all features of X_val or an empty list if more appropriate
            return model, list(X_val.columns) if not X_val.empty else []

        X_important = X_val.loc[:, important_features_names]

        # Retrain a new model on the selected features
        adjusted_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=model.get_params().get('max_iter', 1000))
        adjusted_model.fit(X_important, y_val.values.ravel())

        # Evaluate the adjusted model
        predictions = adjusted_model.predict_proba(X_important)[:, 1]
        ll = log_loss(y_val, predictions)

        print(f"Adjusted Model Log Loss: {ll}")
        print(f"Predictive Features (unique): {list(important_features_names)}")

        return adjusted_model, list(important_features_names)
    except Exception as e:
        print(f"An error occurred during feature reduction (lbfgs): {e}")
        return None, None

def train_model_naive_bayes(X, y):
    """
    Train a Gaussian Naive Bayes model, tuning var_smoothing for F1 score.

    Parameters:
    - X (DataFrame): The input features.
    - y (Series): The target variable.

    Returns:
    - best_model: The best fitted GaussianNB model after GridSearchCV.
    - best_f1_score: The best F1 score from cross-validation.
    - features: List of all feature names.
    """
    print("Training Gaussian Naive Bayes model with GridSearchCV for F1 score (tuning var_smoothing).")
    
    param_grid_nb = {
        'var_smoothing': np.logspace(0, -9, num=100) # Wide range for var_smoothing
    }
    
    nb_model = GaussianNB()
    # Using StratifiedKFold for cross-validation, common for classification
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search_nb = GridSearchCV(estimator=nb_model, 
                                  param_grid=param_grid_nb, 
                                  cv=cv_stratified, 
                                  scoring='f1', 
                                  verbose=0, 
                                  n_jobs=-1)
    
    try:
        grid_search_nb.fit(X, y.values.ravel())
        best_model = grid_search_nb.best_estimator_
        best_f1_score = grid_search_nb.best_score_
        
        # Evaluate on training data with the best model
        predictions_train = best_model.predict(X)
        accuracy_train = accuracy_score(y, predictions_train)
        features = list(X.columns) # All features are considered by GaussianNB prior to any external selection

        print(f"Best GaussianNB Model (tuned for F1): {best_model}")
        print(f"Best F1 Score (CV for var_smoothing): {best_f1_score}")
        print(f"Training Accuracy (with best model): {accuracy_train}")
        print(f"Number of features initially considered by GaussianNB: {len(features)}")
        
        return best_model, best_f1_score, features

    except Exception as e:
        print(f"An error occurred during GaussianNB GridSearchCV: {e}")
        # Fallback to a default GaussianNB if tuning fails, though this might not be ideal
        # Or return None, None, None to indicate failure to tune
        default_model = GaussianNB()
        default_model.fit(X, y.values.ravel())
        predictions_default = default_model.predict(X)
        accuracy_default = accuracy_score(y, predictions_default)
        print("Fell back to default GaussianNB due to error.")
        return default_model, accuracy_default, list(X.columns) # Returning accuracy as score here, not F1

def supervised_feature_reduction_naive_bayes(model, X_val, y_val, top_k=30):
    """
    Perform feature reduction for a Naive Bayes approach using SelectKBest on validation data.
    Retrains a new GaussianNB model on the selected features from X_val.

    Parameters:
    - model: An initial trained GaussianNB model (potentially F1-tuned, used for context but not for selection scores here).
    - X_val (DataFrame): The validation features.
    - y_val (Series): The validation target variable.
    - top_k (int): Number of top features to select.

    Returns:
    - adjusted_model: A new GaussianNB model retrained on the selected features from X_val.
    - important_features_names: Names of the selected features.
    """
    if X_val.empty or y_val.empty:
        print("X_val or y_val is empty. Skipping Naive Bayes feature reduction.")
        return GaussianNB(), list(X_val.columns) if not X_val.empty else [] # Return a default NB and available cols

    if top_k > X_val.shape[1]:
        print(f"Warning: top_k ({top_k}) is greater than the number of features ({X_val.shape[1]}). Selecting all features.")
        top_k = X_val.shape[1]
    
    if top_k == 0:
        print("Warning: top_k is 0. No features will be selected. Returning a default NB model.")
        return GaussianNB(), []


    print(f"Performing feature selection for Naive Bayes using SelectKBest (k={top_k}) on validation data.")
    try:
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selector.fit(X_val, y_val.values.ravel())
        
        selected_features_mask = selector.get_support()
        important_features_names = X_val.columns[selected_features_mask].tolist()
        important_features_names = list(dict.fromkeys(important_features_names)) # Ensure unique

        if not important_features_names:
            print("No features selected by SelectKBest. Using all features from X_val for the new NB model.")
            important_features_names = list(X_val.columns)
            if not important_features_names: 
                 print("X_val has no columns. Cannot train a new Naive Bayes model.")
                 return GaussianNB(), []
        
        X_important_val = X_val.loc[:, important_features_names]

        # Retrain a new default GaussianNB model on the selected features from the validation set
        adjusted_model = GaussianNB() 
        adjusted_model.fit(X_important_val, y_val.values.ravel())
        
        # Evaluate the adjusted model (optional, for consistency)
        if not X_important_val.empty:
            predictions_proba = adjusted_model.predict_proba(X_important_val)
            y_val_aligned = y_val.loc[X_important_val.index]
            if predictions_proba.shape[1] > 1 : 
                 ll = log_loss(y_val_aligned, predictions_proba)
                 f1 = f1_score(y_val_aligned, adjusted_model.predict(X_important_val)) # also calculate F1 for this model
                 print(f"Adjusted Naive Bayes Model (on X_val subset): Log Loss={ll:.4f}, F1 Score={f1:.4f}")
            else: 
                 print("Adjusted Naive Bayes Model (on X_val subset): Log loss/F1 cannot be computed (likely single class in predictions/subset).")
        else:
            print("Adjusted Naive Bayes Model (on X_val subset): X_important_val is empty, cannot evaluate.")


        print(f"Selected {len(important_features_names)} features for Naive Bayes based on X_val: {list(important_features_names)}")
        return adjusted_model, list(important_features_names)

    except Exception as e:
        print(f"An error occurred during Naive Bayes feature reduction: {e}")
        return GaussianNB(), list(X_val.columns) if not X_val.empty else []


def train_all_naive_bayes(feature_reduction=True, phenotypes=PHENOTYPE_LIST):
    models = {}
    for phenotype in phenotypes:
        print(f"--- Training Naive Bayes for Phenotype: {phenotype} ---")
        y_train, X_train, _ = load_and_filter_train(phenotype)
        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
        phenotype_snake = to_snake_case(phenotype)
        model_dir = os.path.join(MODEL_PATH, phenotype_snake, 'naive_bayes')
        os.makedirs(model_dir, exist_ok=True)
        root_dir = os.path.join(MODEL_PATH, phenotype_snake)
        os.makedirs(root_dir, exist_ok=True)

        # Set distribution (root only)
        if not y_train.empty:
            y_train_flat = y_train['y'] if isinstance(y_train, pd.DataFrame) and 'y' in y_train.columns else y_train.squeeze()
            n_train = len(y_train_flat)
            n_train_0 = (y_train_flat == 0).sum()
            n_train_1 = (y_train_flat == 1).sum()
            pct_train_1 = round(n_train_1 / n_train * 100, 2) if n_train > 0 else 0.0
            train_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n_train],
                'y=0': [n_train_0],
                'y=1': [n_train_1],
                '% y=1': [pct_train_1]
            })
            train_dist_path = os.path.join(root_dir, "train_set_distribution.csv")
            save_csv(train_dist_df, train_dist_path)

        y_val, X_val = load_and_filter_val(phenotype, train_features=X_train.columns)
        # Export validation set distribution (root only)
        if not y_val.empty:
            y_val_flat = y_val['y'] if 'y' in y_val.columns else y_val.squeeze()
            n = len(y_val_flat); n_0 = (y_val_flat == 0).sum(); n_1 = (y_val_flat == 1).sum()
            pct_1 = round(n_1 / n * 100, 2) if n > 0 else 0.0
            val_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name], 'n_total': [n], 'y=0': [n_0], 'y=1': [n_1], '% y=1': [pct_1]
            })
            val_dist_path = os.path.join(root_dir, "val_set_distribution.csv")
            save_csv(val_dist_df, val_dist_path)

        initial_model, initial_model_f1_score, initial_model_features = train_model_naive_bayes(X_train, y_train)
        current_model_for_evaluation = initial_model
        current_features = initial_model_features # All X_train.columns initially
        print(f"Initial Naive Bayes model for {phenotype} trained with CV F1 score: {initial_model_f1_score:.4f}")

        if initial_model is None:
            print(f"Skipping feature reduction and saving for {phenotype} (Naive Bayes) due to initial model training failure.")
            models[phenotype] = (None, [])
            continue
        if feature_reduction:
            if y_val.empty or X_val.empty:
                print(f"Skipping Naive Bayes feature reduction for {phenotype} due to empty validation set.")
            else:
                num_features_to_select = min(50, X_val.shape[1])
                if num_features_to_select > 0:
                    # --- SelectKBest logic for feature importances ---
                    selector = SelectKBest(score_func=f_classif, k=num_features_to_select)
                    selector.fit(X_val, y_val.values.ravel())
                    selected_features_mask = selector.get_support()
                    selected_features = X_val.columns[selected_features_mask].tolist()
                    scores = selector.scores_[selected_features_mask]
                    # Save feature scores (with labels)
                    feature_score_df = pd.DataFrame({
                        'feature': selected_features,
                        'score': scores
                    })
                    feature_labels_df = fetch_feature_labels(phenotype, selected_features)
                    feature_score_df = feature_score_df.merge(feature_labels_df, left_on='feature', right_on='id', how='left')
                    feature_score_df = feature_score_df.drop_duplicates(subset=['feature'])
                    feature_score_df = feature_score_df.sort_values(by='score', ascending=False)
                    feature_score_path = os.path.join(model_dir, "feature_importances.csv")
                    save_csv(feature_score_df, feature_score_path)
                    print(f"Saved Naive Bayes feature scores to {feature_score_path}")
                    # --- End SelectKBest logic ---
                    reduced_model_on_gold, features_from_reduction = supervised_feature_reduction_naive_bayes(
                        initial_model, X_val, y_val, top_k=num_features_to_select
                    )
                    if reduced_model_on_gold is not None and features_from_reduction:
                        current_model_for_evaluation = reduced_model_on_gold
                        current_features = features_from_reduction
                        print(f"Naive Bayes model updated with {len(features_from_reduction)} features from validation set selection.")
                    else:
                        print(f"Naive Bayes feature reduction failed or yielded no features for {phenotype}. Using initial model and its features.")
                else:
                    print(f"Not enough features in X_val ({X_val.shape[1]}) to select for Naive Bayes feature reduction. Using initial model.")
        unique_features = list(dict.fromkeys(current_features))
        if not unique_features:
            print(f"No features selected for {phenotype} (Naive Bayes). Skipping saving and storing model.")
            models[phenotype] = (None, [])
            continue
        feature_labels_df = fetch_feature_labels(phenotype, unique_features)
        features_path = os.path.join(model_dir, "features.csv")
        save_csv(feature_labels_df, features_path)
        print(f"Saved selected features for Naive Bayes model of {phenotype} to {features_path}")
        models[phenotype] = (current_model_for_evaluation, unique_features)
        print(f"--- Finished Naive Bayes for Phenotype: {phenotype} ---")
    return models


def train_all_random_forest(feature_reduction=True, imputation_method='simple', phenotypes=PHENOTYPE_LIST):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    import numpy as np

    models = {}
    for phenotype in phenotypes:
        print(f"--- Training Random Forest for Phenotype: {phenotype} ---")
        y_train, X_train, _ = load_and_filter_train(phenotype)
        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
        phenotype_snake = to_snake_case(phenotype)
        model_dir = os.path.join(MODEL_PATH, phenotype_snake, 'random_forest')
        os.makedirs(model_dir, exist_ok=True)
        root_dir = os.path.join(MODEL_PATH, phenotype_snake)
        os.makedirs(root_dir, exist_ok=True)

        # Set distribution (root only)
        if not y_train.empty:
            y_train_flat = y_train['y'] if isinstance(y_train, pd.DataFrame) and 'y' in y_train.columns else y_train.squeeze()
            n_train = len(y_train_flat)
            n_train_0 = (y_train_flat == 0).sum()
            n_train_1 = (y_train_flat == 1).sum()
            pct_train_1 = round(n_train_1 / n_train * 100, 2) if n_train > 0 else 0.0
            train_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n_train],
                'y=0': [n_train_0],
                'y=1': [n_train_1],
                '% y=1': [pct_train_1]
            })
            train_dist_path = os.path.join(root_dir, "train_set_distribution.csv")
            save_csv(train_dist_df, train_dist_path)

        # Impute if needed (sklearn RF does not support NaNs)
        if X_train.isnull().any().any():
            print("Imputing missing values for Random Forest (method: {}):".format(imputation_method))
            X_train, _ = impute_features_data(X_train, method=imputation_method)

        # GridSearchCV for F1 optimization
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        }
        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train.values.ravel())
        best_rf = grid_search.best_estimator_
        print(f"Best RF params: {grid_search.best_params_}, Best F1: {grid_search.best_score_}")

        # Save initial feature importances (with labels)
        importances = best_rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        })
        feature_labels_df = fetch_feature_labels(phenotype, feature_importance_df['feature'].tolist())
        feature_importance_df = feature_importance_df.merge(feature_labels_df, left_on='feature', right_on='id', how='left')
        feature_importance_df = feature_importance_df.drop_duplicates(subset=['feature'])
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        feature_importance_path = os.path.join(model_dir, "feature_importances.csv")
        save_csv(feature_importance_df, feature_importance_path)
        print(f"Saved initial RF feature importances to {feature_importance_path}")

        # Gold standard feature selection on validation set
        y_val, X_val = load_and_filter_val(phenotype, train_features=X_train.columns)
        if X_val.isnull().any().any():
            print("Imputing missing values for Random Forest validation set (method: {}):".format(imputation_method))
            X_val, _ = impute_features_data(X_val, method=imputation_method)
        
        if feature_reduction:
            if not y_val.empty and not X_val.empty:
                print("Performing feature reduction on validation set for Random Forest.")
                # Temporarily fit on validation data to find important features
                rf_val = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **grid_search.best_params_)
                rf_val.fit(X_val, y_val.values.ravel())
                
                val_importances = rf_val.feature_importances_
                val_feature_importance_df = pd.DataFrame({'feature': X_val.columns, 'importance': val_importances})
                
                # Fetch labels and merge
                val_labels_df = fetch_feature_labels(phenotype, val_feature_importance_df['feature'].tolist())
                val_feature_importance_df = val_feature_importance_df.merge(val_labels_df, left_on='feature', right_on='id', how='left')
                val_feature_importance_df = val_feature_importance_df.drop_duplicates(subset=['feature']).sort_values(by='importance', ascending=False)
                
                top_n = min(30, len(val_feature_importance_df))
                top_features = val_feature_importance_df['feature'].iloc[:top_n].tolist()
                
                # Retrain final model on the training data with selected features
                print(f"Retraining final RF model on X_train using top {len(top_features)} features.")
                X_train_reduced = X_train[top_features]
                final_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **grid_search.best_params_)
                final_model.fit(X_train_reduced, y_train.values.ravel())
                
                # Save feature importance and selected features files
                feature_importance_path_final = os.path.join(model_dir, "feature_importances.csv")
                save_csv(val_feature_importance_df.head(top_n), feature_importance_path_final)
                features_path = os.path.join(model_dir, "features.csv")
                save_csv(val_feature_importance_df[['id', 'label']].head(top_n), features_path)
                
                models[phenotype] = (final_model, top_features)
            else:
                print("Skipping RF feature reduction due to empty validation set. Using model trained on all train features.")
                models[phenotype] = (best_rf, list(X_train.columns))
        else:
            models[phenotype] = (best_rf, list(X_train.columns))
    return models

def train_all_gradient_boosting(feature_reduction=True, imputation_method='simple', phenotypes=PHENOTYPE_LIST):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    import numpy as np

    models = {}
    for phenotype in phenotypes:
        print(f"--- Training Gradient Boosting for Phenotype: {phenotype} ---")
        y_train, X_train, _ = load_and_filter_train(phenotype)
        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
        phenotype_snake = to_snake_case(phenotype)
        model_dir = os.path.join(MODEL_PATH, phenotype_snake, 'gradient_boosting')
        os.makedirs(model_dir, exist_ok=True)
        root_dir = os.path.join(MODEL_PATH, phenotype_snake)
        os.makedirs(root_dir, exist_ok=True)

        # Set distribution (root only)
        if not y_train.empty:
            y_train_flat = y_train['y'] if isinstance(y_train, pd.DataFrame) and 'y' in y_train.columns else y_train.squeeze()
            n_train = len(y_train_flat)
            n_train_0 = (y_train_flat == 0).sum()
            n_train_1 = (y_train_flat == 1).sum()
            pct_train_1 = round(n_train_1 / n_train * 100, 2) if n_train > 0 else 0.0
            train_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n_train],
                'y=0': [n_train_0],
                'y=1': [n_train_1],
                '% y=1': [pct_train_1]
            })
            train_dist_path = os.path.join(root_dir, "train_set_distribution.csv")
            save_csv(train_dist_df, train_dist_path)

        # Impute if needed (sklearn GBM does not support NaNs)
        if X_train.isnull().any().any():
            print("Imputing missing values for Gradient Boosting (method: {}):".format(imputation_method))
            X_train, _ = impute_features_data(X_train, method=imputation_method)

        # GridSearchCV for F1 optimization
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
        }
        gbm = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train.values.ravel())
        best_gbm = grid_search.best_estimator_
        print(f"Best GBM params: {grid_search.best_params_}, Best F1: {grid_search.best_score_}")

        # Save initial feature importances (with labels)
        importances = best_gbm.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        })
        feature_labels_df = fetch_feature_labels(phenotype, feature_importance_df['feature'].tolist())
        feature_importance_df = feature_importance_df.merge(feature_labels_df, left_on='feature', right_on='id', how='left')
        feature_importance_df = feature_importance_df.drop_duplicates(subset=['feature'])
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        feature_importance_path = os.path.join(model_dir, "feature_importances.csv")
        save_csv(feature_importance_df, feature_importance_path)
        print(f"Saved initial GBM feature importances to {feature_importance_path}")

        # Gold standard feature selection on validation set
        y_val, X_val = load_and_filter_val(phenotype, train_features=X_train.columns)
        if X_val.isnull().any().any():
            print("Imputing missing values for Gradient Boosting validation set (method: {}):".format(imputation_method))
            X_val, _ = impute_features_data(X_val, method=imputation_method)
        if feature_reduction:
            if not y_val.empty and not X_val.empty:
                best_gbm_val = GradientBoostingClassifier(random_state=42, **grid_search.best_params_)
                best_gbm_val.fit(X_val, y_val.values.ravel())
                val_importances = best_gbm_val.feature_importances_
                val_feature_importance_df = pd.DataFrame({
                    'feature': X_val.columns,
                    'importance': val_importances
                })
                val_labels_df = fetch_feature_labels(phenotype, val_feature_importance_df['feature'].tolist())
                val_feature_importance_df = val_feature_importance_df.merge(val_labels_df, left_on='feature', right_on='id', how='left')
                val_feature_importance_df = val_feature_importance_df.drop_duplicates(subset=['feature'])
                val_feature_importance_df = val_feature_importance_df.sort_values(by='importance', ascending=False)
                top_n = min(30, len(val_feature_importance_df))
                top_features = val_feature_importance_df['feature'].iloc[:top_n].tolist()
                
                # Retrain final model on the training data with selected features
                print(f"Retraining final GBM model on X_train using top {len(top_features)} features.")
                X_train_reduced = X_train[top_features]
                final_model = GradientBoostingClassifier(random_state=42, **grid_search.best_params_)
                final_model.fit(X_train_reduced, y_train.values.ravel())
                
                # Save final importances
                feature_importance_path_final = os.path.join(model_dir, "feature_importances.csv")
                save_csv(val_feature_importance_df.head(top_n), feature_importance_path_final)
                print(f"Saved final GBM feature importances to {feature_importance_path_final}")
                
                # Save selected features (with labels)
                features_labels = val_labels_df[val_labels_df['id'].isin(top_features)].drop_duplicates(subset=['id'])
                features_path = os.path.join(model_dir, "features.csv")
                save_csv(features_labels, features_path)
                print(f"Saved selected features for GBM model of {phenotype} to {features_path}")
                
                models[phenotype] = (final_model, top_features)
            else:
                print("Skipping feature reduction due to empty validation set for GBM. Using model trained on all features.")
                models[phenotype] = (best_gbm, list(X_train.columns))
        else:
            models[phenotype] = (best_gbm, list(X_train.columns))
    return models

def train_all_knn_classifier(feature_reduction=True, imputation_method='simple', phenotypes=PHENOTYPE_LIST):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectKBest, f_classif
    import pandas as pd
    import numpy as np

    models = {}
    for phenotype in phenotypes:
        print(f"--- Training KNN Classifier for Phenotype: {phenotype} ---")
        y_train, X_train, _ = load_and_filter_train(phenotype)
        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
        phenotype_snake = to_snake_case(phenotype)
        model_dir = os.path.join(MODEL_PATH, phenotype_snake, 'knn_classifier')
        os.makedirs(model_dir, exist_ok=True)
        root_dir = os.path.join(MODEL_PATH, phenotype_snake)
        os.makedirs(root_dir, exist_ok=True)

        # Set distribution (root only)
        if not y_train.empty:
            y_train_flat = y_train['y'] if isinstance(y_train, pd.DataFrame) and 'y' in y_train.columns else y_train.squeeze()
            n_train = len(y_train_flat)
            n_train_0 = (y_train_flat == 0).sum()
            n_train_1 = (y_train_flat == 1).sum()
            pct_train_1 = round(n_train_1 / n_train * 100, 2) if n_train > 0 else 0.0
            train_dist_df = pd.DataFrame({
                'phenotype': [display_phenotype_name],
                'n_total': [n_train],
                'y=0': [n_train_0],
                'y=1': [n_train_1],
                '% y=1': [pct_train_1]
            })
            train_dist_path = os.path.join(root_dir, "train_set_distribution.csv")
            save_csv(train_dist_df, train_dist_path)

        # Always impute for KNN
        if X_train.isnull().any().any():
            print("Imputing missing values for KNN (method: {}):".format(imputation_method))
            X_train, _ = impute_features_data(X_train, method=imputation_method)

        # GridSearchCV for F1 optimization
        param_grid = {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
        }
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train.values.ravel())
        best_knn = grid_search.best_estimator_
        print(f"Best KNN params: {grid_search.best_params_}, Best F1: {grid_search.best_score_}")

        # Gold standard feature selection on validation set
        y_val, X_val = load_and_filter_val(phenotype, train_features=X_train.columns)
        if X_val.isnull().any().any():
            print("Imputing missing values for KNN validation set (method: {}):".format(imputation_method))
            X_val, _ = impute_features_data(X_val, method=imputation_method)
        if feature_reduction:
            top_n = min(30, X_val.shape[1])
            selector = SelectKBest(score_func=f_classif, k=top_n)
            selector.fit(X_val, y_val.values.ravel())
            selected_mask = selector.get_support()
            selected_features = X_val.columns[selected_mask].tolist()
            scores = selector.scores_[selected_mask]
            # Save feature scores (with labels)
            feature_score_df = pd.DataFrame({
                'feature': selected_features,
                'score': scores
            })
            feature_labels_df = fetch_feature_labels(phenotype, selected_features)
            feature_score_df = feature_score_df.merge(feature_labels_df, left_on='feature', right_on='id', how='left')
            feature_score_df = feature_score_df.drop_duplicates(subset=['feature'])
            feature_score_df = feature_score_df.sort_values(by='score', ascending=False)
            feature_score_path = os.path.join(model_dir, "feature_importances.csv")
            save_csv(feature_score_df, feature_score_path)
            print(f"Saved KNN feature scores to {feature_score_path}")
            # Save selected features (with labels)
            features_labels = feature_labels_df[feature_labels_df['id'].isin(selected_features)].drop_duplicates(subset=['id'])
            features_path = os.path.join(model_dir, "features.csv")
            save_csv(features_labels, features_path)
            print(f"Saved selected features for KNN model of {phenotype} to {features_path}")
            models[phenotype] = (best_knn, selected_features)
        else:
            models[phenotype] = (best_knn, list(X_train.columns))
    return models

def train_all_models(phenotypes=PHENOTYPE_LIST, imputation_method_lr='knn', imputation_method_tree='simple', imputation_method_knn='simple'):
    """
    Trains all model types for all phenotypes, saves outputs in models/{phenotype}/{model_type}/, evaluates, and aggregates metrics.
    Uses evaluate_all(models) for evaluation.
    """
    import os
    import pandas as pd
    from collections import defaultdict
    from scripts.utils import save_csv

    model_types = [
        ('lr', train_all_lr, {'imputation_method': imputation_method_lr}),
        ('naive_bayes', train_all_naive_bayes, {}),
        ('random_forest', train_all_random_forest, {'imputation_method': imputation_method_tree}),
        ('gradient_boosting', train_all_gradient_boosting, {'imputation_method': imputation_method_tree}),
        ('knn_classifier', train_all_knn_classifier, {'imputation_method': imputation_method_knn}),
    ]

    all_metrics = []
    all_models = defaultdict(dict)

    for model_type, train_func, extra_kwargs in model_types:
        print(f"\n=== Training {model_type} models for all phenotypes ===")
        models = train_func(feature_reduction=True, phenotypes=phenotypes, **extra_kwargs)
        all_models[model_type] = models
        # Evaluate all models using evaluate_all
        metrics_df = evaluate_all(models, model_type=model_type)
        if metrics_df is not None and not metrics_df.empty:
            # Add model_type and phenotype columns if not present
            if 'model_type' not in metrics_df.columns:
                metrics_df['model_type'] = model_type
            if 'Phenotype' not in metrics_df.columns and 'phenotype' in metrics_df.columns:
                metrics_df.rename(columns={'phenotype': 'Phenotype'}, inplace=True)
            # Save metrics for each phenotype in the appropriate directory
            for _, row in metrics_df.iterrows():
                phenotype = row['Phenotype']
                phenotype_snake = to_snake_case(phenotype)
                model_dir = os.path.join(MODEL_PATH, phenotype_snake, model_type)
                os.makedirs(model_dir, exist_ok=True)
                metrics_path = os.path.join(model_dir, 'model_performance.csv')
                row_df = pd.DataFrame([row])
                save_csv(row_df, metrics_path)
                print(f"Saved {model_type} performance metrics for {phenotype} to {metrics_path}")
            all_metrics.append(metrics_df)
        else:
            print(f"No metrics to aggregate for {model_type}.")

    # Aggregate all metrics into a single CSV
    if all_metrics:
        #overall_metrics_df = pd.concat(all_metrics, ignore_index=True)
        #overall_metrics_path = os.path.join(MODEL_PATH, "overall_model_performance.csv")
        #save_csv(overall_metrics_df, overall_metrics_path)
        print(f"\nSaved overall model performance metrics to {overall_metrics_path}")
    else:
        print("No metrics to aggregate.")
    return all_models



# You'll need to modify generate_model_path and add a save_model utility if they don't exist
# or handle model type distinction.
# For now, I'm focusing on the structure.
# The main script would then call:
# models_nb = train_all_naive_bayes(feature_reduction=True)
# evaluate_all(models_nb) # evaluate_all should work if models_nb has the same structure

#models = train_all_l1(feature_reduction=True)
import time



PHENOTYPES_DEMO = ["hypertension"]
PHENOTYPES_DEMO_2 = ["hypertension", "sle", "obesity"]

print("TASK COMPLETED")
#models_nb = train_all_naive_bayes(feature_reduction=True)
#evaluate_all(models_nb)
