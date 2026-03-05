import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import *
from scripts.constants import *
from scripts.model.feature_utils import aggregate_duplicate_columns, one_hot_encode_non_numeric, process_numerical_columns, impute_features_data
from scripts.utils import load_baseline
import re

## This module contains functions to evaluate the performance of the models on the test dataset
## Also includes evaluation of rule-based models and LLM models



def load_and_filter_test(phenotype, train_features, notes=True, nlp=True, train=False, noisy_label=True, corrector_column='los', method='zero'):
    """
    Load all pivot tables and filter them based on hadm_id for test data from shared paths.
    Ensure that all the features in the test set are the same as the training set.
    Y and hadm_id selection for test set come from a phenotype-specific index file.
    Prioritizes keeping all hadm_ids from the index file.

    Parameters:
    - phenotype (str): The phenotype to load.
    - train_features (list-like): The columns/features used in training (for alignment).
    - notes, nlp, train, noisy_label, corrector_column, method: Same as load_and_filter_val.

    Returns:
    - y_test (pd.DataFrame): The target variable for the test set.
    - X_test (pd.DataFrame): The test features, aligned to train_features.
    """
    # 1. Load hadm_ids and y for the test set from the phenotype-specific index file
    index_file_path = os.path.join(INDEXES_PATH, f"{phenotype}_indexes.csv")
    if not path_exists(index_file_path):
        print(f"[SKIPPING TEST DATA] Index file does not exist: {index_file_path}")
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
            raise ValueError(f"Index file {index_file_path} must contain a '{set_col_name}' column (or a recognized alternative like {alternatives}) to identify test records.")
    if 'y' not in index_df.columns:
        raise ValueError(f"Index file {index_file_path} must contain a 'y' column for the target variable.")

    test_index_df = index_df[index_df[set_col_name].astype(str).str.lower() == 'test']
    if not test_index_df.empty and test_index_df[id_col_name].duplicated().any():
        print(f"Warning: Duplicate '{id_col_name}' entries found in 'test' set of {index_file_path}. Keeping first occurrence.")
        test_index_df = test_index_df.drop_duplicates(subset=[id_col_name], keep='first')
        
    if test_index_df.empty:
        print(f"No 'test' entries found in '{set_col_name}' column of {index_file_path} for phenotype {phenotype}.")
        empty_y = pd.DataFrame(columns=['y'], index=pd.Index([], name=id_col_name))
        empty_X = pd.DataFrame(columns=train_features, index=pd.Index([], name=id_col_name))
        return empty_y, empty_X

    indexed_hadm_ids = pd.Index(test_index_df[id_col_name].unique(), name=id_col_name)
    y_test_initial = test_index_df[[id_col_name, 'y']].set_index(id_col_name).reindex(indexed_hadm_ids)
    print(f"Initial set from index: {len(indexed_hadm_ids)} test hadm_ids for phenotype {phenotype}.")

    # 2. Load shared baseline features and corrector
    try:
        features_from_shared_baseline, corrector_from_shared_baseline = load_baseline(
            phenotype, train=False, noisy_label=noisy_label, use_shared_paths=True
        )
    except ValueError as e:
        print(f"Error loading shared baseline for {phenotype}: {e}. Check if shared baseline file exists and is correctly formatted.")
        empty_X = pd.DataFrame(columns=train_features, index=indexed_hadm_ids)
        return y_test_initial, empty_X

    # 3. Core hadm_id Set Definition
    core_hadm_ids = indexed_hadm_ids.intersection(features_from_shared_baseline.index).intersection(corrector_from_shared_baseline.index)
    if core_hadm_ids.empty:
        print(f"No common hadm_ids after intersecting index file with baseline and corrector for {phenotype}. Returning empty DataFrames.")
        empty_y = pd.DataFrame(columns=['y'], index=pd.Index([], name=id_col_name))
        empty_X = pd.DataFrame(columns=train_features, index=pd.Index([], name=id_col_name))
        return empty_y, empty_X

    y_test = y_test_initial.reindex(core_hadm_ids) # Filter y_test to core_hadm_ids
    print(f"Core set: {len(core_hadm_ids)} test hadm_ids after ensuring presence in baseline/corrector.")
    print(f"Shape of y_test after filtering to core_hadm_ids: {y_test.shape}")

    # 4. Prepare Baseline for Merging (reindexed to core_hadm_ids)
    baseline_features_core = features_from_shared_baseline.reindex(core_hadm_ids)
    corrector_test_core = corrector_from_shared_baseline.reindex(core_hadm_ids)

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
        return y_test, pd.DataFrame(columns=train_features, index=core_hadm_ids)

    temp_combined_features = pd.concat(all_feature_dfs, axis=1, join='outer')
    
    # 8. Final Reindex to core_hadm_ids
    combined_test_features = temp_combined_features.reindex(core_hadm_ids)
    print(f"Shape of combined_test_features after outer concat and final reindex to core_hadm_ids: {combined_test_features.shape}")

    # 9. Perform processing steps:
    combined_test_features = aggregate_duplicate_columns(combined_test_features)
    combined_test_features = one_hot_encode_non_numeric(combined_test_features)

    missing_cols = [col for col in train_features if col not in combined_test_features.columns]
    for col in missing_cols:
        combined_test_features[col] = 0
    combined_test_features = combined_test_features.reindex(columns=train_features)
    print(f"Shape of combined_test_features after OHE and aligning with train_features: {combined_test_features.shape}")

    # Impute missing values. For the test set, we ideally should use the imputer
    # fitted on the training set. This needs to be passed into this function.
    # For now, if no imputer is passed, it will fit a new one based on 'method'.
    # We will pass combined_test_features for imputation first, then add 'los' and 'admissions_number'.
    combined_features_imputed, _ = impute_features_data(combined_test_features, method=method) # Potentially pass fitted imputer here in future

    # Prepare 'los' and 'admissions_number' for normalization from corrector_test_core
    # These are added *after* imputation to ensure they are present for process_numerical_columns
    # and are not dropped by an imputer (if one were used from training) that wasn't trained on them.
    los_admissions_for_norm = corrector_test_core[['los', 'admissions_number']].astype(float).reindex(combined_features_imputed.index)

    if 'los' not in combined_features_imputed.columns and 'los' in los_admissions_for_norm.columns:
        combined_features_imputed['los'] = los_admissions_for_norm['los']
    if 'admissions_number' not in combined_features_imputed.columns and 'admissions_number' in los_admissions_for_norm.columns:
        combined_features_imputed['admissions_number'] = los_admissions_for_norm['admissions_number']
    
    combined_features_processed = process_numerical_columns(combined_features_imputed, corrector_column)
    print(f"Shape of combined_features after numerical processing: {combined_features_processed.shape}") 
    
    final_test_features = combined_features_processed.reindex(columns=train_features, fill_value=0)
    
    # Final alignment of y_test and features by index
    common_final_ids = core_hadm_ids.intersection(final_test_features.index).sort_values()
    
    final_test_features = final_test_features.loc[common_final_ids]
    final_y_test = y_test.loc[common_final_ids] # y_test was already on core_hadm_ids, further filtering if final_test_features dropped some
    
    print(f"Final shape of y_test: {final_y_test.shape}, final_test_features: {final_test_features.shape}")
    if not final_test_features.index.equals(final_y_test.index):
        print("Warning: Index mismatch between final features and y_test. Aligning them now.") # Should be handled by common_final_ids

    if final_test_features.empty and not final_y_test.empty:
        print(f"Warning: final_test_features is empty but final_y_test is not for {phenotype}. Check data pipeline.")
    elif not final_test_features.empty and final_y_test.empty:
        print(f"Warning: final_test_features is not empty but final_y_test is for {phenotype}. Check data pipeline.")
    return final_y_test, final_test_features

def evaluate_classification_model(model, X_test, y_test, phenotype):
    """
    Evaluate a classification model using various performance metrics and optionally save the results to a CSV file.

    Parameters:
    - model: Trained classification model.
    - X_test (DataFrame): Input features for the model.
    - y_test (Series): True labels.
    - phenotype (str): The phenotype being evaluated.

    Returns:
    - metrics_df (DataFrame): DataFrame containing various performance metrics.
    """
    # Handle empty y_test: if y_test is empty, metrics cannot be calculated.
    if y_test.empty:
        print(f"Warning: y_test is empty for {phenotype}. Returning NaN for metrics.")
        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
        metrics = {
            'Phenotype': [display_phenotype_name],
            'Accuracy': [np.nan],
            'Precision': [np.nan],
            'Recall': [np.nan],
            'Specificity': [np.nan],
            'F1 Score': [np.nan],
            'TP': [np.nan],
            'TN': [np.nan],
            'FP': [np.nan],
            'FN': [np.nan],
            'Baseline Accuracy': [np.nan]
        }
        return pd.DataFrame(metrics)

    # Calculate Baseline Accuracy (predicting the majority class)
    # Ensure y_test is a Series for mode() and comparison
    y_true_series = y_test['y'] if isinstance(y_test, pd.DataFrame) and 'y' in y_test.columns else y_test.squeeze()
    display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
    if y_true_series.empty:
        # This case should ideally be caught by the y_test.empty check above, but as a safeguard:
        baseline_accuracy = np.nan
    else:
        majority_class = y_true_series.mode()
        if not majority_class.empty:
            majority_class = majority_class[0]
            baseline_accuracy = (y_true_series == majority_class).mean()
        else: # Should not happen if y_true_series is not empty
            baseline_accuracy = np.nan
            
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Ensure cm has 4 values (tn, fp, fn, tp) for binary classification
    # If only one class is present in y_test or y_pred, cm might be smaller
    if cm.size == 1: # Only one class predicted and present (e.g., all TN or all TP)
        if y_test.nunique() == 1:
            if y_test.iloc[0] == 0: # All true negatives
                tn = cm[0][0]
                fp = 0
                fn = 0
                tp = 0
            else: # All true positives
                tn = 0
                fp = 0
                fn = 0
                tp = cm[0][0]
        else: # Should not happen if y_test has variation and model predicts single class
            # Defaulting to 0 for safety, but this case indicates an issue or edge case
            tn, fp, fn, tp = 0, 0, 0, 0 
            print(f"Warning: Confusion matrix for {phenotype} has unexpected size 1 with varied y_test. Check predictions.")
    elif cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else: # Should not happen in binary classification
        tn, fp, fn, tp = 0,0,0,0 # Or handle as an error
        print(f"Warning: Confusion matrix for {phenotype} has unexpected shape: {cm.shape}. Setting TP/TN/FP/FN to 0.")

    # Convert the classification report to a DataFrame for easier handling
    # classification_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

    metrics = {
        'Phenotype': [display_phenotype_name],
        'Accuracy': [accuracy_score(y_test, y_pred)],
        'Precision': [precision_score(y_test, y_pred, zero_division=0)],
        'Recall': [recall_score(y_test, y_pred, zero_division=0)], # Added zero_division for recall too
        'Specificity': [tn / (tn + fp) if (tn + fp) > 0 else 0],
        'F1 Score': [f1_score(y_test, y_pred, zero_division=0)], # Added zero_division for F1
        'TP': [tp],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'Baseline Accuracy': [baseline_accuracy] # Add baseline accuracy here
        # 'Confusion Matrix': [cm.tolist()],  # Remove this line
        #'Classification Report': classification_report_df
    }

    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics)

    # Optionally, you can concatenate the classification_report_df with metrics_df if needed
    # For simplicity, we're not doing that here since it's not clear how you want to handle the report

    print(f'[SAVING INDIVIDUAL PERFORMANCE METRICS FOR {phenotype}]')

    # Note: Persisting the metrics CSV is handled by the caller (e.g., train_all_models)
    # to guarantee the file is written inside the correct phenotype/model_type folder.

    return metrics_df

def evaluate_all(models, model_type=None):
    """
    Evaluate all models using the test set and return a DataFrame of evaluation metrics.
    Input is a dictionary containing phenotype as key and (model, features) as item:
    {'phenotype1': (model1, features1), 'phenotype2': (model2, features2),...}

    Parameters
    ----------
    models : dict
        Mapping phenotype → (fitted_model, feature_list).
    model_type : str, optional
        Name of the model family being evaluated (e.g. 'lr', 'naive_bayes'). If provided, the
        function will immediately persist the resulting *model_performance.csv* for each phenotype
        inside  models/<phenotype>/<model_type>/ .  When omitted the function behaves as before
        and only returns the aggregated DataFrame.
    """
    # Helper here to avoid circular import with modeling.py
    def _to_snake_case(name: str) -> str:
        name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        name = name.replace(' ', '_').replace('-', '_')
        name = re.sub(r'_+', '_', name)
        return name.lower()

    evaluation_results = []

    for phenotype, model_features in models.items():
        # Unpack model and features
        if isinstance(model_features, tuple) and len(model_features) == 2:
            model, features = model_features
        else:
            model = model_features
            features = None

        display_phenotype_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)

        if model is None or features is None or len(features) == 0:
            print(f"Skipping evaluation for {display_phenotype_name} due to missing model or features.")
            continue
        try:
            y_test, X_test = load_and_filter_test(phenotype, train_features=features)
            metrics_df = evaluate_classification_model(model, X_test, y_test, phenotype)

            # Persist immediately if model_type is given
            if model_type:
                phenotype_snake = _to_snake_case(display_phenotype_name)
                model_dir = os.path.join(MODEL_PATH, phenotype_snake, model_type)
                os.makedirs(model_dir, exist_ok=True)
                metrics_path = os.path.join(model_dir, 'model_performance.csv')
                save_csv(metrics_df, metrics_path)
                print(f"[evaluate_all] Saved {model_type} performance metrics for {display_phenotype_name} to {metrics_path}")

            evaluation_results.append(metrics_df)
        except Exception as e:
            print(f"Error evaluating model for {phenotype}: {e}")

    if not evaluation_results:
        print("No evaluation results to aggregate.")
        return pd.DataFrame()

    aggregated_df = pd.concat(evaluation_results, ignore_index=True)

    # Save overall metrics
    try:
        overall_path = generate_overall_model_path()
        save_csv(aggregated_df, overall_path)
        print(f"[evaluate_all] Saved overall evaluation metrics to {overall_path}")
    except Exception as e:
        print(f"Error saving the overall evaluation results: {e}")

    return aggregated_df

def plot_sine_function():
    # Set the style to a seaborn darkgrid style
    sns.set_style("darkgrid")

    # Generate some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(x, y, label='sin(x)', color='skyblue')

    # Set the labels and title
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('sin(x)', fontsize=14)
    ax.set_title('A Simple Sin Function Plot', fontsize=16)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_roc_curve(y_test, y_score, phenotype):
    """
    Plot ROC curve for a binary classification problem.

    Parameters:
    - y_test (array): True binary labels.
    - y_score (array): Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
    - phenotype (str): The phenotype being evaluated.
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {phenotype}')
    plt.legend(loc="lower right")
    plt.show()
